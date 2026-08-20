import unittest

from dataclasses import replace
from datetime import date

import pandas as pd

from src.application.decision_support import build_decision_support
from src.application.forecast_service import (
    ForecastProductContext,
    build_forecast_product_context,
)
from src.models.inference import InferenceResult, InferenceWarning
from src.models.modeling_v2_common import load_modeling_v2_config
from src.models.prediction_intervals_v2 import PredictionIntervalResult
from src.visualization.dashboard_data import (
    DashboardContext,
    DashboardLineage,
    TerritoryValidationMetrics,
    load_canonical_validation_bundle,
    load_gold_history,
)


class TestDecisionSupport(unittest.TestCase):
    """La regla estacional depende del point y del historico, no del intervalo."""

    @staticmethod
    def _product(
        values_by_year: dict[int, float],
        *,
        forecast: float = 550.0,
        target_month_id: str = "2026-09",
        interval_margin: float = 100.0,
        covid_years: set[int] | None = None,
        provisional_years: set[int] | None = None,
        warnings: tuple[InferenceWarning, ...] = (),
        extra_rows: list[dict[str, object]] | None = None,
    ) -> ForecastProductContext:
        covid_years = set() if covid_years is None else covid_years
        provisional_years = (
            set() if provisional_years is None else provisional_years
        )
        rows = [
            {
                "territory_id": "ES-PROV-01",
                "territory_name": "Provincia A",
                "month_id": f"{year}-09",
                "date_month": pd.Timestamp(f"{year}-09-01"),
                "overnight_stays_total": value,
                "covid_period": year in covid_years,
                "data_status": (
                    "provisional"
                    if year in provisional_years
                    else "final_or_not_marked_provisional"
                ),
                "is_provisional": year in provisional_years,
                "coverage_quality": "high",
            }
            for year, value in values_by_year.items()
        ]
        if extra_rows:
            rows.extend(extra_rows)
        history = pd.DataFrame.from_records(rows).sort_values("date_month")
        target = pd.Period(target_month_id, freq="M")
        reference = target - 12
        inference = InferenceResult(
            territory_id="ES-PROV-01",
            territory_name="Provincia A",
            target_month_id=str(target),
            business_origin_month_id=str(target - 1),
            latest_available_month_id=str(target - 3),
            forecast_horizon_months=1,
            effective_model_horizon_steps=3,
            predicted_overnight_stays_total=forecast,
            selected_model_id="holt_winters_additive_damped_v1",
            selection_status="provisional_validation_champion",
            actual_model_used="holt_winters_additive_damped_v1",
            fallback_used=False,
            fallback_reason="not_used",
            baseline_reference_month_id=str(reference),
            baseline_prediction=forecast,
            baseline_reference_is_provisional=bool(warnings),
            ets_raw_prediction=forecast,
            clipping_applied=False,
            training_start="2017-01",
            training_end=str(target - 3),
            training_rows=114,
            source_snapshot_id="snapshot",
            pipeline_run_id="run",
            data_version="gold-v1",
            operational_status="forecast_ready",
            warnings=warnings,
        )
        interval = PredictionIntervalResult(
            territory_id="ES-PROV-01",
            target_month_id=str(target),
            point_prediction=forecast,
            lower=max(0.0, forecast - interval_margin),
            upper=forecast + interval_margin,
            nominal_level=0.8,
            method_id=(
                "operational_prequential_scaled_absolute_residual_interval_v1"
            ),
            calibration_scores_n=100,
            calibration_origins_n=20,
            calibration_max_target_month_id=str(target - 3),
            calibration_quantile=0.4,
            interval_available=True,
            unavailable_reason=None,
        )
        dashboard = DashboardContext(
            territory_id="ES-PROV-01",
            territory_name="Provincia A",
            history=history.reset_index(drop=True),
            validation_metrics=TerritoryValidationMetrics(
                territory_id="ES-PROV-01",
                territory_name="Provincia A",
                validation_mae=10.0,
                validation_rmse=12.0,
                validation_wape_pct=20.0,
                validation_bias=1.0,
                validation_rows=35,
                selected_model_id="holt_winters_additive_damped_v1",
                selection_status="provisional_validation_champion",
                evidence_scope="canonical_rolling_validation",
                prediction_column="operational_prediction",
                candidate_available_rows=35,
                availability_fallback_rows=0,
            ),
            lineage=DashboardLineage(
                operational_source_snapshot_id="snapshot",
                operational_pipeline_run_id="run",
                operational_data_version="gold-v1",
                operational_dataset_path="data/gold/gold.parquet",
                evaluation_artifact_path="data/model_outputs/ets.parquet",
                evaluation_metadata_path="data/metadata/ets.yml",
                evaluation_artifact_sha256="a" * 64,
                evaluation_metadata_sha256="b" * 64,
                evaluation_logical_prediction_sha256="c" * 64,
                evaluation_generator_commit_sha="d" * 40,
                evaluation_github_run_id="123",
                evaluation_config_sha256="e" * 64,
                evaluation_source_snapshot_ids=("evaluation-snapshot",),
                evaluation_data_versions=("evaluation-v1",),
                evaluation_scope="canonical_rolling_validation",
            ),
        )
        return ForecastProductContext(
            as_of_date=date(2026, 8, 20),
            forecast_origin_month_id="2026-08",
            forecast=inference,
            prediction_interval=interval,
            dashboard=dashboard,
        )

    def setUp(self) -> None:
        self.reference = {
            year: float((year - 2011) * 100) for year in range(2012, 2022)
        }

    def test_low_usual_and_high_are_distinct(self) -> None:
        cases = ((50.0, "low"), (550.0, "usual"), (1_100.0, "high"))

        for forecast, expected in cases:
            with self.subTest(forecast=forecast):
                support = build_decision_support(
                    self._product(self.reference, forecast=forecast)
                )
                self.assertEqual(support.activity_level, expected)
                self.assertEqual(support.historical_sample_size, 10)

    def test_q25_and_q75_are_included_in_usual(self) -> None:
        cases = (
            (324.99, "low"),
            (325.0, "usual"),
            (775.0, "usual"),
            (775.01, "high"),
        )

        for forecast, expected in cases:
            with self.subTest(forecast=forecast):
                support = build_decision_support(
                    self._product(self.reference, forecast=forecast)
                )
                self.assertEqual(support.historical_q25, 325.0)
                self.assertEqual(support.historical_q75, 775.0)
                self.assertEqual(support.activity_level, expected)

    def test_interval_width_does_not_change_decision_support(self) -> None:
        narrow = self._product(
            self.reference,
            forecast=550.0,
            interval_margin=10.0,
        )
        wide = self._product(
            self.reference,
            forecast=550.0,
            interval_margin=1_000.0,
        )

        first = build_decision_support(narrow)
        second = build_decision_support(wide)
        self.assertNotEqual(
            narrow.prediction_interval.upper,
            wide.prediction_interval.upper,
        )
        self.assertEqual(first.activity_level, second.activity_level)
        self.assertEqual(
            first.historical_percentile_pct,
            second.historical_percentile_pct,
        )
        self.assertEqual(first.action_guidance, second.action_guidance)

    def test_exclusions_and_recent_window_are_traceable(self) -> None:
        values = {year: float(year) for year in range(2012, 2026)}
        support = build_decision_support(
            self._product(
                values,
                forecast=2024.0,
                covid_years={2020, 2021},
                provisional_years={2025},
            )
        )

        self.assertEqual(support.historical_sample_size, 10)
        self.assertEqual(support.excluded_covid_month_ids, ("2020-09", "2021-09"))
        self.assertEqual(support.excluded_provisional_month_ids, ("2025-09",))
        self.assertEqual(support.omitted_older_month_ids, ("2012-09",))
        self.assertEqual(
            support.rule_id,
            "seasonal_q25_q75_last_10_final_non_covid_v1",
        )

    def test_provisional_forecast_warning_is_preserved(self) -> None:
        warning = InferenceWarning(
            code="provisional_reference_data",
            message="La referencia es provisional.",
        )
        support = build_decision_support(
            self._product(self.reference, warnings=(warning,))
        )

        self.assertIn(
            "provisional_reference_data",
            {item.code for item in support.warnings},
        )

    def test_same_month_gap_is_reported_without_using_other_months(self) -> None:
        values = dict(self.reference)
        values.pop(2016)

        support = build_decision_support(self._product(values))

        self.assertIn(2016, support.missing_comparison_years)
        self.assertTrue(
            all(month_id.endswith("-09") for month_id in support.comparison_month_ids)
        )
        self.assertIn(
            "historical_comparison_gaps",
            {item.code for item in support.warnings},
        )

    def test_small_sample_is_not_classified(self) -> None:
        support = build_decision_support(
            self._product(
                {2018: 100.0, 2019: 200.0, 2020: 300.0, 2021: 400.0}
            )
        )

        self.assertEqual(support.activity_level, "insufficient")
        self.assertIsNone(support.action_guidance)
        self.assertIn(
            "insufficient_seasonal_history",
            {item.code for item in support.warnings},
        )

    def test_flat_history_and_outside_range_are_explicit(self) -> None:
        flat = {year: 100.0 for year in range(2012, 2022)}
        usual = build_decision_support(self._product(flat, forecast=100.0))
        high = build_decision_support(self._product(flat, forecast=101.0))

        self.assertEqual(usual.activity_level, "usual")
        self.assertEqual(usual.historical_percentile_pct, 50.0)
        self.assertEqual(high.activity_level, "high")
        self.assertIn(
            "forecast_outside_historical_range",
            {item.code for item in high.warnings},
        )

    def test_validation_metrics_do_not_drive_activity_level(self) -> None:
        product = self._product(self.reference, forecast=550.0)
        changed = replace(
            product,
            dashboard=replace(
                product.dashboard,
                validation_metrics=replace(
                    product.dashboard.validation_metrics,
                    validation_wape_pct=999.0,
                ),
            ),
        )

        first = build_decision_support(product)
        second = build_decision_support(changed)
        self.assertEqual(first.activity_level, second.activity_level)
        self.assertEqual(
            first.historical_percentile_pct,
            second.historical_percentile_pct,
        )


class TestDecisionSupportB5Integration(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_modeling_v2_config()
        cls.bundle = load_canonical_validation_bundle(cls.config)
        cls.gold = load_gold_history(cls.config)
        cls.product = build_forecast_product_context(
            "ES-PROV-01",
            as_of_date="2026-08-20",
            gold=cls.gold,
            canonical_bundle=cls.bundle,
            config=cls.config,
        )
        cls.historical_product = build_forecast_product_context(
            "ES-PROV-01",
            as_of_date="2023-01-20",
            history_months=24,
            gold=cls.gold,
            canonical_bundle=cls.bundle,
            config=cls.config,
        )

    def test_real_araba_signal_uses_point_not_interval_as_historical_position(
        self,
    ) -> None:
        support = build_decision_support(self.product)

        self.assertEqual(support.activity_level, "high")
        self.assertEqual(support.historical_sample_size, 10)
        self.assertEqual(support.historical_percentile_pct, 90.0)
        self.assertEqual(
            support.rule_id,
            "seasonal_q25_q75_last_10_final_non_covid_v1",
        )
        self.assertTrue(self.product.prediction_interval.interval_available)

    def test_historical_signal_uses_only_comparisons_known_at_cutoff(self) -> None:
        support = build_decision_support(self.historical_product)
        cutoff = self.historical_product.forecast.latest_available_month_id

        self.assertTrue(support.comparison_month_ids)
        self.assertTrue(
            all(month_id <= cutoff for month_id in support.comparison_month_ids)
        )
        self.assertEqual(max(support.comparison_month_ids), "2022-02")
        self.assertEqual(support.activity_level, "insufficient")


if __name__ == "__main__":
    unittest.main()
