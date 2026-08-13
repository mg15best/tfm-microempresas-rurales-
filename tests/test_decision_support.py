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
from src.models.modeling_common import load_config
from src.visualization.dashboard_data import (
    DashboardContext,
    DashboardLineage,
    TerritoryValidationMetrics,
    load_official_validation_metrics,
    load_validation_predictions,
)


class TestDecisionSupport(unittest.TestCase):
    """Pruebas unitarias de la regla estacional transparente."""

    @staticmethod
    def _product(
        values_by_year: dict[int, float],
        *,
        forecast: float = 550.0,
        target_month_id: str = "2026-09",
        covid_years: set[int] | None = None,
        provisional_years: set[int] | None = None,
        warnings: tuple[InferenceWarning, ...] = (),
        validation_wape_pct: float = 20.0,
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
        history = pd.DataFrame.from_records(rows)
        history = history.sort_values("date_month").reset_index(drop=True)

        target = pd.Period(target_month_id, freq="M")
        reference = target - 12
        inference = InferenceResult(
            territory_id="ES-PROV-01",
            territory_name="Provincia A",
            target_month_id=str(target),
            forecast_horizon_months=1,
            predicted_overnight_stays_total=forecast,
            reference_month_id=str(reference),
            reference_overnight_stays_total=forecast,
            reference_is_provisional=bool(warnings),
            model_name="seasonal_naive_lag_12",
            source_snapshot_id="snapshot",
            pipeline_run_id="run",
            data_version="gold-v1",
            latest_available_month_id=str(target - 3),
            operational_status="forecast_ready",
            warnings=warnings,
        )
        dashboard = DashboardContext(
            territory_id="ES-PROV-01",
            territory_name="Provincia A",
            history=history,
            validation_metrics=TerritoryValidationMetrics(
                territory_id="ES-PROV-01",
                territory_name="Provincia A",
                validation_mae=10.0,
                validation_rmse=12.0,
                validation_wape_pct=validation_wape_pct,
                validation_bias=1.0,
                validation_rows=35,
            ),
            lineage=DashboardLineage(
                operational_source_snapshot_id="snapshot",
                operational_pipeline_run_id="run",
                operational_data_version="gold-v1",
                evaluation_source_snapshot_id="evaluation-snapshot",
                evaluation_pipeline_run_id="evaluation-run",
                evaluation_data_version="modeling-v1",
                operational_dataset_path="data/gold/gold.parquet",
                evaluation_dataset_path="data/gold/modeling.parquet",
                evaluation_predictions_path="data/predictions.parquet",
                evaluation_metrics_path="data/metrics.csv",
            ),
        )
        return ForecastProductContext(
            as_of_date=date(2026, 8, 13),
            forecast_origin_month_id="2026-08",
            forecast=inference,
            dashboard=dashboard,
        )

    def setUp(self) -> None:
        self.reference = {
            year: float((year - 2011) * 100)
            for year in range(2012, 2022)
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

    def test_only_same_calendar_month_is_compared(self) -> None:
        august = {
            "territory_id": "ES-PROV-01",
            "territory_name": "Provincia A",
            "month_id": "2021-08",
            "date_month": pd.Timestamp("2021-08-01"),
            "overnight_stays_total": 1_000_000.0,
            "covid_period": False,
            "data_status": "final_or_not_marked_provisional",
            "is_provisional": False,
            "coverage_quality": "high",
        }
        support = build_decision_support(
            self._product(self.reference, extra_rows=[august])
        )

        self.assertEqual(support.comparison_calendar_month, 9)
        self.assertNotIn("2021-08", support.comparison_month_ids)
        self.assertEqual(support.historical_median, 550.0)

    def test_exclusions_and_recent_window_are_traceable(self) -> None:
        values = {
            year: float(year)
            for year in range(2012, 2026)
        }
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
        self.assertEqual(support.comparison_month_ids[0], "2013-09")
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

    def test_small_sample_is_not_classified(self) -> None:
        support = build_decision_support(
            self._product({2018: 100.0, 2019: 200.0, 2020: 300.0, 2021: 400.0})
        )

        self.assertEqual(support.activity_level, "insufficient")
        self.assertEqual(support.historical_sample_size, 4)
        self.assertIsNone(support.historical_q25)
        self.assertIsNone(support.action_guidance)
        self.assertIn(
            "insufficient_seasonal_history",
            {item.code for item in support.warnings},
        )

    def test_historical_gap_is_reported(self) -> None:
        support = build_decision_support(
            self._product(
                {
                    2015: 100.0,
                    2016: 200.0,
                    2018: 300.0,
                    2019: 400.0,
                    2020: 500.0,
                }
            )
        )

        self.assertIn(2017, support.missing_comparison_years)
        self.assertIn(
            "historical_comparison_gaps",
            {item.code for item in support.warnings},
        )

    def test_flat_and_outside_range_cases_are_explicit(self) -> None:
        flat = {year: 100.0 for year in range(2012, 2022)}
        usual = build_decision_support(self._product(flat, forecast=100.0))
        high = build_decision_support(self._product(flat, forecast=101.0))

        self.assertEqual(usual.activity_level, "usual")
        self.assertEqual(usual.historical_percentile_pct, 50.0)
        self.assertIn(
            "flat_comparison_history",
            {item.code for item in usual.warnings},
        )
        self.assertEqual(high.activity_level, "high")
        self.assertIn(
            "forecast_outside_historical_range",
            {item.code for item in high.warnings},
        )

    def test_activity_level_does_not_depend_on_validation_wape(self) -> None:
        product = self._product(self.reference, forecast=550.0)
        altered_metrics = replace(
            product.dashboard.validation_metrics,
            validation_wape_pct=999.0,
        )
        altered = replace(
            product,
            dashboard=replace(
                product.dashboard,
                validation_metrics=altered_metrics,
            ),
        )

        original = build_decision_support(product)
        changed = build_decision_support(altered)
        self.assertEqual(original.activity_level, changed.activity_level)
        self.assertEqual(
            original.historical_percentile_pct,
            changed.historical_percentile_pct,
        )


class TestDecisionSupportIntegration(unittest.TestCase):
    """Integracion con los artefactos reales de operacion y evaluacion."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config()
        cls.gold = pd.read_parquet(cls.config["source_dataset"]["path"])
        cls.predictions = load_validation_predictions()
        cls.official_metrics = load_official_validation_metrics()

    def _build(self, territory_id: str):
        product = build_forecast_product_context(
            territory_id,
            as_of_date="2026-08-13",
            gold=self.gold,
            predictions=self.predictions,
            official_metrics=self.official_metrics,
            config=self.config,
        )
        return build_decision_support(product)

    def test_real_araba_signal(self) -> None:
        support = self._build("ES-PROV-01")

        self.assertEqual(support.activity_level, "high")
        self.assertEqual(support.historical_sample_size, 10)
        self.assertEqual(support.historical_percentile_pct, 90.0)
        self.assertEqual(support.excluded_covid_month_ids, ("2020-09", "2021-09"))
        self.assertEqual(support.excluded_provisional_month_ids, ("2025-09",))
        self.assertIn(
            "provisional_reference_data",
            {item.code for item in support.warnings},
        )

    def test_real_signal_is_available_for_all_fifty_provinces(self) -> None:
        territory_ids = sorted(self.gold["territory_id"].astype(str).unique())
        supports = [self._build(territory_id) for territory_id in territory_ids]
        distribution = pd.Series(
            [support.activity_level for support in supports]
        ).value_counts()

        self.assertEqual(len(supports), 50)
        self.assertTrue(
            all(support.historical_sample_size == 10 for support in supports)
        )
        self.assertNotIn("insufficient", distribution.index)
        self.assertEqual(distribution.to_dict(), {"high": 31, "usual": 15, "low": 4})


if __name__ == "__main__":
    unittest.main()
