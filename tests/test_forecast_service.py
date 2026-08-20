import unittest

from dataclasses import replace
from datetime import date
from unittest.mock import patch

import pandas as pd

from src.application import forecast_service
from src.application.decision_support import build_decision_support
from src.application.forecast_service import (
    ProductCompositionError,
    build_forecast_product_context,
    prepare_forecast_service_resources,
)
from src.models.inference import FALLBACK_MODEL_ID, SELECTED_MODEL_ID
from src.models.modeling_v2_common import load_modeling_v2_config
from src.models.prediction_intervals_v2 import PredictionIntervalResult
from src.visualization.dashboard_data import (
    CANONICAL_ARTIFACT_SHA256,
    CANONICAL_LOGICAL_SHA256,
    EVIDENCE_SCOPE,
    CanonicalArtifactError,
    load_canonical_validation_bundle,
    load_gold_history,
)


class TestForecastServiceB5(unittest.TestCase):
    """Integracion del point B5, intervalo y dashboard canonico."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_modeling_v2_config()
        cls.bundle = load_canonical_validation_bundle(cls.config)
        cls.gold = load_gold_history(cls.config)
        cls.resources = prepare_forecast_service_resources(
            cls.config,
            canonical_bundle=cls.bundle,
        )
        cls.araba = build_forecast_product_context(
            "ES-PROV-01",
            as_of_date="2026-08-20",
            history_months=24,
            gold=cls.gold,
            prepared_resources=cls.resources,
            config=cls.config,
        )
        cls.badajoz = build_forecast_product_context(
            "ES-PROV-06",
            as_of_date="2026-08-20",
            gold=cls.gold,
            prepared_resources=cls.resources,
            config=cls.config,
        )
        cls.historical = build_forecast_product_context(
            "ES-PROV-01",
            as_of_date="2023-01-20",
            gold=cls.gold,
            prepared_resources=cls.resources,
            config=cls.config,
        )
        cls.historical_24 = build_forecast_product_context(
            "ES-PROV-01",
            as_of_date="2023-01-20",
            history_months=24,
            gold=cls.gold,
            prepared_resources=cls.resources,
            config=cls.config,
        )

    def _validate(self, product=None) -> None:
        selected = self.araba if product is None else product
        forecast_service._validate_composition(
            selected.forecast,
            selected.prediction_interval,
            selected.dashboard,
            self.config,
            self.bundle,
        )

    def test_default_service_loads_v2_and_reuses_one_gold(self) -> None:
        with (
            patch(
                "src.application.forecast_service.load_modeling_v2_config",
                return_value=self.config,
            ) as load_v2,
            patch(
                "src.application.forecast_service.load_gold_history",
                return_value=self.gold,
            ) as load_gold,
            patch(
                "src.application.forecast_service.prepare_forecast_service_resources",
                return_value=self.resources,
            ) as prepare_resources,
        ):
            product = build_forecast_product_context(
                "ES-PROV-01",
                as_of_date="2026-08-20",
            )

        load_v2.assert_called_once()
        load_gold.assert_called_once_with(self.config)
        prepare_resources.assert_called_once_with(
            self.config,
            canonical_bundle=None,
        )
        self.assertEqual(product.forecast.target_month_id, "2026-09")
        self.assertFalse(hasattr(forecast_service, "load_config"))

    def test_araba_uses_selected_ets_and_available_interval(self) -> None:
        product = self.araba
        forecast = product.forecast
        interval = product.prediction_interval

        self.assertEqual(product.as_of_date, date(2026, 8, 20))
        self.assertEqual(product.forecast_origin_month_id, "2026-08")
        self.assertEqual(forecast.territory_name, "Araba/Álava")
        self.assertEqual(forecast.selected_model_id, SELECTED_MODEL_ID)
        self.assertEqual(forecast.actual_model_used, SELECTED_MODEL_ID)
        self.assertFalse(forecast.fallback_used)
        self.assertTrue(interval.interval_available)
        self.assertLessEqual(
            product.dashboard.history["month_id"].astype(str).max(),
            forecast.latest_available_month_id,
        )
        self.assertLessEqual(
            interval.lower,
            forecast.predicted_overnight_stays_total,
        )
        self.assertGreaterEqual(
            interval.upper,
            forecast.predicted_overnight_stays_total,
        )

    def test_badajoz_uses_availability_fallback_without_router(self) -> None:
        forecast = self.badajoz.forecast

        self.assertEqual(forecast.selected_model_id, SELECTED_MODEL_ID)
        self.assertEqual(forecast.actual_model_used, FALLBACK_MODEL_ID)
        self.assertTrue(forecast.fallback_used)
        self.assertEqual(forecast.fallback_reason, "training_gap_unsupported")
        self.assertEqual(forecast.predicted_overnight_stays_total, 14214.0)
        self.assertTrue(self.badajoz.prediction_interval.interval_available)
        self.assertLessEqual(
            self.badajoz.dashboard.history["month_id"].astype(str).max(),
            forecast.latest_available_month_id,
        )
        self.assertFalse(
            self.config["operational_selection"]["fallback"][
                "performance_based"
            ]
        )

    def test_historical_public_context_is_point_in_time_safe(self) -> None:
        product = self.historical
        history_months = product.dashboard.history["month_id"].astype(str)

        self.assertEqual(product.forecast.target_month_id, "2023-02")
        self.assertEqual(product.forecast.latest_available_month_id, "2022-11")
        self.assertEqual(history_months.max(), "2022-11")
        self.assertFalse(history_months.gt("2022-11").any())

    def test_historical_natural_window_is_anchored_to_cutoff(self) -> None:
        product = self.historical_24
        periods = pd.PeriodIndex(
            product.dashboard.history["month_id"].astype(str),
            freq="M",
        )

        self.assertEqual(str(periods.min()), "2020-12")
        self.assertEqual(str(periods.max()), "2022-11")
        self.assertTrue((periods >= pd.Period("2020-12", freq="M")).all())
        self.assertTrue((periods <= pd.Period("2022-11", freq="M")).all())

    def test_future_gold_mutation_cannot_change_historical_product(self) -> None:
        changed_gold = self.gold.copy()
        future = changed_gold["month_id"].astype(str).gt("2022-11")
        changed_gold.loc[future, "overnight_stays_total"] += 1_000_000.0
        for column in ("covid_period", "is_provisional"):
            changed_gold.loc[future, column] = ~changed_gold.loc[future, column]
        if "complete_month_available" in changed_gold.columns:
            changed_gold.loc[future, "complete_month_available"] = ~changed_gold.loc[
                future,
                "complete_month_available",
            ]

        changed = build_forecast_product_context(
            "ES-PROV-01",
            as_of_date="2023-01-20",
            gold=changed_gold,
            canonical_bundle=self.bundle,
            config=self.config,
        )

        self.assertAlmostEqual(
            changed.forecast.predicted_overnight_stays_total,
            self.historical.forecast.predicted_overnight_stays_total,
        )
        self.assertEqual(
            changed.forecast.latest_available_month_id,
            self.historical.forecast.latest_available_month_id,
        )
        pd.testing.assert_frame_equal(
            changed.dashboard.history,
            self.historical.dashboard.history,
        )
        self.assertEqual(
            build_decision_support(changed),
            build_decision_support(self.historical),
        )

    def test_interval_point_is_exactly_the_operational_forecast(self) -> None:
        for product in (self.araba, self.badajoz):
            with self.subTest(territory=product.forecast.territory_id):
                self.assertEqual(
                    product.prediction_interval.point_prediction,
                    product.forecast.predicted_overnight_stays_total,
                )

    def test_canonical_and_operational_lineages_are_both_exposed(self) -> None:
        lineage = self.araba.dashboard.lineage

        self.assertEqual(
            lineage.operational_source_snapshot_id,
            self.araba.forecast.source_snapshot_id,
        )
        self.assertEqual(
            lineage.evaluation_artifact_sha256,
            CANONICAL_ARTIFACT_SHA256,
        )
        self.assertEqual(
            lineage.evaluation_logical_prediction_sha256,
            CANONICAL_LOGICAL_SHA256,
        )
        self.assertEqual(lineage.evaluation_scope, EVIDENCE_SCOPE)
        self.assertTrue(lineage.operational_source_snapshot_id)
        self.assertTrue(lineage.evaluation_source_snapshot_ids)

    def test_operational_vintage_need_not_equal_evaluation_vintage(self) -> None:
        newer = self.gold.copy()
        newer["source_snapshot_id"] = "newer-operational-snapshot"
        newer["pipeline_run_id"] = "newer-operational-run"
        newer["data_version"] = "newer-operational-version"

        product = build_forecast_product_context(
            "ES-PROV-01",
            as_of_date="2026-08-20",
            gold=newer,
            canonical_bundle=self.bundle,
            config=self.config,
        )

        lineage = product.dashboard.lineage
        self.assertEqual(
            lineage.operational_source_snapshot_id,
            "newer-operational-snapshot",
        )
        self.assertEqual(
            lineage.evaluation_source_snapshot_ids,
            self.bundle.evaluation_source_snapshot_ids,
        )

    def test_invalid_or_corrupted_canonical_resource_fails_closed(self) -> None:
        for message in ("invalid canonical hash", "corrupted artifact"):
            with (
                self.subTest(message=message),
                patch(
                    "src.application.forecast_service.prepare_forecast_service_resources",
                    side_effect=CanonicalArtifactError(message),
                ),
                self.assertRaisesRegex(CanonicalArtifactError, message),
            ):
                build_forecast_product_context(
                    "ES-PROV-01",
                    as_of_date="2026-08-20",
                    gold=self.gold,
                    config=self.config,
                )

    def test_arbitrary_dataframe_cannot_impersonate_canonical_bundle(self) -> None:
        with self.assertRaisesRegex(CanonicalArtifactError, "DataFrame"):
            build_forecast_product_context(
                "ES-PROV-01",
                as_of_date="2026-08-20",
                gold=self.gold,
                canonical_bundle=pd.DataFrame(),  # type: ignore[arg-type]
                config=self.config,
            )

    def test_mutated_validated_bundle_cannot_impersonate_official_evidence(
        self,
    ) -> None:
        predictions = self.bundle.predictions.copy()
        predictions.loc[0, "operational_prediction"] += 1.0
        mutated = replace(self.bundle, predictions=predictions)

        with self.assertRaises(CanonicalArtifactError):
            build_forecast_product_context(
                "ES-PROV-01",
                as_of_date="2026-08-20",
                gold=self.gold,
                canonical_bundle=mutated,
                config=self.config,
            )

    def test_unavailable_interval_keeps_valid_point_forecast(self) -> None:
        def unavailable(**kwargs):
            return PredictionIntervalResult(
                territory_id=str(kwargs["territory_id"]),
                target_month_id=str(kwargs["target_month_id"]),
                point_prediction=float(kwargs["point_prediction"]),
                lower=None,
                upper=None,
                nominal_level=0.8,
                method_id=(
                    "operational_prequential_scaled_absolute_residual_interval_v1"
                ),
                calibration_scores_n=0,
                calibration_origins_n=0,
                calibration_max_target_month_id=None,
                calibration_quantile=None,
                interval_available=False,
                unavailable_reason="insufficient_calibration_origins",
            )

        with patch(
            "src.application.forecast_service.calculate_current_operational_interval",
            side_effect=unavailable,
        ):
            product = build_forecast_product_context(
                "ES-PROV-01",
                as_of_date="2026-08-20",
                gold=self.gold,
                canonical_bundle=self.bundle,
                config=self.config,
            )

        self.assertTrue(product.forecast.is_operational)
        self.assertFalse(product.prediction_interval.interval_available)
        self.assertEqual(
            product.prediction_interval.point_prediction,
            product.forecast.predicted_overnight_stays_total,
        )

    def test_territory_mismatch_is_blocking(self) -> None:
        mismatched = replace(
            self.araba.dashboard,
            territory_id="ES-PROV-02",
        )
        product = replace(self.araba, dashboard=mismatched)

        with self.assertRaisesRegex(ProductCompositionError, "territorio"):
            self._validate(product)

    def test_operational_lineage_mismatch_is_blocking(self) -> None:
        lineage = replace(
            self.araba.dashboard.lineage,
            operational_source_snapshot_id="another-snapshot",
        )
        product = replace(
            self.araba,
            dashboard=replace(self.araba.dashboard, lineage=lineage),
        )

        with self.assertRaisesRegex(
            ProductCompositionError,
            "provenance operacional",
        ):
            self._validate(product)

    def test_selected_model_mismatch_is_blocking(self) -> None:
        product = replace(
            self.araba,
            forecast=replace(self.araba.forecast, selected_model_id="ridge"),
        )

        with self.assertRaisesRegex(ProductCompositionError, "seleccionado"):
            self._validate(product)

    def test_fallback_semantics_mismatch_is_blocking(self) -> None:
        product = replace(
            self.araba,
            forecast=replace(self.araba.forecast, fallback_used=True),
        )

        with self.assertRaisesRegex(ProductCompositionError, "fallback"):
            self._validate(product)

    def test_interval_territory_and_point_mismatches_are_blocking(self) -> None:
        cases = (
            replace(
                self.araba.prediction_interval,
                territory_id="ES-PROV-02",
            ),
            replace(
                self.araba.prediction_interval,
                point_prediction=(
                    self.araba.prediction_interval.point_prediction + 1.0
                ),
            ),
        )
        for interval in cases:
            product = replace(self.araba, prediction_interval=interval)
            with (
                self.subTest(interval=interval),
                self.assertRaises(ProductCompositionError),
            ):
                self._validate(product)

    def test_dashboard_must_use_operational_prediction(self) -> None:
        metrics = replace(
            self.araba.dashboard.validation_metrics,
            prediction_column="ets_candidate_prediction",
        )
        product = replace(
            self.araba,
            dashboard=replace(self.araba.dashboard, validation_metrics=metrics),
        )

        with self.assertRaisesRegex(
            ProductCompositionError,
            "operational_prediction",
        ):
            self._validate(product)


if __name__ == "__main__":
    unittest.main()
