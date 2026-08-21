import unittest
from copy import deepcopy
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.models.ets_v2 import ETSForecastResult
from src.models.inference import (
    EFFECTIVE_MODEL_HORIZON_STEPS,
    FALLBACK_MODEL_ID,
    SELECTED_MODEL_ID,
    SELECTION_STATUS,
    EmptyInferenceDatasetError,
    InferenceConfigurationError,
    InferenceDataError,
    InvalidTerritoryError,
    MissingReferenceError,
    UnsupportedHorizonError,
    load_inference_dataset,
    predict_next_month,
)
from src.models.modeling_v2_common import (
    MODELING_V2_CONFIG_PATH,
    load_modeling_v2_config,
)


def synthetic_gold() -> pd.DataFrame:
    months = pd.period_range("2017-01", "2026-12", freq="M")
    rows: list[dict[str, object]] = []
    for territory_id, territory_name, offset in (
        ("ES-PROV-01", "Provincia A", 0.0),
        ("ES-PROV-02", "Provincia B", 500.0),
    ):
        for index, month in enumerate(months):
            rows.append(
                {
                    "territory_id": territory_id,
                    "territory_name": territory_name,
                    "territory_level": "province",
                    "month_id": str(month),
                    "date_month": month.to_timestamp(),
                    "overnight_stays_total": (
                        1000.0 + offset + 8.0 * (index % 12) + index
                    ),
                    "complete_month_available": True,
                    "is_provisional": month >= pd.Period("2026-01", freq="M"),
                    "source_snapshot_id": "snapshot-v2-test",
                    "pipeline_run_id": "pipeline-v2-test",
                    "data_version": "gold-v2-test",
                }
            )
    return pd.DataFrame(rows)


def ets_result(
    *,
    available: bool,
    reason: str | None = None,
    prediction: float | None = 4321.5,
    raw_prediction: float | None = 4321.5,
    clipping_applied: bool = False,
    target: str = "2026-09",
    cutoff: str = "2026-06",
    territory_id: str = "ES-PROV-01",
) -> ETSForecastResult:
    return ETSForecastResult(
        territory_id=territory_id,
        target_month_id=target,
        latest_available_month_id=cutoff,
        effective_horizon_steps=3,
        prediction=prediction if available else None,
        raw_prediction=raw_prediction if available else None,
        candidate_available=available,
        unavailable_reason=None if available else reason,
        training_rows=114,
        training_observed_rows=114,
        training_start="2017-01",
        training_end=cutoff,
        model_id=SELECTED_MODEL_ID,
        clipping_applied=clipping_applied,
        fit_attempted=True,
        fit_seconds=0.01,
        imputed_months_n=0,
        imputed_month_ids=(),
        fit_warning_count=0,
        fit_warning_messages=(),
    )


class TestOperationalInferenceB5(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_modeling_v2_config()
        self.dataframe = synthetic_gold()

    def predict_with(
        self,
        forecast: ETSForecastResult,
        *,
        dataframe: pd.DataFrame | None = None,
        config: dict | None = None,
        as_of_date: str = "2026-08-20",
    ):
        with patch(
            "src.models.inference.fit_ets_forecast",
            return_value=forecast,
        ):
            return predict_next_month(
                "ES-PROV-01",
                as_of_date=as_of_date,
                dataframe=self.dataframe if dataframe is None else dataframe,
                config=self.config if config is None else config,
            )

    def test_config_selects_provisional_ets_and_availability_fallback(self) -> None:
        selection = self.config["operational_selection"]

        self.assertEqual(selection["selected_model_id"], SELECTED_MODEL_ID)
        self.assertEqual(selection["status"], SELECTION_STATUS)
        self.assertFalse(selection["independent_test_confirmed"])
        self.assertEqual(
            selection["fallback"]["model_id"], FALLBACK_MODEL_ID
        )
        self.assertEqual(
            selection["fallback"]["policy"], "availability_only"
        )
        self.assertFalse(selection["fallback"]["performance_based"])

    def test_default_loader_uses_modeling_v2_config(self) -> None:
        with (
            patch(
                "src.models.inference.load_modeling_v2_config",
                return_value=self.config,
            ) as loader,
            patch(
                "src.models.inference.fit_ets_forecast",
                return_value=ets_result(available=True),
            ),
        ):
            result = predict_next_month(
                "ES-PROV-01",
                as_of_date="2026-08-20",
                dataframe=self.dataframe,
            )

        loader.assert_called_once_with(MODELING_V2_CONFIG_PATH)
        self.assertEqual(result.selected_model_id, SELECTED_MODEL_ID)

    def test_target_business_origin_cutoff_and_horizons_are_v2(self) -> None:
        result = self.predict_with(ets_result(available=True))

        self.assertEqual(result.target_month_id, "2026-09")
        self.assertEqual(result.business_origin_month_id, "2026-08")
        self.assertEqual(result.latest_available_month_id, "2026-06")
        self.assertEqual(result.forecast_horizon_months, 1)
        self.assertEqual(
            result.effective_model_horizon_steps,
            EFFECTIVE_MODEL_HORIZON_STEPS,
        )

    def test_available_ets_is_the_point_forecast(self) -> None:
        result = self.predict_with(ets_result(available=True))

        self.assertEqual(result.actual_model_used, SELECTED_MODEL_ID)
        self.assertEqual(result.predicted_overnight_stays_total, 4321.5)
        self.assertEqual(result.ets_raw_prediction, 4321.5)
        self.assertFalse(result.fallback_used)
        self.assertEqual(result.fallback_reason, "not_used")
        self.assertTrue(np.isfinite(result.predicted_overnight_stays_total))
        self.assertGreaterEqual(result.predicted_overnight_stays_total, 0)
        self.assertTrue(result.is_operational)
        self.assertIn(
            "provisional_training_data",
            {warning.code for warning in result.warnings},
        )
        with self.assertRaises(FrozenInstanceError):
            result.fallback_used = True

    def test_historical_ets_without_provisional_training_has_no_warning(
        self,
    ) -> None:
        result = self.predict_with(
            ets_result(available=True, target="2025-09", cutoff="2025-06"),
            as_of_date="2025-08-20",
        )

        self.assertEqual(result.actual_model_used, SELECTED_MODEL_ID)
        self.assertNotIn(
            "provisional_training_data",
            {warning.code for warning in result.warnings},
        )

    def test_future_provisional_data_cannot_activate_training_warning(
        self,
    ) -> None:
        forecast = ets_result(
            available=True,
            target="2025-09",
            cutoff="2025-06",
        )
        baseline = self.predict_with(
            forecast,
            as_of_date="2025-08-20",
        )
        changed = self.dataframe.copy()
        future = (
            changed["territory_id"].eq("ES-PROV-01")
            & changed["month_id"].gt("2025-06")
        )
        changed.loc[future, "is_provisional"] = True
        mutated = self.predict_with(
            forecast,
            dataframe=changed,
            as_of_date="2025-08-20",
        )

        self.assertEqual(
            mutated.predicted_overnight_stays_total,
            baseline.predicted_overnight_stays_total,
        )
        self.assertEqual(mutated.warnings, baseline.warnings)
        self.assertNotIn(
            "provisional_training_data",
            {warning.code for warning in mutated.warnings},
        )

    def test_training_end_never_exceeds_information_cutoff(self) -> None:
        result = self.predict_with(ets_result(available=True))

        self.assertLessEqual(
            pd.Period(result.training_end, freq="M"),
            pd.Period(result.latest_available_month_id, freq="M"),
        )

    def test_fit_receives_one_territory_and_no_future_rows(self) -> None:
        captured: dict[str, pd.DataFrame] = {}

        def fake_fit(history, origin, config):
            captured["history"] = history.copy()
            self.assertEqual(origin.latest_available_month_id, "2026-06")
            return ets_result(available=True)

        with patch(
            "src.models.inference.fit_ets_forecast",
            side_effect=fake_fit,
        ):
            predict_next_month(
                "ES-PROV-01",
                as_of_date="2026-08-20",
                dataframe=self.dataframe,
                config=self.config,
            )

        history = captured["history"]
        self.assertEqual(history["territory_id"].nunique(), 1)
        self.assertEqual(set(history["territory_id"]), {"ES-PROV-01"})
        self.assertLessEqual(
            pd.PeriodIndex(history["month_id"], freq="M").max(),
            pd.Period("2026-06", freq="M"),
        )

    def test_future_analytical_flags_cannot_change_inference(self) -> None:
        calls: list[pd.DataFrame] = []

        def causal_fit(history, origin, config):
            calls.append(history.reset_index(drop=True).copy())
            return ets_result(available=True)

        mutations = (
            ("complete_month_available", None),
            ("is_provisional", None),
            ("complete_month_available", "invalid"),
            ("is_provisional", "invalid"),
        )
        results = []
        with patch(
            "src.models.inference.fit_ets_forecast",
            side_effect=causal_fit,
        ):
            baseline = predict_next_month(
                "ES-PROV-01",
                as_of_date="2026-08-20",
                dataframe=self.dataframe,
                config=self.config,
            )
            for column, invalid_value in mutations:
                changed = self.dataframe.copy()
                changed[column] = changed[column].astype(object)
                future = (
                    changed["territory_id"].eq("ES-PROV-01")
                    & changed["month_id"].eq("2026-07")
                )
                changed.loc[future, column] = invalid_value
                results.append(
                    predict_next_month(
                        "ES-PROV-01",
                        as_of_date="2026-08-20",
                        dataframe=changed,
                        config=self.config,
                    )
                )

        forecast_rtol = float(
            self.config["numerical_reproducibility"]
            ["forecast_relative_tolerance"]
        )
        for index, result in enumerate(results, start=1):
            with self.subTest(mutation=mutations[index - 1]):
                pd.testing.assert_frame_equal(calls[0], calls[index])
                self.assertEqual(result.selected_model_id, baseline.selected_model_id)
                self.assertEqual(result.actual_model_used, baseline.actual_model_used)
                self.assertEqual(result.fallback_used, baseline.fallback_used)
                self.assertEqual(result.fallback_reason, baseline.fallback_reason)
                self.assertEqual(
                    result.latest_available_month_id,
                    baseline.latest_available_month_id,
                )
                self.assertEqual(result.training_end, baseline.training_end)
                self.assertEqual(result.warnings, baseline.warnings)
                np.testing.assert_allclose(
                    result.predicted_overnight_stays_total,
                    baseline.predicted_overnight_stays_total,
                    rtol=forecast_rtol,
                    atol=0.0,
                )

    def test_invalid_analytical_flags_within_cutoff_remain_blocking(self) -> None:
        mutations = (
            ("complete_month_available", None),
            ("is_provisional", None),
            ("complete_month_available", "invalid"),
            ("is_provisional", "invalid"),
        )
        for column, invalid_value in mutations:
            with self.subTest(column=column, invalid_value=invalid_value):
                changed = self.dataframe.copy()
                changed[column] = changed[column].astype(object)
                known = (
                    changed["territory_id"].eq("ES-PROV-01")
                    & changed["month_id"].eq("2026-06")
                )
                changed.loc[known, column] = invalid_value
                with (
                    patch("src.models.inference.fit_ets_forecast") as fit,
                    self.assertRaisesRegex(InferenceDataError, column),
                ):
                    predict_next_month(
                        "ES-PROV-01",
                        as_of_date="2026-08-20",
                        dataframe=changed,
                        config=self.config,
                    )
                fit.assert_not_called()

    def test_ets_unavailable_uses_lag12_only_for_availability(self) -> None:
        result = self.predict_with(
            ets_result(
                available=False,
                reason="training_gap_unsupported",
            )
        )
        reference = self.dataframe.loc[
            self.dataframe["territory_id"].eq("ES-PROV-01")
            & self.dataframe["month_id"].eq("2025-09"),
            "overnight_stays_total",
        ].iloc[0]

        self.assertEqual(result.actual_model_used, FALLBACK_MODEL_ID)
        self.assertTrue(result.fallback_used)
        self.assertEqual(result.fallback_reason, "training_gap_unsupported")
        self.assertEqual(result.baseline_reference_month_id, "2025-09")
        self.assertEqual(result.baseline_prediction, reference)
        self.assertEqual(result.predicted_overnight_stays_total, reference)
        self.assertIn(
            "availability_fallback_used",
            {warning.code for warning in result.warnings},
        )

    def test_all_approved_unavailability_reasons_can_fallback(self) -> None:
        for reason in (
            "insufficient_history",
            "training_gap_unsupported",
            "fit_failure",
            "invalid_forecast",
        ):
            with self.subTest(reason=reason):
                result = self.predict_with(
                    ets_result(available=False, reason=reason)
                )
                self.assertEqual(result.fallback_reason, reason)
                self.assertTrue(result.fallback_used)

    def test_expected_numerical_fit_failure_allows_availability_fallback(
        self,
    ) -> None:
        with patch(
            "src.models.ets_v2.ExponentialSmoothing",
            side_effect=FloatingPointError("expected numerical failure"),
        ):
            result = predict_next_month(
                "ES-PROV-01",
                as_of_date="2026-08-20",
                dataframe=self.dataframe,
                config=self.config,
            )

        self.assertEqual(result.actual_model_used, FALLBACK_MODEL_ID)
        self.assertTrue(result.fallback_used)
        self.assertEqual(result.fallback_reason, "fit_failure")

    def test_unexpected_programming_defect_propagates_without_fallback(
        self,
    ) -> None:
        with (
            patch(
                "src.models.ets_v2.ExponentialSmoothing",
                side_effect=RuntimeError("simulated programming defect"),
            ),
            self.assertRaisesRegex(RuntimeError, "programming defect"),
        ):
            predict_next_month(
                "ES-PROV-01",
                as_of_date="2026-08-20",
                dataframe=self.dataframe,
                config=self.config,
            )

    def test_performance_based_fallback_configuration_is_rejected(self) -> None:
        config = deepcopy(self.config)
        config["operational_selection"]["fallback"][
            "performance_based"
        ] = True

        with self.assertRaisesRegex(
            InferenceConfigurationError, "performance"
        ):
            self.predict_with(
                ets_result(available=False, reason="fit_failure"),
                config=config,
            )

    def test_unknown_ets_reason_does_not_route_to_baseline(self) -> None:
        with self.assertRaisesRegex(
            InferenceDataError, "no soportado"
        ):
            self.predict_with(
                ets_result(available=False, reason="low_expected_skill")
            )

    def test_unavailable_ets_and_missing_lag12_fails_closed(self) -> None:
        dataframe = self.dataframe.copy()
        reference = (
            dataframe["territory_id"].eq("ES-PROV-01")
            & dataframe["month_id"].eq("2025-09")
        )
        dataframe.loc[reference, "complete_month_available"] = False
        dataframe.loc[reference, "overnight_stays_total"] = np.nan

        with self.assertRaisesRegex(MissingReferenceError, "lag-12"):
            self.predict_with(
                ets_result(
                    available=False,
                    reason="training_gap_unsupported",
                ),
                dataframe=dataframe,
            )

    def test_available_ets_survives_missing_interval_scale(self) -> None:
        dataframe = self.dataframe.copy()
        reference = (
            dataframe["territory_id"].eq("ES-PROV-01")
            & dataframe["month_id"].eq("2025-09")
        )
        dataframe.loc[reference, "complete_month_available"] = False
        dataframe.loc[reference, "overnight_stays_total"] = np.nan

        result = self.predict_with(
            ets_result(available=True),
            dataframe=dataframe,
        )

        self.assertEqual(result.predicted_overnight_stays_total, 4321.5)
        self.assertIsNone(result.baseline_prediction)
        self.assertFalse(result.fallback_used)

    def test_provisional_lag12_warning_only_when_fallback_uses_it(self) -> None:
        dataframe = self.dataframe.copy()
        reference = (
            dataframe["territory_id"].eq("ES-PROV-01")
            & dataframe["month_id"].eq("2025-09")
        )
        dataframe.loc[reference, "is_provisional"] = True
        result = self.predict_with(
            ets_result(available=False, reason="fit_failure"),
            dataframe=dataframe,
        )

        self.assertTrue(result.baseline_reference_is_provisional)
        self.assertIn(
            "provisional_reference_data",
            {warning.code for warning in result.warnings},
        )
        self.assertNotIn(
            "provisional_training_data",
            {warning.code for warning in result.warnings},
        )

    def test_invalid_territory_is_blocking_before_fit(self) -> None:
        with (
            patch("src.models.inference.fit_ets_forecast") as fit,
            self.assertRaisesRegex(InvalidTerritoryError, "ES-PROV-99"),
        ):
            predict_next_month(
                "ES-PROV-99",
                as_of_date="2026-08-20",
                dataframe=self.dataframe,
                config=self.config,
            )
        fit.assert_not_called()

    def test_duplicate_keys_are_blocking(self) -> None:
        dataframe = pd.concat(
            [self.dataframe, self.dataframe.iloc[[0]]],
            ignore_index=True,
        )

        with self.assertRaisesRegex(InferenceDataError, "duplicadas"):
            self.predict_with(ets_result(available=True), dataframe=dataframe)

    def test_invalid_lineage_is_blocking(self) -> None:
        dataframe = self.dataframe.copy()
        dataframe.loc[0, "source_snapshot_id"] = "another-snapshot"

        with self.assertRaisesRegex(InferenceDataError, "source_snapshot_id"):
            self.predict_with(ets_result(available=True), dataframe=dataframe)

    def test_empty_dataset_and_unsupported_horizon_are_blocking(self) -> None:
        with self.assertRaises(EmptyInferenceDatasetError):
            predict_next_month(
                "ES-PROV-01",
                as_of_date="2026-08-20",
                dataframe=self.dataframe.iloc[0:0],
                config=self.config,
            )
        with self.assertRaises(UnsupportedHorizonError):
            predict_next_month(
                "ES-PROV-01",
                as_of_date="2026-08-20",
                forecast_horizon_months=2,
                dataframe=self.dataframe,
                config=self.config,
            )

    def test_source_loader_uses_v2_gold_not_evaluation_or_final_test(self) -> None:
        with patch(
            "src.models.inference.pd.read_parquet",
            return_value=self.dataframe,
        ) as reader:
            loaded = load_inference_dataset(self.config)

        called_path = Path(reader.call_args.args[0])
        self.assertEqual(
            called_path,
            Path(__file__).resolve().parents[1]
            / self.config["source"]["path"],
        )
        self.assertNotIn("model_outputs", str(called_path))
        self.assertNotIn("test", called_path.name.lower())
        self.assertEqual(len(loaded), len(self.dataframe))


class TestOperationalInferenceIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.araba = predict_next_month(
            "ES-PROV-01", as_of_date="2026-08-20"
        )
        cls.badajoz = predict_next_month(
            "ES-PROV-06", as_of_date="2026-08-20"
        )

    def test_current_gold_araba_returns_finite_causal_ets(self) -> None:
        result = self.araba

        self.assertEqual(result.target_month_id, "2026-09")
        self.assertEqual(result.latest_available_month_id, "2026-06")
        self.assertEqual(result.actual_model_used, SELECTED_MODEL_ID)
        self.assertFalse(result.fallback_used)
        self.assertTrue(np.isfinite(result.predicted_overnight_stays_total))
        self.assertGreaterEqual(result.predicted_overnight_stays_total, 0)
        self.assertLessEqual(
            pd.Period(result.training_end, freq="M"),
            pd.Period(result.latest_available_month_id, freq="M"),
        )
        self.assertIn(
            "provisional_training_data",
            {warning.code for warning in result.warnings},
        )

    def test_current_gold_badajoz_is_structural_availability_fallback(self) -> None:
        result = self.badajoz

        self.assertEqual(result.target_month_id, "2026-09")
        self.assertEqual(result.actual_model_used, FALLBACK_MODEL_ID)
        self.assertTrue(result.fallback_used)
        self.assertEqual(result.fallback_reason, "training_gap_unsupported")
        self.assertEqual(
            result.predicted_overnight_stays_total,
            result.baseline_prediction,
        )
        self.assertTrue(np.isfinite(result.predicted_overnight_stays_total))


if __name__ == "__main__":
    unittest.main()
