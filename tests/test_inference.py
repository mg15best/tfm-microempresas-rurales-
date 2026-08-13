import unittest

from copy import deepcopy
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.models.inference import (
    EmptyInferenceDatasetError,
    GlobalReferenceGapError,
    InferenceConfigurationError,
    InferenceDataError,
    InvalidTerritoryError,
    TerritorialReferenceGapError,
    UnsupportedHorizonError,
    predict_next_month,
)
from src.models.modeling_common import load_config


class TestInference(unittest.TestCase):
    """Pruebas de la inferencia estacional operacional."""

    def setUp(self) -> None:
        self.config = {
            "problem": {
                "forecast_horizon_months": 1,
            },
            "target": {
                "source_column": "overnight_stays_total",
            },
            "source_dataset": {
                "path": "data/gold/gold_tourism_demand_monthly.parquet",
            },
            "baseline": {
                "name": "seasonal_naive_lag_12",
                "prediction_feature": "lag_12_overnight_stays",
            },
            "fallback": {
                "if_no_candidate_beats_baseline": {
                    "selected_solution": "seasonal_naive_lag_12",
                },
            },
        }
        self.dataframe = self._build_dataframe()

    @staticmethod
    def _build_dataframe() -> pd.DataFrame:
        rows = [
            ("ES-PROV-01", "Provincia A", "2025-01", 101.0, False),
            ("ES-PROV-01", "Provincia A", "2025-05", 100.0, False),
            ("ES-PROV-01", "Provincia A", "2025-06", 118.0, False),
            ("ES-PROV-01", "Provincia A", "2025-07", 999.0, False),
            ("ES-PROV-01", "Provincia A", "2025-09", 123.0, False),
            ("ES-PROV-01", "Provincia A", "2026-05", 500.0, True),
            ("ES-PROV-02", "Provincia B", "2025-09", 456.0, False),
            ("ES-PROV-02", "Provincia B", "2026-05", 600.0, True),
        ]
        return pd.DataFrame(
            {
                "territory_id": [row[0] for row in rows],
                "territory_name": [row[1] for row in rows],
                "month_id": [row[2] for row in rows],
                "date_month": pd.to_datetime([row[2] for row in rows]),
                "overnight_stays_total": [row[3] for row in rows],
                "is_provisional": [row[4] for row in rows],
                "source_snapshot_id": "snapshot-001",
                "pipeline_run_id": "run-001",
                "data_version": "gold-test-v1",
            }
        )

    def _predict(
        self,
        dataframe: pd.DataFrame | None = None,
        **kwargs: object,
    ):
        kwargs.setdefault("as_of_date", "2026-08-13")
        return predict_next_month(
            "ES-PROV-01",
            dataframe=self.dataframe if dataframe is None else dataframe,
            config=self.config,
            **kwargs,
        )

    def test_valid_territory_returns_structured_result(self) -> None:
        result = self._predict()

        self.assertEqual(result.territory_id, "ES-PROV-01")
        self.assertEqual(result.territory_name, "Provincia A")
        self.assertEqual(result.model_name, "seasonal_naive_lag_12")
        self.assertEqual(result.source_snapshot_id, "snapshot-001")
        self.assertEqual(result.pipeline_run_id, "run-001")
        self.assertEqual(result.data_version, "gold-test-v1")

    def test_invalid_territory_is_blocking(self) -> None:
        with self.assertRaisesRegex(InvalidTerritoryError, "ES-PROV-99"):
            predict_next_month(
                "ES-PROV-99",
                dataframe=self.dataframe,
                config=self.config,
                as_of_date="2026-08-13",
            )

    def test_reference_month_is_exactly_target_minus_twelve(self) -> None:
        result = self._predict()

        self.assertEqual(result.target_month_id, "2026-09")
        self.assertEqual(result.reference_month_id, "2025-09")

    def test_prediction_equals_exact_reference_value(self) -> None:
        result = self._predict()

        self.assertEqual(result.predicted_overnight_stays_total, 123.0)
        self.assertEqual(result.reference_overnight_stays_total, 123.0)

    def test_global_reference_gap_is_blocking(self) -> None:
        dataframe = self.dataframe.loc[
            ~self.dataframe["month_id"].eq("2025-09")
        ]

        with self.assertRaisesRegex(GlobalReferenceGapError, "gap global"):
            self._predict(dataframe)

    def test_territorial_reference_gap_is_blocking(self) -> None:
        dataframe = self.dataframe.loc[
            ~(
                self.dataframe["territory_id"].eq("ES-PROV-01")
                & self.dataframe["month_id"].eq("2025-09")
            )
        ]

        with self.assertRaisesRegex(
            TerritorialReferenceGapError,
            "gap territorial",
        ):
            self._predict(dataframe)

    def test_missing_reference_does_not_fall_back_to_adjacent_month(self) -> None:
        dataframe = self.dataframe.loc[
            ~(
                self.dataframe["territory_id"].eq("ES-PROV-01")
                & self.dataframe["month_id"].eq("2025-09")
            )
        ]

        with self.assertRaises(TerritorialReferenceGapError):
            self._predict(dataframe)

    def test_observed_zero_is_preserved(self) -> None:
        dataframe = self.dataframe.copy()
        reference_mask = (
            dataframe["territory_id"].eq("ES-PROV-01")
            & dataframe["month_id"].eq("2025-09")
        )
        dataframe.loc[reference_mask, "overnight_stays_total"] = 0.0

        result = self._predict(dataframe)

        self.assertEqual(result.predicted_overnight_stays_total, 0.0)

    def test_provisional_reference_is_propagated_as_warning(self) -> None:
        dataframe = self.dataframe.copy()
        reference_mask = (
            dataframe["territory_id"].eq("ES-PROV-01")
            & dataframe["month_id"].eq("2025-09")
        )
        dataframe.loc[reference_mask, "is_provisional"] = True

        result = self._predict(dataframe)

        self.assertTrue(result.reference_is_provisional)
        self.assertIn(
            "provisional_reference_data",
            {warning.code for warning in result.warnings},
        )

    def test_mid_month_defines_an_operational_next_month_forecast(self) -> None:
        result = self._predict()

        self.assertEqual(result.target_month_id, "2026-09")
        self.assertEqual(result.operational_status, "forecast_ready")
        self.assertTrue(result.is_operational)

    def test_last_day_of_month_still_targets_next_month(self) -> None:
        result = self._predict(as_of_date="2026-08-31")

        self.assertEqual(result.target_month_id, "2026-09")
        self.assertEqual(result.reference_month_id, "2025-09")

    def test_latest_available_month_is_context_not_forecast_origin(self) -> None:
        result = self._predict(as_of_date="2026-08-13")

        self.assertEqual(result.latest_available_month_id, "2026-05")
        self.assertEqual(result.target_month_id, "2026-09")
        self.assertTrue(result.is_operational)

    def test_injected_as_of_date_controls_target(self) -> None:
        august = self._predict(as_of_date="2026-08-13")
        may = self._predict(as_of_date="2026-05-31")

        self.assertEqual(august.target_month_id, "2026-09")
        self.assertEqual(may.target_month_id, "2026-06")

    def test_default_forecast_origin_uses_local_system_date(self) -> None:
        with patch(
            "src.models.inference._local_today",
            return_value=date(2026, 8, 13),
        ):
            result = predict_next_month(
                "ES-PROV-01",
                dataframe=self.dataframe,
                config=self.config,
            )

        self.assertEqual(result.target_month_id, "2026-09")
        self.assertEqual(result.reference_month_id, "2025-09")

    def test_month_arithmetic_crosses_year_boundary(self) -> None:
        result = self._predict(
            as_of_date="2025-12-31",
        )

        self.assertEqual(result.target_month_id, "2026-01")
        self.assertEqual(result.reference_month_id, "2025-01")
        self.assertEqual(result.predicted_overnight_stays_total, 101.0)

    def test_reference_lookup_ignores_time_within_month_start(self) -> None:
        dataframe = self.dataframe.copy()
        reference_mask = dataframe["month_id"].eq("2025-09")
        dataframe.loc[reference_mask, "date_month"] += pd.Timedelta(hours=12)

        result = self._predict(dataframe)

        self.assertEqual(result.reference_month_id, "2025-09")
        self.assertEqual(result.predicted_overnight_stays_total, 123.0)

    def test_empty_dataset_is_blocking(self) -> None:
        with self.assertRaises(EmptyInferenceDatasetError):
            self._predict(self.dataframe.iloc[0:0])

    def test_one_month_horizon_is_preserved(self) -> None:
        result = self._predict(forecast_horizon_months=1)

        self.assertEqual(result.forecast_horizon_months, 1)

    def test_unsupported_horizon_is_blocking(self) -> None:
        with self.assertRaisesRegex(UnsupportedHorizonError, "un mes"):
            self._predict(forecast_horizon_months=2)

    def test_missing_configuration_file_is_blocking(self) -> None:
        with self.assertRaises(InferenceConfigurationError):
            predict_next_month(
                "ES-PROV-01",
                dataframe=self.dataframe,
                config_path=Path("configuration-that-does-not-exist.yml"),
                as_of_date="2026-08-13",
            )

    def test_incompatible_baseline_configuration_is_blocking(self) -> None:
        config = deepcopy(self.config)
        config["baseline"]["name"] = "ridge"

        with self.assertRaisesRegex(
            InferenceConfigurationError,
            "seasonal_naive_lag_12",
        ):
            predict_next_month(
                "ES-PROV-01",
                dataframe=self.dataframe,
                config=config,
                as_of_date="2026-08-13",
            )

    def test_incompatible_selected_solution_is_blocking(self) -> None:
        config = deepcopy(self.config)
        config["fallback"]["if_no_candidate_beats_baseline"][
            "selected_solution"
        ] = "ridge"

        with self.assertRaisesRegex(
            InferenceConfigurationError,
            "solucion seleccionada",
        ):
            predict_next_month(
                "ES-PROV-01",
                dataframe=self.dataframe,
                config=config,
                as_of_date="2026-08-13",
            )

    def test_missing_required_column_is_blocking(self) -> None:
        dataframe = self.dataframe.drop(columns="source_snapshot_id")

        with self.assertRaisesRegex(InferenceDataError, "source_snapshot_id"):
            self._predict(dataframe)

    def test_duplicate_territory_month_key_is_blocking(self) -> None:
        dataframe = pd.concat(
            [self.dataframe, self.dataframe.iloc[[0]]],
            ignore_index=True,
        )

        with self.assertRaisesRegex(InferenceDataError, "duplicadas"):
            self._predict(dataframe)

    def test_null_reference_value_is_blocking(self) -> None:
        dataframe = self.dataframe.copy()
        reference_mask = (
            dataframe["territory_id"].eq("ES-PROV-01")
            & dataframe["month_id"].eq("2025-09")
        )
        dataframe.loc[reference_mask, "overnight_stays_total"] = None

        with self.assertRaisesRegex(InferenceDataError, "nulo"):
            self._predict(dataframe)

    def test_invalid_numeric_reference_values_are_blocking(self) -> None:
        invalid_values = [float("nan"), float("inf"), float("-inf"), -1.0]

        for invalid_value in invalid_values:
            with self.subTest(invalid_value=invalid_value):
                dataframe = self.dataframe.copy()
                reference_mask = (
                    dataframe["territory_id"].eq("ES-PROV-01")
                    & dataframe["month_id"].eq("2025-09")
                )
                dataframe.loc[
                    reference_mask,
                    "overnight_stays_total",
                ] = invalid_value

                with self.assertRaises(InferenceDataError):
                    self._predict(dataframe)

    def test_incompatible_lineage_values_are_blocking(self) -> None:
        dataframe = self.dataframe.copy()
        dataframe.loc[dataframe.index[0], "source_snapshot_id"] = "snapshot-002"

        with self.assertRaisesRegex(InferenceDataError, "source_snapshot_id"):
            self._predict(dataframe)

    def test_empty_lineage_values_are_blocking(self) -> None:
        for column in (
            "source_snapshot_id",
            "pipeline_run_id",
            "data_version",
        ):
            with self.subTest(column=column):
                dataframe = self.dataframe.copy()
                dataframe[column] = ""

                with self.assertRaisesRegex(InferenceDataError, column):
                    self._predict(dataframe)


class TestInferenceIntegration(unittest.TestCase):
    """Integracion segura con la Gold operacional versionada."""

    def test_real_gold_supports_forecast_from_august_2026(self) -> None:
        config = load_config()
        gold = pd.read_parquet(config["source_dataset"]["path"])
        reference_rows = gold.loc[
            gold["month_id"].astype(str).eq("2025-09")
            & gold["overnight_stays_total"].notna()
        ].sort_values("territory_id")
        self.assertFalse(reference_rows.empty)
        reference = reference_rows.iloc[0]

        result = predict_next_month(
            str(reference["territory_id"]),
            as_of_date="2026-08-13",
        )

        self.assertEqual(result.target_month_id, "2026-09")
        self.assertEqual(result.reference_month_id, "2025-09")
        self.assertEqual(result.latest_available_month_id, "2026-06")
        self.assertNotEqual(
            pd.Period(result.target_month_id, freq="M"),
            pd.Period(result.latest_available_month_id, freq="M") + 1,
        )
        self.assertEqual(
            result.predicted_overnight_stays_total,
            float(reference["overnight_stays_total"]),
        )
        self.assertEqual(result.operational_status, "forecast_ready")
        self.assertTrue(result.is_operational)


if __name__ == "__main__":
    unittest.main()
