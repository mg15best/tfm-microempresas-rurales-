import unittest

from dataclasses import asdict
from math import sqrt

import numpy as np
import pandas as pd

from src.visualization.dashboard_data import (
    BaselineReplicaError,
    DashboardDataError,
    InvalidTerritoryError,
    LineageMismatchError,
    MetricsReconciliationError,
    build_dashboard_context,
    calculate_territory_validation_metrics,
    get_territory_history,
    get_territory_validation_metrics,
    load_official_validation_metrics,
    load_validation_predictions,
    prepare_baseline_validation_predictions,
    reconcile_official_pooled_metrics,
    validate_lineage_compatibility,
)
from src.models.modeling_common import load_config


class TestDashboardData(unittest.TestCase):
    """Pruebas de datos historicos y metricas para presentacion."""

    def setUp(self) -> None:
        self.config = {
            "source_dataset": {
                "path": "data/gold/gold_tourism_demand_monthly.parquet",
            },
            "modeling_dataset": {
                "path": "data/gold/gold_modeling_dataset_monthly.parquet",
            },
            "baseline": {
                "name": "seasonal_naive_lag_12",
            },
            "fallback": {
                "if_no_candidate_beats_baseline": {
                    "selected_solution": "seasonal_naive_lag_12",
                },
            },
            "validation": {
                "folds": [
                    {
                        "name": "validation_1",
                    },
                ],
            },
        }
        self.gold = self._build_gold()
        self.predictions = self._build_predictions()
        self.official_metrics = self._build_official_metrics()

    @staticmethod
    def _build_gold() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "territory_id": [
                    "ES-PROV-01",
                    "ES-PROV-01",
                    "ES-PROV-01",
                    "ES-PROV-02",
                ],
                "territory_name": [
                    "Provincia A",
                    "Provincia A",
                    "Provincia A",
                    "Provincia B",
                ],
                "month_id": ["2023-03", "2023-01", "2023-04", "2023-01"],
                "date_month": pd.to_datetime(
                    ["2023-03-01", "2023-01-01", "2023-04-01", "2023-01-01"]
                ),
                "overnight_stays_total": [130, 100, 140, 0],
                "data_status": [
                    "final_or_not_marked_provisional",
                    "final_or_not_marked_provisional",
                    "provisional",
                    "final_or_not_marked_provisional",
                ],
                "is_provisional": [False, False, True, False],
                "coverage_quality": "high",
                "source_snapshot_id": "snapshot-001",
                "pipeline_run_id": "gold-run-001",
                "data_version": "gold-v1",
            }
        )

    @staticmethod
    def _build_predictions() -> pd.DataFrame:
        observations = [
            (
                "ES-PROV-01",
                "Provincia A",
                "2023-01",
                100.0,
                90.0,
            ),
            (
                "ES-PROV-01",
                "Provincia A",
                "2023-02",
                200.0,
                220.0,
            ),
            (
                "ES-PROV-02",
                "Provincia B",
                "2023-01",
                0.0,
                0.0,
            ),
        ]
        records: list[dict[str, object]] = []
        for model in ("ridge_alpha_1", "hgb_raw_02"):
            for (
                territory_id,
                territory_name,
                month_id,
                actual,
                baseline,
            ) in observations:
                records.append(
                    {
                        "territory_id": territory_id,
                        "territory_name": territory_name,
                        "target_month_id": month_id,
                        "target_date_month": pd.Timestamp(f"{month_id}-01"),
                        "evaluation_split": "validation_1",
                        "source_snapshot_id": "snapshot-001",
                        "pipeline_run_id": "model-run-001",
                        "data_version": "modeling-v1",
                        "created_at": pd.Timestamp("2026-08-03", tz="UTC"),
                        "model": model,
                        "baseline_id": "seasonal_naive_lag_12",
                        "dataset_path": (
                            "data/gold/gold_modeling_dataset_monthly.parquet"
                        ),
                        "actual": actual,
                        "baseline_prediction": baseline,
                        "validation_start": "2023-01",
                        "structural_train_end": "2022-12",
                        "availability_train_end": "2022-10",
                        "effective_train_end": "2022-10",
                    }
                )
        return pd.DataFrame.from_records(records)

    @staticmethod
    def _build_official_metrics() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "evaluation_split": ["validation_pooled"],
                "model": ["seasonal_naive_lag_12"],
                "rows": [3],
                "MAE": [10.0],
                "RMSE": [sqrt(500.0 / 3.0)],
                "WAPE_pct": [10.0],
                "mean_bias": [10.0 / 3.0],
            }
        )

    def test_valid_territory_history_has_expected_columns(self) -> None:
        history = get_territory_history(
            "ES-PROV-01",
            dataframe=self.gold,
            config=self.config,
        )

        self.assertEqual(
            history.columns.tolist(),
            [
                "territory_id",
                "territory_name",
                "month_id",
                "date_month",
                "overnight_stays_total",
                "data_status",
                "is_provisional",
                "coverage_quality",
            ],
        )
        self.assertEqual(len(history), 3)

    def test_invalid_territory_is_blocking(self) -> None:
        with self.assertRaises(InvalidTerritoryError):
            get_territory_history(
                "ES-PROV-99",
                dataframe=self.gold,
                config=self.config,
            )

    def test_history_is_chronological_and_month_limit_uses_latest(self) -> None:
        history = get_territory_history(
            "ES-PROV-01",
            months=2,
            dataframe=self.gold,
            config=self.config,
        )

        self.assertEqual(history["month_id"].tolist(), ["2023-03", "2023-04"])

    def test_month_limit_is_a_natural_month_window_with_gaps(self) -> None:
        gold = self.gold.copy()
        april = gold["month_id"].eq("2023-04")
        gold.loc[april, "month_id"] = "2023-05"
        gold.loc[april, "date_month"] = pd.Timestamp("2023-05-01")

        history = get_territory_history(
            "ES-PROV-01",
            months=2,
            dataframe=gold,
            config=self.config,
        )

        self.assertEqual(history["month_id"].tolist(), ["2023-05"])

    def test_invalid_month_limits_are_rejected(self) -> None:
        for months in (0, -1, True, 1.5):
            with self.subTest(months=months):
                with self.assertRaises(ValueError):
                    get_territory_history(
                        "ES-PROV-01",
                        months=months,
                        dataframe=self.gold,
                        config=self.config,
                    )

    def test_month_limit_larger_than_history_returns_all_rows(self) -> None:
        history = get_territory_history(
            "ES-PROV-01",
            months=120,
            dataframe=self.gold,
            config=self.config,
        )

        self.assertEqual(len(history), 3)

    def test_empty_gold_is_blocking(self) -> None:
        with self.assertRaises(DashboardDataError):
            get_territory_history(
                "ES-PROV-01",
                dataframe=self.gold.iloc[0:0],
                config=self.config,
            )

    def test_duplicate_gold_territory_month_is_blocking(self) -> None:
        gold = pd.concat(
            [self.gold, self.gold.iloc[[0]]],
            ignore_index=True,
        )

        with self.assertRaisesRegex(DashboardDataError, "duplicadas"):
            get_territory_history(
                "ES-PROV-01",
                dataframe=gold,
                config=self.config,
            )

    def test_history_preserves_provisionality(self) -> None:
        history = get_territory_history(
            "ES-PROV-01",
            dataframe=self.gold,
            config=self.config,
        )

        self.assertEqual(history["is_provisional"].tolist(), [False, False, True])
        self.assertEqual(history.iloc[-1]["data_status"], "provisional")

    def test_history_does_not_impute_missing_months(self) -> None:
        history = get_territory_history(
            "ES-PROV-01",
            dataframe=self.gold,
            config=self.config,
        )

        self.assertEqual(
            history["month_id"].tolist(),
            ["2023-01", "2023-03", "2023-04"],
        )
        self.assertNotIn("2023-02", history["month_id"].tolist())

    def test_baseline_replicas_are_identified_by_candidate(self) -> None:
        unique = prepare_baseline_validation_predictions(
            self.predictions,
            self.config,
        )

        self.assertEqual(len(self.predictions), 6)
        self.assertEqual(len(unique), 3)
        self.assertEqual(
            set(self.predictions["model"]),
            {"ridge_alpha_1", "hgb_raw_02"},
        )
        self.assertNotIn("model", unique.columns)

    def test_candidate_order_does_not_change_deduplicated_baseline(self) -> None:
        first = prepare_baseline_validation_predictions(
            self.predictions,
            self.config,
        )
        reversed_predictions = self.predictions.iloc[::-1].reset_index(drop=True)
        second = prepare_baseline_validation_predictions(
            reversed_predictions,
            self.config,
        )

        pd.testing.assert_frame_equal(first, second)
        self.assertNotIn("model", first.columns)

    def test_different_baseline_replica_is_blocking(self) -> None:
        predictions = self.predictions.copy()
        mask = (
            predictions["model"].eq("hgb_raw_02")
            & predictions["territory_id"].eq("ES-PROV-01")
            & predictions["target_month_id"].eq("2023-01")
        )
        predictions.loc[mask, "baseline_prediction"] = 91.0

        with self.assertRaisesRegex(
            BaselineReplicaError,
            "baseline_prediction",
        ):
            prepare_baseline_validation_predictions(
                predictions,
                self.config,
            )

    def test_missing_candidate_replica_is_blocking(self) -> None:
        predictions = self.predictions.drop(index=self.predictions.index[-1])

        with self.assertRaisesRegex(BaselineReplicaError, "candidatos"):
            prepare_baseline_validation_predictions(
                predictions,
                self.config,
            )

    def test_duplicate_observation_candidate_is_blocking(self) -> None:
        predictions = pd.concat(
            [self.predictions, self.predictions.iloc[[0]]],
            ignore_index=True,
        )

        with self.assertRaisesRegex(BaselineReplicaError, "candidato"):
            prepare_baseline_validation_predictions(
                predictions,
                self.config,
            )

    def test_invalid_actual_or_prediction_is_blocking(self) -> None:
        invalid_values = [float("nan"), float("inf"), float("-inf"), -1.0]

        for column in ("actual", "baseline_prediction"):
            for value in invalid_values:
                with self.subTest(column=column, value=value):
                    predictions = self.predictions.copy()
                    logical_observation = (
                        predictions["territory_id"].eq("ES-PROV-01")
                        & predictions["target_month_id"].eq("2023-01")
                    )
                    predictions.loc[logical_observation, column] = value

                    with self.assertRaises(BaselineReplicaError):
                        prepare_baseline_validation_predictions(
                            predictions,
                            self.config,
                        )

    def test_unique_validation_lineage_is_accepted(self) -> None:
        unique = prepare_baseline_validation_predictions(
            self.predictions,
            self.config,
        )

        self.assertEqual(
            unique["source_snapshot_id"].unique().tolist(),
            ["snapshot-001"],
        )

    def test_multiple_validation_lineage_is_blocking(self) -> None:
        for column in (
            "source_snapshot_id",
            "pipeline_run_id",
            "data_version",
        ):
            with self.subTest(column=column):
                predictions = self.predictions.copy()
                territory = predictions["territory_id"].eq("ES-PROV-02")
                predictions.loc[territory, column] = "incompatible-value"

                with self.assertRaises(LineageMismatchError):
                    calculate_territory_validation_metrics(
                        predictions,
                        self.config,
                    )

    def test_empty_or_null_validation_lineage_is_blocking(self) -> None:
        for column in (
            "source_snapshot_id",
            "pipeline_run_id",
            "data_version",
        ):
            for value in ("", None):
                with self.subTest(column=column, value=value):
                    predictions = self.predictions.copy()
                    predictions[column] = value

                    with self.assertRaises(LineageMismatchError):
                        get_territory_validation_metrics(
                            "ES-PROV-01",
                            dataframe=predictions,
                            config=self.config,
                        )

    def test_territory_mae_is_correct(self) -> None:
        metrics = get_territory_validation_metrics(
            "ES-PROV-01",
            dataframe=self.predictions,
            config=self.config,
        )

        self.assertAlmostEqual(metrics.validation_mae, 15.0)

    def test_territory_rmse_is_correct(self) -> None:
        metrics = get_territory_validation_metrics(
            "ES-PROV-01",
            dataframe=self.predictions,
            config=self.config,
        )

        self.assertAlmostEqual(metrics.validation_rmse, sqrt(250.0))

    def test_territory_wape_is_correct(self) -> None:
        metrics = get_territory_validation_metrics(
            "ES-PROV-01",
            dataframe=self.predictions,
            config=self.config,
        )

        self.assertAlmostEqual(metrics.validation_wape_pct, 10.0)

    def test_territory_bias_is_correct(self) -> None:
        metrics = get_territory_validation_metrics(
            "ES-PROV-01",
            dataframe=self.predictions,
            config=self.config,
        )

        self.assertAlmostEqual(metrics.validation_bias, 5.0)

    def test_territory_sample_size_is_correct(self) -> None:
        metrics = get_territory_validation_metrics(
            "ES-PROV-01",
            dataframe=self.predictions,
            config=self.config,
        )

        self.assertEqual(metrics.validation_rows, 2)

    def test_metrics_are_calculated_once_per_territory(self) -> None:
        metrics = calculate_territory_validation_metrics(
            self.predictions,
            self.config,
        )

        self.assertEqual(metrics["territory_id"].tolist(), ["ES-PROV-01", "ES-PROV-02"])
        self.assertFalse(metrics["territory_id"].duplicated().any())

    def test_territory_without_predictions_is_blocking(self) -> None:
        with self.assertRaisesRegex(InvalidTerritoryError, "evaluables"):
            get_territory_validation_metrics(
                "ES-PROV-99",
                dataframe=self.predictions,
                config=self.config,
            )

    def test_zero_wape_denominator_returns_nan(self) -> None:
        metrics = get_territory_validation_metrics(
            "ES-PROV-02",
            dataframe=self.predictions,
            config=self.config,
        )

        self.assertTrue(np.isnan(metrics.validation_wape_pct))
        self.assertEqual(metrics.validation_rows, 1)

    def test_official_metrics_are_reconciled(self) -> None:
        unique = prepare_baseline_validation_predictions(
            self.predictions,
            self.config,
        )

        calculated = reconcile_official_pooled_metrics(
            unique,
            self.official_metrics,
            self.config,
        )

        self.assertEqual(calculated["rows"], 3)
        self.assertAlmostEqual(calculated["WAPE_pct"], 10.0)

    def test_official_metrics_mismatch_is_blocking(self) -> None:
        unique = prepare_baseline_validation_predictions(
            self.predictions,
            self.config,
        )
        official = self.official_metrics.copy()
        official.loc[0, "MAE"] = 999.0

        with self.assertRaises(MetricsReconciliationError):
            reconcile_official_pooled_metrics(
                unique,
                official,
                self.config,
            )

    def test_compatible_lineage_is_returned(self) -> None:
        unique = prepare_baseline_validation_predictions(
            self.predictions,
            self.config,
        )

        lineage = validate_lineage_compatibility(
            self.gold,
            unique,
            self.config,
        )

        self.assertEqual(lineage.source_snapshot_id, "snapshot-001")
        self.assertEqual(lineage.gold_pipeline_run_id, "gold-run-001")
        self.assertEqual(lineage.validation_pipeline_run_id, "model-run-001")

    def test_incompatible_lineage_is_blocking(self) -> None:
        gold = self.gold.copy()
        gold["source_snapshot_id"] = "snapshot-002"
        unique = prepare_baseline_validation_predictions(
            self.predictions,
            self.config,
        )

        with self.assertRaises(LineageMismatchError):
            validate_lineage_compatibility(
                gold,
                unique,
                self.config,
            )

    def test_context_is_exportable_as_simple_tables(self) -> None:
        context = build_dashboard_context(
            "ES-PROV-01",
            gold=self.gold,
            predictions=self.predictions,
            official_metrics=self.official_metrics,
            config=self.config,
        )

        export = context.to_export_frames()

        self.assertEqual(
            set(export),
            {"history", "validation_metrics", "lineage"},
        )
        self.assertEqual(len(export["validation_metrics"]), 1)
        self.assertEqual(
            asdict(context.validation_metrics)["validation_rows"],
            2,
        )

    def test_context_rejects_inconsistent_territory_name(self) -> None:
        gold = self.gold.copy()
        gold.loc[
            gold["territory_id"].eq("ES-PROV-01"),
            "territory_name",
        ] = "Nombre diferente"

        with self.assertRaisesRegex(DashboardDataError, "nombre territorial"):
            build_dashboard_context(
                "ES-PROV-01",
                gold=gold,
                predictions=self.predictions,
                official_metrics=self.official_metrics,
                config=self.config,
            )

    def test_incompatible_selected_solution_is_blocking(self) -> None:
        config = dict(self.config)
        config["fallback"] = {
            "if_no_candidate_beats_baseline": {
                "selected_solution": "ridge",
            },
        }

        with self.assertRaises(DashboardDataError):
            prepare_baseline_validation_predictions(
                self.predictions,
                config,
            )

    def test_non_operational_baseline_and_selection_are_blocking(self) -> None:
        config = dict(self.config)
        config["baseline"] = {"name": "ridge"}
        config["fallback"] = {
            "if_no_candidate_beats_baseline": {
                "selected_solution": "ridge",
            },
        }
        predictions = self.predictions.copy()
        predictions["baseline_id"] = "ridge"

        with self.assertRaisesRegex(DashboardDataError, "seasonal_naive_lag_12"):
            prepare_baseline_validation_predictions(
                predictions,
                config,
            )


class TestDashboardDataIntegration(unittest.TestCase):
    """Integracion con los artefactos versionados vigentes."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config()
        source_path = cls.config["source_dataset"]["path"]
        cls.gold = pd.read_parquet(source_path)
        cls.predictions = load_validation_predictions()
        cls.official = load_official_validation_metrics()

    def test_real_baseline_replication_contract(self) -> None:
        key = [
            "territory_id",
            "target_month_id",
            "evaluation_split",
        ]
        unique = prepare_baseline_validation_predictions(
            self.predictions,
            self.config,
        )
        replica_counts = self.predictions.groupby(key).size()

        self.assertEqual(len(self.predictions), 12_250)
        self.assertEqual(len(unique), 1_750)
        self.assertEqual(replica_counts.unique().tolist(), [7])
        self.assertEqual(self.predictions["model"].nunique(), 7)
        self.assertNotIn("model", unique.columns)

    def test_real_baseline_reproduces_official_pooled_metrics(self) -> None:
        unique = prepare_baseline_validation_predictions(
            self.predictions,
            self.config,
        )

        calculated = reconcile_official_pooled_metrics(
            unique,
            self.official,
            self.config,
        )

        official = self.official.loc[
            self.official["evaluation_split"].eq("validation_pooled")
            & self.official["model"].eq("seasonal_naive_lag_12")
        ].iloc[0]
        self.assertEqual(calculated["rows"], int(official["rows"]))
        for column in ("MAE", "RMSE", "WAPE_pct", "mean_bias"):
            self.assertAlmostEqual(
                calculated[column],
                float(official[column]),
                delta=1e-6,
            )

    def test_real_territory_metrics_cover_expected_panel(self) -> None:
        metrics = calculate_territory_validation_metrics(
            self.predictions,
            self.config,
        )

        self.assertEqual(len(metrics), 50)
        self.assertEqual(metrics["validation_rows"].unique().tolist(), [35])
        self.assertFalse(metrics["territory_id"].duplicated().any())

    def test_real_gold_and_validation_lineage_are_compatible(self) -> None:
        unique = prepare_baseline_validation_predictions(
            self.predictions,
            self.config,
        )

        lineage = validate_lineage_compatibility(
            self.gold,
            unique,
            self.config,
        )

        self.assertEqual(
            lineage.source_snapshot_id,
            self.gold["source_snapshot_id"].iloc[0],
        )


if __name__ == "__main__":
    unittest.main()
