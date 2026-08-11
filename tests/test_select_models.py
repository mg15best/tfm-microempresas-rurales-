import unittest

import numpy as np
import pandas as pd

from src.models.modeling_common import (
    common_evaluable_mask,
    get_model_inputs,
    get_validation_folds,
    load_config,
)
from src.models.select_models import (
    add_baseline_metrics,
    evaluate_model_fold,
    select_solution_by_validation,
)


class RecordingPipeline:
    """Pipeline mínimo para observar filas train sin entrenar sklearn."""

    fitted_rows: dict[str, int] = {}

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id

    def fit(self, features: pd.DataFrame, target: np.ndarray) -> None:
        self.fitted_rows[self.model_id] = len(target)

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.full(len(features), 90.0)


class TestSelectModels(unittest.TestCase):
    """Pruebas de la regla de selección basada solo en validación."""

    @staticmethod
    def selection_config() -> dict:
        return {
            "model_selection": {
                "minimum_mae_improvement_pct": 5.0,
            },
            "fallback": {
                "if_no_candidate_beats_baseline": {
                    "selected_solution": "seasonal_naive_lag_12",
                },
            },
        }

    def test_improvement_below_threshold_keeps_baseline(self) -> None:
        selected, improvement = select_solution_by_validation(
            self.selection_config(),
            "candidate",
            baseline_mae=100.0,
            candidate_mae=96.0,
        )

        self.assertAlmostEqual(improvement, 4.0)
        self.assertEqual(selected, "seasonal_naive_lag_12")

    def test_improvement_at_threshold_selects_candidate(self) -> None:
        selected, improvement = select_solution_by_validation(
            self.selection_config(),
            "candidate",
            baseline_mae=100.0,
            candidate_mae=95.0,
        )

        self.assertAlmostEqual(improvement, 5.0)
        self.assertEqual(selected, "candidate")

    @staticmethod
    def modeling_rows() -> pd.DataFrame:
        config = load_config()
        _, _, _, features = get_model_inputs(config)
        dates = pd.to_datetime(
            [
                "2022-02-01",
                "2022-03-01",
                "2022-04-01",
                "2022-05-01",
                "2022-06-01",
            ]
        )
        dataframe = pd.DataFrame(
            {
                "territory_id": ["ES01"] * 5,
                "territory_name": ["Territory"] * 5,
                "target_month_id": dates.strftime("%Y-%m"),
                "target_date_month": dates,
                "evaluation_split": ["train"] * 4 + ["validation_2"],
                "is_provisional": [False] * 5,
                "target_overnight_stays_total": [100.0] * 5,
                "lag_12_overnight_stays": [80.0] * 5,
                "source_snapshot_id": ["snapshot"] * 5,
                "pipeline_run_id": ["run"] * 5,
                "data_version": ["version"] * 5,
                "created_at": pd.to_datetime(["2026-01-01"] * 5),
            }
        )
        for feature in features:
            if feature not in dataframe:
                dataframe[feature] = (
                    "ES01" if feature == "territory_id" else 1.0
                )
        return dataframe

    def test_ridge_and_hgb_use_same_effective_cutoff(self) -> None:
        config = load_config()
        dataframe = self.modeling_rows()
        evaluable = common_evaluable_mask(dataframe, config)
        fold = next(
            item
            for item in get_validation_folds(config, include_test=False)
            if item["name"] == "validation_2"
        )

        rows = []
        for model_id in ["ridge_alpha_100", "hgb_raw_02"]:
            _, metrics = evaluate_model_fold(
                dataframe,
                evaluable,
                config,
                fold,
                model_id,
                lambda model_id=model_id: RecordingPipeline(model_id),
            )
            rows.append(metrics)

        self.assertEqual(
            {row["effective_train_end"] for row in rows},
            {"2022-03"},
        )
        self.assertEqual(
            RecordingPipeline.fitted_rows,
            {"ridge_alpha_100": 2, "hgb_raw_02": 2},
        )

    def test_baseline_metrics_do_not_depend_on_training_purge(self) -> None:
        predictions = pd.DataFrame(
            {
                "territory_id": ["ES01"],
                "target_month_id": ["2022-06"],
                "evaluation_split": ["validation_2"],
                "actual": [100.0],
                "baseline_prediction": [80.0],
            }
        )
        metrics_rows: list[dict] = []

        add_baseline_metrics(
            predictions,
            metrics_rows,
            "seasonal_naive_lag_12",
        )

        self.assertEqual(metrics_rows[0]["train_end"], "not_applicable")
        self.assertAlmostEqual(metrics_rows[0]["MAE"], 20.0)

    def test_selection_folds_exclude_test(self) -> None:
        folds = get_validation_folds(load_config(), include_test=False)

        self.assertEqual(
            [fold["name"] for fold in folds],
            ["validation_1", "validation_2", "validation_3"],
        )
        self.assertNotIn("test", {fold["name"] for fold in folds})


if __name__ == "__main__":
    unittest.main()
