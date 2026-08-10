import unittest

import numpy as np
import pandas as pd

from src.models.modeling_common import (
    calculate_improvement_pct,
    calculate_metrics,
    common_evaluable_mask,
    get_model_inputs,
    get_validation_folds,
)


class TestModelingCommon(unittest.TestCase):
    """Pruebas de utilidades compartidas de modelado."""

    def test_get_model_inputs_returns_configured_features(self) -> None:
        config = {
            "model_inputs": {
                "numeric_features": [
                    "year",
                    "is_summer",
                ],
                "categorical_features": [
                    "territory_id",
                ],
                "boolean_features": [
                    "is_summer",
                ],
            }
        }

        (
            numeric,
            categorical,
            boolean,
            feature_columns,
        ) = get_model_inputs(config)

        self.assertEqual(
            numeric,
            ["year", "is_summer"],
        )
        self.assertEqual(
            categorical,
            ["territory_id"],
        )
        self.assertEqual(
            boolean,
            ["is_summer"],
        )
        self.assertEqual(
            feature_columns,
            [
                "year",
                "is_summer",
                "territory_id",
            ],
        )

    def test_get_model_inputs_rejects_duplicate_predictors(self) -> None:
        config = {
            "model_inputs": {
                "numeric_features": [
                    "year",
                ],
                "categorical_features": [
                    "year",
                ],
                "boolean_features": [],
            }
        }

        with self.assertRaises(ValueError):
            get_model_inputs(config)

    def test_get_validation_folds_adds_final_test(self) -> None:
        config = {
            "validation": {
                "folds": [
                    {
                        "name": "validation_1",
                        "train_end": "2021-05",
                    }
                ],
                "final_test": {
                    "start": "2024-06",
                },
            }
        }

        folds = get_validation_folds(
            config,
            include_test=True,
        )

        self.assertEqual(
            folds,
            [
                {
                    "name": "validation_1",
                    "train_end": "2021-05-01",
                },
                {
                    "name": "test",
                    "train_end": "2024-05-01",
                },
            ],
        )

    def test_common_evaluable_mask_requires_comparable_rows(self) -> None:
        config = {
            "target": {
                "column": "target",
            },
            "baseline": {
                "prediction_feature": "baseline",
            },
        }

        dataframe = pd.DataFrame(
            {
                "target": [
                    100.0,
                    np.nan,
                    100.0,
                    100.0,
                ],
                "baseline": [
                    90.0,
                    90.0,
                    np.nan,
                    90.0,
                ],
                "is_provisional": [
                    False,
                    False,
                    False,
                    True,
                ],
            }
        )

        mask = common_evaluable_mask(
            dataframe,
            config,
        )

        self.assertEqual(
            mask.tolist(),
            [True, False, False, False],
        )

    def test_calculate_metrics_returns_expected_values(self) -> None:
        metrics = calculate_metrics(
            np.array([100.0, 200.0]),
            np.array([110.0, 180.0]),
        )

        self.assertEqual(metrics["rows"], 2)
        self.assertAlmostEqual(metrics["MAE"], 15.0)
        self.assertAlmostEqual(
            metrics["RMSE"],
            np.sqrt(250.0),
        )
        self.assertAlmostEqual(metrics["WAPE_pct"], 10.0)
        self.assertAlmostEqual(metrics["mean_bias"], -5.0)

    def test_calculate_improvement_pct(self) -> None:
        self.assertAlmostEqual(
            calculate_improvement_pct(
                100.0,
                80.0,
            ),
            20.0,
        )

        self.assertTrue(
            np.isnan(
                calculate_improvement_pct(
                    0.0,
                    0.0,
                )
            )
        )


if __name__ == "__main__":
    unittest.main()
