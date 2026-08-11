import unittest

from copy import deepcopy

import numpy as np
import pandas as pd

from src.models.modeling_common import (
    calculate_improvement_pct,
    calculate_metrics,
    common_evaluable_mask,
    get_model_inputs,
    get_validation_folds,
    load_config,
)


class TestModelingCommon(unittest.TestCase):
    """Pruebas de utilidades compartidas de modelado."""

    def test_get_model_inputs_returns_configured_features(self) -> None:
        config = {
            "problem": {
                "forecast_horizon_months": 1,
            },
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
            },
            "point_in_time_availability": {
                "forecast_horizon_months": 1,
                "forecast_origin": "end_of_month_t_before_target_t_plus_1",
                "canonical_forecast_origin": (
                    "end_of_month_t_before_target_t_plus_1"
                ),
                "minimum_safe_eotr_lag_months": 3,
                "known_in_advance_predictors": [
                    "year",
                    "is_summer",
                    "territory_id",
                ],
                "eotr_predictor_lags": {},
                "unavailable_at_forecast_origin": [],
            },
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

    def test_current_operational_inputs_are_point_in_time_safe(self) -> None:
        config = load_config()

        _, _, _, feature_columns = get_model_inputs(config)

        self.assertEqual(
            feature_columns,
            [
                "year",
                "is_summer",
                "is_christmas_period",
                "lag_3_overnight_stays",
                "lag_12_overnight_stays",
                "lag_12_occupancy_rate_pct",
                "lag_12_average_stay",
                "territory_id",
                "month",
                "quarter",
            ],
        )

    def test_incorrect_forecast_origin_is_rejected(self) -> None:
        config = deepcopy(load_config())
        config["point_in_time_availability"][
            "forecast_origin"
        ] = "anything_nonempty"

        with self.assertRaisesRegex(ValueError, "forecast origin"):
            get_model_inputs(config)

    def test_unclassified_operational_input_is_rejected(self) -> None:
        config = deepcopy(load_config())
        config["model_inputs"]["numeric_features"].append(
            "unclassified_eotr_predictor"
        )

        with self.assertRaisesRegex(ValueError, "sin clasificacion"):
            get_model_inputs(config)

    def test_conflicting_availability_classification_is_rejected(self) -> None:
        config = deepcopy(load_config())
        config["point_in_time_availability"][
            "known_in_advance_predictors"
        ].append("lag_1_overnight_stays")

        with self.assertRaisesRegex(ValueError, "simultaneamente"):
            get_model_inputs(config)

    def test_unavailable_inputs_are_rejected(self) -> None:
        unavailable_predictors = [
            "lag_1_overnight_stays",
            "lag_1_occupancy_rate_pct",
            "rolling_mean_3m_overnight_stays",
            "rolling_mean_12m_overnight_stays",
            "yoy_change_overnight_stays",
        ]

        for predictor in unavailable_predictors:
            with self.subTest(predictor=predictor):
                config = deepcopy(load_config())
                config["model_inputs"]["numeric_features"].append(
                    predictor
                )

                with self.assertRaisesRegex(
                    ValueError,
                    "no disponibles",
                ):
                    get_model_inputs(config)

    def test_eotr_predictor_below_minimum_lag_is_rejected(self) -> None:
        config = deepcopy(load_config())
        predictor = "example_eotr_lag_2"
        config["model_inputs"]["numeric_features"].append(predictor)
        config["point_in_time_availability"][
            "eotr_predictor_lags"
        ][predictor] = 2

        with self.assertRaisesRegex(ValueError, "minimo seguro de 3 meses"):
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
