import unittest

from copy import deepcopy

import numpy as np
import pandas as pd

from src.models.modeling_common import (
    calculate_training_label_cutoffs,
    calculate_improvement_pct,
    calculate_metrics,
    common_evaluable_mask,
    ensure_test_window_is_untouched,
    get_model_inputs,
    get_minimum_safe_training_label_lag,
    get_validation_folds,
    load_config,
    training_label_masks,
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
            "point_in_time_availability": {
                "minimum_safe_eotr_lag_months": 3,
                "training_labels": {
                    "require_published_before_forecast_origin": True,
                    "cutoff_uses_minimum_safe_eotr_lag_months": True,
                },
            },
            "validation": {
                "folds": [
                    {
                        "name": "validation_1",
                        "train_end": "2021-05",
                        "validation_start": "2021-06",
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
                    "validation_start": "2021-06",
                    "structural_train_end": "2021-05",
                    "availability_train_end": "2021-03",
                    "effective_train_end": "2021-03",
                    "train_end": "2021-03-01",
                },
                {
                    "name": "test",
                    "validation_start": "2024-06",
                    "structural_train_end": "2024-05",
                    "availability_train_end": "2024-03",
                    "effective_train_end": "2024-03",
                    "train_end": "2024-03-01",
                },
            ],
        )

    def test_training_label_cutoff_uses_monthly_publication_lag(self) -> None:
        cutoffs = calculate_training_label_cutoffs(
            "2022-06",
            "2022-05",
            3,
        )

        self.assertEqual(str(cutoffs["availability_train_end"]), "2022-03")
        self.assertEqual(str(cutoffs["effective_train_end"]), "2022-03")

    def test_structural_cutoff_prevails_when_earlier(self) -> None:
        cutoffs = calculate_training_label_cutoffs(
            "2022-06",
            "2022-02",
            3,
        )

        self.assertEqual(str(cutoffs["effective_train_end"]), "2022-02")

    def test_availability_cutoff_prevails_when_earlier(self) -> None:
        cutoffs = calculate_training_label_cutoffs(
            "2022-06",
            "2022-05",
            3,
        )

        self.assertEqual(str(cutoffs["effective_train_end"]), "2022-03")

    def test_training_mask_excludes_labels_after_effective_cutoff(self) -> None:
        dataframe = pd.DataFrame(
            {
                "target_date_month": pd.to_datetime(
                    ["2022-02-01", "2022-03-01", "2022-04-01", "2022-05-01"]
                )
            }
        )
        fold = {
            "name": "validation_2",
            "structural_train_end": "2022-05",
            "availability_train_end": "2022-03",
            "effective_train_end": "2022-03",
        }

        before, after = training_label_masks(
            dataframe,
            pd.Series(True, index=dataframe.index),
            fold,
        )

        self.assertEqual(before.tolist(), [True, True, True, True])
        self.assertEqual(after.tolist(), [True, True, False, False])

    def test_inconsistent_training_label_policy_is_rejected(self) -> None:
        config = deepcopy(load_config())
        config["point_in_time_availability"]["training_labels"][
            "cutoff_uses_minimum_safe_eotr_lag_months"
        ] = False

        with self.assertRaisesRegex(ValueError, "cutoff de etiquetas"):
            get_minimum_safe_training_label_lag(config)

    def test_eotr_lag_must_match_temporal_lineage(self) -> None:
        config = deepcopy(load_config())
        config["point_in_time_availability"]["eotr_predictor_lags"][
            "lag_3_overnight_stays"
        ] = 4

        with self.assertRaisesRegex(ValueError, "temporal_integrity"):
            get_model_inputs(config)

    def test_opened_test_guard_rejects_before_data_access(self) -> None:
        config = deepcopy(load_config())

        with self.assertRaisesRegex(RuntimeError, "already opened"):
            ensure_test_window_is_untouched(
                config,
                purpose="baseline evaluation",
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
