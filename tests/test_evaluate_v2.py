import unittest

import numpy as np
import pandas as pd

from src.models.evaluate_v2 import (
    assert_validation_targets_are_final,
    build_baseline_predictions,
    calculate_fold_metrics,
    calculate_month_metrics,
    calculate_origin_metrics,
    calculate_pooled_metrics,
    calculate_skill_mae_pct,
    calculate_territory_metrics,
    compare_with_frozen_v1,
    comparable_predictions,
    first_fold_stress_evidence,
    load_gold_history,
    validate_expected_baseline_rows,
)
from src.models.modeling_v2_common import load_modeling_v2_config


class TestEvaluateV2Integration(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_modeling_v2_config()
        cls.gold = load_gold_history()
        cls.predictions = build_baseline_predictions(cls.gold, cls.config)
        cls.comparable = comparable_predictions(cls.predictions)

    def test_baseline_reproduces_expected_rows(self) -> None:
        counts = validate_expected_baseline_rows(
            self.predictions, self.config
        )

        self.assertEqual(
            counts,
            {
                "validation_1": 550,
                "validation_2": 600,
                "validation_3": 600,
            },
        )
        self.assertEqual(len(self.comparable), 1750)
        self.assertEqual(len(self.predictions), 1800)

    def test_real_araba_prediction_equals_exact_lag_twelve(self) -> None:
        row = self.predictions.loc[
            self.predictions["territory_id"].eq("ES-PROV-01")
            & self.predictions["target_month_id"].eq("2023-09")
        ].iloc[0]
        expected = self.gold.loc[
            self.gold["territory_id"].eq("ES-PROV-01")
            & self.gold["month_id"].astype(str).eq("2022-09"),
            "overnight_stays_total",
        ].iloc[0]

        self.assertEqual(row["seasonal_reference_month_id"], "2022-09")
        self.assertEqual(float(row["prediction"]), float(expected))
        self.assertEqual(float(row["reference"]), float(expected))

    def test_november_2021_gap_is_not_imputed(self) -> None:
        gap = self.predictions.loc[
            self.predictions["target_month_id"].eq("2021-11")
        ]

        self.assertEqual(len(gap), 50)
        self.assertFalse(gap["prediction_available"].any())
        self.assertTrue(gap["prediction"].isna().all())
        self.assertEqual(
            gap["availability_reason"].unique().tolist(),
            ["missing_seasonal_reference"],
        )

    def test_all_validation_targets_are_final(self) -> None:
        assert_validation_targets_are_final(self.predictions)

        self.assertFalse(
            self.predictions["target_is_provisional"].fillna(True).any()
        )
        self.assertEqual(
            set(self.predictions["target_data_status"].astype(str)),
            {"final_or_not_marked_provisional"},
        )

    def test_fold_metrics_reproduce_known_baseline(self) -> None:
        metrics = calculate_fold_metrics(self.predictions, self.config).set_index(
            "fold_id"
        )
        expected = {
            "validation_1": (
                550,
                9116.563636,
                16454.667811,
                44.598291,
                -8782.887273,
            ),
            "validation_2": (
                600,
                3071.713333,
                6094.109335,
                14.912184,
                -909.253333,
            ),
            "validation_3": (
                600,
                3209.141667,
                5552.102022,
                15.158831,
                -571.431667,
            ),
        }

        for fold_id, (n, mae, rmse, wape, bias) in expected.items():
            self.assertEqual(int(metrics.loc[fold_id, "n"]), n)
            self.assertAlmostEqual(metrics.loc[fold_id, "MAE"], mae, places=5)
            self.assertAlmostEqual(
                metrics.loc[fold_id, "RMSE"], rmse, places=5
            )
            self.assertAlmostEqual(
                metrics.loc[fold_id, "WAPE_pct"], wape, places=5
            )
            self.assertAlmostEqual(
                metrics.loc[fold_id, "bias"], bias, places=5
            )
            self.assertEqual(metrics.loc[fold_id, "skill_mae_pct"], 0.0)

    def test_pooled_metrics_are_reconstructed_from_rows(self) -> None:
        pooled = calculate_pooled_metrics(self.predictions).iloc[0]

        self.assertEqual(int(pooled["n"]), 1750)
        self.assertAlmostEqual(pooled["MAE"], 5018.641714, places=5)
        self.assertAlmostEqual(pooled["RMSE"], 10411.374392, places=5)
        self.assertAlmostEqual(pooled["WAPE_pct"], 24.191817, places=5)
        self.assertAlmostEqual(pooled["bias"], -3267.999429, places=5)
        self.assertEqual(pooled["skill_mae_pct"], 0.0)

    def test_territory_month_and_origin_metrics_preserve_panel_shape(self) -> None:
        territory = calculate_territory_metrics(self.predictions)
        month = calculate_month_metrics(self.predictions).set_index(
            "month_number"
        )
        origin = calculate_origin_metrics(self.predictions).set_index(
            "target_month_id"
        )

        self.assertEqual(len(territory), 50)
        self.assertEqual(int(territory["n"].min()), 35)
        self.assertEqual(int(territory["n"].max()), 35)
        self.assertEqual(month.index.tolist(), list(range(1, 13)))
        self.assertEqual(int(month.loc[11, "n"]), 100)
        self.assertTrue((month.drop(index=11)["n"] == 150).all())
        self.assertEqual(len(origin), 36)
        self.assertEqual(int(origin["n_territories"].min()), 0)
        self.assertEqual(int(origin["n_territories"].max()), 50)
        self.assertEqual(int(origin.loc["2021-11", "n_territories"]), 0)

    def test_frozen_v1_baseline_matches_exactly(self) -> None:
        comparison = compare_with_frozen_v1(self.predictions)

        self.assertTrue(comparison.exact_match)
        self.assertEqual(comparison.v2_rows, 1750)
        self.assertEqual(comparison.v1_comparable_rows, 1750)
        self.assertEqual(comparison.common_rows, 1750)
        self.assertEqual(comparison.missing_in_v1, 0)
        self.assertEqual(comparison.extra_in_v1, 0)
        self.assertEqual(comparison.prediction_mismatches, 0)
        self.assertEqual(comparison.actual_mismatches, 0)

    def test_first_fold_is_reported_as_stress_without_causal_claim(self) -> None:
        evidence = first_fold_stress_evidence(self.predictions)

        self.assertEqual(evidence["interpretation"], "stress_period")
        self.assertEqual(evidence["n"], 550)
        self.assertEqual(evidence["missing_origin_months"], ["2021-11"])
        self.assertEqual(evidence["missing_prediction_rows"], 50)
        self.assertGreater(evidence["actual_reference_ratio"], 1.0)

    def test_predictions_are_prequentially_ordered(self) -> None:
        expected = self.predictions.sort_values(
            ["target_month_id", "territory_id"], ignore_index=True
        )

        pd.testing.assert_frame_equal(self.predictions, expected)

    def test_skill_contract_is_zero_for_baseline_against_itself(self) -> None:
        self.assertEqual(calculate_skill_mae_pct(10.0, 10.0), 0.0)
        self.assertGreater(calculate_skill_mae_pct(10.0, 9.0), 0.0)
        self.assertLess(calculate_skill_mae_pct(10.0, 11.0), 0.0)
        self.assertTrue(np.isnan(calculate_skill_mae_pct(0.0, 1.0)))


if __name__ == "__main__":
    unittest.main()
