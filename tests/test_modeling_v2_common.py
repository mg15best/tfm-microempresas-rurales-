import unittest

import pandas as pd

from src.models.modeling_v2_common import (
    assert_training_labels_within_cutoff,
    build_backtest_origins,
    cutoff_policy_from_config,
    filter_history_to_information_cutoff,
    load_feature_availability_v2,
    load_modeling_v2_config,
    purge_training_labels,
    resolve_information_cutoff,
)


class TestModelingV2TemporalContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_modeling_v2_config()
        cls.policy = cutoff_policy_from_config(cls.config)

    def test_september_contract_uses_centralized_target_lag_three(self) -> None:
        origin = resolve_information_cutoff("2026-09", self.policy)

        self.assertEqual(origin.business_origin_month_id, "2026-08")
        self.assertEqual(origin.latest_available_month_id, "2026-06")
        self.assertEqual(origin.max_training_target_month_id, "2026-06")
        self.assertEqual(
            origin.cutoff_policy_id, "conservative_target_lag_3_v1"
        )

    def test_january_contract_crosses_year_boundary(self) -> None:
        origin = resolve_information_cutoff("2026-01", self.policy)

        self.assertEqual(origin.business_origin_month_id, "2025-12")
        self.assertEqual(origin.latest_available_month_id, "2025-10")

    def test_march_latest_available_is_previous_december(self) -> None:
        origin = resolve_information_cutoff("2026-03", self.policy)

        self.assertEqual(origin.business_origin_month_id, "2026-02")
        self.assertEqual(origin.latest_available_month_id, "2025-12")

    def test_all_origins_precede_business_origin(self) -> None:
        origins = build_backtest_origins(self.config)

        latest = pd.PeriodIndex(origins["latest_available_month_id"], freq="M")
        business = pd.PeriodIndex(
            origins["business_origin_month_id"], freq="M"
        )
        training = pd.PeriodIndex(
            origins["max_training_target_month_id"], freq="M"
        )
        self.assertTrue((latest < business).all())
        self.assertTrue((training <= latest).all())
        self.assertEqual(len(origins), 36)
        self.assertTrue(origins["target_month_id"].is_unique)

    def test_training_label_purge_rejects_t_minus_one_and_t_minus_two(self) -> None:
        origin = resolve_information_cutoff("2026-09", self.policy)
        labels = pd.DataFrame(
            {
                "month_id": ["2026-06", "2026-07", "2026-08"],
                "target": [1.0, 2.0, 3.0],
            }
        )

        purged = purge_training_labels(labels, origin)

        self.assertEqual(purged["month_id"].tolist(), ["2026-06"])
        assert_training_labels_within_cutoff(purged, origin)
        with self.assertRaisesRegex(AssertionError, "posteriores al cutoff"):
            assert_training_labels_within_cutoff(labels, origin)

    def test_future_rolling_history_cannot_cross_information_cutoff(self) -> None:
        origin = resolve_information_cutoff("2026-09", self.policy)
        history = pd.DataFrame(
            {
                "month_id": ["2026-05", "2026-06", "2026-07", "2026-08"],
                "value": [1.0, 2.0, 3.0, 4.0],
            }
        )

        available = filter_history_to_information_cutoff(history, origin)

        self.assertEqual(
            available["month_id"].tolist(), ["2026-05", "2026-06"]
        )

    def test_feature_availability_spec_matches_temporal_policy(self) -> None:
        spec = load_feature_availability_v2()

        self.assertEqual(
            spec["cutoff_policy"]["id"], self.policy.policy_id
        )
        self.assertEqual(
            spec["laggable_fields"]["overnight_stays_total"]
            ["b1_permitted_lags_months"],
            [12],
        )
        self.assertEqual(
            spec["laggable_fields"]["overnight_stays_total"]
            ["availability_policy_id"],
            self.policy.policy_id,
        )
        limitation = spec["vintage_limitation"]
        self.assertTrue(limitation["publication_availability_corrected"])
        self.assertFalse(
            limitation["exact_historical_vintages_reconstructed"]
        )
        self.assertEqual(
            limitation["id"], "availability_correct_latest_vintage"
        )

    def test_operational_selection_promotes_ets_provisionally_only(self) -> None:
        selection = self.config["operational_selection"]

        self.assertEqual(
            selection["selected_model_id"],
            "holt_winters_additive_damped_v1",
        )
        self.assertEqual(
            selection["status"], "provisional_validation_champion"
        )
        self.assertEqual(
            selection["evidence_scope"], "canonical_rolling_validation"
        )
        self.assertFalse(selection["independent_test_confirmed"])
        self.assertEqual(
            selection["fallback"],
            {
                "model_id": "seasonal_naive_lag_12",
                "policy": "availability_only",
                "performance_based": False,
            },
        )
        self.assertEqual(
            self.config["ets_candidate"]["screening_status"],
            "passed_screening",
        )

    def test_operational_interval_is_separate_from_b3_interval(self) -> None:
        baseline_interval = self.config["prediction_interval"]
        operational = self.config["operational_prediction_interval"]

        self.assertEqual(
            baseline_interval["method_id"],
            "prequential_scaled_absolute_residual_interval_v1",
        )
        self.assertEqual(
            operational["method_id"],
            "operational_prequential_scaled_absolute_residual_interval_v1",
        )
        self.assertNotEqual(
            baseline_interval["method_id"], operational["method_id"]
        )
        self.assertEqual(
            operational["score_prediction_column"],
            "operational_prediction",
        )
        self.assertEqual(
            operational["scale_column"], "baseline_prediction"
        )
        self.assertEqual(operational["nominal_level"], 0.8)
        self.assertEqual(operational["minimum_calibration_origins"], 12)


if __name__ == "__main__":
    unittest.main()
