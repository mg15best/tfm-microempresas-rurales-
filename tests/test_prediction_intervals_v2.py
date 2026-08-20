import unittest
from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd

from src.models.evaluate_v2 import (
    assert_interval_points_unchanged,
    calculate_interval_metrics,
    compare_with_frozen_v1,
    load_gold_history,
    reproduce_prediction_intervals_in_memory,
)
from src.models.modeling_v2_common import (
    cutoff_policy_from_config,
    load_modeling_v2_config,
    resolve_information_cutoff,
)
from src.models.prediction_intervals_v2 import (
    OPERATIONAL_INTERVAL_METHOD_ID,
    apply_operational_prequential_intervals,
    build_historical_baseline_score_bank,
    build_operational_score_bank,
    calculate_current_operational_interval,
    calculate_interval_score,
    calculate_prediction_interval,
    calculate_scaled_absolute_residual,
    eligible_calibration_scores,
    eligible_operational_calibration_scores,
    evaluate_operational_prequential_intervals,
    finite_sample_order_quantile,
    validate_operational_interval_config,
)


BASELINE_ID = "seasonal_naive_lag_12"


def score_bank(months: list[str], scores: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "territory_id": ["ES-PROV-01"] * len(months),
            "target_month_id": months,
            "score": scores,
            "baseline_id": [BASELINE_ID] * len(months),
        }
    )


class TestPredictionIntervalFormula(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_modeling_v2_config()
        cls.interval_config = cls.config["prediction_interval"]
        cls.policy = cutoff_policy_from_config(cls.config)

    def test_scaled_absolute_residual_uses_reference_scale(self) -> None:
        self.assertEqual(
            calculate_scaled_absolute_residual(120.0, 100.0, 100.0),
            0.2,
        )
        self.assertEqual(
            calculate_scaled_absolute_residual(5.0, 0.0, 0.0),
            5.0,
        )

    def test_finite_sample_quantile_uses_one_based_order_statistic(self) -> None:
        quantile, k = finite_sample_order_quantile([4, 1, 3, 2], 0.8)

        self.assertEqual(k, 4)
        self.assertEqual(quantile, 4.0)

    def test_quantile_is_deterministic_and_does_not_interpolate(self) -> None:
        scores = [0.4, 0.1, 0.5, 0.3, 0.2]

        first = finite_sample_order_quantile(scores, 0.8)
        second = finite_sample_order_quantile(scores, 0.8)

        self.assertEqual(first, second)
        self.assertEqual(first, (0.5, 5))

    def test_calibration_excludes_t_minus_one_t_minus_two_and_target(self) -> None:
        bank = score_bank(
            ["2026-06", "2026-07", "2026-08", "2026-09"],
            [0.1, 0.2, 0.3, 0.4],
        )

        eligible = eligible_calibration_scores(
            bank,
            latest_available_month_id="2026-06",
            baseline_id=BASELINE_ID,
        )

        self.assertEqual(eligible["target_month_id"].tolist(), ["2026-06"])

    def test_minimum_refers_to_twelve_distinct_origins(self) -> None:
        origin = resolve_information_cutoff("2026-09", self.policy)
        months = pd.period_range("2025-01", "2025-11", freq="M").astype(str)
        bank = score_bank(list(months) * 2, [0.2] * 22)

        result = calculate_prediction_interval(
            territory_id="ES-PROV-01",
            target_month_id="2026-09",
            point_prediction=100.0,
            seasonal_reference=100.0,
            origin=origin,
            score_bank=bank,
            interval_config=self.interval_config,
        )

        self.assertFalse(result.interval_available)
        self.assertEqual(result.calibration_origins_n, 11)
        self.assertEqual(result.calibration_scores_n, 22)
        self.assertEqual(
            result.unavailable_reason, "insufficient_calibration_origins"
        )
        self.assertIsNone(result.lower)
        self.assertIsNone(result.upper)

    def test_interval_formula_contains_point_and_clips_lower_at_zero(self) -> None:
        origin = resolve_information_cutoff("2026-09", self.policy)
        months = pd.period_range("2024-01", periods=12, freq="M").astype(str)
        bank = score_bank(list(months), [2.0] * 12)

        result = calculate_prediction_interval(
            territory_id="ES-PROV-01",
            target_month_id="2026-09",
            point_prediction=10.0,
            seasonal_reference=10.0,
            origin=origin,
            score_bank=bank,
            interval_config=self.interval_config,
        )

        self.assertTrue(result.interval_available)
        self.assertEqual(result.lower, 0.0)
        self.assertEqual(result.upper, 30.0)
        self.assertLessEqual(result.lower, result.point_prediction)
        self.assertLessEqual(result.point_prediction, result.upper)
        with self.assertRaises(FrozenInstanceError):
            result.lower = 1.0

    def test_score_bank_excludes_actuals_not_yet_complete(self) -> None:
        history = pd.DataFrame(
            {
                "territory_id": ["ES-PROV-01", "ES-PROV-01"],
                "territory_level": ["province", "province"],
                "month_id": ["2024-01", "2025-01"],
                "overnight_stays_total": [100.0, 120.0],
                "complete_month_available": [True, False],
                "is_provisional": [False, True],
            }
        )

        bank = build_historical_baseline_score_bank(
            history, baseline_id=BASELINE_ID
        )

        self.assertTrue(bank.empty)

    def test_interval_score_penalizes_below_and_above_misses(self) -> None:
        self.assertEqual(calculate_interval_score(5, 0, 10, 0.2), 10.0)
        self.assertEqual(calculate_interval_score(-2, 0, 10, 0.2), 30.0)
        self.assertEqual(calculate_interval_score(12, 0, 10, 0.2), 30.0)

    def test_interval_metrics_distinguish_below_and_above(self) -> None:
        frame = pd.DataFrame(
            {
                "interval_available": [True, True, True],
                "nominal_level": [0.8, 0.8, 0.8],
                "covered": [True, False, False],
                "width": [10.0, 10.0, 10.0],
                "normalized_width": [1.0, 2.0, 3.0],
                "interval_score": [10.0, 30.0, 30.0],
                "miss_below": [False, True, False],
                "miss_above": [False, False, True],
            }
        )

        metrics = calculate_interval_metrics(frame)

        self.assertAlmostEqual(metrics["empirical_coverage_pct"], 100 / 3)
        self.assertEqual(metrics["misses_below"], 1)
        self.assertEqual(metrics["misses_above"], 1)
        self.assertEqual(metrics["median_normalized_width"], 2.0)
        self.assertAlmostEqual(metrics["mean_interval_score"], 70 / 3)

    def test_configuration_freezes_single_eighty_percent_method(self) -> None:
        self.assertEqual(
            self.interval_config["method_id"],
            "prequential_scaled_absolute_residual_interval_v1",
        )
        self.assertEqual(self.interval_config["nominal_level"], 0.8)
        self.assertEqual(self.interval_config["alpha"], 0.2)
        self.assertEqual(self.interval_config["minimum_calibration_origins"], 12)
        self.assertFalse(
            self.interval_config["exact_iid_coverage_guarantee"]
        )
        self.assertEqual(
            self.config["candidate"]["screening_status"],
            "rejected_screening",
        )


class TestPredictionIntervalsV2Integration(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_modeling_v2_config()
        cls.gold = load_gold_history()
        cls.result = reproduce_prediction_intervals_in_memory(
            cls.gold, cls.config
        )
        cls.intervals = cls.result["interval_predictions"]

    def test_first_calibration_and_eligible_targets_are_derived(self) -> None:
        self.assertEqual(
            self.result["first_calibration_target"], "2006-01"
        )
        self.assertEqual(
            self.result["first_interval_eligible_target"], "2007-03"
        )

    def test_same_1750_points_match_b1_and_frozen_v1(self) -> None:
        invariant = self.result["point_invariant"]

        self.assertEqual(invariant["baseline_rows"], 1750)
        self.assertEqual(invariant["interval_rows"], 1750)
        self.assertEqual(invariant["common_rows"], 1750)
        self.assertEqual(invariant["missing_keys"], 0)
        self.assertEqual(invariant["extra_keys"], 0)
        self.assertEqual(invariant["point_mismatches"], 0)
        self.assertEqual(invariant["actual_mismatches"], 0)
        comparison = compare_with_frozen_v1(
            self.result["baseline"]["predictions"]
        )
        self.assertTrue(comparison.exact_match)

    def test_all_evaluation_intervals_are_available_and_valid(self) -> None:
        self.assertTrue(self.intervals["interval_available"].all())
        self.assertTrue((self.intervals["lower"] >= 0).all())
        self.assertTrue(
            (self.intervals["lower"] <= self.intervals["point_prediction"]).all()
        )
        self.assertTrue(
            (self.intervals["point_prediction"] <= self.intervals["upper"]).all()
        )
        self.assertTrue(
            np.allclose(
                self.intervals["normalized_width"],
                self.intervals["width"]
                / self.intervals["point_prediction"].clip(lower=1),
            )
        )

    def test_every_calibration_bank_is_strictly_prequential(self) -> None:
        maximum = pd.PeriodIndex(
            self.intervals["calibration_max_target_month_id"], freq="M"
        )
        latest = pd.PeriodIndex(
            self.intervals["latest_available_month_id"], freq="M"
        )
        target = pd.PeriodIndex(self.intervals["target_month_id"], freq="M")

        self.assertTrue((maximum <= latest).all())
        self.assertTrue((maximum <= target - 3).all())
        self.assertTrue((maximum < target - 2).all())

    def test_fold_territory_month_and_origin_diagnostics_are_complete(self) -> None:
        self.assertEqual(
            self.result["fold_metrics"]["fold_id"].tolist(),
            ["validation_1", "validation_2", "validation_3"],
        )
        self.assertEqual(len(self.result["territory_metrics"]), 50)
        self.assertEqual(
            self.result["month_metrics"]["month_number"].tolist(),
            list(range(1, 13)),
        )
        self.assertEqual(len(self.result["origin_metrics"]), 35)
        self.assertAlmostEqual(
            self.result["pooled_metrics"]["empirical_coverage_pct"],
            71.54285714285714,
        )
        self.assertAlmostEqual(
            self.result["post_stress_metrics"]["empirical_coverage_pct"],
            92.16666666666667,
        )

    def test_current_araba_interval_is_reproducible_and_baseline_only(self) -> None:
        current = self.result["current_intervals"]
        araba = current.loc[current["territory_id"].eq("ES-PROV-01")].iloc[0]

        self.assertEqual(len(current), 50)
        self.assertEqual(araba["point_prediction"], 7691.0)
        self.assertAlmostEqual(araba["lower"], 4408.964501510574)
        self.assertAlmostEqual(araba["upper"], 10973.035498489426)
        self.assertAlmostEqual(araba["calibration_quantile"], 0.42673716012084595)
        self.assertEqual(araba["calibration_scores_n"], 11987)
        self.assertEqual(araba["calibration_origins_n"], 240)
        self.assertEqual(araba["calibration_max_target_month_id"], "2026-06")
        self.assertTrue(araba["interval_available"])
        self.assertLessEqual(araba["lower"], araba["point_prediction"])
        self.assertLessEqual(araba["point_prediction"], araba["upper"])
        self.assertEqual(
            set(self.result["score_bank"]["baseline_id"]), {BASELINE_ID}
        )
        self.assertNotIn("candidate_prediction", self.result["score_bank"].columns)

    def test_interval_application_is_deterministic(self) -> None:
        repeated = reproduce_prediction_intervals_in_memory(
            self.gold, self.config
        )["interval_predictions"]

        pd.testing.assert_frame_equal(self.intervals, repeated)

    def test_point_invariant_rejects_a_changed_prediction(self) -> None:
        changed = self.intervals.copy()
        changed.loc[0, "point_prediction"] += 1

        with self.assertRaisesRegex(AssertionError, "B3 altero"):
            assert_interval_points_unchanged(
                self.result["baseline"]["comparable_predictions"],
                changed,
            )


def operational_artifact(
    months: list[str],
    *,
    territories: int = 1,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for month_index, target in enumerate(months):
        target_period = pd.Period(target, freq="M")
        for territory_index in range(territories):
            baseline = 100.0 + territory_index
            point = baseline + month_index
            rows.append(
                {
                    "fold_id": "validation_synthetic",
                    "territory_id": f"ES-PROV-{territory_index + 1:02d}",
                    "target_month_id": str(target_period),
                    "latest_available_month_id": str(target_period - 3),
                    "actual": point + 20.0,
                    "baseline_prediction": baseline,
                    "operational_prediction": point,
                    "availability_fallback_used": territory_index == 0,
                }
            )
    return pd.DataFrame(rows)


class TestOperationalPredictionIntervalFormula(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_modeling_v2_config()
        cls.interval_config = cls.config["operational_prediction_interval"]
        cls.policy = cutoff_policy_from_config(cls.config)

    def test_operational_config_is_fixed_and_not_the_b3_method(self) -> None:
        validate_operational_interval_config(self.interval_config)

        self.assertEqual(
            self.interval_config["method_id"],
            OPERATIONAL_INTERVAL_METHOD_ID,
        )
        self.assertEqual(self.interval_config["nominal_level"], 0.8)
        self.assertEqual(self.interval_config["alpha"], 0.2)
        self.assertEqual(
            self.interval_config["minimum_calibration_origins"], 12
        )
        self.assertNotEqual(
            self.interval_config["method_id"],
            self.config["prediction_interval"]["method_id"],
        )

    def test_score_uses_operational_point_and_baseline_scale(self) -> None:
        artifact = operational_artifact(["2024-01"])
        artifact.loc[0, ["actual", "operational_prediction"]] = [130.0, 100.0]
        artifact.loc[0, "baseline_prediction"] = 50.0

        bank = build_operational_score_bank(artifact)

        self.assertEqual(bank.loc[0, "score"], 0.6)
        self.assertEqual(
            bank.loc[0, "score_prediction_column"],
            "operational_prediction",
        )
        self.assertEqual(bank.loc[0, "scale_column"], "baseline_prediction")

    def test_artifact_schema_and_unique_keys_are_enforced(self) -> None:
        artifact = operational_artifact(["2024-01"])
        with self.assertRaisesRegex(ValueError, "baseline_prediction"):
            build_operational_score_bank(
                artifact.drop(columns="baseline_prediction")
            )
        with self.assertRaisesRegex(ValueError, "duplicadas"):
            build_operational_score_bank(
                pd.concat([artifact, artifact], ignore_index=True)
            )

    def test_calibration_includes_cutoff_and_excludes_future_scores(self) -> None:
        months = pd.period_range("2024-01", periods=4, freq="M").astype(str)
        bank = build_operational_score_bank(
            operational_artifact(list(months))
        )

        eligible = eligible_operational_calibration_scores(
            bank,
            latest_available_month_id="2024-02",
        )

        self.assertEqual(
            eligible["target_month_id"].tolist(),
            ["2024-01", "2024-02"],
        )
        self.assertNotIn("2024-03", set(eligible["target_month_id"]))

    def test_minimum_counts_distinct_origins_not_province_rows(self) -> None:
        months = pd.period_range("2024-01", periods=11, freq="M").astype(str)
        bank = build_operational_score_bank(
            operational_artifact(list(months), territories=2)
        )
        origin = resolve_information_cutoff("2026-09", self.policy)

        interval = calculate_current_operational_interval(
            territory_id="ES-PROV-01",
            target_month_id="2026-09",
            point_prediction=100.0,
            baseline_prediction=100.0,
            origin=origin,
            score_bank=bank,
            interval_config=self.interval_config,
        )

        self.assertFalse(interval.interval_available)
        self.assertEqual(interval.calibration_scores_n, 22)
        self.assertEqual(interval.calibration_origins_n, 11)
        self.assertEqual(
            interval.unavailable_reason,
            "insufficient_calibration_origins",
        )

    def test_operational_interval_uses_finite_quantile_and_contains_point(self) -> None:
        months = pd.period_range("2024-01", periods=12, freq="M").astype(str)
        artifact = operational_artifact(list(months))
        artifact["actual"] = artifact["operational_prediction"] + 200.0
        artifact["baseline_prediction"] = 100.0
        bank = build_operational_score_bank(artifact)
        origin = resolve_information_cutoff("2026-09", self.policy)

        interval = calculate_current_operational_interval(
            territory_id="ES-PROV-01",
            target_month_id="2026-09",
            point_prediction=10.0,
            baseline_prediction=10.0,
            origin=origin,
            score_bank=bank,
            interval_config=self.interval_config,
        )

        self.assertTrue(interval.interval_available)
        self.assertEqual(interval.calibration_quantile, 2.0)
        self.assertEqual(interval.lower, 0.0)
        self.assertEqual(interval.upper, 30.0)
        self.assertLessEqual(interval.lower, interval.point_prediction)
        self.assertLessEqual(interval.point_prediction, interval.upper)

    def test_missing_scale_keeps_point_and_marks_interval_unavailable(
        self,
    ) -> None:
        months = pd.period_range("2024-01", periods=12, freq="M").astype(str)
        bank = build_operational_score_bank(
            operational_artifact(list(months))
        )
        origin = resolve_information_cutoff("2026-09", self.policy)

        interval = calculate_current_operational_interval(
            territory_id="ES-PROV-01",
            target_month_id="2026-09",
            point_prediction=4321.5,
            baseline_prediction=None,
            origin=origin,
            score_bank=bank,
            interval_config=self.interval_config,
        )

        self.assertEqual(interval.point_prediction, 4321.5)
        self.assertFalse(interval.interval_available)
        self.assertEqual(interval.unavailable_reason, "missing_interval_scale")
        self.assertIsNone(interval.lower)
        self.assertIsNone(interval.upper)


class TestOperationalPredictionIntervalCanonicalValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_modeling_v2_config()
        cls.artifact = pd.read_parquet(
            "data/model_outputs/ets_v2_rolling_validation_predictions.parquet"
        )
        cls.result = evaluate_operational_prequential_intervals(
            cls.artifact,
            cls.config["operational_prediction_interval"],
        )
        cls.intervals = cls.result["interval_predictions"]

    def test_canonical_counts_and_first_evaluable_origin(self) -> None:
        self.assertEqual(self.result["total_rows"], 1750)
        self.assertEqual(self.result["interval_evaluable_rows"], 1050)
        self.assertEqual(self.result["non_evaluable_rows"], 700)
        self.assertEqual(self.result["evaluable_origins"], 21)
        self.assertEqual(self.result["first_evaluable_target"], "2022-09")

    def test_canonical_coverage_width_and_misses_are_reproduced(self) -> None:
        self.assertAlmostEqual(
            self.result["observed_coverage_pct"], 96.28571428571429
        )
        self.assertAlmostEqual(
            self.result["median_width"], 19591.00123517526
        )
        self.assertAlmostEqual(
            self.result["mean_width"], 32638.206451737242
        )
        self.assertEqual(self.result["misses_below"], 22)
        self.assertEqual(self.result["misses_above"], 17)

    def test_every_available_interval_is_valid_and_prequential(self) -> None:
        available = self.intervals.loc[self.intervals["interval_available"]]
        maximum = pd.PeriodIndex(
            available["calibration_max_target_month_id"], freq="M"
        )
        cutoff = pd.PeriodIndex(
            available["latest_available_month_id"], freq="M"
        )

        self.assertTrue((available["lower"] >= 0).all())
        self.assertTrue(
            (available["lower"] <= available["point_prediction"]).all()
        )
        self.assertTrue(
            (available["point_prediction"] <= available["upper"]).all()
        )
        self.assertTrue((maximum <= cutoff).all())
        self.assertTrue(
            (
                pd.PeriodIndex(available["target_month_id"], freq="M")
                > cutoff
            ).all()
        )

    def test_fallback_rows_receive_the_same_operational_interval(self) -> None:
        available_fallbacks = self.intervals.loc[
            self.intervals["interval_available"]
            & self.intervals["availability_fallback_used"]
        ]

        self.assertEqual(len(available_fallbacks), 21)
        self.assertTrue(available_fallbacks["lower"].notna().all())
        self.assertTrue(available_fallbacks["upper"].notna().all())
        self.assertEqual(self.result["fallback_coverage_pct"], 100.0)
        self.assertEqual(
            set(available_fallbacks["method_id"]),
            {OPERATIONAL_INTERVAL_METHOD_ID},
        )

    def test_application_is_read_only_and_deterministic(self) -> None:
        before = self.artifact.copy(deep=True)
        repeated = apply_operational_prequential_intervals(
            self.artifact,
            self.config["operational_prediction_interval"],
        )

        pd.testing.assert_frame_equal(self.artifact, before)
        pd.testing.assert_frame_equal(self.intervals, repeated)


if __name__ == "__main__":
    unittest.main()
