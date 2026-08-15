import unittest

import numpy as np
import pandas as pd

from src.models.evaluate_v2 import (
    assert_candidate_rows_are_paired,
    build_current_candidate_illustration,
    calculate_paired_metrics,
    load_gold_history,
    reproduce_seasonal_trend_in_memory,
    screen_seasonal_trend_candidate,
)
from src.models.modeling_v2_common import (
    cutoff_policy_from_config,
    load_feature_availability_v2,
    load_modeling_v2_config,
    resolve_information_cutoff,
)
from src.models.seasonal_trend_v2 import (
    apply_availability_fallback,
    calculate_seasonal_trend_forecast,
    resolve_candidate_window,
)


class TestSeasonalTrendFormula(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_modeling_v2_config()
        cls.candidate = cls.config["candidate"]
        cls.policy = cutoff_policy_from_config(cls.config)

    def window(self, target: str):
        return resolve_candidate_window(
            resolve_information_cutoff(target, self.policy),
            window_months=self.candidate["window_months"],
            seasonal_reference_lag_months=self.candidate[
                "seasonal_reference_lag_months"
            ],
        )

    def test_september_2026_uses_only_exact_published_calendar_months(self) -> None:
        window = self.window("2026-09")

        self.assertEqual(window.cutoff_month_id, "2026-06")
        self.assertEqual(window.seasonal_reference_month_id, "2025-09")
        self.assertEqual(
            window.recent_window_month_ids,
            ("2026-04", "2026-05", "2026-06"),
        )
        self.assertEqual(
            window.prior_year_window_month_ids,
            ("2025-04", "2025-05", "2025-06"),
        )
        self.assertNotIn("2026-07", window.recent_window_month_ids)
        self.assertNotIn("2026-08", window.recent_window_month_ids)

    def test_candidate_and_feature_contract_are_frozen_before_evaluation(self) -> None:
        spec = load_feature_availability_v2()

        self.assertEqual(
            self.candidate["id"], "seasonal_trend_adjusted_3m_yoy_v1"
        )
        self.assertEqual(self.candidate["window_months"], 3)
        self.assertFalse(self.candidate["clipping"])
        dependencies = spec["candidate_dependencies"]
        self.assertEqual(dependencies["candidate_id"], self.candidate["id"])
        self.assertEqual(
            dependencies["recent_published_window"]
            ["relative_to_cutoff_months"],
            [-2, -1, 0],
        )
        self.assertEqual(
            dependencies["prior_year_recent_window"]
            ["relative_to_cutoff_months"],
            [-14, -13, -12],
        )
        self.assertTrue(dependencies["raw_ratio_without_clipping"])

    def test_january_february_and_march_cross_years_exactly(self) -> None:
        expected = {
            "2026-01": (
                "2025-01",
                ("2025-08", "2025-09", "2025-10"),
                ("2024-08", "2024-09", "2024-10"),
            ),
            "2026-02": (
                "2025-02",
                ("2025-09", "2025-10", "2025-11"),
                ("2024-09", "2024-10", "2024-11"),
            ),
            "2026-03": (
                "2025-03",
                ("2025-10", "2025-11", "2025-12"),
                ("2024-10", "2024-11", "2024-12"),
            ),
        }
        for target, (anchor, recent, prior) in expected.items():
            with self.subTest(target=target):
                window = self.window(target)
                self.assertEqual(window.seasonal_reference_month_id, anchor)
                self.assertEqual(window.recent_window_month_ids, recent)
                self.assertEqual(window.prior_year_window_month_ids, prior)

    def test_formula_is_raw_three_month_yoy_times_seasonal_anchor(self) -> None:
        window = self.window("2026-09")
        values = {
            "2025-09": 120.0,
            "2026-04": 30.0,
            "2026-05": 40.0,
            "2026-06": 50.0,
            "2025-04": 20.0,
            "2025-05": 30.0,
            "2025-06": 40.0,
        }

        forecast = calculate_seasonal_trend_forecast(values, window)

        self.assertTrue(forecast.candidate_available)
        self.assertEqual(forecast.recent_sum, 120.0)
        self.assertEqual(forecast.prior_year_sum, 90.0)
        self.assertAlmostEqual(forecast.trend_factor, 4 / 3)
        self.assertAlmostEqual(forecast.candidate_prediction, 160.0)

    def test_missing_recent_window_is_explicit(self) -> None:
        window = self.window("2026-09")
        values = {
            window.seasonal_reference_month_id: 100.0,
            **{month: 10.0 for month in window.prior_year_window_month_ids},
        }

        forecast = calculate_seasonal_trend_forecast(values, window)

        self.assertFalse(forecast.candidate_available)
        self.assertEqual(forecast.fallback_reason, "missing_recent_window")

    def test_missing_prior_year_window_is_explicit(self) -> None:
        window = self.window("2026-09")
        values = {
            window.seasonal_reference_month_id: 100.0,
            **{month: 10.0 for month in window.recent_window_month_ids},
        }

        forecast = calculate_seasonal_trend_forecast(values, window)

        self.assertFalse(forecast.candidate_available)
        self.assertEqual(forecast.fallback_reason, "missing_prior_year_window")

    def test_zero_prior_year_denominator_is_explicit(self) -> None:
        window = self.window("2026-09")
        values = {
            window.seasonal_reference_month_id: 100.0,
            **{month: 10.0 for month in window.recent_window_month_ids},
            **{month: 0.0 for month in window.prior_year_window_month_ids},
        }

        forecast = calculate_seasonal_trend_forecast(values, window)

        self.assertFalse(forecast.candidate_available)
        self.assertEqual(
            forecast.fallback_reason, "non_positive_prior_year_sum"
        )

    def test_negative_or_non_finite_inputs_are_invalid(self) -> None:
        window = self.window("2026-09")
        for invalid in (-1.0, np.inf):
            values = {
                window.seasonal_reference_month_id: 100.0,
                **{month: 10.0 for month in window.recent_window_month_ids},
                **{
                    month: 10.0
                    for month in window.prior_year_window_month_ids
                },
            }
            values[window.recent_window_month_ids[0]] = invalid
            with self.subTest(invalid=invalid):
                forecast = calculate_seasonal_trend_forecast(values, window)
                self.assertFalse(forecast.candidate_available)
                self.assertEqual(
                    forecast.fallback_reason, "invalid_trend_input"
                )

    def test_availability_fallback_uses_baseline_without_performance_rule(self) -> None:
        window = self.window("2026-09")
        forecast = calculate_seasonal_trend_forecast(
            {window.seasonal_reference_month_id: 80.0}, window
        )

        operational = apply_availability_fallback(forecast, 80.0)

        self.assertTrue(operational.fallback_used)
        self.assertEqual(operational.operational_prediction, 80.0)
        self.assertEqual(operational.fallback_reason, "missing_recent_window")

    def test_valid_formula_cannot_produce_negative_prediction(self) -> None:
        window = self.window("2026-09")
        values = {
            window.seasonal_reference_month_id: 100.0,
            **{month: 0.0 for month in window.recent_window_month_ids},
            **{month: 10.0 for month in window.prior_year_window_month_ids},
        }

        forecast = calculate_seasonal_trend_forecast(values, window)

        self.assertTrue(forecast.candidate_available)
        self.assertEqual(forecast.candidate_prediction, 0.0)

    def test_window_size_is_frozen_and_not_tuned(self) -> None:
        origin = resolve_information_cutoff("2026-09", self.policy)

        with self.assertRaisesRegex(ValueError, "unica ventana"):
            resolve_candidate_window(
                origin,
                window_months=2,
                seasonal_reference_lag_months=12,
            )


class TestSeasonalTrendEvaluationIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_modeling_v2_config()
        cls.gold = load_gold_history()
        cls.result = reproduce_seasonal_trend_in_memory(cls.gold, cls.config)
        cls.candidate = cls.result["candidate_predictions"]

    def test_operational_candidate_has_same_1750_baseline_keys(self) -> None:
        invariant = self.result["paired_invariant"]

        self.assertEqual(len(self.candidate), 1750)
        self.assertEqual(invariant["baseline_rows"], 1750)
        self.assertEqual(invariant["candidate_rows"], 1750)
        self.assertEqual(invariant["common_rows"], 1750)
        self.assertEqual(invariant["missing_keys"], 0)
        self.assertEqual(invariant["extra_keys"], 0)
        self.assertEqual(invariant["actual_mismatches"], 0)
        self.assertEqual(invariant["baseline_mismatches"], 0)

    def test_pure_candidate_rows_are_strictly_paired(self) -> None:
        pure = self.result["pure_candidate_predictions"]
        metrics = calculate_paired_metrics(
            pure, candidate_prediction_column="candidate_prediction"
        )

        self.assertEqual(metrics["n"], len(pure))
        self.assertTrue(pure["candidate_available"].all())
        self.assertTrue(pure["candidate_prediction"].notna().all())
        self.assertTrue(pure["baseline_prediction"].notna().all())

    def test_operational_fallback_is_only_availability_based(self) -> None:
        fallback = self.candidate.loc[self.candidate["fallback_used"]]

        self.assertTrue(
            np.allclose(
                fallback["operational_prediction"],
                fallback["baseline_prediction"],
            )
        )
        self.assertTrue(
            set(fallback["fallback_reason"]).issubset(
                {
                    "missing_recent_window",
                    "missing_prior_year_window",
                    "non_positive_prior_year_sum",
                    "invalid_trend_input",
                }
            )
        )
        self.assertFalse(self.config["candidate"]["fallback"]["performance_based"])

    def test_real_availability_coverage_and_gap_reasons_are_reproduced(self) -> None:
        fallback = self.candidate.loc[self.candidate["fallback_used"]]

        self.assertEqual(int(self.candidate["candidate_available"].sum()), 1400)
        self.assertEqual(len(fallback), 350)
        self.assertEqual(
            fallback["fallback_reason"].value_counts().to_dict(),
            {"missing_prior_year_window": 350},
        )
        by_target = fallback.groupby("target_month_id").size().to_dict()
        self.assertEqual(
            by_target,
            {
                "2021-07": 50,
                "2021-08": 50,
                "2021-09": 50,
                "2021-10": 50,
                "2022-02": 50,
                "2022-03": 50,
                "2022-04": 50,
            },
        )

    def test_all_candidate_predictions_are_non_negative(self) -> None:
        pure = self.result["pure_candidate_predictions"]

        self.assertTrue((pure["candidate_prediction"] >= 0).all())
        self.assertTrue((self.candidate["operational_prediction"] >= 0).all())

    def test_fold_territory_month_and_origin_metrics_are_complete(self) -> None:
        fold = self.result["fold_metrics"]
        territory = self.result["territory_metrics"]
        month = self.result["month_metrics"]
        origin = self.result["origin_metrics"]

        self.assertEqual(fold["fold_id"].tolist(), [
            "validation_1", "validation_2", "validation_3"
        ])
        self.assertEqual(len(territory), 50)
        self.assertEqual(month["month_number"].tolist(), list(range(1, 13)))
        self.assertEqual(len(origin), 35)
        self.assertTrue(
            {"mae_skill_pct", "wape_skill_pct", "outcome"}.issubset(
                fold.columns
            )
        )

        self.assertEqual(fold["outcome"].value_counts().to_dict(), {"loss": 3})
        self.assertEqual(
            territory["outcome"].value_counts().to_dict(), {"loss": 50}
        )
        self.assertEqual(
            month["outcome"].value_counts().to_dict(),
            {"loss": 8, "win": 4},
        )
        self.assertEqual(
            origin["outcome"].value_counts().to_dict(),
            {"loss": 19, "win": 9, "tie": 7},
        )

    def test_operational_and_fold_metrics_are_reproducible(self) -> None:
        operational = self.result["operational_metrics"]
        fold = self.result["fold_metrics"].set_index("fold_id")

        self.assertEqual(operational["n"], 1750)
        self.assertAlmostEqual(
            operational["candidate_MAE"], 13632.833659, places=5
        )
        self.assertAlmostEqual(
            operational["mae_skill_pct"], -171.643892, places=5
        )
        self.assertAlmostEqual(
            fold.loc["validation_1", "mae_skill_pct"],
            -19.625619,
            places=5,
        )
        self.assertAlmostEqual(
            fold.loc["validation_2", "mae_skill_pct"],
            -736.030092,
            places=5,
        )
        self.assertAlmostEqual(
            fold.loc["validation_3", "mae_skill_pct"],
            -27.294166,
            places=5,
        )

    def test_screening_is_deterministic_and_uses_all_six_checks(self) -> None:
        pooled = {
            "mae_skill_pct": 1.0,
            "baseline_bias": -10.0,
            "candidate_bias": -9.0,
        }
        folds = pd.DataFrame({"mae_skill_pct": [1.0, 2.0, -1.0]})
        territories = pd.DataFrame(
            {"mae_skill_pct": [1.0, 2.0, 3.0, -1.0]}
        )

        first = screen_seasonal_trend_candidate(
            pooled, folds, territories, self.config
        )
        second = screen_seasonal_trend_candidate(
            pooled, folds, territories, self.config
        )

        self.assertEqual(first, second)
        self.assertEqual(
            first["conclusion"],
            "SEASONAL TREND CANDIDATE PASSES SCREENING",
        )
        checks = [
            first[key]
            for key in first
            if key.startswith(tuple("ABCDEF"))
        ]
        self.assertTrue(all(checks))

    def test_real_screening_matches_recalculation(self) -> None:
        recalculated = screen_seasonal_trend_candidate(
            self.result["operational_metrics"],
            self.result["fold_metrics"],
            self.result["territory_metrics"],
            self.config,
        )

        self.assertEqual(recalculated, self.result["screening"])
        self.assertEqual(
            recalculated["conclusion"],
            "SEASONAL TREND CANDIDATE FAILS SCREENING",
        )
        check_values = [
            value
            for key, value in recalculated.items()
            if key.startswith(tuple("ABCDEF"))
        ]
        self.assertEqual(check_values, [False] * 6)

    def test_current_araba_illustration_uses_expected_arithmetic(self) -> None:
        current = build_current_candidate_illustration(
            self.gold, target_month_id="2026-09", config=self.config
        )
        araba = current.loc[current["territory_id"].eq("ES-PROV-01")].iloc[0]

        self.assertTrue(araba["candidate_available"])
        self.assertAlmostEqual(
            araba["trend_factor"],
            araba["recent_sum"] / araba["prior_year_sum"],
        )
        self.assertAlmostEqual(
            araba["candidate_prediction"],
            araba["baseline_prediction"] * araba["trend_factor"],
        )
        self.assertEqual(len(current), 50)
        self.assertEqual(araba["baseline_prediction"], 7691.0)
        self.assertEqual(araba["recent_sum"], 24090.0)
        self.assertEqual(araba["prior_year_sum"], 23429.0)
        self.assertAlmostEqual(araba["trend_factor"], 1.0282128985)
        self.assertAlmostEqual(
            araba["candidate_prediction"], 7907.985403, places=5
        )

    def test_pairing_invariant_rejects_missing_operational_key(self) -> None:
        baseline = self.result["baseline"]["comparable_predictions"]
        incomplete = self.candidate.iloc[:-1].copy()

        with self.assertRaisesRegex(AssertionError, "missing=1"):
            assert_candidate_rows_are_paired(baseline, incomplete)


if __name__ == "__main__":
    unittest.main()
