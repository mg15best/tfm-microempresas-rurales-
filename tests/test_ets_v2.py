import unittest
from dataclasses import FrozenInstanceError
import hashlib
import platform
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import patsy
import scipy
import statsmodels

from src.models.ets_v2 import (
    fit_ets_forecast,
    prepare_ets_training_series,
    resolve_effective_horizon,
    validate_ets_config,
)
from src.models.evaluate_v2 import (
    assert_candidate_rows_are_paired,
    build_current_ets_illustration,
    calculate_paired_metrics,
    load_gold_history,
    reproduce_ets_in_memory,
    screen_ets_candidate,
)
from src.models.modeling_v2_common import (
    cutoff_policy_from_config,
    load_modeling_v2_config,
    resolve_information_cutoff,
)


def numerical_runtime_summary() -> str:
    return (
        f"platform={platform.platform()}, "
        f"python={platform.python_version()}, "
        f"numpy={np.__version__}, scipy={scipy.__version__}, "
        f"pandas={pd.__version__}, patsy={patsy.__version__}, "
        f"statsmodels={statsmodels.__version__}"
    )


def synthetic_history(
    start: str = "2018-01",
    end: str = "2026-08",
    *,
    territory_id: str = "ES-PROV-01",
) -> pd.DataFrame:
    months = pd.period_range(start, end, freq="M")
    values = 100 + np.arange(len(months), dtype=float)
    return pd.DataFrame(
        {
            "territory_id": territory_id,
            "territory_name": "Synthetic",
            "territory_level": "province",
            "month_id": months.astype(str),
            "overnight_stays_total": values,
            "complete_month_available": True,
        }
    )


class TestETSPointInTimeFormula(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_modeling_v2_config()
        cls.candidate = cls.config["ets_candidate"]
        cls.policy = cutoff_policy_from_config(cls.config)

    def origin(self, target: str = "2026-09"):
        return resolve_information_cutoff(target, self.policy)

    def test_single_ets_specification_is_frozen(self) -> None:
        validate_ets_config(self.candidate)

        self.assertEqual(self.candidate["trend"], "add")
        self.assertTrue(self.candidate["damped_trend"])
        self.assertEqual(self.candidate["seasonal"], "add")
        self.assertEqual(self.candidate["seasonal_periods"], 12)
        self.assertEqual(self.candidate["library_version"], "0.14.6")
        self.assertEqual(
            self.candidate["screening_status"], "passed_screening"
        )

    def test_numerical_stack_is_explicitly_frozen(self) -> None:
        reproducibility = self.config["numerical_reproducibility"]
        canonical = reproducibility["canonical_environment"]

        self.assertEqual(np.__version__, canonical["numpy"])
        self.assertEqual(scipy.__version__, canonical["scipy"])
        self.assertEqual(pd.__version__, canonical["pandas"])
        self.assertEqual(patsy.__version__, canonical["patsy"])
        self.assertEqual(statsmodels.__version__, canonical["statsmodels"])
        self.assertEqual(
            ".".join(platform.python_version_tuple()[:2]),
            reproducibility["local_python_major_minor"],
        )

    def test_effective_horizon_is_three_and_business_horizon_is_one(self) -> None:
        origin = self.origin()

        self.assertEqual(
            resolve_effective_horizon(origin, self.candidate), 3
        )
        self.assertEqual(origin.business_origin_month_id, "2026-08")
        self.assertEqual(origin.latest_available_month_id, "2026-06")

    def test_effective_horizon_crosses_year_boundary(self) -> None:
        origin = self.origin("2022-01")

        self.assertEqual(origin.latest_available_month_id, "2021-10")
        self.assertEqual(resolve_effective_horizon(origin, self.candidate), 3)

    def test_training_ends_exactly_at_cutoff_and_excludes_t1_t2(self) -> None:
        history = synthetic_history()
        training = prepare_ets_training_series(
            history, self.origin(), self.candidate
        )

        self.assertTrue(training.available)
        self.assertEqual(training.training_end, "2026-06")
        expected_last = history.loc[
            history["month_id"].eq("2026-06"),
            "overnight_stays_total",
        ].iloc[0]
        self.assertEqual(training.values[-1], expected_last)
        self.assertNotIn("2026-07", training.imputed_month_ids)
        self.assertNotIn("2026-08", training.imputed_month_ids)

    def test_future_july_august_values_cannot_change_forecast(self) -> None:
        history = synthetic_history()
        changed = history.copy()
        changed.loc[
            changed["month_id"].isin(["2026-07", "2026-08"]),
            "overnight_stays_total",
        ] = 10_000_000.0

        first = fit_ets_forecast(history, self.origin(), self.candidate)
        second = fit_ets_forecast(changed, self.origin(), self.candidate)

        self.assertTrue(first.candidate_available)
        self.assertAlmostEqual(first.prediction, second.prediction, places=8)

    def test_minimum_history_is_sixty_observed_months(self) -> None:
        history = synthetic_history("2021-08", "2026-06")

        result = fit_ets_forecast(history, self.origin(), self.candidate)

        self.assertFalse(result.candidate_available)
        self.assertFalse(result.fit_attempted)
        self.assertEqual(result.unavailable_reason, "insufficient_history")

    def test_gap_uses_only_observed_seasonal_lag_twelve(self) -> None:
        history = synthetic_history("2018-01", "2026-06")
        source = history.loc[
            history["month_id"].eq("2021-04"),
            "overnight_stays_total",
        ].iloc[0]
        history = history.loc[~history["month_id"].eq("2022-04")]

        training = prepare_ets_training_series(
            history, self.origin(), self.candidate
        )

        self.assertTrue(training.available)
        self.assertEqual(training.imputed_month_ids, ("2022-04",))
        index = pd.Period("2022-04", freq="M").ordinal - pd.Period(
            training.training_start, freq="M"
        ).ordinal
        self.assertEqual(training.values[index], source)

    def test_gap_without_observed_lag_is_explicitly_unsupported(self) -> None:
        history = synthetic_history("2018-01", "2026-06")
        history = history.loc[
            ~history["month_id"].isin(["2021-04", "2022-04"])
        ]

        result = fit_ets_forecast(history, self.origin(), self.candidate)

        self.assertFalse(result.candidate_available)
        self.assertEqual(
            result.unavailable_reason, "training_gap_unsupported"
        )

    def test_fit_rejects_cross_territory_training(self) -> None:
        mixed = pd.concat(
            [
                synthetic_history(territory_id="ES-PROV-01"),
                synthetic_history(territory_id="ES-PROV-02"),
            ],
            ignore_index=True,
        )

        with self.assertRaisesRegex(ValueError, "exactamente un territorio"):
            fit_ets_forecast(mixed, self.origin(), self.candidate)

    def test_negative_raw_forecast_is_clipped_once_at_zero(self) -> None:
        fitted = Mock()
        fitted.forecast.return_value = np.array([-1.0, -2.0, -3.0])
        fitted.mle_retvals = {"success": True}
        model = Mock()
        model.fit.return_value = fitted

        with patch(
            "src.models.ets_v2.ExponentialSmoothing", return_value=model
        ):
            result = fit_ets_forecast(
                synthetic_history(), self.origin(), self.candidate
            )

        self.assertEqual(result.raw_prediction, -3.0)
        self.assertEqual(result.prediction, 0.0)
        self.assertTrue(result.clipping_applied)

    def test_output_is_deterministic_within_numeric_tolerance(self) -> None:
        history = synthetic_history()

        first = fit_ets_forecast(history, self.origin(), self.candidate)
        second = fit_ets_forecast(history, self.origin(), self.candidate)

        self.assertAlmostEqual(first.prediction, second.prediction, places=8)
        with self.assertRaises(FrozenInstanceError):
            first.prediction = 1.0


class TestETSV2Integration(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_modeling_v2_config()
        cls.gold = load_gold_history()
        cls.result = reproduce_ets_in_memory(cls.gold, cls.config)
        cls.candidate = cls.result["candidate_predictions"]

    def test_operational_candidate_has_same_1750_baseline_keys(self) -> None:
        invariant = self.result["paired_invariant"]

        self.assertEqual(len(self.candidate), 1750)
        self.assertEqual(invariant["baseline_rows"], 1750)
        self.assertEqual(invariant["candidate_rows"], 1750)
        self.assertEqual(invariant["common_rows"], 1750)
        self.assertEqual(invariant["missing_keys"], 0)
        self.assertEqual(invariant["extra_keys"], 0)

    def test_pure_rows_are_exactly_paired_and_available(self) -> None:
        pure = self.result["pure_candidate_predictions"]
        metrics = calculate_paired_metrics(
            pure, candidate_prediction_column="candidate_prediction"
        )

        self.assertEqual(metrics["n"], len(pure))
        self.assertTrue(pure["candidate_available"].all())
        self.assertTrue(pure["candidate_prediction"].notna().all())

    def test_every_fit_respects_cutoff_and_three_steps(self) -> None:
        latest = pd.PeriodIndex(
            self.candidate["latest_available_month_id"], freq="M"
        )
        target = pd.PeriodIndex(
            self.candidate["target_month_id"], freq="M"
        )

        self.assertTrue(
            self.candidate["training_end"].eq(
                self.candidate["latest_available_month_id"]
            ).all()
        )
        self.assertTrue((target.asi8 - latest.asi8 == 3).all())
        self.assertTrue(self.candidate["effective_horizon_steps"].eq(3).all())

    def test_real_training_input_is_order_independent_and_hashed(self) -> None:
        policy = cutoff_policy_from_config(self.config)
        origin = resolve_information_cutoff("2026-09", policy)
        history = self.gold.loc[
            self.gold["territory_id"].astype(str).eq("ES-PROV-01")
        ].copy()
        shuffled = history.sample(frac=1.0, random_state=42)

        ordered_training = prepare_ets_training_series(
            history, origin, self.config["ets_candidate"]
        )
        shuffled_training = prepare_ets_training_series(
            shuffled, origin, self.config["ets_candidate"]
        )
        ordered_values = np.asarray(ordered_training.values, dtype="<f8")
        shuffled_values = np.asarray(shuffled_training.values, dtype="<f8")

        np.testing.assert_array_equal(ordered_values, shuffled_values)
        self.assertEqual(
            hashlib.sha256(ordered_values.tobytes()).hexdigest(),
            "3cdb1b74ec3f92e2402e52549165e7bec774a2088469957a6e1832983a650b98",
        )

    def test_availability_fallback_never_uses_performance(self) -> None:
        fallback = self.candidate.loc[self.candidate["fallback_used"]]

        self.assertTrue(
            np.allclose(
                fallback["operational_prediction"],
                fallback["baseline_prediction"],
            )
        )
        self.assertFalse(
            self.config["ets_candidate"]["fallback"]["performance_based"]
        )
        self.assertEqual(len(fallback), 35)
        self.assertEqual(
            fallback["fallback_reason"].value_counts().to_dict(),
            {"training_gap_unsupported": 35},
        )
        self.assertEqual(set(fallback["territory_id"]), {"ES-PROV-06"})

    def test_fold_province_month_and_origin_metrics_are_complete(self) -> None:
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
        fold = self.result["fold_metrics"].set_index("fold_id")
        self.assertEqual(
            fold["outcome"].to_dict(),
            {
                "validation_1": "win",
                "validation_2": "win",
                "validation_3": "loss",
            },
        )
        self.assertAlmostEqual(
            fold.loc["validation_1", "mae_skill_pct"],
            31.62470499,
            delta=float(
                self.config["numerical_reproducibility"]
                ["fold_skill_absolute_tolerance_percentage_points"]
            ),
            msg=numerical_runtime_summary(),
        )
        self.assertAlmostEqual(
            fold.loc["validation_2", "mae_skill_pct"],
            5.77449852,
            delta=float(
                self.config["numerical_reproducibility"]
                ["fold_skill_absolute_tolerance_percentage_points"]
            ),
            msg=numerical_runtime_summary(),
        )
        self.assertAlmostEqual(
            fold.loc["validation_3", "mae_skill_pct"],
            -3.87382573,
            delta=float(
                self.config["numerical_reproducibility"]
                ["fold_skill_absolute_tolerance_percentage_points"]
            ),
            msg=numerical_runtime_summary(),
        )

    def test_screening_is_deterministic(self) -> None:
        first = screen_ets_candidate(
            self.result["operational_metrics"],
            self.result["fold_metrics"],
            self.result["territory_metrics"],
            self.config,
        )
        second = screen_ets_candidate(
            self.result["operational_metrics"],
            self.result["fold_metrics"],
            self.result["territory_metrics"],
            self.config,
        )

        self.assertEqual(first, second)
        self.assertEqual(first, self.result["screening"])
        self.assertEqual(first["conclusion"], "ETS CANDIDATE PASSES SCREENING")
        self.assertTrue(
            all(first[key] for key in first if key.startswith(tuple("ABCDEF")))
        )
        self.assertAlmostEqual(
            self.result["operational_metrics"]["mae_skill_pct"],
            18.41742698,
            delta=float(
                self.config["numerical_reproducibility"]
                ["pooled_skill_absolute_tolerance_percentage_points"]
            ),
            msg=numerical_runtime_summary(),
        )
        self.assertEqual(
            self.result["territory_metrics"]["outcome"]
            .value_counts()
            .to_dict(),
            {"win": 47, "loss": 2, "tie": 1},
        )

    def test_current_araba_uses_june_cutoff_and_three_steps(self) -> None:
        current = build_current_ets_illustration(
            self.gold, target_month_id="2026-09", config=self.config
        )
        araba = current.loc[current["territory_id"].eq("ES-PROV-01")].iloc[0]

        self.assertEqual(len(current), 50)
        self.assertEqual(araba["training_end"], "2026-06")
        self.assertEqual(araba["effective_horizon_steps"], 3)
        self.assertEqual(araba["baseline_prediction"], 7691.0)
        self.assertTrue(araba["candidate_available"])
        forecast_rtol = float(
            self.config["numerical_reproducibility"]
            ["forecast_relative_tolerance"]
        )
        np.testing.assert_allclose(
            araba["raw_prediction"],
            7794.63521722,
            rtol=forecast_rtol,
            atol=0.0,
            err_msg=numerical_runtime_summary(),
        )
        np.testing.assert_allclose(
            araba["prediction"],
            7794.63521722,
            rtol=forecast_rtol,
            atol=0.0,
            err_msg=numerical_runtime_summary(),
        )
        self.assertEqual(araba["training_rows"], 258)
        self.assertEqual(araba["training_start"], "2005-01")

    def test_pairing_invariant_rejects_missing_key(self) -> None:
        baseline = self.result["baseline"]["comparable_predictions"]
        incomplete = self.candidate.iloc[:-1].copy()

        with self.assertRaisesRegex(AssertionError, "missing=1"):
            assert_candidate_rows_are_paired(baseline, incomplete)


if __name__ == "__main__":
    unittest.main()
