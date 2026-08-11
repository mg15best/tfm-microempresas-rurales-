import unittest

from src.models.select_models import select_solution_by_validation


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


if __name__ == "__main__":
    unittest.main()
