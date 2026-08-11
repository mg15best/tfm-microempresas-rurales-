import unittest

from src.models.evaluate_final_candidate import (
    ensure_final_test_is_untouched,
)


class TestEvaluateFinalCandidateGuard(unittest.TestCase):
    """Prueba la protección sin cargar ni evaluar filas de test."""

    def test_already_opened_test_is_rejected(self) -> None:
        config = {
            "validation": {
                "final_test": {
                    "test_status": "already_opened",
                },
            },
        }

        with self.assertRaisesRegex(RuntimeError, "already opened"):
            ensure_final_test_is_untouched(config)


if __name__ == "__main__":
    unittest.main()
