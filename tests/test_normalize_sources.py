from pathlib import Path
import unittest

import pandas as pd
import yaml

from src.data.normalize_sources import (
    PROVISIONAL_WINDOW_MONTHS,
    add_time_columns,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TestProvisionalityByVintage(unittest.TestCase):
    """Pruebas de la ventana provisional móvil de cada vintage."""

    @staticmethod
    def _classify(*periods: str) -> dict[str, bool]:
        result = add_time_columns(
            pd.DataFrame({"source_period": list(periods)})
        )
        return dict(
            zip(
                result["source_period"].tolist(),
                result["is_provisional"].astype(bool).tolist(),
                strict=True,
            )
        )

    def test_repository_vintage_keeps_june_2025_to_may_2026_provisional(
        self,
    ) -> None:
        classified = self._classify("2025M05", "2025M06", "2026M05")

        self.assertFalse(classified["2025M05"])
        self.assertTrue(classified["2025M06"])
        self.assertTrue(classified["2026M05"])

    def test_next_vintage_moves_window_without_a_manual_date_change(
        self,
    ) -> None:
        classified = self._classify(
            "2025M06",
            "2025M07",
            "2026M06",
        )

        self.assertFalse(classified["2025M06"])
        self.assertTrue(classified["2025M07"])
        self.assertTrue(classified["2026M06"])

    def test_metadata_and_normalizer_share_the_twelve_month_contract(
        self,
    ) -> None:
        with (PROJECT_ROOT / "data/metadata/validation_rules.yml").open(
            encoding="utf-8"
        ) as rules_file:
            validation_rules = yaml.safe_load(rules_file)
        with (PROJECT_ROOT / "data/metadata/data_sources.yml").open(
            encoding="utf-8"
        ) as sources_file:
            sources = yaml.safe_load(sources_file)

        self.assertEqual(
            validation_rules["dataset"]["provisional_window_months"],
            PROVISIONAL_WINDOW_MONTHS,
        )
        provincial_sources = sources["source_groups"]["rural_occupancy"][:2]
        self.assertEqual(
            [
                source["provisionality"]["window_months"]
                for source in provincial_sources
            ],
            [PROVISIONAL_WINDOW_MONTHS, PROVISIONAL_WINDOW_MONTHS],
        )

    def test_real_gold_matches_the_current_vintage_window(self) -> None:
        gold = pd.read_parquet(
            PROJECT_ROOT
            / "data/gold/gold_tourism_demand_monthly.parquet"
        )
        months = pd.to_datetime(gold["date_month"]).dt.to_period("M")
        latest_month = months.max()
        provisional_from = latest_month - (PROVISIONAL_WINDOW_MONTHS - 1)

        self.assertEqual(str(latest_month), "2026-05")
        self.assertTrue(
            gold["is_provisional"].astype(bool).eq(
                months.ge(provisional_from)
            ).all()
        )


if __name__ == "__main__":
    unittest.main()
