import unittest

import pandas as pd

from src.data.build_gold import add_tourism_pressure_index


class TestTourismPressureIndex(unittest.TestCase):
    """Pruebas del índice descriptivo de presión turística."""

    def test_requires_all_four_components(self) -> None:
        """
        El índice solo debe calcularse cuando están disponibles
        sus cuatro componentes.
        """
        dataframe = pd.DataFrame(
            {
                "month_id": [
                    "2024-08",
                    "2024-08",
                    "2024-08",
                ],
                "occupancy_rate_pct": [
                    70.0,
                    65.0,
                    80.0,
                ],
                "overnight_stays_per_place": [
                    12.0,
                    10.0,
                    15.0,
                ],
                "overnight_stays_yoy_change_pct": [
                    5.0,
                    float("nan"),
                    8.0,
                ],
                "weekend_occupancy_rate_pct": [
                    75.0,
                    70.0,
                    85.0,
                ],
            }
        )

        result = add_tourism_pressure_index(dataframe)

        self.assertFalse(
            pd.isna(
                result.loc[
                    0,
                    "tourism_pressure_index",
                ]
            )
        )

        self.assertTrue(
            pd.isna(
                result.loc[
                    1,
                    "tourism_pressure_index",
                ]
            )
        )

        self.assertFalse(
            pd.isna(
                result.loc[
                    2,
                    "tourism_pressure_index",
                ]
            )
        )


if __name__ == "__main__":
    unittest.main()