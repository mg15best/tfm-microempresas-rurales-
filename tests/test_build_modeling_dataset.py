import unittest

import pandas as pd

from src.features.build_modeling_dataset import (
    add_calendar_lag,
    add_calendar_rolling_mean,
    safe_percentage_change,
)


class TestCalendarFeatureEngineering(unittest.TestCase):
    """Pruebas de construcción temporal sin leakage."""

    def setUp(self) -> None:
        self.source = pd.DataFrame(
            {
                "territory_id": [
                    "A",
                    "A",
                    "A",
                    "A",
                ],
                "date_month": pd.to_datetime(
                    [
                        "2024-01-01",
                        "2024-02-01",
                        "2024-03-01",
                        "2024-05-01",
                    ]
                ),
                "overnight_stays_total": [
                    10.0,
                    20.0,
                    30.0,
                    50.0,
                ],
            }
        )

    def test_lag_uses_exact_calendar_month(self) -> None:
        """
        Un lag debe buscar el mes calendario exacto.

        Abril encuentra marzo, pero mayo no debe utilizar marzo
        como si fuera el mes inmediatamente anterior, porque abril
        no está presente.
        """
        targets = pd.DataFrame(
            {
                "territory_id": ["A", "A", "A"],
                "target_date_month": pd.to_datetime(
                    [
                        "2024-04-01",
                        "2024-05-01",
                        "2024-06-01",
                    ]
                ),
            }
        )

        result = add_calendar_lag(
            dataframe=targets,
            source=self.source,
            source_column="overnight_stays_total",
            offset_months=1,
            output_column="lag_1_overnight_stays",
        )

        self.assertEqual(
            result.loc[0, "lag_1_overnight_stays"],
            30.0,
        )

        self.assertTrue(
            pd.isna(
                result.loc[1, "lag_1_overnight_stays"]
            )
        )

        self.assertEqual(
            result.loc[2, "lag_1_overnight_stays"],
            50.0,
        )

    def test_calendar_lag_does_not_zero_fill_gap(self) -> None:
        """Un mes ausente debe producir un nulo y nunca un cero."""
        target = pd.DataFrame(
            {
                "territory_id": ["A"],
                "target_date_month": pd.to_datetime(
                    ["2024-05-01"]
                ),
            }
        )

        result = add_calendar_lag(
            dataframe=target,
            source=self.source,
            source_column="overnight_stays_total",
            offset_months=1,
            output_column="lag_1_overnight_stays",
        )

        value = result.loc[
            0,
            "lag_1_overnight_stays",
        ]

        self.assertTrue(pd.isna(value))
        self.assertNotEqual(value, 0)

    def test_rolling_mean_requires_complete_calendar_window(
        self,
    ) -> None:
        """
        La media móvil solo se calcula cuando están presentes
        todos los meses calendario de la ventana.
        """
        targets = pd.DataFrame(
            {
                "territory_id": ["A", "A"],
                "target_date_month": pd.to_datetime(
                    [
                        "2024-04-01",
                        "2024-06-01",
                    ]
                ),
            }
        )

        result = add_calendar_rolling_mean(
            dataframe=targets,
            source=self.source,
            source_column="overnight_stays_total",
            window_months=3,
            output_column=(
                "rolling_mean_3m_overnight_stays"
            ),
        )

        self.assertEqual(
            result.loc[
                0,
                "rolling_mean_3m_overnight_stays",
            ],
            20.0,
        )

        self.assertTrue(
            pd.isna(
                result.loc[
                    1,
                    "rolling_mean_3m_overnight_stays",
                ]
            )
        )

    def test_rolling_mean_excludes_target_month(self) -> None:
        """
        La ventana de abril debe utilizar enero, febrero y marzo;
        el valor del propio abril no puede intervenir.
        """
        source = pd.concat(
            [
                self.source,
                pd.DataFrame(
                    {
                        "territory_id": ["A"],
                        "date_month": pd.to_datetime(
                            ["2024-04-01"]
                        ),
                        "overnight_stays_total": [1000.0],
                    }
                ),
            ],
            ignore_index=True,
        )

        target = pd.DataFrame(
            {
                "territory_id": ["A"],
                "target_date_month": pd.to_datetime(
                    ["2024-04-01"]
                ),
            }
        )

        result = add_calendar_rolling_mean(
            dataframe=target,
            source=source,
            source_column="overnight_stays_total",
            window_months=3,
            output_column=(
                "rolling_mean_3m_overnight_stays"
            ),
        )

        self.assertEqual(
            result.loc[
                0,
                "rolling_mean_3m_overnight_stays",
            ],
            20.0,
        )

    def test_safe_percentage_change(self) -> None:
        """La variación evita divisiones por cero o datos ausentes."""
        current = pd.Series(
            [120.0, 50.0, pd.NA],
            dtype="Float64",
        )

        previous = pd.Series(
            [100.0, 0.0, 20.0],
            dtype="Float64",
        )

        result = safe_percentage_change(
            current,
            previous,
        )

        self.assertAlmostEqual(
            float(result.iloc[0]),
            20.0,
        )

        self.assertTrue(pd.isna(result.iloc[1]))
        self.assertTrue(pd.isna(result.iloc[2]))


if __name__ == "__main__":
    unittest.main()