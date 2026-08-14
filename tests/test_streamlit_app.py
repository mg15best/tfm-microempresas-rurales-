import unittest

from datetime import date
from io import BytesIO
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

from src.visualization.streamlit_app import (
    build_download_csv,
    build_query,
    format_activity_level,
    format_spanish_date,
    format_spanish_month,
    format_spanish_number,
    format_spanish_percent,
    prepare_history_chart,
    read_app_resources,
    territory_options,
    translate_warning,
)


class TestStreamlitPresentationFunctions(unittest.TestCase):
    """Pruebas puras del contrato de presentacion."""

    def test_spanish_formatters_are_deterministic(self) -> None:
        self.assertEqual(format_spanish_number(1_226_939), "1.226.939")
        self.assertEqual(format_spanish_number(24.392, 1), "24,4")
        self.assertEqual(format_spanish_percent(24.392), "24,4 %")
        self.assertEqual(format_spanish_percent(None), "No disponible")
        self.assertEqual(format_spanish_month("2026-09"), "Septiembre 2026")
        self.assertEqual(
            format_spanish_date(date(2026, 8, 13)),
            "13 de agosto de 2026",
        )

    def test_activity_levels_are_translated_without_new_categories(self) -> None:
        expected = {
            "low": "Baja",
            "usual": "Habitual",
            "high": "Alta",
            "insufficient": "Contexto insuficiente",
        }

        self.assertEqual(
            {key: format_activity_level(key) for key in expected},
            expected,
        )
        with self.assertRaises(ValueError):
            format_activity_level("very_high")

    def test_known_warning_has_a_clear_spanish_message(self) -> None:
        message = translate_warning("provisional_reference_data")

        self.assertIn("provisional", message)
        self.assertIn("INE", message)

    def test_territory_selector_is_derived_and_sorted_from_data(self) -> None:
        gold = pd.DataFrame(
            {
                "territory_id": ["ES-PROV-02", "ES-PROV-01", "ES-PROV-02"],
                "territory_name": ["Zamora", "Álava", "Zamora"],
            }
        )

        options = territory_options(gold)

        self.assertEqual(
            [(item.territory_id, item.territory_name) for item in options],
            [("ES-PROV-01", "Álava"), ("ES-PROV-02", "Zamora")],
        )

    def test_history_uses_natural_months_and_preserves_a_gap(self) -> None:
        history = pd.DataFrame(
            {
                "date_month": pd.to_datetime(["2026-01-01", "2026-03-01"]),
                "overnight_stays_total": [100.0, 300.0],
            }
        )

        chart = prepare_history_chart(
            history,
            "2026-06",
            600.0,
            months=3,
        )

        self.assertEqual(
            chart.history["date_month"].dt.strftime("%Y-%m").tolist(),
            ["2026-01", "2026-02", "2026-03"],
        )
        self.assertTrue(pd.isna(chart.history.iloc[1]["overnight_stays_total"]))
        self.assertEqual(chart.forecast_date, pd.Timestamp("2026-06-01"))
        self.assertNotIn(chart.forecast_date, chart.history["date_month"].tolist())


class TestStreamlitRealProduct(unittest.TestCase):
    """Integracion reproducible con los artefactos reales congelados."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.resources = read_app_resources()
        cls.view = build_query(
            cls.resources,
            "ES-PROV-01",
            as_of_date="2026-08-13",
        )

    def test_real_selector_contains_fifty_provinces(self) -> None:
        options = territory_options(self.resources.gold)

        self.assertEqual(len(options), 50)
        self.assertEqual(len({item.territory_id for item in options}), 50)
        self.assertIn(
            ("ES-PROV-01", "Araba/Álava"),
            {(item.territory_id, item.territory_name) for item in options},
        )

    def test_real_araba_query_reproduces_expected_product(self) -> None:
        view = self.view

        self.assertEqual(view.territory_name, "Araba/Álava")
        self.assertEqual(view.as_of_date, date(2026, 8, 13))
        self.assertEqual(view.target_month_id, "2026-09")
        self.assertEqual(view.forecast_value, 7_691.0)
        self.assertEqual(view.reference_month_id, "2025-09")
        self.assertTrue(view.reference_is_provisional)
        self.assertEqual(view.latest_available_month_id, "2026-06")
        self.assertEqual(view.activity_level, "high")
        self.assertEqual(view.historical_percentile_pct, 90.0)
        self.assertEqual(view.historical_sample_size, 10)
        self.assertAlmostEqual(view.validation_wape_pct, 24.392342351526025)
        self.assertEqual(view.validation_rows, 35)
        self.assertIn(
            "provisional_reference_data",
            {warning.code for warning in view.warnings},
        )

    def test_download_is_utf8_csv_built_in_memory(self) -> None:
        payload = build_download_csv(self.view)
        exported = pd.read_csv(BytesIO(payload))

        self.assertTrue(payload.startswith(b"\xef\xbb\xbf"))
        self.assertEqual(len(exported), 1)
        self.assertEqual(exported.iloc[0]["territory_id"], "ES-PROV-01")
        self.assertEqual(exported.iloc[0]["target_month_id"], "2026-09")
        self.assertEqual(
            exported.iloc[0]["warning_codes"],
            "provisional_reference_data",
        )


class TestStreamlitNativeSmoke(unittest.TestCase):
    """Smoke test del arbol nativo renderizado por Streamlit."""

    def test_app_renders_real_default_query_without_exceptions(self) -> None:
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        app = AppTest.from_file(app_path, default_timeout=120).run()

        self.assertEqual(list(app.exception), [])
        self.assertEqual(len(app.selectbox), 1)
        self.assertEqual(app.selectbox[0].label, "Provincia")
        self.assertEqual(len(app.selectbox[0].options), 50)
        metric_labels = {metric.label for metric in app.metric}
        self.assertIn("Pernoctaciones provinciales previstas", metric_labels)
        self.assertIn("Actividad territorial relativa", metric_labels)
        self.assertIn("Error histórico WAPE", metric_labels)
        self.assertIn("Datos oficiales disponibles hasta", metric_labels)
        self.assertGreaterEqual(len(app.warning), 1)
        self.assertEqual(len(app.download_button), 1)


if __name__ == "__main__":
    unittest.main()
