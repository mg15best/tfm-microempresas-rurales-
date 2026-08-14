import unittest

from datetime import date
from io import BytesIO
from pathlib import Path
from textwrap import dedent

import pandas as pd
from streamlit.testing.v1 import AppTest

from src.visualization.streamlit_app import (
    build_download_csv,
    build_history_summary,
    build_query,
    format_activity_level,
    format_error_message,
    format_percentile_position,
    format_spanish_date,
    format_spanish_month,
    format_spanish_number,
    format_spanish_percent,
    make_history_figure,
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
        self.assertEqual(
            format_spanish_month("2026-09"),
            "septiembre de 2026",
        )
        self.assertEqual(
            format_spanish_date(date(2026, 8, 13)),
            "13 de agosto de 2026",
        )

    def test_activity_levels_are_translated_without_new_categories(self) -> None:
        expected = {
            "low": "Por debajo de lo habitual",
            "usual": "Dentro de lo habitual",
            "high": "Por encima de lo habitual",
            "insufficient": "Contexto histórico insuficiente",
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

    def test_outside_range_warning_names_the_comparable_month(self) -> None:
        message = translate_warning(
            "forecast_outside_historical_range",
            target_month_id="2026-09",
        )

        self.assertIn("septiembres históricos comparables", message)
        self.assertIn("no a todo el gráfico", message)

    def test_percentile_is_expressed_as_historical_position(self) -> None:
        self.assertEqual(
            format_percentile_position(90.0),
            "Más alto que el 90 % de los meses comparables",
        )
        self.assertNotIn("probabilidad", format_percentile_position(90.0))

    def test_error_messages_are_simple_and_actionable(self) -> None:
        self.assertEqual(
            format_error_message("load"),
            "No hemos podido cargar los datos necesarios. Vuelve a intentarlo.",
        )
        self.assertIn(
            "Falta el dato histórico",
            format_error_message("missing_reference"),
        )
        self.assertNotIn("artefact", format_error_message("load").casefold())
        with self.assertRaises(ValueError):
            format_error_message("unknown")

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

        figure = make_history_figure(chart)
        self.assertEqual(len(figure.data), 2)
        self.assertFalse(figure.data[0].connectgaps)
        self.assertEqual(figure.data[0].mode, "lines+markers")
        self.assertEqual(figure.data[1].mode, "markers")
        self.assertEqual(list(figure.data[1].x), [pd.Timestamp("2026-06-01")])
        self.assertNotIn(pd.Timestamp("2026-04-01"), list(figure.data[0].x))
        self.assertNotIn(pd.Timestamp("2026-05-01"), list(figure.data[0].x))
        self.assertIn("Estado del dato", figure.data[0].hovertemplate)
        self.assertIn("Previsión para", figure.data[1].hovertemplate)
        self.assertIn("ene 2026", list(figure.layout.xaxis.ticktext))
        self.assertEqual(figure.layout.yaxis.tickformat, ",.0f")


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

    def test_real_chart_and_text_keep_forecast_separate(self) -> None:
        figure = make_history_figure(self.view.chart)
        historical_months = {
            pd.Timestamp(value).strftime("%Y-%m")
            for value in figure.data[0].x
        }

        self.assertEqual(figure.data[1].mode, "markers")
        self.assertFalse(figure.data[0].connectgaps)
        self.assertEqual(
            [pd.Timestamp(value).strftime("%Y-%m") for value in figure.data[1].x],
            ["2026-09"],
        )
        self.assertNotIn("2026-07", historical_months)
        self.assertNotIn("2026-08", historical_months)

        summary = build_history_summary(self.view)
        self.assertIn("Araba/Álava", summary)
        self.assertIn("julio de 2024", summary)
        self.assertIn("junio de 2026", summary)
        self.assertIn("septiembre de 2026", summary)
        self.assertIn("7.691 pernoctaciones provinciales", summary)
        self.assertIn("interrupciones", summary)


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
        self.assertNotIn("Error histórico WAPE", metric_labels)

        subheaders = [item.value for item in app.subheader]
        self.assertLess(
            subheaders.index("Orientación para la planificación"),
            subheaders.index("Histórico provincial"),
        )
        self.assertIn("Cómo interpretar la previsión", subheaders)

        visible_text = " ".join(
            str(item.value)
            for kind in (
                "caption",
                "info",
                "markdown",
                "subheader",
                "warning",
            )
            for item in app.get(kind)
        )
        self.assertIn("Por encima de lo habitual", visible_text)
        self.assertIn("Error porcentual histórico (WAPE)", visible_text)
        self.assertNotIn("precisión", visible_text.casefold())
        self.assertNotIn("Mes de referencia (t-12)", visible_text)
        self.assertNotIn("Alta", visible_text)
        self.assertIn("dato utilizado como referencia es provisional", visible_text)
        self.assertIn("septiembres históricos comparables", visible_text)
        self.assertIn("Los meses sin datos", visible_text)

        self.assertEqual(len(app.warning), 1)
        self.assertGreaterEqual(len(app.info), 2)
        self.assertEqual(len(app.get("plotly_chart")), 1)
        self.assertEqual(
            [item.label for item in app.expander],
            ["Rendimiento histórico del modelo", "Metodología y trazabilidad"],
        )
        self.assertEqual(len(app.download_button), 1)


class TestStreamlitControlledStates(unittest.TestCase):
    """Estados de UI reproducidos solo con datos en memoria."""

    def test_load_error_is_simple_and_does_not_render_partial_results(self) -> None:
        script = dedent(
            """
            import pandas as pd
            from src.visualization.streamlit_app import (
                AppResources,
                render_app,
            )

            resources = AppResources(
                config={},
                gold=pd.DataFrame(),
                predictions=pd.DataFrame(),
                official_metrics=pd.DataFrame(),
            )
            render_app(resources=resources)
            """
        )

        app = AppTest.from_string(script, default_timeout=30).run()

        self.assertEqual(list(app.exception), [])
        self.assertEqual(
            [item.value for item in app.error],
            [format_error_message("load")],
        )
        self.assertEqual(len(app.metric), 0)

    def test_missing_reference_is_a_clear_blocking_state(self) -> None:
        script = dedent(
            """
            from src.visualization.streamlit_app import (
                AppResources,
                read_app_resources,
                render_app,
            )

            resources = read_app_resources()
            missing_reference = (
                resources.gold["territory_id"].astype(str).eq("ES-PROV-02")
                & resources.gold["month_id"].astype(str).eq("2025-09")
            )
            controlled = AppResources(
                config=resources.config,
                gold=resources.gold.loc[~missing_reference].copy(),
                predictions=resources.predictions,
                official_metrics=resources.official_metrics,
            )
            render_app(resources=controlled, as_of_date="2026-08-13")
            """
        )

        app = AppTest.from_string(script, default_timeout=60).run()

        self.assertEqual(list(app.exception), [])
        self.assertEqual(
            [item.value for item in app.error],
            [format_error_message("missing_reference")],
        )
        self.assertEqual(len(app.metric), 0)

    def test_insufficient_history_is_visible_and_non_blocking(self) -> None:
        script = dedent(
            """
            from src.visualization.streamlit_app import (
                AppResources,
                read_app_resources,
                render_app,
            )

            resources = read_app_resources()
            dates = resources.gold["date_month"]
            territory = resources.gold["territory_id"].astype(str).eq(
                "ES-PROV-02"
            )
            old_comparables = (
                territory
                & dates.dt.month.eq(9)
                & dates.dt.year.lt(2022)
            )
            controlled = AppResources(
                config=resources.config,
                gold=resources.gold.loc[~old_comparables].copy(),
                predictions=resources.predictions,
                official_metrics=resources.official_metrics,
            )
            render_app(resources=controlled, as_of_date="2026-08-13")
            """
        )

        app = AppTest.from_string(script, default_timeout=60).run()

        self.assertEqual(list(app.exception), [])
        self.assertIn(
            "Contexto histórico insuficiente",
            " ".join(item.value for item in app.markdown),
        )
        self.assertTrue(
            any("suficiente histórico" in item.value for item in app.warning)
        )
        self.assertTrue(
            any("ofrecer orientación" in item.value for item in app.info)
        )
        self.assertEqual(len(app.error), 0)


if __name__ == "__main__":
    unittest.main()
