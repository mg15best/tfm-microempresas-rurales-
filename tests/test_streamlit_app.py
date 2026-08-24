import unittest

from dataclasses import replace
from datetime import date
from io import BytesIO
from pathlib import Path
from textwrap import dedent
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

from src.models import inference
from src.models.inference import FALLBACK_MODEL_ID, SELECTED_MODEL_ID
from src.visualization.streamlit_app import (
    AppResources,
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
    load_app_resources,
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

        training_message = translate_warning("provisional_training_data")
        self.assertIn("estimación ETS", training_message)
        self.assertIn("datos provisionales del INE", training_message)

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
        self.assertEqual(figure.layout.height, 290)


class TestStreamlitRealProduct(unittest.TestCase):
    """Integracion reproducible con los artefactos reales congelados."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.resources = read_app_resources()
        cls.araba = build_query(
            cls.resources,
            "ES-PROV-01",
            as_of_date="2026-08-20",
        )
        cls.badajoz = build_query(
            cls.resources,
            "ES-PROV-06",
            as_of_date="2026-08-20",
        )

    def test_app_resources_are_b5_only_and_isolate_score_bank(self) -> None:
        resources = self.resources

        self.assertTrue(resources.config_v2)
        self.assertFalse(hasattr(resources, "predictions"))
        self.assertFalse(hasattr(resources, "official_metrics"))
        self.assertEqual(
            resources.forecast_resources.canonical_bundle.selected_model_id,
            SELECTED_MODEL_ID,
        )
        first = resources.forecast_resources.operational_score_bank
        original = first.iloc[0]["score"]
        first.loc[first.index[0], "score"] = -999.0
        second = resources.forecast_resources.operational_score_bank
        self.assertEqual(second.iloc[0]["score"], original)

    def test_real_selector_contains_fifty_provinces(self) -> None:
        options = territory_options(self.resources.gold)

        self.assertEqual(len(options), 50)
        self.assertEqual(len({item.territory_id for item in options}), 50)
        self.assertIn(
            ("ES-PROV-01", "Araba/Álava"),
            {(item.territory_id, item.territory_name) for item in options},
        )

    def test_real_araba_query_reproduces_expected_product(self) -> None:
        view = self.araba

        self.assertEqual(view.territory_name, "Araba/Álava")
        self.assertEqual(view.as_of_date, date(2026, 8, 20))
        self.assertEqual(view.target_month_id, "2026-09")
        self.assertGreaterEqual(view.forecast_value, 0)
        self.assertEqual(view.baseline_reference_month_id, "2025-09")
        self.assertTrue(view.baseline_reference_is_provisional)
        self.assertEqual(view.latest_available_month_id, "2026-06")
        self.assertEqual(view.selected_model_id, SELECTED_MODEL_ID)
        self.assertEqual(view.actual_model_used, SELECTED_MODEL_ID)
        self.assertFalse(view.fallback_used)
        self.assertEqual(view.selection_status, "provisional_validation_champion")
        self.assertTrue(view.interval_available)
        self.assertLessEqual(view.interval_lower, view.forecast_value)
        self.assertGreaterEqual(view.interval_upper, view.forecast_value)
        self.assertEqual(view.activity_level, "high")
        self.assertEqual(view.historical_percentile_pct, 90.0)
        self.assertEqual(view.historical_sample_size, 10)
        self.assertAlmostEqual(view.validation_wape_pct, 23.354961719573705)
        self.assertEqual(view.validation_rows, 35)
        self.assertEqual(view.evaluation_scope, "canonical_rolling_validation")

    def test_download_is_utf8_csv_built_in_memory(self) -> None:
        payload = build_download_csv(self.araba)
        exported = pd.read_csv(BytesIO(payload))

        self.assertTrue(payload.startswith(b"\xef\xbb\xbf"))
        self.assertEqual(len(exported), 1)
        self.assertEqual(exported.iloc[0]["territory_id"], "ES-PROV-01")
        self.assertEqual(exported.iloc[0]["target_month_id"], "2026-09")
        required = {
            "selected_model_id",
            "selection_status",
            "actual_model_used",
            "fallback_used",
            "fallback_reason",
            "baseline_reference_month_id",
            "baseline_prediction",
            "interval_available",
            "interval_lower",
            "interval_upper",
            "interval_nominal_level",
            "interval_method_id",
            "evaluation_scope",
            "evaluation_generator_commit_sha",
            "evaluation_logical_prediction_sha256",
        }
        self.assertTrue(required.issubset(exported.columns))
        self.assertIn(
            "provisional_training_data",
            exported.iloc[0]["warning_codes"],
        )

    def test_real_chart_and_text_keep_forecast_separate(self) -> None:
        figure = make_history_figure(self.araba.chart)
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

        self.assertTrue(figure.data[1].error_y.visible)
        summary = build_history_summary(self.araba)
        self.assertIn("Araba/Álava", summary)
        self.assertIn("julio de 2024", summary)
        self.assertIn("junio de 2026", summary)
        self.assertIn("septiembre de 2026", summary)
        self.assertIn("pernoctaciones provinciales", summary)
        self.assertIn("interrupciones", summary)

    def test_badajoz_exposes_selected_model_and_availability_fallback(self) -> None:
        view = self.badajoz

        self.assertEqual(view.selected_model_id, SELECTED_MODEL_ID)
        self.assertEqual(view.actual_model_used, FALLBACK_MODEL_ID)
        self.assertTrue(view.fallback_used)
        self.assertEqual(view.fallback_reason, "training_gap_unsupported")
        self.assertEqual(view.forecast_value, 14_214.0)
        self.assertTrue(view.interval_available)
        self.assertEqual(view.chart.history["date_month"].max(), pd.Timestamp("2026-06-01"))

    def test_unavailable_interval_keeps_point_and_has_no_chart_range(self) -> None:
        view = replace(
            self.araba,
            interval_available=False,
            interval_lower=None,
            interval_upper=None,
            interval_unavailable_reason="insufficient_calibration_origins",
            chart=replace(
                self.araba.chart,
                interval_available=False,
                interval_lower=None,
                interval_upper=None,
            ),
        )

        figure = make_history_figure(view.chart)

        self.assertGreaterEqual(view.forecast_value, 0)
        self.assertFalse(view.interval_available)
        self.assertFalse(figure.data[1].error_y.visible)

    def test_precomputed_resources_avoid_revalidation_and_score_rebuild(self) -> None:
        with (
            patch(
                "src.visualization.dashboard_data.validate_canonical_validation_bundle"
            ) as validate_bundle,
            patch(
                "src.application.forecast_service.build_operational_score_bank"
            ) as build_bank,
            patch(
                "src.models.inference.fit_ets_forecast",
                wraps=inference.fit_ets_forecast,
            ) as fit_ets,
        ):
            build_query(
                self.resources,
                "ES-PROV-01",
                as_of_date="2026-08-20",
            )

        validate_bundle.assert_not_called()
        build_bank.assert_not_called()
        fit_ets.assert_called_once()

    def test_streamlit_resource_cache_reads_once_and_returns_isolated_data(
        self,
    ) -> None:
        load_app_resources.clear()
        with patch(
            "src.visualization.streamlit_app.read_app_resources",
            return_value=self.resources,
        ) as read_resources:
            first = load_app_resources()
            second = load_app_resources()

        read_resources.assert_called_once()
        first.gold.loc[first.gold.index[0], "territory_name"] = "mutated"
        self.assertNotEqual(
            second.gold.loc[second.gold.index[0], "territory_name"],
            "mutated",
        )
        load_app_resources.clear()


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
        self.assertTrue(
            {
                "Pronóstico puntual",
                "Intervalo empírico 80 %",
                "Posición histórica",
                "Modelo realmente usado",
                "Mes objetivo",
            }.issubset(metric_labels)
        )
        self.assertNotIn("Error histórico WAPE", metric_labels)

        dashboard_headings = [
            item.value
            for item in app.markdown
            if str(item.value).startswith("#")
        ]
        self.assertLess(
            dashboard_headings.index("### Resumen ejecutivo"),
            dashboard_headings.index("#### Evolución provincial y previsión"),
        )
        self.assertIn("#### Evolución provincial y previsión", dashboard_headings)

        visible_text = " ".join(
            str(item.value)
            for kind in (
                "caption",
                "info",
                "markdown",
                "metric",
                "warning",
            )
            for item in app.get(kind)
        )
        self.assertIn("por encima de lo habitual", visible_text.casefold())
        self.assertIn("Error WAPE en validación", visible_text)
        self.assertIn("seleccionado provisionalmente", visible_text)
        self.assertIn("distinta del intervalo predictivo", visible_text)
        self.assertIn("Lectura rápida", visible_text)
        self.assertIn("Avisos", visible_text)
        self.assertIn("Señal provincial", visible_text)
        self.assertNotIn("precisión", visible_text.casefold())
        self.assertNotIn("confidence", visible_text.casefold())
        self.assertNotIn("Mes de referencia (t-12)", visible_text)
        self.assertNotIn("Alta", visible_text)
        self.assertIn("septiembres históricos comparables", visible_text)
        self.assertIn("Los meses sin datos", visible_text)
        self.assertIn(
            "La estimación ETS utiliza datos provisionales del INE",
            visible_text,
        )

        self.assertEqual(len(app.warning), 1)
        self.assertGreaterEqual(len(app.info), 2)
        self.assertEqual(len(app.get("plotly_chart")), 1)
        self.assertEqual(
            [item.label for item in app.tabs],
            [
                "Histórico comparable",
                "Metodología y trazabilidad",
                "Exportación",
            ],
        )
        self.assertEqual(
            [item.label for item in app.expander],
            ["Detalle y trazabilidad"],
        )
        self.assertEqual(len(app.download_button), 1)

    def test_badajoz_rerun_shows_human_fallback_warning(self) -> None:
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        app = AppTest.from_file(app_path, default_timeout=120).run()

        app.selectbox[0].select("Badajoz").run(timeout=120)

        self.assertEqual(list(app.exception), [])
        visible_warnings = " ".join(item.value for item in app.warning)
        self.assertIn("ETS no está disponible con este histórico", visible_warnings)
        self.assertIn("mismo mes del año anterior", visible_warnings)
        self.assertIn("dato de referencia utilizado es provisional", visible_warnings)
        self.assertNotIn("training_gap_unsupported", visible_warnings)
        model_metric = next(
            item for item in app.metric if item.label == "Modelo realmente usado"
        )
        self.assertEqual(model_metric.value, "lag-12")

    def test_interval_unavailable_copy_does_not_block_point(self) -> None:
        script = dedent(
            """
            from dataclasses import replace
            from src.visualization.streamlit_app import (
                _render_primary_result,
                build_query,
                read_app_resources,
            )

            view = build_query(
                read_app_resources(),
                "ES-PROV-01",
                as_of_date="2026-08-20",
            )
            controlled = replace(
                view,
                interval_available=False,
                interval_lower=None,
                interval_upper=None,
                interval_unavailable_reason="insufficient_calibration_origins",
            )
            _render_primary_result(controlled)
            """
        )

        app = AppTest.from_string(script, default_timeout=120).run()

        self.assertEqual(list(app.exception), [])
        text = " ".join(
            str(item.value)
            for kind in ("markdown", "caption", "metric")
            for item in app.get(kind)
        )
        self.assertIn("Intervalo no disponible", text)
        self.assertIn(
            "Pronóstico puntual",
            [item.label for item in app.metric],
        )


class TestStreamlitControlledStates(unittest.TestCase):
    """Estados de UI reproducidos solo con datos en memoria."""

    def test_load_error_is_simple_and_does_not_render_partial_results(self) -> None:
        script = dedent(
            """
            from dataclasses import replace
            from src.visualization.streamlit_app import (
                read_app_resources,
                render_app,
            )

            loaded = read_app_resources()
            resources = replace(loaded, gold=loaded.gold.iloc[0:0].copy())
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

    def test_canonical_integrity_error_has_a_distinct_blocking_message(self) -> None:
        script = dedent(
            """
            from unittest.mock import patch
            from src.visualization.dashboard_data import CanonicalArtifactError
            from src.visualization.streamlit_app import render_app

            with patch(
                "src.visualization.streamlit_app.load_app_resources",
                side_effect=CanonicalArtifactError("invalid hash"),
            ):
                render_app()
            """
        )

        app = AppTest.from_string(script, default_timeout=30).run()

        self.assertEqual(list(app.exception), [])
        self.assertEqual(
            [item.value for item in app.error],
            [format_error_message("canonical")],
        )
        self.assertEqual(len(app.metric), 0)

    def test_missing_reference_is_a_clear_blocking_state(self) -> None:
        script = dedent(
            """
            from dataclasses import replace
            from src.visualization.streamlit_app import (
                read_app_resources,
                render_app,
            )

            resources = read_app_resources()
            missing_reference = (
                resources.gold["month_id"].astype(str).eq("2025-09")
            )
            controlled = replace(
                resources,
                gold=resources.gold.loc[~missing_reference].copy(),
            )
            render_app(resources=controlled, as_of_date="2026-08-13")
            """
        )

        app = AppTest.from_string(script, default_timeout=60).run()
        app.selectbox[0].select("Badajoz").run(timeout=60)

        self.assertEqual(list(app.exception), [])
        self.assertEqual(
            [item.value for item in app.error],
            [format_error_message("missing_reference")],
        )
        self.assertEqual(len(app.metric), 0)

    def test_insufficient_history_is_visible_and_non_blocking(self) -> None:
        script = dedent(
            """
            from dataclasses import replace
            from src.visualization.streamlit_app import (
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
            controlled = replace(
                resources,
                gold=resources.gold.loc[~old_comparables].copy(),
            )
            render_app(resources=controlled, as_of_date="2026-08-13")
            """
        )

        app = AppTest.from_string(script, default_timeout=60).run()

        self.assertEqual(list(app.exception), [])
        self.assertIn(
            "contexto histórico insuficiente",
            " ".join(item.value for item in app.markdown).casefold(),
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
