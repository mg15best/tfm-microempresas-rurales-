"""Presentacion Streamlit del producto provincial de forecasting."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from io import StringIO
import logging
from typing import Any, Mapping
import unicodedata

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
import pandas as pd
import streamlit as st

from src.application.decision_support import (
    DecisionSupport,
    DecisionSupportError,
    build_decision_support,
)
from src.application.forecast_service import (
    ForecastProductContext,
    ProductCompositionError,
    build_forecast_product_context,
)
from src.models.inference import AsOfDate, InferenceError
from src.models.modeling_common import load_config
from src.visualization.dashboard_data import (
    DashboardDataError,
    load_gold_history,
    load_official_validation_metrics,
    load_validation_predictions,
)


__all__ = [
    "AppResources",
    "ForecastViewModel",
    "HistoryChartData",
    "TerritoryOption",
    "build_download_csv",
    "build_query",
    "build_view_model",
    "format_activity_level",
    "format_spanish_date",
    "format_spanish_month",
    "format_spanish_number",
    "format_spanish_percent",
    "load_app_resources",
    "make_history_figure",
    "prepare_history_chart",
    "read_app_resources",
    "render_app",
    "territory_options",
    "translate_warning",
]

LOGGER = logging.getLogger(__name__)

SPANISH_MONTHS = (
    "enero",
    "febrero",
    "marzo",
    "abril",
    "mayo",
    "junio",
    "julio",
    "agosto",
    "septiembre",
    "octubre",
    "noviembre",
    "diciembre",
)

ACTIVITY_LABELS = {
    "low": "Baja",
    "usual": "Habitual",
    "high": "Alta",
    "insufficient": "Contexto insuficiente",
}

WARNING_MESSAGES = {
    "provisional_reference_data": (
        "El dato de referencia utilizado es provisional y puede ser "
        "revisado por el INE."
    ),
    "historical_comparison_gaps": (
        "Faltan observaciones del mes comparable en algunos años."
    ),
    "invalid_historical_observations_excluded": (
        "Se excluyeron valores históricos no válidos de la comparación."
    ),
    "insufficient_seasonal_history": (
        "No hay suficiente histórico comparable para asignar un nivel."
    ),
    "flat_comparison_history": (
        "Todas las observaciones de la referencia estacional son iguales."
    ),
    "forecast_outside_historical_range": (
        "La previsión queda fuera del rango de la muestra histórica "
        "comparable."
    ),
}


@dataclass(frozen=True)
class AppResources:
    """Artefactos inyectables que una sesión solo debe cargar una vez."""

    config: Mapping[str, Any]
    gold: pd.DataFrame
    predictions: pd.DataFrame
    official_metrics: pd.DataFrame


@dataclass(frozen=True, order=True)
class TerritoryOption:
    """Opción del selector: label visible y clave productiva estable."""

    sort_key: str
    territory_name: str
    territory_id: str


@dataclass(frozen=True)
class HistoryChartData:
    """Serie real con gaps y punto de forecast deliberadamente separado."""

    history: pd.DataFrame
    forecast_date: pd.Timestamp
    forecast_value: float


@dataclass(frozen=True)
class WarningView:
    """Warning analítico con copy de presentación determinista."""

    code: str
    message: str


@dataclass(frozen=True)
class ForecastViewModel:
    """Datos reales necesarios para renderizar una consulta provincial."""

    territory_id: str
    territory_name: str
    as_of_date: date
    target_month_id: str
    forecast_value: float
    reference_month_id: str
    reference_value: float
    reference_is_provisional: bool
    latest_available_month_id: str
    operational_status: str
    model_name: str
    forecast_horizon_months: int
    activity_level: str
    historical_percentile_pct: float | None
    historical_median: float | None
    historical_q25: float | None
    historical_q75: float | None
    historical_sample_size: int
    action_guidance: str | None
    rule_id: str
    comparison_month_ids: tuple[str, ...]
    excluded_covid_month_ids: tuple[str, ...]
    excluded_provisional_month_ids: tuple[str, ...]
    warnings: tuple[WarningView, ...]
    limitations: tuple[str, ...]
    validation_wape_pct: float
    validation_mae: float
    validation_rmse: float
    validation_bias: float
    validation_rows: int
    operational_source_snapshot_id: str
    operational_pipeline_run_id: str
    operational_data_version: str
    evaluation_source_snapshot_id: str
    evaluation_pipeline_run_id: str
    evaluation_data_version: str
    chart: HistoryChartData


def _alphabetical_key(value: str) -> str:
    """Ordena labels de forma estable sin depender del locale del sistema."""
    normalized = unicodedata.normalize("NFKD", value.casefold())
    return "".join(char for char in normalized if not unicodedata.combining(char))


def format_spanish_number(value: float, decimals: int = 0) -> str:
    """Formatea un número sin alterar su valor subyacente."""
    if decimals < 0:
        raise ValueError("decimals no puede ser negativo.")
    formatted = f"{float(value):,.{decimals}f}"
    return formatted.replace(",", "_").replace(".", ",").replace("_", ".")


def format_spanish_percent(value: float | None, decimals: int = 1) -> str:
    """Formatea un porcentaje o explicita que no está disponible."""
    if value is None:
        return "No disponible"
    return f"{format_spanish_number(value, decimals)} %"


def format_spanish_month(month_id: str) -> str:
    """Convierte YYYY-MM en un label mensual en español."""
    try:
        period = pd.Period(month_id, freq="M")
    except (TypeError, ValueError) as error:
        raise ValueError(f"Mes no válido: {month_id!r}.") from error
    return f"{SPANISH_MONTHS[period.month - 1].capitalize()} {period.year}"


def format_spanish_date(value: date) -> str:
    """Presenta una fecha local con mes en español."""
    return f"{value.day} de {SPANISH_MONTHS[value.month - 1]} de {value.year}"


def format_activity_level(activity_level: str) -> str:
    """Traduce exclusivamente los niveles definidos por DecisionSupport."""
    try:
        return ACTIVITY_LABELS[activity_level]
    except KeyError as error:
        raise ValueError(
            f"Nivel de actividad no soportado: {activity_level!r}."
        ) from error


def translate_warning(code: str, fallback_message: str = "") -> str:
    """Traduce un warning real sin crear señales analíticas nuevas."""
    return WARNING_MESSAGES.get(code, fallback_message or code)


def territory_options(gold: pd.DataFrame) -> tuple[TerritoryOption, ...]:
    """Deriva el selector de los territorios reales, nunca de un mapping manual."""
    required = {"territory_id", "territory_name"}
    missing = required.difference(gold.columns)
    if missing:
        raise ValueError(
            "Faltan columnas territoriales: " + ", ".join(sorted(missing))
        )

    pairs = gold.loc[:, ["territory_id", "territory_name"]].drop_duplicates()
    if pairs.empty:
        raise ValueError("No hay territorios disponibles.")
    if pairs["territory_id"].astype(str).duplicated().any():
        raise ValueError("Un territory_id tiene más de un nombre.")

    options = (
        TerritoryOption(
            sort_key=_alphabetical_key(str(row.territory_name)),
            territory_name=str(row.territory_name),
            territory_id=str(row.territory_id),
        )
        for row in pairs.itertuples(index=False)
    )
    return tuple(sorted(options))


def read_app_resources() -> AppResources:
    """Lee una vez cada artefacto que el backend permite inyectar."""
    config = load_config()
    return AppResources(
        config=config,
        gold=load_gold_history(config),
        predictions=load_validation_predictions(),
        official_metrics=load_official_validation_metrics(),
    )


@st.cache_data(show_spinner=False)
def load_app_resources() -> AppResources:
    """Cachea cargas deterministas; no cachea consultas por usuario."""
    return read_app_resources()


def prepare_history_chart(
    history: pd.DataFrame,
    target_month_id: str,
    forecast_value: float,
    *,
    months: int = 24,
) -> HistoryChartData:
    """Prepara una ventana natural preservando gaps como nulos."""
    if not isinstance(months, int) or isinstance(months, bool) or months < 1:
        raise ValueError("months debe ser un entero positivo.")
    required = {"date_month", "overnight_stays_total"}
    missing = required.difference(history.columns)
    if missing:
        raise ValueError(
            "Faltan columnas históricas: " + ", ".join(sorted(missing))
        )
    if history.empty:
        raise ValueError("El histórico está vacío.")

    prepared = history.loc[:, list(required)].copy()
    prepared["date_month"] = pd.to_datetime(
        prepared["date_month"],
        errors="raise",
    )
    prepared["overnight_stays_total"] = pd.to_numeric(
        prepared["overnight_stays_total"],
        errors="coerce",
    )
    prepared = prepared.sort_values("date_month")
    last_month = prepared["date_month"].max().to_period("M")
    first_month = last_month - (months - 1)
    calendar = pd.date_range(
        start=first_month.to_timestamp(),
        end=last_month.to_timestamp(),
        freq="MS",
    )
    window = prepared.loc[
        prepared["date_month"].dt.to_period("M").between(
            first_month,
            last_month,
        )
    ]
    window = (
        window.set_index("date_month")
        .reindex(calendar)
        .rename_axis("date_month")
        .reset_index()
    )
    return HistoryChartData(
        history=window,
        forecast_date=pd.Period(target_month_id, freq="M").to_timestamp(),
        forecast_value=float(forecast_value),
    )


def build_view_model(
    product: ForecastProductContext,
    support: DecisionSupport,
) -> ForecastViewModel:
    """Compone el contrato de presentación sin recalcular analítica."""
    forecast = product.forecast
    metrics = product.dashboard.validation_metrics
    lineage = product.dashboard.lineage
    warnings = tuple(
        WarningView(
            code=warning.code,
            message=translate_warning(warning.code, warning.message),
        )
        for warning in support.warnings
    )
    return ForecastViewModel(
        territory_id=forecast.territory_id,
        territory_name=forecast.territory_name,
        as_of_date=product.as_of_date,
        target_month_id=forecast.target_month_id,
        forecast_value=float(forecast.predicted_overnight_stays_total),
        reference_month_id=forecast.reference_month_id,
        reference_value=float(forecast.reference_overnight_stays_total),
        reference_is_provisional=forecast.reference_is_provisional,
        latest_available_month_id=forecast.latest_available_month_id,
        operational_status=forecast.operational_status,
        model_name=forecast.model_name,
        forecast_horizon_months=forecast.forecast_horizon_months,
        activity_level=support.activity_level,
        historical_percentile_pct=support.historical_percentile_pct,
        historical_median=support.historical_median,
        historical_q25=support.historical_q25,
        historical_q75=support.historical_q75,
        historical_sample_size=support.historical_sample_size,
        action_guidance=support.action_guidance,
        rule_id=support.rule_id,
        comparison_month_ids=support.comparison_month_ids,
        excluded_covid_month_ids=support.excluded_covid_month_ids,
        excluded_provisional_month_ids=(
            support.excluded_provisional_month_ids
        ),
        warnings=warnings,
        limitations=support.limitations,
        validation_wape_pct=metrics.validation_wape_pct,
        validation_mae=metrics.validation_mae,
        validation_rmse=metrics.validation_rmse,
        validation_bias=metrics.validation_bias,
        validation_rows=metrics.validation_rows,
        operational_source_snapshot_id=(
            lineage.operational_source_snapshot_id
        ),
        operational_pipeline_run_id=lineage.operational_pipeline_run_id,
        operational_data_version=lineage.operational_data_version,
        evaluation_source_snapshot_id=lineage.evaluation_source_snapshot_id,
        evaluation_pipeline_run_id=lineage.evaluation_pipeline_run_id,
        evaluation_data_version=lineage.evaluation_data_version,
        chart=prepare_history_chart(
            product.dashboard.history,
            forecast.target_month_id,
            forecast.predicted_overnight_stays_total,
        ),
    )


def build_query(
    resources: AppResources,
    territory_id: str,
    *,
    as_of_date: AsOfDate | None = None,
) -> ForecastViewModel:
    """Construye producto, decision support y view model para una provincia."""
    product = build_forecast_product_context(
        territory_id,
        as_of_date=as_of_date,
        gold=resources.gold,
        predictions=resources.predictions,
        official_metrics=resources.official_metrics,
        config=resources.config,
    )
    support = build_decision_support(product)
    return build_view_model(product, support)


def make_history_figure(chart: HistoryChartData) -> Figure:
    """Crea un gráfico sobrio sin conectar historia y forecast."""
    figure, axis = plt.subplots(figsize=(10, 4.2), layout="constrained")
    axis.plot(
        chart.history["date_month"],
        chart.history["overnight_stays_total"],
        color="#2563EB",
        linewidth=2,
        marker="o",
        markersize=3.5,
        label="Observaciones oficiales",
    )
    axis.scatter(
        [chart.forecast_date],
        [chart.forecast_value],
        color="#D97706",
        marker="D",
        s=70,
        zorder=3,
        label="Previsión",
    )
    axis.set_xlabel("Fecha")
    axis.set_ylabel("Pernoctaciones provinciales")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False, loc="upper left")
    locator = mdates.AutoDateLocator(minticks=5, maxticks=9)
    axis.xaxis.set_major_locator(locator)
    axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    axis.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _: format_spanish_number(value))
    )
    return figure


def build_download_csv(view: ForecastViewModel) -> bytes:
    """Genera en memoria un resumen auditable de la consulta."""
    record = {
        "territory_id": view.territory_id,
        "territory_name": view.territory_name,
        "as_of_date": view.as_of_date.isoformat(),
        "target_month_id": view.target_month_id,
        "predicted_overnight_stays_total": view.forecast_value,
        "reference_month_id": view.reference_month_id,
        "reference_overnight_stays_total": view.reference_value,
        "reference_is_provisional": view.reference_is_provisional,
        "latest_available_month_id": view.latest_available_month_id,
        "activity_level": view.activity_level,
        "historical_percentile_pct": view.historical_percentile_pct,
        "historical_median": view.historical_median,
        "historical_sample_size": view.historical_sample_size,
        "validation_wape_pct": view.validation_wape_pct,
        "validation_mae": view.validation_mae,
        "validation_rmse": view.validation_rmse,
        "validation_bias": view.validation_bias,
        "validation_rows": view.validation_rows,
        "model_name": view.model_name,
        "decision_support_rule_id": view.rule_id,
        "warning_codes": "|".join(warning.code for warning in view.warnings),
        "operational_source_snapshot_id": (
            view.operational_source_snapshot_id
        ),
        "operational_pipeline_run_id": view.operational_pipeline_run_id,
        "operational_data_version": view.operational_data_version,
        "evaluation_source_snapshot_id": view.evaluation_source_snapshot_id,
        "evaluation_pipeline_run_id": view.evaluation_pipeline_run_id,
        "evaluation_data_version": view.evaluation_data_version,
    }
    output = StringIO()
    pd.DataFrame([record]).to_csv(output, index=False)
    return output.getvalue().encode("utf-8-sig")


def _render_warnings(view: ForecastViewModel) -> None:
    for warning in view.warnings:
        st.warning(warning.message, icon="⚠️")


def _render_primary_result(view: ForecastViewModel) -> None:
    st.subheader(f"Previsión para {format_spanish_month(view.target_month_id)}")
    forecast_column, level_column = st.columns(2)
    forecast_column.metric(
        "Pernoctaciones provinciales previstas",
        format_spanish_number(view.forecast_value),
    )
    level_column.metric(
        "Actividad territorial relativa",
        format_activity_level(view.activity_level),
        help=(
            "Compara esta previsión con el mismo mes de años históricos "
            "comparables de la provincia."
        ),
    )
    wape_column, latest_column = st.columns(2)
    wape_column.metric(
        "Error histórico WAPE",
        format_spanish_percent(view.validation_wape_pct),
        help=(
            "Error porcentual absoluto agregado observado durante las "
            "validaciones históricas. No es una medida de precisión."
        ),
    )
    latest_column.metric(
        "Datos oficiales disponibles hasta",
        format_spanish_month(view.latest_available_month_id),
    )

    reference_column, reference_value_column, query_date_column = st.columns(3)
    reference_column.metric(
        "Mes de referencia (t-12)",
        format_spanish_month(view.reference_month_id),
    )
    reference_value_column.metric(
        "Pernoctaciones de referencia",
        format_spanish_number(view.reference_value),
    )
    query_date_column.metric(
        "Fecha de consulta",
        format_spanish_date(view.as_of_date),
    )


def _render_history(view: ForecastViewModel) -> None:
    st.subheader("Histórico provincial")
    st.caption(
        "Últimos 24 meses naturales. Los meses sin observación permanecen "
        "como gaps; la previsión es un punto separado."
    )
    figure = make_history_figure(view.chart)
    st.pyplot(figure, width="stretch")
    plt.close(figure)


def _render_interpretation(view: ForecastViewModel) -> None:
    st.subheader("Interpretación estadística")
    st.caption(
        "Posición de la previsión frente al mismo mes de años comparables; "
        "el percentil no representa confianza estadística."
    )
    percentile_column, median_column, sample_column = st.columns(3)
    percentile_column.metric(
        "Percentil histórico",
        format_spanish_percent(view.historical_percentile_pct, decimals=0),
    )
    median_column.metric(
        "Mediana del mismo mes",
        (
            format_spanish_number(view.historical_median)
            if view.historical_median is not None
            else "No disponible"
        ),
    )
    sample_column.metric(
        "Observaciones comparables",
        str(view.historical_sample_size),
    )

    st.subheader("Orientación para la planificación")
    if view.action_guidance:
        st.info(view.action_guidance, icon="ℹ️")
    else:
        st.info(
            "No hay suficiente contexto histórico para ofrecer orientación.",
            icon="ℹ️",
        )


def _render_secondary_metrics(view: ForecastViewModel) -> None:
    with st.expander("Rendimiento histórico del modelo"):
        st.caption(
            "Métricas observadas en las validaciones históricas; se muestran "
            "separadas del nivel de actividad."
        )
        mae_column, rmse_column, bias_column, rows_column = st.columns(4)
        mae_column.metric("MAE", format_spanish_number(view.validation_mae, 1))
        rmse_column.metric(
            "RMSE",
            format_spanish_number(view.validation_rmse, 1),
        )
        bias_column.metric(
            "Sesgo medio",
            format_spanish_number(view.validation_bias, 1),
        )
        rows_column.metric("Observaciones de validación", str(view.validation_rows))


def _render_methodology(view: ForecastViewModel) -> None:
    with st.expander("Metodología y trazabilidad"):
        st.markdown(
            f"""
**Modelo:** `{view.model_name}`  
**Regla de previsión:** mismo mes del año anterior  
**Horizonte:** {view.forecast_horizon_months} mes  
**Relación:** {format_spanish_month(view.target_month_id)} ← \
{format_spanish_month(view.reference_month_id)}  
**Estado operacional:** `{view.operational_status}`  
**Regla de interpretación:** `{view.rule_id}`
"""
        )
        st.markdown(
            f"""
**Q25:** {
    format_spanish_number(view.historical_q25)
    if view.historical_q25 is not None else "No disponible"
}  
**Q75:** {
    format_spanish_number(view.historical_q75)
    if view.historical_q75 is not None else "No disponible"
}  
**Meses comparables:** {", ".join(view.comparison_month_ids) or "Ninguno"}  
**COVID excluido:** {
    ", ".join(view.excluded_covid_month_ids) or "Ninguno"
}  
**Provisionales excluidos del benchmark:** {
    ", ".join(view.excluded_provisional_month_ids) or "Ninguno"
}
"""
        )
        st.markdown("**Datos operacionales**")
        st.code(
            "\n".join(
                (
                    f"snapshot: {view.operational_source_snapshot_id}",
                    f"pipeline: {view.operational_pipeline_run_id}",
                    f"version: {view.operational_data_version}",
                )
            )
        )
        st.markdown("**Evaluación congelada**")
        st.code(
            "\n".join(
                (
                    f"snapshot: {view.evaluation_source_snapshot_id}",
                    f"pipeline: {view.evaluation_pipeline_run_id}",
                    f"version: {view.evaluation_data_version}",
                )
            )
        )
        st.markdown("**Limitaciones**")
        for limitation in view.limitations:
            st.markdown(f"- {limitation}")


def _render_download(view: ForecastViewModel) -> None:
    st.download_button(
        "Descargar datos de esta consulta",
        data=build_download_csv(view),
        file_name=(
            f"consulta_{view.territory_id}_{view.target_month_id}.csv"
        ),
        mime="text/csv",
    )


def _render_blocking_error(message: str) -> None:
    st.error(message, icon="🚫")
    st.stop()


def render_app(
    *,
    resources: AppResources | None = None,
    as_of_date: AsOfDate | None = None,
) -> None:
    """Renderiza la aplicación; producción usa la fecha local actual."""
    st.set_page_config(
        page_title="Planificación de demanda turística rural",
        page_icon="🌿",
        layout="wide",
    )
    st.title("Planificación de demanda turística rural")
    st.caption(
        "Señal territorial provincial para apoyar la planificación. No es "
        "una previsión de reservas, ventas o demanda de un establecimiento."
    )

    try:
        effective_resources = resources or load_app_resources()
        options = territory_options(effective_resources.gold)
    except (OSError, ValueError, KeyError, DashboardDataError):
        LOGGER.exception("No se pudieron cargar los artefactos del producto.")
        _render_blocking_error(
            "No se pudieron cargar y validar los datos necesarios. "
            "Inténtalo de nuevo o revisa los artefactos del producto."
        )
        return

    selected = st.selectbox(
        "Provincia",
        options=options,
        format_func=lambda option: option.territory_name,
        key="territory_selector",
    )
    if selected is None:
        _render_blocking_error("Selecciona una provincia para continuar.")
        return

    try:
        with st.spinner("Construyendo la consulta provincial..."):
            view = build_query(
                effective_resources,
                selected.territory_id,
                as_of_date=as_of_date,
            )
    except InferenceError:
        LOGGER.exception("La inferencia provincial no está disponible.")
        _render_blocking_error(
            "No se puede generar una previsión válida para esta provincia "
            "y fecha de consulta."
        )
        return
    except (DashboardDataError, ProductCompositionError):
        LOGGER.exception("El contexto provincial no supera los guardrails.")
        _render_blocking_error(
            "Los datos operacionales y la evaluación no forman un contexto "
            "coherente. No se muestra un resultado parcial."
        )
        return
    except DecisionSupportError:
        LOGGER.exception("No se pudo interpretar el contexto provincial.")
        _render_blocking_error(
            "No se puede construir un contexto histórico válido para esta "
            "consulta."
        )
        return
    except (OSError, ValueError, KeyError):
        LOGGER.exception("Fallo al construir la consulta de producto.")
        _render_blocking_error(
            "La consulta no pudo completarse con garantías metodológicas."
        )
        return

    _render_primary_result(view)
    _render_warnings(view)
    _render_history(view)
    _render_interpretation(view)
    _render_secondary_metrics(view)
    _render_methodology(view)
    _render_download(view)
