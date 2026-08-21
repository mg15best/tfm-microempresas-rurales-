"""Presentacion Streamlit del producto provincial de forecasting."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import logging
from io import StringIO
from typing import Any, Mapping
import unicodedata

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.application.decision_support import (
    DecisionSupport,
    DecisionSupportError,
    build_decision_support,
)
from src.application.forecast_service import (
    ForecastProductContext,
    ForecastServiceResources,
    ProductCompositionError,
    build_forecast_product_context,
    prepare_forecast_service_resources,
)
from src.models.inference import AsOfDate, InferenceError, MissingReferenceError
from src.models.modeling_v2_common import load_modeling_v2_config
from src.visualization.dashboard_data import (
    CanonicalArtifactError,
    DashboardDataError,
    InvalidTerritoryError,
    load_gold_history,
)


__all__ = [
    "AppResources",
    "ForecastViewModel",
    "HistoryChartData",
    "TerritoryOption",
    "build_download_csv",
    "build_history_summary",
    "build_query",
    "build_view_model",
    "format_error_message",
    "format_activity_level",
    "format_percentile_position",
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
    "low": "Por debajo de lo habitual",
    "usual": "Dentro de lo habitual",
    "high": "Por encima de lo habitual",
    "insufficient": "Contexto histórico insuficiente",
}

WARNING_MESSAGES = {
    "provisional_training_data": (
        "La estimación ETS utiliza datos provisionales del INE que pueden "
        "revisarse."
    ),
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
        "La previsión queda fuera del rango observado en el mismo mes "
        "de los años históricos comparables."
    ),
}

ERROR_MESSAGES = {
    "load": (
        "No hemos podido cargar los datos necesarios. Vuelve a intentarlo."
    ),
    "missing_reference": (
        "Falta el dato histórico necesario para calcular la previsión de "
        "esta provincia. Selecciona otra provincia."
    ),
    "inference": (
        "No podemos generar una previsión validada para esta provincia en "
        "este momento. Selecciona otra provincia o vuelve a intentarlo."
    ),
    "invalid_territory": (
        "La provincia seleccionada no está disponible. Selecciona otra."
    ),
    "canonical": (
        "La evidencia de validación no supera los controles de integridad. "
        "Vuelve a intentarlo más tarde."
    ),
    "composition": (
        "Los datos disponibles no permiten mostrar un resultado validado "
        "en este momento. Vuelve a intentarlo."
    ),
    "decision_support": (
        "No hay un histórico comparable válido para interpretar esta "
        "previsión. Selecciona otra provincia."
    ),
    "query": "No hemos podido completar la consulta. Vuelve a intentarlo.",
}


@dataclass(frozen=True)
class AppResources:
    """Artefactos inyectables que una sesión solo debe cargar una vez."""

    config_v2: Mapping[str, Any]
    gold: pd.DataFrame
    forecast_resources: ForecastServiceResources


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
    interval_available: bool = False
    interval_lower: float | None = None
    interval_upper: float | None = None


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
    baseline_reference_month_id: str
    baseline_prediction: float | None
    baseline_reference_is_provisional: bool
    latest_available_month_id: str
    operational_status: str
    selected_model_id: str
    selection_status: str
    actual_model_used: str
    fallback_used: bool
    fallback_reason: str
    ets_raw_prediction: float | None
    clipping_applied: bool
    forecast_horizon_months: int
    interval_available: bool
    interval_lower: float | None
    interval_upper: float | None
    interval_nominal_level: float
    interval_method_id: str
    interval_calibration_scores_n: int
    interval_calibration_origins_n: int
    interval_calibration_max_target_month_id: str | None
    interval_unavailable_reason: str | None
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
    evaluation_source_snapshot_ids: tuple[str, ...]
    evaluation_data_versions: tuple[str, ...]
    evaluation_scope: str
    evaluation_artifact_path: str
    evaluation_logical_prediction_sha256: str
    evaluation_generator_commit_sha: str
    evaluation_github_run_id: str
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
    return f"{SPANISH_MONTHS[period.month - 1]} de {period.year}"


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


def format_percentile_position(value: float | None) -> str:
    """Expresa posición histórica sin sugerir probabilidad o confianza."""
    if value is None:
        return "Posición histórica no disponible"
    percentage = format_spanish_number(value, decimals=0)
    return f"Más alto que el {percentage} % de los meses comparables"


def format_error_message(error_code: str) -> str:
    """Devuelve copy empresarial estable para un estado de error conocido."""
    try:
        return ERROR_MESSAGES[error_code]
    except KeyError as error:
        raise ValueError(
            f"Estado de error no soportado: {error_code!r}."
        ) from error


def translate_warning(
    code: str,
    fallback_message: str = "",
    *,
    target_month_id: str | None = None,
) -> str:
    """Traduce un warning real sin crear señales analíticas nuevas."""
    if code == "forecast_outside_historical_range" and target_month_id:
        period = pd.Period(target_month_id, freq="M")
        comparable_month = f"{SPANISH_MONTHS[period.month - 1]}s"
        return (
            "La previsión queda fuera del rango observado entre los "
            f"{comparable_month} históricos comparables. La comparación "
            "se limita al mismo mes, no a todo el gráfico."
        )
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
    config = load_modeling_v2_config()
    return AppResources(
        config_v2=config,
        gold=load_gold_history(config),
        forecast_resources=prepare_forecast_service_resources(config),
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
    interval_lower: float | None = None,
    interval_upper: float | None = None,
) -> HistoryChartData:
    """Prepara una ventana natural preservando huecos como nulos."""
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

    columns = ["date_month", "overnight_stays_total"]
    if "is_provisional" in history.columns:
        columns.append("is_provisional")
    prepared = history.loc[:, columns].copy()
    if "is_provisional" not in prepared.columns:
        prepared["is_provisional"] = False
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
    interval_available = interval_lower is not None and interval_upper is not None
    if interval_available and not (
        float(interval_lower) <= float(forecast_value) <= float(interval_upper)
    ):
        raise ValueError("El point forecast debe quedar dentro del intervalo.")
    return HistoryChartData(
        history=window,
        forecast_date=pd.Period(target_month_id, freq="M").to_timestamp(),
        forecast_value=float(forecast_value),
        interval_available=interval_available,
        interval_lower=(float(interval_lower) if interval_available else None),
        interval_upper=(float(interval_upper) if interval_available else None),
    )


def build_view_model(
    product: ForecastProductContext,
    support: DecisionSupport,
) -> ForecastViewModel:
    """Compone el contrato de presentación sin recalcular analítica."""
    forecast = product.forecast
    interval = product.prediction_interval
    metrics = product.dashboard.validation_metrics
    lineage = product.dashboard.lineage
    warnings = tuple(
        WarningView(
            code=warning.code,
            message=translate_warning(
                warning.code,
                warning.message,
                target_month_id=forecast.target_month_id,
            ),
        )
        for warning in support.warnings
    )
    return ForecastViewModel(
        territory_id=forecast.territory_id,
        territory_name=forecast.territory_name,
        as_of_date=product.as_of_date,
        target_month_id=forecast.target_month_id,
        forecast_value=float(forecast.predicted_overnight_stays_total),
        baseline_reference_month_id=forecast.baseline_reference_month_id,
        baseline_prediction=(
            float(forecast.baseline_prediction)
            if forecast.baseline_prediction is not None
            else None
        ),
        baseline_reference_is_provisional=(
            forecast.baseline_reference_is_provisional
        ),
        latest_available_month_id=forecast.latest_available_month_id,
        operational_status=forecast.operational_status,
        selected_model_id=forecast.selected_model_id,
        selection_status=forecast.selection_status,
        actual_model_used=forecast.actual_model_used,
        fallback_used=forecast.fallback_used,
        fallback_reason=forecast.fallback_reason,
        ets_raw_prediction=forecast.ets_raw_prediction,
        clipping_applied=forecast.clipping_applied,
        forecast_horizon_months=forecast.forecast_horizon_months,
        interval_available=interval.interval_available,
        interval_lower=interval.lower,
        interval_upper=interval.upper,
        interval_nominal_level=interval.nominal_level,
        interval_method_id=interval.method_id,
        interval_calibration_scores_n=interval.calibration_scores_n,
        interval_calibration_origins_n=interval.calibration_origins_n,
        interval_calibration_max_target_month_id=(
            interval.calibration_max_target_month_id
        ),
        interval_unavailable_reason=interval.unavailable_reason,
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
        evaluation_source_snapshot_ids=lineage.evaluation_source_snapshot_ids,
        evaluation_data_versions=lineage.evaluation_data_versions,
        evaluation_scope=lineage.evaluation_scope,
        evaluation_artifact_path=lineage.evaluation_artifact_path,
        evaluation_logical_prediction_sha256=(
            lineage.evaluation_logical_prediction_sha256
        ),
        evaluation_generator_commit_sha=(
            lineage.evaluation_generator_commit_sha
        ),
        evaluation_github_run_id=lineage.evaluation_github_run_id,
        chart=prepare_history_chart(
            product.dashboard.history,
            forecast.target_month_id,
            forecast.predicted_overnight_stays_total,
            interval_lower=interval.lower,
            interval_upper=interval.upper,
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
        prepared_resources=resources.forecast_resources,
        config=resources.config_v2,
    )
    support = build_decision_support(product)
    return build_view_model(product, support)


def make_history_figure(chart: HistoryChartData) -> go.Figure:
    """Crea trazas separadas sin conectar historia y previsión."""
    tick_values = list(
        pd.date_range(
            start=chart.history["date_month"].min(),
            end=chart.forecast_date,
            freq="4MS",
        )
    )
    if chart.forecast_date not in tick_values:
        tick_values.append(chart.forecast_date)
    tick_text = [
        f"{SPANISH_MONTHS[value.month - 1][:3]} {value.year}"
        for value in tick_values
    ]
    history_customdata = [
        [
            format_spanish_month(str(row.date_month.to_period("M"))),
            (
                format_spanish_number(row.overnight_stays_total)
                if pd.notna(row.overnight_stays_total)
                else "No disponible"
            ),
            (
                "Provisional"
                if pd.notna(row.is_provisional) and bool(row.is_provisional)
                else "No provisional"
            ),
        ]
        for row in chart.history.itertuples(index=False)
    ]
    figure = go.Figure()
    error_y: dict[str, Any] | None = None
    if chart.interval_available:
        if chart.interval_lower is None or chart.interval_upper is None:
            raise ValueError("Intervalo disponible sin límites.")
        error_y = {
            "type": "data",
            "symmetric": False,
            "array": [chart.interval_upper - chart.forecast_value],
            "arrayminus": [chart.forecast_value - chart.interval_lower],
            "visible": True,
            "thickness": 2,
            "width": 8,
        }
    figure.add_trace(
        go.Scatter(
            x=chart.history["date_month"],
            y=chart.history["overnight_stays_total"],
            customdata=history_customdata,
            mode="lines+markers",
            name="Histórico oficial",
            connectgaps=False,
            marker={"symbol": "circle", "size": 7},
            line={"width": 2},
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Pernoctaciones oficiales: %{customdata[1]}<br>"
                "Estado del dato: %{customdata[2]}<extra></extra>"
            ),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[chart.forecast_date],
            y=[chart.forecast_value],
            customdata=[
                [
                    format_spanish_month(str(chart.forecast_date.to_period("M"))),
                    format_spanish_number(chart.forecast_value),
                ]
            ],
            mode="markers",
            name=(
                "Previsión e intervalo empírico al 80 %"
                if chart.interval_available
                else "Previsión"
            ),
            marker={"symbol": "diamond", "size": 13},
            error_y=error_y,
            hovertemplate=(
                "<b>Previsión para %{customdata[0]}</b><br>"
                "Pernoctaciones provinciales previstas: "
                "%{customdata[1]}<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        height=430,
        hovermode="closest",
        legend={"orientation": "h", "y": 1.12, "x": 0},
        margin={"l": 20, "r": 20, "t": 55, "b": 20},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        separators=",.",
        xaxis={
            "title": "Mes",
            "showgrid": False,
            "tickmode": "array",
            "tickvals": tick_values,
            "ticktext": tick_text,
            "automargin": True,
        },
        yaxis={
            "title": "Pernoctaciones provinciales",
            "rangemode": "tozero",
            "tickformat": ",.0f",
            "automargin": True,
        },
    )
    return figure


def build_history_summary(view: ForecastViewModel) -> str:
    """Resume el gráfico en texto con los mismos datos de presentación."""
    observed = view.chart.history.dropna(
        subset=["overnight_stays_total"]
    )
    first_month = str(view.chart.history["date_month"].min().to_period("M"))
    last_month = str(view.chart.history["date_month"].max().to_period("M"))
    latest = observed.iloc[-1]
    latest_month = str(latest["date_month"].to_period("M"))
    latest_value = format_spanish_number(latest["overnight_stays_total"])
    forecast_value = format_spanish_number(view.forecast_value)
    return (
        f"Histórico de {view.territory_name} desde "
        f"{format_spanish_month(first_month)} hasta "
        f"{format_spanish_month(last_month)}. El último dato observado es "
        f"{format_spanish_month(latest_month)}, con {latest_value} "
        f"pernoctaciones. La previsión para "
        f"{format_spanish_month(view.target_month_id)} es de "
        f"{forecast_value} pernoctaciones provinciales. Los meses sin datos "
        "se muestran como interrupciones y no se estiman valores."
    )


def build_download_csv(view: ForecastViewModel) -> bytes:
    """Genera en memoria un resumen auditable de la consulta."""
    record = {
        "territory_id": view.territory_id,
        "territory_name": view.territory_name,
        "as_of_date": view.as_of_date.isoformat(),
        "target_month_id": view.target_month_id,
        "predicted_overnight_stays_total": view.forecast_value,
        "selected_model_id": view.selected_model_id,
        "selection_status": view.selection_status,
        "actual_model_used": view.actual_model_used,
        "fallback_used": view.fallback_used,
        "fallback_reason": view.fallback_reason,
        "baseline_reference_month_id": view.baseline_reference_month_id,
        "baseline_prediction": view.baseline_prediction,
        "baseline_reference_is_provisional": (
            view.baseline_reference_is_provisional
        ),
        "ets_raw_prediction": view.ets_raw_prediction,
        "clipping_applied": view.clipping_applied,
        "interval_available": view.interval_available,
        "interval_lower": view.interval_lower,
        "interval_upper": view.interval_upper,
        "interval_nominal_level": view.interval_nominal_level,
        "interval_method_id": view.interval_method_id,
        "interval_calibration_scores_n": (
            view.interval_calibration_scores_n
        ),
        "interval_calibration_origins_n": (
            view.interval_calibration_origins_n
        ),
        "interval_calibration_max_target_month_id": (
            view.interval_calibration_max_target_month_id
        ),
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
        "decision_support_rule_id": view.rule_id,
        "warning_codes": "|".join(warning.code for warning in view.warnings),
        "operational_source_snapshot_id": (
            view.operational_source_snapshot_id
        ),
        "operational_pipeline_run_id": view.operational_pipeline_run_id,
        "operational_data_version": view.operational_data_version,
        "evaluation_scope": view.evaluation_scope,
        "evaluation_source_snapshot_ids": "|".join(
            view.evaluation_source_snapshot_ids
        ),
        "evaluation_data_versions": "|".join(
            view.evaluation_data_versions
        ),
        "evaluation_artifact_path": view.evaluation_artifact_path,
        "evaluation_logical_prediction_sha256": (
            view.evaluation_logical_prediction_sha256
        ),
        "evaluation_generator_commit_sha": (
            view.evaluation_generator_commit_sha
        ),
        "evaluation_github_run_id": view.evaluation_github_run_id,
    }
    output = StringIO()
    pd.DataFrame([record]).to_csv(output, index=False)
    return output.getvalue().encode("utf-8-sig")


def _render_temporal_context(view: ForecastViewModel) -> None:
    available_column, query_column = st.columns(2)
    available_column.markdown(
        "**Datos oficiales disponibles hasta**  \n"
        f"{format_spanish_month(view.latest_available_month_id)}"
    )
    query_column.markdown(
        "**Fecha de consulta**  \n"
        f"{format_spanish_date(view.as_of_date)}"
    )


def _render_analytical_warnings(view: ForecastViewModel) -> None:
    for warning in view.warnings:
        if warning.code not in {
            "provisional_reference_data",
            "availability_fallback_used",
        }:
            st.warning(warning.message, icon="⚠️")


def _render_primary_result(view: ForecastViewModel) -> None:
    st.subheader(
        "Previsión provincial para "
        f"{format_spanish_month(view.target_month_id)}"
    )
    forecast_column, level_column = st.columns([2, 1])
    forecast_column.metric(
        "Pernoctaciones previstas en turismo rural",
        format_spanish_number(view.forecast_value),
    )
    level_column.markdown("**Posición frente al histórico comparable**")
    level_column.markdown(
        f"**{format_activity_level(view.activity_level)}**"
    )
    level_column.write(
        "Comparación con el mismo mes de años históricos comparables."
    )

    with st.container(border=True):
        st.markdown("**Intervalo predictivo empírico al 80 %**")
        if view.interval_available:
            st.write(
                f"De **{format_spanish_number(view.interval_lower)}** a "
                f"**{format_spanish_number(view.interval_upper)}** "
                "pernoctaciones, alrededor de la previsión puntual."
            )
        else:
            st.write(
                "Intervalo no disponible con la información de calibración "
                "actual. La previsión puntual sigue siendo válida."
            )
        st.caption(
            "Representa un rango empírico de incertidumbre construido con "
            "errores de validación temporal. No garantiza que el valor real "
            "quede dentro del rango."
        )

    if view.fallback_used:
        st.warning(
            "Para esta provincia no fue posible generar la estimación ETS "
            "con el histórico disponible. Se utiliza como respaldo el valor "
            "del mismo mes del año anterior.",
            icon="⚠️",
        )


def _render_guidance(view: ForecastViewModel) -> None:
    st.subheader("Orientación para la planificación")
    st.write(
        "Orientación basada en una regla histórica para contrastar con "
        "reservas propias, capacidad, eventos y contexto local; no es una orden "
        "automatizada."
    )
    if view.action_guidance:
        st.info(view.action_guidance, icon="ℹ️")
    else:
        st.info(
            "No hay suficiente contexto histórico para ofrecer orientación.",
            icon="ℹ️",
        )


def _render_history(view: ForecastViewModel) -> None:
    st.subheader("Histórico provincial")
    st.write(build_history_summary(view))
    figure = make_history_figure(view.chart)
    st.plotly_chart(
        figure,
        width="stretch",
        theme="streamlit",
        config={
            "displaylogo": False,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        },
    )


def _render_interpretation(view: ForecastViewModel) -> None:
    st.subheader("Posición frente al histórico")
    st.caption(
        "Esta posición histórica es distinta del intervalo predictivo."
    )
    st.write(
        format_percentile_position(view.historical_percentile_pct) + "."
    )
    median_column, sample_column = st.columns(2)
    median_column.metric(
        "Mediana de los meses comparables",
        (
            format_spanish_number(view.historical_median)
            if view.historical_median is not None
            else "No disponible"
        ),
    )
    sample_column.metric(
        "Años comparables",
        str(view.historical_sample_size),
    )


def _render_secondary_metrics(view: ForecastViewModel) -> None:
    with st.expander("Resultados en validación temporal canónica"):
        st.markdown(
            "**Error WAPE en validación: "
            f"{format_spanish_percent(view.validation_wape_pct)}**"
        )
        st.write(
            "Métricas provinciales del sistema operacional seleccionado, "
            "calculadas sobre `operational_prediction`."
        )
        mae_column, rmse_column = st.columns(2)
        mae_column.metric(
            "Error absoluto medio (MAE)",
            format_spanish_number(view.validation_mae, 1),
        )
        rmse_column.metric(
            "Raíz del error cuadrático medio (RMSE)",
            format_spanish_number(view.validation_rmse, 1),
        )
        bias_column, rows_column = st.columns(2)
        bias_column.metric(
            "Sesgo medio",
            format_spanish_number(view.validation_bias, 1),
            help=(
                "Un valor negativo indica tendencia histórica a "
                "infrapredecir; uno positivo, a sobrepredecir."
            ),
        )
        bias_column.write(
            "Negativo: tendencia histórica a infrapredecir. Positivo: "
            "tendencia histórica a sobrepredecir."
        )
        rows_column.metric(
            "Observaciones de validación",
            str(view.validation_rows),
        )


def _render_methodology(view: ForecastViewModel) -> None:
    with st.expander("Metodología y trazabilidad"):
        st.info(
            "El modelo ha sido seleccionado provisionalmente mediante "
            "validación temporal point-in-time y todavía no dispone de "
            "confirmación en una nueva ventana final intacta.",
            icon="ℹ️",
        )
        st.markdown(
            "\n".join(
                (
                    f"- **Modelo seleccionado:** `{view.selected_model_id}`",
                    f"- **Estado de selección:** `{view.selection_status}`",
                    "- **Modelo usado en esta consulta:** "
                    f"`{view.actual_model_used}`",
                    f"- **Fallback por disponibilidad:** `{view.fallback_used}`",
                    f"- **Motivo de fallback:** `{view.fallback_reason}`",
                    f"- **Horizonte:** {view.forecast_horizon_months} mes",
                    "- **Referencia estacional de escala/respaldo:** "
                    + format_spanish_month(view.baseline_reference_month_id),
                    f"- **Estado operacional:** `{view.operational_status}`",
                    f"- **Regla de interpretación:** `{view.rule_id}`",
                )
            )
        )
        if view.baseline_prediction is not None:
            st.write(
                "Valor de la referencia estacional: "
                f"{format_spanish_number(view.baseline_prediction)}."
            )
        if view.baseline_reference_is_provisional:
            st.caption(
                "La referencia estacional es provisional y puede ser "
                "revisada por el INE."
            )
        st.markdown(
            "\n".join(
                (
                    f"- **Intervalo:** `{view.interval_method_id}`",
                    "- **Nivel nominal:** "
                    + format_spanish_percent(
                        view.interval_nominal_level * 100
                    ),
                    "- **Scores de calibración:** "
                    f"{view.interval_calibration_scores_n}",
                    "- **Orígenes de calibración:** "
                    f"{view.interval_calibration_origins_n}",
                    "- **Máximo target de calibración:** "
                    + (
                        view.interval_calibration_max_target_month_id
                        or "No disponible"
                    ),
                )
            )
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
**Provisionales excluidos de la comparación:** {
    ", ".join(view.excluded_provisional_month_ids) or "Ninguno"
}
"""
        )
        st.markdown("**Linaje operacional**")
        st.code(
            "\n".join(
                (
                    f"snapshot: {view.operational_source_snapshot_id}",
                    f"pipeline: {view.operational_pipeline_run_id}",
                    f"version: {view.operational_data_version}",
                )
            )
        )
        st.markdown("**Linaje de evaluación canónica**")
        st.code(
            "\n".join(
                (
                    f"scope: {view.evaluation_scope}",
                    f"artifact: {view.evaluation_artifact_path}",
                    "generator commit: "
                    f"{view.evaluation_generator_commit_sha[:12]}",
                    f"GitHub run: {view.evaluation_github_run_id}",
                    "data versions: "
                    f"{', '.join(view.evaluation_data_versions)}",
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
        "Previsión provincial de pernoctaciones en turismo rural para apoyar "
        "la planificación del próximo mes. Úsala como señal de contexto "
        "junto con tus reservas, capacidad, eventos locales y conocimiento "
        "del negocio; no estima la demanda de un establecimiento concreto."
    )

    try:
        effective_resources = resources or load_app_resources()
        options = territory_options(effective_resources.gold)
    except CanonicalArtifactError:
        LOGGER.exception("La evidencia canónica no supera integridad.")
        _render_blocking_error(format_error_message("canonical"))
        return
    except (OSError, ValueError, KeyError, DashboardDataError):
        LOGGER.exception("No se pudieron cargar los artefactos del producto.")
        _render_blocking_error(format_error_message("load"))
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
    except MissingReferenceError:
        LOGGER.exception("Falta la referencia histórica de la provincia.")
        _render_blocking_error(format_error_message("missing_reference"))
        return
    except InvalidTerritoryError:
        LOGGER.exception("La provincia no existe en los recursos B5.")
        _render_blocking_error(format_error_message("invalid_territory"))
        return
    except CanonicalArtifactError:
        LOGGER.exception("La evidencia canónica no supera integridad.")
        _render_blocking_error(format_error_message("canonical"))
        return
    except InferenceError:
        LOGGER.exception("La inferencia provincial no está disponible.")
        _render_blocking_error(format_error_message("inference"))
        return
    except (DashboardDataError, ProductCompositionError):
        LOGGER.exception("El contexto provincial no supera los guardrails.")
        _render_blocking_error(format_error_message("composition"))
        return
    except DecisionSupportError:
        LOGGER.exception("No se pudo interpretar el contexto provincial.")
        _render_blocking_error(format_error_message("decision_support"))
        return
    except (OSError, ValueError, KeyError):
        LOGGER.exception("Fallo al construir la consulta de producto.")
        _render_blocking_error(format_error_message("query"))
        return

    _render_temporal_context(view)
    _render_primary_result(view)
    _render_analytical_warnings(view)
    _render_guidance(view)
    _render_history(view)
    _render_interpretation(view)
    _render_secondary_metrics(view)
    _render_methodology(view)
    _render_download(view)
