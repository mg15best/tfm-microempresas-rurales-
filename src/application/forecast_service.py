"""Composicion del forecast operacional y su contexto de producto."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from src.models.inference import (
    AsOfDate,
    InferenceError,
    InferenceResult,
    MODEL_NAME,
    SUPPORTED_FORECAST_HORIZON_MONTHS,
    predict_next_month,
)
from src.models.modeling_common import CONFIG_PATH, load_config
from src.visualization.dashboard_data import (
    DashboardContext,
    build_dashboard_context,
    load_gold_history,
)


__all__ = [
    "ForecastProductContext",
    "ProductCompositionError",
    "build_forecast_product_context",
]


class ProductCompositionError(RuntimeError):
    """La inferencia y el contexto no forman un producto coherente."""


@dataclass(frozen=True)
class ForecastProductContext:
    """Resultado compuesto de una consulta provincial del producto."""

    as_of_date: date
    forecast_origin_month_id: str
    forecast: InferenceResult
    dashboard: DashboardContext


def _effective_as_of_date(as_of_date: AsOfDate | None) -> date:
    """Fija una fecha efectiva unica para toda la consulta compuesta."""
    if as_of_date is None:
        return date.today()
    try:
        timestamp = pd.Timestamp(as_of_date)
    except (TypeError, ValueError) as error:
        raise InferenceError(
            f"as_of_date no es una fecha valida: {as_of_date!r}."
        ) from error
    if pd.isna(timestamp):
        raise InferenceError("as_of_date no puede ser nulo.")
    return timestamp.date()


def _validate_composition(
    forecast: InferenceResult,
    dashboard: DashboardContext,
) -> None:
    """Impide combinar territorios, modelos o vintages incompatibles."""
    if (
        forecast.territory_id != dashboard.territory_id
        or forecast.territory_name != dashboard.territory_name
    ):
        raise ProductCompositionError(
            "Forecast y dashboard no representan el mismo territorio."
        )

    lineage = dashboard.lineage
    operational_lineage = (
        forecast.source_snapshot_id,
        forecast.pipeline_run_id,
        forecast.data_version,
    )
    dashboard_lineage = (
        lineage.operational_source_snapshot_id,
        lineage.operational_pipeline_run_id,
        lineage.operational_data_version,
    )
    if operational_lineage != dashboard_lineage:
        raise ProductCompositionError(
            "Forecast y dashboard no comparten provenance operacional."
        )

    if (
        forecast.model_name != MODEL_NAME
        or forecast.forecast_horizon_months
        != SUPPORTED_FORECAST_HORIZON_MONTHS
    ):
        raise ProductCompositionError(
            "El producto requiere seasonal_naive_lag_12 a horizonte uno."
        )


def build_forecast_product_context(
    territory_id: str,
    *,
    as_of_date: AsOfDate | None = None,
    history_months: int | None = None,
    gold: pd.DataFrame | None = None,
    predictions: pd.DataFrame | None = None,
    official_metrics: pd.DataFrame | None = None,
    config: Mapping[str, Any] | None = None,
    config_path: Path = CONFIG_PATH,
) -> ForecastProductContext:
    """Construye una consulta de producto reutilizando una sola carga Gold."""
    effective_as_of_date = _effective_as_of_date(as_of_date)
    effective_config = load_config(config_path) if config is None else config
    shared_gold = (
        load_gold_history(effective_config)
        if gold is None
        else gold
    )

    forecast = predict_next_month(
        territory_id,
        as_of_date=effective_as_of_date,
        dataframe=shared_gold,
        config=effective_config,
    )
    dashboard = build_dashboard_context(
        territory_id,
        history_months=history_months,
        gold=shared_gold,
        predictions=predictions,
        official_metrics=official_metrics,
        config=effective_config,
    )
    _validate_composition(forecast, dashboard)

    return ForecastProductContext(
        as_of_date=effective_as_of_date,
        forecast_origin_month_id=effective_as_of_date.strftime("%Y-%m"),
        forecast=forecast,
        dashboard=dashboard,
    )
