"""Composicion B5 del forecast, intervalo y evidencia de producto."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.models.inference import (
    AsOfDate,
    FALLBACK_MODEL_ID,
    InferenceError,
    InferenceResult,
    SELECTED_MODEL_ID,
    SELECTION_STATUS,
    SUPPORTED_FORECAST_HORIZON_MONTHS,
    predict_next_month,
)
from src.models.modeling_v2_common import (
    MODELING_V2_CONFIG_PATH,
    cutoff_policy_from_config,
    load_modeling_v2_config,
    resolve_information_cutoff,
)
from src.models.prediction_intervals_v2 import (
    PredictionIntervalResult,
    build_operational_score_bank,
    calculate_current_operational_interval,
)
from src.visualization.dashboard_data import (
    EVIDENCE_SCOPE,
    PREDICTION_COLUMN,
    CanonicalValidationBundle,
    DashboardContext,
    PreparedCanonicalValidation,
    build_dashboard_context,
    load_gold_history,
    load_prepared_canonical_validation,
    prepare_canonical_validation,
    validate_b5_lifecycle,
)


__all__ = [
    "ForecastProductContext",
    "ForecastServiceResources",
    "ProductCompositionError",
    "build_forecast_product_context",
    "prepare_forecast_service_resources",
]


class ProductCompositionError(RuntimeError):
    """Forecast, intervalo y dashboard no forman un producto coherente."""


@dataclass(frozen=True)
class ForecastProductContext:
    """Resultado B5 compuesto de una consulta provincial."""

    as_of_date: date
    forecast_origin_month_id: str
    forecast: InferenceResult
    prediction_interval: PredictionIntervalResult
    dashboard: DashboardContext


@dataclass(frozen=True, init=False)
class ForecastServiceResources:
    """Evidencia B5 validada y score bank precomputado para consultas."""

    canonical_validation: PreparedCanonicalValidation
    _operational_score_bank: pd.DataFrame

    def __init__(self) -> None:
        raise TypeError(
            "Use prepare_forecast_service_resources() para crear recursos."
        )

    @classmethod
    def _from_prepared(
        cls,
        canonical_validation: PreparedCanonicalValidation,
        operational_score_bank: pd.DataFrame,
    ) -> ForecastServiceResources:
        instance = object.__new__(cls)
        object.__setattr__(
            instance,
            "canonical_validation",
            canonical_validation,
        )
        object.__setattr__(
            instance,
            "_operational_score_bank",
            operational_score_bank.copy(deep=True),
        )
        return instance

    @property
    def canonical_bundle(self) -> CanonicalValidationBundle:
        """Bundle oficial asociado al recurso validado."""

        return self.canonical_validation.bundle

    @property
    def operational_score_bank(self) -> pd.DataFrame:
        """Entrega una copia aislada para impedir contaminación entre sesiones."""

        return self._operational_score_bank.copy(deep=True)


def prepare_forecast_service_resources(
    config: Mapping[str, Any],
    *,
    canonical_bundle: CanonicalValidationBundle | None = None,
) -> ForecastServiceResources:
    """Prepara una vez evidencia canónica y calibración operacional."""

    validate_b5_lifecycle(config)
    canonical_validation = (
        load_prepared_canonical_validation(config)
        if canonical_bundle is None
        else prepare_canonical_validation(canonical_bundle, config)
    )
    score_bank = build_operational_score_bank(
        canonical_validation.bundle.predictions
    )
    return ForecastServiceResources._from_prepared(
        canonical_validation,
        score_bank,
    )


def _effective_as_of_date(as_of_date: AsOfDate | None) -> date:
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


def _required_mapping(
    mapping: Mapping[str, Any],
    key: str,
) -> Mapping[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, Mapping):
        raise ProductCompositionError(
            f"Falta la seccion de configuracion '{key}'."
        )
    return value


def _validate_composition(
    forecast: InferenceResult,
    prediction_interval: PredictionIntervalResult,
    dashboard: DashboardContext,
    config: Mapping[str, Any],
    bundle: CanonicalValidationBundle,
) -> None:
    """Valida territorio, lineage, seleccion, fallback e intervalo."""

    if (
        forecast.territory_id != dashboard.territory_id
        or forecast.territory_name != dashboard.territory_name
    ):
        raise ProductCompositionError(
            "Forecast y dashboard no representan el mismo territorio."
        )

    lineage = dashboard.lineage
    operational_forecast = (
        forecast.source_snapshot_id,
        forecast.pipeline_run_id,
        forecast.data_version,
    )
    operational_dashboard = (
        lineage.operational_source_snapshot_id,
        lineage.operational_pipeline_run_id,
        lineage.operational_data_version,
    )
    if operational_forecast != operational_dashboard:
        raise ProductCompositionError(
            "Forecast y dashboard no comparten provenance operacional."
        )

    try:
        history_months = pd.PeriodIndex(
            dashboard.history["month_id"].astype(str),
            freq="M",
        )
        latest_available = pd.Period(
            forecast.latest_available_month_id,
            freq="M",
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ProductCompositionError(
            "El historico dashboard no tiene un contrato temporal valido."
        ) from error
    if history_months.empty or history_months.max() > latest_available:
        raise ProductCompositionError(
            "El historico dashboard contiene meses posteriores al cutoff."
        )

    selection = _required_mapping(config, "operational_selection")
    fallback = _required_mapping(selection, "fallback")
    if forecast.selected_model_id != selection.get("selected_model_id"):
        raise ProductCompositionError(
            "El modelo seleccionado no coincide con config V2."
        )
    if forecast.selection_status != selection.get("status"):
        raise ProductCompositionError(
            "El status de seleccion no coincide con config V2."
        )
    if forecast.forecast_horizon_months != SUPPORTED_FORECAST_HORIZON_MONTHS:
        raise ProductCompositionError("El producto B5 requiere horizonte uno.")
    permitted_models = {
        str(selection.get("selected_model_id")),
        str(fallback.get("model_id")),
    }
    if forecast.actual_model_used not in permitted_models:
        raise ProductCompositionError("actual_model_used no esta permitido.")
    expected_fallback = forecast.actual_model_used == FALLBACK_MODEL_ID
    if forecast.fallback_used != expected_fallback:
        raise ProductCompositionError("Semantica fallback/modelo incoherente.")
    if expected_fallback and forecast.fallback_reason == "not_used":
        raise ProductCompositionError("Fallback activo sin razon de disponibilidad.")
    if not expected_fallback and forecast.fallback_reason != "not_used":
        raise ProductCompositionError("ETS activo con fallback_reason espurio.")
    if fallback.get("performance_based") is not False:
        raise ProductCompositionError("B5 prohibe performance routing.")

    if prediction_interval.territory_id != forecast.territory_id:
        raise ProductCompositionError("Intervalo y forecast difieren en territorio.")
    if prediction_interval.target_month_id != forecast.target_month_id:
        raise ProductCompositionError("Intervalo y forecast difieren en target.")
    if not np.isclose(
        prediction_interval.point_prediction,
        forecast.predicted_overnight_stays_total,
        rtol=0,
        atol=1e-9,
    ):
        raise ProductCompositionError("El intervalo no conserva el point forecast.")
    if prediction_interval.interval_available:
        if prediction_interval.lower is None or prediction_interval.upper is None:
            raise ProductCompositionError("Intervalo disponible sin limites.")
        if not (
            prediction_interval.lower
            <= prediction_interval.point_prediction
            <= prediction_interval.upper
        ):
            raise ProductCompositionError("El point queda fuera del intervalo.")
    elif (
        prediction_interval.lower is not None
        or prediction_interval.upper is not None
    ):
        raise ProductCompositionError("Intervalo no disponible con limites.")

    metrics = dashboard.validation_metrics
    if metrics.prediction_column != PREDICTION_COLUMN:
        raise ProductCompositionError(
            "Las metricas no usan operational_prediction."
        )
    if (
        metrics.selected_model_id != SELECTED_MODEL_ID
        or metrics.selection_status != SELECTION_STATUS
        or metrics.evidence_scope != EVIDENCE_SCOPE
    ):
        raise ProductCompositionError("Evidencia dashboard incompatible con B5.")
    if lineage.evaluation_scope != EVIDENCE_SCOPE:
        raise ProductCompositionError("El scope de evaluacion no es canonico.")
    evaluation_bundle = (
        lineage.evaluation_artifact_sha256,
        lineage.evaluation_metadata_sha256,
        lineage.evaluation_logical_prediction_sha256,
        lineage.evaluation_generator_commit_sha,
        lineage.evaluation_github_run_id,
    )
    expected_bundle = (
        bundle.artifact_sha256,
        bundle.metadata_sha256,
        bundle.logical_prediction_sha256,
        bundle.generator_commit_sha,
        bundle.github_run_id,
    )
    if evaluation_bundle != expected_bundle:
        raise ProductCompositionError(
            "Dashboard y bundle no comparten provenance de evaluacion."
        )


def build_forecast_product_context(
    territory_id: str,
    *,
    as_of_date: AsOfDate | None = None,
    history_months: int | None = None,
    gold: pd.DataFrame | None = None,
    canonical_bundle: CanonicalValidationBundle | None = None,
    prepared_resources: ForecastServiceResources | None = None,
    config: Mapping[str, Any] | None = None,
    config_path: Path = MODELING_V2_CONFIG_PATH,
) -> ForecastProductContext:
    """Compone una consulta B5 con una sola Gold y evidencia validada."""

    effective_date = _effective_as_of_date(as_of_date)
    effective_config = (
        load_modeling_v2_config(config_path) if config is None else config
    )
    validate_b5_lifecycle(effective_config)
    shared_gold = (
        load_gold_history(effective_config) if gold is None else gold
    )
    forecast = predict_next_month(
        territory_id,
        as_of_date=effective_date,
        dataframe=shared_gold,
        config=effective_config,
    )
    policy = cutoff_policy_from_config(effective_config)
    origin = resolve_information_cutoff(forecast.target_month_id, policy)
    if (
        origin.business_origin_month_id != forecast.business_origin_month_id
        or origin.latest_available_month_id
        != forecast.latest_available_month_id
    ):
        raise ProductCompositionError(
            "Inference y servicio no comparten origin/cutoff V2."
        )
    if canonical_bundle is not None and prepared_resources is not None:
        raise ProductCompositionError(
            "No se puede inyectar bundle y recursos preparados a la vez."
        )
    resources = prepared_resources
    if resources is None:
        resources = prepare_forecast_service_resources(
            effective_config,
            canonical_bundle=canonical_bundle,
        )
    elif not isinstance(resources, ForecastServiceResources):
        raise ProductCompositionError(
            "Se requieren recursos B5 preparados y validados."
        )
    bundle = resources.canonical_bundle
    dashboard = build_dashboard_context(
        territory_id,
        history_months=history_months,
        latest_available_month_id=forecast.latest_available_month_id,
        gold=shared_gold,
        prepared_validation=resources.canonical_validation,
        config=effective_config,
    )
    interval_config = _required_mapping(
        effective_config,
        "operational_prediction_interval",
    )
    prediction_interval = calculate_current_operational_interval(
        territory_id=forecast.territory_id,
        target_month_id=forecast.target_month_id,
        point_prediction=forecast.predicted_overnight_stays_total,
        baseline_prediction=forecast.baseline_prediction,
        origin=origin,
        score_bank=resources.operational_score_bank,
        interval_config=interval_config,
    )
    _validate_composition(
        forecast,
        prediction_interval,
        dashboard,
        effective_config,
        bundle,
    )
    return ForecastProductContext(
        as_of_date=effective_date,
        forecast_origin_month_id=effective_date.strftime("%Y-%m"),
        forecast=forecast,
        prediction_interval=prediction_interval,
        dashboard=dashboard,
    )
