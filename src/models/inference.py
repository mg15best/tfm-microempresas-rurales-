"""Inferencia operacional B5 gobernada por el contrato point-in-time V2.

La seleccion es el ETS congelado en B4. El baseline estacional solo se usa
como fallback por indisponibilidad tecnica del ETS y como escala del intervalo
operacional que se incorpora en una capa separada.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from math import isfinite
from pathlib import Path
from typing import Any, Literal, Mapping

import pandas as pd

from src.models.ets_v2 import ETSForecastResult, fit_ets_forecast
from src.models.modeling_v2_common import (
    MODELING_V2_CONFIG_PATH,
    PROJECT_ROOT,
    cutoff_policy_from_config,
    filter_history_to_information_cutoff,
    load_modeling_v2_config,
    resolve_information_cutoff,
)


SELECTED_MODEL_ID = "holt_winters_additive_damped_v1"
SELECTION_STATUS = "provisional_validation_champion"
FALLBACK_MODEL_ID = "seasonal_naive_lag_12"
# Alias historico conservado para imports de solo lectura durante B5-B1.
SUPPORTED_FORECAST_HORIZON_MONTHS = 1
EFFECTIVE_MODEL_HORIZON_STEPS = 3
AVAILABILITY_FALLBACK_REASONS = frozenset(
    {
        "insufficient_history",
        "training_gap_unsupported",
        "fit_failure",
        "invalid_forecast",
    }
)

REQUIRED_SOURCE_COLUMNS = {
    "territory_id",
    "territory_name",
    "territory_level",
    "month_id",
    "date_month",
    "overnight_stays_total",
    "complete_month_available",
    "is_provisional",
    "source_snapshot_id",
    "pipeline_run_id",
    "data_version",
}

OperationalStatus = Literal["forecast_ready"]
AsOfDate = str | date | datetime | pd.Timestamp


class InferenceError(RuntimeError):
    """Error bloqueante de la capa de inferencia."""


class InferenceConfigurationError(InferenceError):
    """La configuracion no permite una inferencia B5 reproducible."""


class InferenceDataError(InferenceError):
    """El dataset no cumple el contrato necesario para inferencia."""


class EmptyInferenceDatasetError(InferenceDataError):
    """El dataset de inferencia no contiene observaciones."""


class InvalidTerritoryError(InferenceError):
    """El territorio solicitado no existe en el dataset."""


class MissingReferenceError(InferenceError):
    """No existe un fallback lag-12 valido y la inferencia falla cerrada."""


class GlobalReferenceGapError(MissingReferenceError):
    """El mes de referencia falta para todos los territorios."""


class TerritorialReferenceGapError(MissingReferenceError):
    """El mes de referencia falta para el territorio solicitado."""


class UnsupportedHorizonError(InferenceError):
    """El horizonte solicitado no esta permitido por esta solucion."""


@dataclass(frozen=True)
class InferenceWarning:
    """Warning no bloqueante y apto para consumo por un frontal."""

    code: str
    message: str


@dataclass(frozen=True)
class InferenceResult:
    """Resultado tipado y trazable de una prediccion mensual provincial."""

    territory_id: str
    territory_name: str
    target_month_id: str
    business_origin_month_id: str
    latest_available_month_id: str
    forecast_horizon_months: int
    effective_model_horizon_steps: int
    predicted_overnight_stays_total: float
    selected_model_id: str
    selection_status: str
    actual_model_used: str
    fallback_used: bool
    fallback_reason: str
    baseline_reference_month_id: str
    baseline_prediction: float | None
    baseline_reference_is_provisional: bool
    ets_raw_prediction: float | None
    clipping_applied: bool
    training_start: str | None
    training_end: str | None
    training_rows: int
    source_snapshot_id: str
    pipeline_run_id: str
    data_version: str
    operational_status: OperationalStatus
    warnings: tuple[InferenceWarning, ...]

    @property
    def is_operational(self) -> bool:
        """Indica si existe un point forecast valido para el proximo mes."""
        return self.operational_status == "forecast_ready"

def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _required_mapping(
    config: Mapping[str, Any],
    section: str,
) -> Mapping[str, Any]:
    value = config.get(section)
    if not isinstance(value, Mapping):
        raise InferenceConfigurationError(
            f"Falta la seccion de configuracion '{section}'."
        )
    return value


def _validate_inference_config(config: Mapping[str, Any]) -> None:
    """Valida que B5 tenga una unica seleccion y ningun router de performance."""

    source = _required_mapping(config, "source")
    _required_mapping(config, "cutoff_policy")
    ets = _required_mapping(config, "ets_candidate")
    selection = _required_mapping(config, "operational_selection")
    fallback = _required_mapping(selection, "fallback")

    expected_selection = {
        "selected_model_id": SELECTED_MODEL_ID,
        "status": SELECTION_STATUS,
        "evidence_scope": "canonical_rolling_validation",
        "independent_test_confirmed": False,
    }
    mismatches = {
        key: (selection.get(key), expected)
        for key, expected in expected_selection.items()
        if selection.get(key) != expected
    }
    if mismatches:
        raise InferenceConfigurationError(
            f"Seleccion operacional B5 no soportada: {mismatches}."
        )
    if ets.get("id") != SELECTED_MODEL_ID:
        raise InferenceConfigurationError(
            "ets_candidate no coincide con el modelo operacional seleccionado."
        )
    if fallback.get("model_id") != FALLBACK_MODEL_ID:
        raise InferenceConfigurationError(
            "El fallback B5 debe ser seasonal_naive_lag_12."
        )
    if fallback.get("policy") != "availability_only":
        raise InferenceConfigurationError(
            "El fallback B5 solo puede activarse por disponibilidad."
        )
    if fallback.get("performance_based") is not False:
        raise InferenceConfigurationError(
            "B5 prohibe el routing o fallback basado en performance."
        )
    if source.get("target_column") != "overnight_stays_total":
        raise InferenceConfigurationError(
            "source.target_column debe ser overnight_stays_total."
        )
    if not isinstance(source.get("path"), str) or not source["path"].strip():
        raise InferenceConfigurationError("Falta una ruta valida en source.path.")

    try:
        policy = cutoff_policy_from_config(config)
    except (KeyError, TypeError, ValueError) as error:
        raise InferenceConfigurationError(
            f"cutoff_policy V2 no es valida: {error}"
        ) from error
    if (
        policy.business_origin_lag_months != 1
        or policy.latest_available_lag_months != 3
        or policy.max_training_target_lag_months != 3
    ):
        raise InferenceConfigurationError(
            "B5 requiere business origin target-1 y cutoff target-3."
        )
    try:
        business_horizon = int(ets.get("business_horizon_months", -1))
        effective_horizon = int(ets.get("effective_horizon_steps", -1))
    except (TypeError, ValueError) as error:
        raise InferenceConfigurationError(
            "Los horizontes ETS deben ser enteros."
        ) from error
    if business_horizon != 1:
        raise InferenceConfigurationError(
            "El horizonte de negocio ETS configurado no es un mes."
        )
    if effective_horizon != 3:
        raise InferenceConfigurationError(
            "El horizonte efectivo ETS configurado no es tres pasos."
        )


def _load_inference_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise InferenceConfigurationError(
            "No se encontro modeling_v2_config.yml: "
            f"{_display_path(config_path)}."
        )
    try:
        return load_modeling_v2_config(config_path)
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise InferenceConfigurationError(
            f"No se pudo cargar modeling_v2_config.yml: {error}"
        ) from error


def _resolve_project_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_inference_dataset(config: Mapping[str, Any]) -> pd.DataFrame:
    """Carga el Gold declarado exclusivamente por modeling_v2_config.yml."""

    _validate_inference_config(config)
    source = _required_mapping(config, "source")
    dataset_path = _resolve_project_path(str(source["path"]))
    if not dataset_path.exists():
        raise InferenceDataError(
            f"No se encontro el dataset de inferencia: {_display_path(dataset_path)}."
        )
    try:
        return pd.read_parquet(dataset_path)
    except (OSError, ValueError) as error:
        raise InferenceDataError(
            f"No se pudo leer el dataset de inferencia: {error}"
        ) from error


def _valid_boolean_series(series: pd.Series, *, column: str) -> pd.Series:
    if series.isna().any():
        raise InferenceDataError(f"{column} contiene valores nulos.")
    valid = series.map(
        lambda value: isinstance(value, (bool, int)) and value in (0, 1)
    )
    if not bool(valid.all()):
        raise InferenceDataError(
            f"{column} debe contener solo booleanos no nulos."
        )
    return series.astype(bool)


def _prepare_dataset(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Valida schema, claves, provincia y lineage sin completar observaciones."""

    if not isinstance(dataframe, pd.DataFrame):
        raise InferenceDataError("dataframe debe ser un pandas.DataFrame.")
    if dataframe.empty:
        raise EmptyInferenceDatasetError("El dataset de inferencia esta vacio.")

    missing = sorted(REQUIRED_SOURCE_COLUMNS.difference(dataframe.columns))
    if missing:
        raise InferenceDataError(
            "Faltan columnas requeridas para inferencia: " + ", ".join(missing)
        )
    result = dataframe.loc[:, sorted(REQUIRED_SOURCE_COLUMNS)].copy()
    try:
        result["date_month"] = pd.to_datetime(
            result["date_month"], errors="raise"
        )
    except (TypeError, ValueError) as error:
        raise InferenceDataError("date_month contiene fechas no validas.") from error
    if result["date_month"].isna().any():
        raise InferenceDataError("date_month contiene valores nulos.")
    if not bool(result["date_month"].dt.is_month_start.all()):
        raise InferenceDataError(
            "date_month debe representar el primer dia de cada mes."
        )

    for column in ("territory_id", "territory_name", "territory_level", "month_id"):
        if result[column].isna().any():
            raise InferenceDataError(f"{column} contiene valores nulos.")
        result[column] = result[column].astype("string")
        if result[column].str.strip().eq("").any():
            raise InferenceDataError(f"{column} contiene valores vacios.")

    if not result["territory_level"].eq("province").all():
        raise InferenceDataError(
            "La inferencia B5 admite exclusivamente nivel province."
        )
    expected_month_ids = result["date_month"].dt.strftime("%Y-%m")
    if not result["month_id"].eq(expected_month_ids).all():
        raise InferenceDataError(
            "month_id y date_month no representan el mismo mes."
        )
    duplicate_count = int(
        result.duplicated(["territory_id", "month_id"]).sum()
    )
    if duplicate_count:
        raise InferenceDataError(
            f"El dataset contiene {duplicate_count} claves territorio-mes duplicadas."
        )
    inconsistent_names = (
        result.groupby("territory_id", observed=True)["territory_name"]
        .nunique()
        .gt(1)
    )
    if bool(inconsistent_names.any()):
        raise InferenceDataError(
            "Un territory_id no puede tener varios territory_name."
        )

    result["overnight_stays_total"] = pd.to_numeric(
        result["overnight_stays_total"], errors="coerce"
    )

    for column in ("source_snapshot_id", "pipeline_run_id", "data_version"):
        if result[column].isna().any():
            raise InferenceDataError(
                f"{column} debe identificar un unico lineage no nulo."
            )
        values = result[column].astype(str)
        if values.str.strip().eq("").any() or values.nunique() != 1:
            raise InferenceDataError(
                f"{column} debe identificar un unico lineage no nulo."
            )
        result[column] = values.astype("string")

    return result.sort_values(
        ["territory_id", "date_month"], ignore_index=True
    )


def _validate_available_observations(history: pd.DataFrame) -> None:
    """Valida valores solo dentro del conjunto causal entregado al modelo."""

    history["complete_month_available"] = _valid_boolean_series(
        history["complete_month_available"],
        column="complete_month_available",
    )
    history["is_provisional"] = _valid_boolean_series(
        history["is_provisional"],
        column="is_provisional",
    )
    numeric = history["overnight_stays_total"]
    complete = history["complete_month_available"]
    finite = numeric.map(
        lambda value: isfinite(float(value)) if pd.notna(value) else False
    )
    invalid_complete = complete & (
        numeric.isna() | ~finite | numeric.lt(0)
    )
    if bool(invalid_complete.any()):
        raise InferenceDataError(
            "Una observacion completa conocida en el cutoff tiene "
            "overnight_stays_total invalido."
        )


def _local_today() -> date:
    return date.today()


def _as_month(as_of_date: AsOfDate | None) -> pd.Period:
    try:
        timestamp = pd.Timestamp(
            _local_today() if as_of_date is None else as_of_date
        )
    except (TypeError, ValueError) as error:
        raise InferenceError(
            f"as_of_date no es una fecha valida: {as_of_date!r}."
        ) from error
    if pd.isna(timestamp):
        raise InferenceError("as_of_date no puede ser nulo.")
    return timestamp.to_period("M")


def _baseline_reference(
    territory_history: pd.DataFrame,
    target_month: pd.Period,
) -> tuple[pd.Series | None, str]:
    reference_month = target_month - 12
    rows = territory_history.loc[
        territory_history["month_id"].eq(str(reference_month))
    ]
    if rows.empty:
        return None, str(reference_month)
    row = rows.iloc[0]
    value = row["overnight_stays_total"]
    valid = (
        bool(row["complete_month_available"])
        and pd.notna(value)
        and isfinite(float(value))
        and float(value) >= 0
    )
    return (row if valid else None), str(reference_month)


def _validate_ets_result(
    forecast: ETSForecastResult,
    *,
    territory_id: str,
    target_month_id: str,
    latest_available_month_id: str,
) -> None:
    if forecast.territory_id != territory_id:
        raise InferenceDataError("El ETS devolvio otro territorio.")
    if forecast.target_month_id != target_month_id:
        raise InferenceDataError("El ETS devolvio otro target.")
    if forecast.latest_available_month_id != latest_available_month_id:
        raise InferenceDataError("El ETS devolvio otro cutoff.")
    if forecast.effective_horizon_steps != EFFECTIVE_MODEL_HORIZON_STEPS:
        raise InferenceDataError("El ETS devolvio un horizonte efectivo invalido.")
    if forecast.training_end is not None and (
        pd.Period(forecast.training_end, freq="M")
        > pd.Period(latest_available_month_id, freq="M")
    ):
        raise InferenceDataError("training_end ETS supera el cutoff point-in-time.")


def _ets_training_uses_provisional_data(
    history: pd.DataFrame,
    forecast: ETSForecastResult,
) -> bool:
    """Detecta inputs provisionales observados en el training ETS real."""

    if not forecast.candidate_available:
        return False
    if forecast.training_start is None or forecast.training_end is None:
        raise InferenceDataError(
            "El ETS disponible no declara su intervalo de entrenamiento."
        )

    start = pd.Period(forecast.training_start, freq="M")
    end = pd.Period(forecast.training_end, freq="M")
    months = pd.PeriodIndex(history["month_id"], freq="M")
    values = pd.to_numeric(history["overnight_stays_total"], errors="coerce")
    finite = values.map(
        lambda value: isfinite(float(value)) if pd.notna(value) else False
    )
    in_training_window = pd.Series(
        (months >= start) & (months <= end),
        index=history.index,
    )
    observed_training = (
        history["complete_month_available"]
        & values.notna()
        & finite
        & values.ge(0)
        & in_training_window
    )
    return bool((observed_training & history["is_provisional"]).any())


def predict_next_month(
    territory_id: str,
    *,
    as_of_date: AsOfDate | None = None,
    forecast_horizon_months: int = SUPPORTED_FORECAST_HORIZON_MONTHS,
    dataframe: pd.DataFrame | None = None,
    config: Mapping[str, Any] | None = None,
    config_path: Path = MODELING_V2_CONFIG_PATH,
) -> InferenceResult:
    """Predice target+1 con ETS y fallback lag-12 solo por disponibilidad."""

    effective_config: Mapping[str, Any] = (
        _load_inference_config(config_path) if config is None else config
    )
    _validate_inference_config(effective_config)
    if (
        not isinstance(forecast_horizon_months, int)
        or isinstance(forecast_horizon_months, bool)
        or forecast_horizon_months != SUPPORTED_FORECAST_HORIZON_MONTHS
    ):
        raise UnsupportedHorizonError(
            "El sistema B5 solo admite un horizonte de negocio de un mes."
        )

    raw = (
        load_inference_dataset(effective_config)
        if dataframe is None
        else dataframe
    )
    source = _prepare_dataset(raw)
    requested_id = str(territory_id).strip()
    territory_history = source.loc[
        source["territory_id"].eq(requested_id)
    ].copy()
    if territory_history.empty:
        raise InvalidTerritoryError(
            f"No existe el territory_id '{requested_id}'."
        )
    territory_name = str(territory_history["territory_name"].iloc[0])

    as_of_month = _as_month(as_of_date)
    target_month = as_of_month + forecast_horizon_months
    policy = cutoff_policy_from_config(effective_config)
    origin = resolve_information_cutoff(target_month, policy)
    if origin.business_origin_month_id != str(as_of_month):
        raise InferenceConfigurationError(
            "El business origin V2 no coincide con as_of_month."
        )
    if origin.max_training_target_month_id != origin.latest_available_month_id:
        raise InferenceConfigurationError(
            "El purge de labels no coincide con el cutoff operacional B5."
        )

    cutoff_history = filter_history_to_information_cutoff(
        territory_history,
        origin,
        observation_month_column="month_id",
    )
    if cutoff_history.empty:
        raise MissingReferenceError(
            "No existe historia provincial compatible con el cutoff."
        )
    _validate_available_observations(cutoff_history)
    if pd.PeriodIndex(cutoff_history["month_id"], freq="M").max() > pd.Period(
        origin.latest_available_month_id, freq="M"
    ):
        raise AssertionError("La inferencia ETS recibio datos futuros.")

    reference_row, reference_month_id = _baseline_reference(
        cutoff_history,
        target_month,
    )
    baseline_prediction = (
        float(reference_row["overnight_stays_total"])
        if reference_row is not None
        else None
    )
    baseline_provisional = (
        bool(reference_row["is_provisional"])
        if reference_row is not None
        else False
    )

    ets_forecast = fit_ets_forecast(
        cutoff_history,
        origin,
        _required_mapping(effective_config, "ets_candidate"),
    )
    _validate_ets_result(
        ets_forecast,
        territory_id=requested_id,
        target_month_id=str(target_month),
        latest_available_month_id=origin.latest_available_month_id,
    )

    warnings: list[InferenceWarning] = []
    if ets_forecast.candidate_available:
        prediction = ets_forecast.prediction
        if prediction is None or not isfinite(float(prediction)) or prediction < 0:
            raise InferenceDataError(
                "El ETS marcado disponible no contiene un forecast valido."
            )
        point = float(prediction)
        actual_model = SELECTED_MODEL_ID
        fallback_used = False
        fallback_reason = "not_used"
        if _ets_training_uses_provisional_data(
            cutoff_history,
            ets_forecast,
        ):
            warnings.append(
                InferenceWarning(
                    code="provisional_training_data",
                    message=(
                        "La estimacion ETS utiliza datos provisionales del "
                        "INE que pueden revisarse."
                    ),
                )
            )
    else:
        reason = str(ets_forecast.unavailable_reason)
        if reason not in AVAILABILITY_FALLBACK_REASONS:
            raise InferenceDataError(
                f"Motivo de indisponibilidad ETS no soportado: {reason!r}."
            )
        if reference_row is None or baseline_prediction is None:
            raise MissingReferenceError(
                "ETS no disponible y no existe una referencia lag-12 valida "
                f"para {requested_id} en {reference_month_id}."
            )
        point = baseline_prediction
        actual_model = FALLBACK_MODEL_ID
        fallback_used = True
        fallback_reason = reason
        warnings.append(
            InferenceWarning(
                code="availability_fallback_used",
                message=(
                    "ETS no disponible; se usa seasonal_naive_lag_12 "
                    f"por {reason}."
                ),
            )
        )
        if baseline_provisional:
            warnings.append(
                InferenceWarning(
                    code="provisional_reference_data",
                    message=(
                        "La referencia lag-12 usada por el fallback es provisional."
                    ),
                )
            )

    if ets_forecast.clipping_applied:
        warnings.append(
            InferenceWarning(
                code="ets_negative_prediction_clipped",
                message="La prediccion ETS negativa se trunco a cero.",
            )
        )
    if ets_forecast.fit_warning_count:
        warnings.append(
            InferenceWarning(
                code="ets_fit_warnings",
                message=(
                    f"El ajuste ETS registro {ets_forecast.fit_warning_count} "
                    "warnings no bloqueantes."
                ),
            )
        )

    lineage_row = cutoff_history.iloc[-1]
    return InferenceResult(
        territory_id=requested_id,
        territory_name=territory_name,
        target_month_id=str(target_month),
        business_origin_month_id=origin.business_origin_month_id,
        latest_available_month_id=origin.latest_available_month_id,
        forecast_horizon_months=SUPPORTED_FORECAST_HORIZON_MONTHS,
        effective_model_horizon_steps=EFFECTIVE_MODEL_HORIZON_STEPS,
        predicted_overnight_stays_total=point,
        selected_model_id=SELECTED_MODEL_ID,
        selection_status=SELECTION_STATUS,
        actual_model_used=actual_model,
        fallback_used=fallback_used,
        fallback_reason=fallback_reason,
        baseline_reference_month_id=reference_month_id,
        baseline_prediction=(
            float(baseline_prediction)
            if baseline_prediction is not None
            else None
        ),
        baseline_reference_is_provisional=baseline_provisional,
        ets_raw_prediction=(
            float(ets_forecast.raw_prediction)
            if ets_forecast.raw_prediction is not None
            else None
        ),
        clipping_applied=bool(ets_forecast.clipping_applied),
        training_start=ets_forecast.training_start,
        training_end=ets_forecast.training_end,
        training_rows=int(ets_forecast.training_rows),
        source_snapshot_id=str(lineage_row["source_snapshot_id"]),
        pipeline_run_id=str(lineage_row["pipeline_run_id"]),
        data_version=str(lineage_row["data_version"]),
        operational_status="forecast_ready",
        warnings=tuple(warnings),
    )
