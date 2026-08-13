"""Inferencia operacional para el baseline estacional seleccionado."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from math import isfinite
from pathlib import Path
from typing import Any, Literal, Mapping

import pandas as pd

from src.models.modeling_common import (
    CONFIG_PATH,
    PROJECT_ROOT,
    load_config,
    resolve_project_path,
)


MODEL_NAME = "seasonal_naive_lag_12"
PREDICTION_FEATURE = "lag_12_overnight_stays"
SUPPORTED_FORECAST_HORIZON_MONTHS = 1

REQUIRED_SOURCE_COLUMNS = {
    "territory_id",
    "territory_name",
    "month_id",
    "date_month",
    "overnight_stays_total",
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
    """La configuracion no permite una inferencia reproducible."""


class InferenceDataError(InferenceError):
    """El dataset no cumple el contrato necesario para inferencia."""


class EmptyInferenceDatasetError(InferenceDataError):
    """El dataset de inferencia no contiene observaciones."""


class InvalidTerritoryError(InferenceError):
    """El territorio solicitado no existe en el dataset."""


class MissingReferenceError(InferenceError):
    """No existe la observacion exacta requerida en target menos 12 meses."""


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
    """Resultado trazable de una prediccion mensual provincial."""

    territory_id: str
    territory_name: str
    target_month_id: str
    forecast_horizon_months: int
    predicted_overnight_stays_total: float
    reference_month_id: str
    reference_overnight_stays_total: float
    reference_is_provisional: bool
    model_name: str
    source_snapshot_id: str
    pipeline_run_id: str
    data_version: str
    latest_available_month_id: str
    operational_status: OperationalStatus
    warnings: tuple[InferenceWarning, ...]

    @property
    def is_operational(self) -> bool:
        """Indica si existe un forecast valido para el proximo mes."""
        return self.operational_status == "forecast_ready"


def _display_path(path: Path) -> str:
    """Devuelve una ruta legible aunque este fuera de la raiz del proyecto."""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _load_inference_config(config_path: Path) -> dict[str, Any]:
    """Carga la configuracion y traduce sus fallos a errores de inferencia."""
    if not config_path.exists():
        raise InferenceConfigurationError(
            "No se encontro la configuracion necesaria para inferencia: "
            f"{_display_path(config_path)}."
        )

    try:
        return load_config(config_path)
    except (OSError, TypeError, ValueError) as error:
        raise InferenceConfigurationError(
            "No se pudo cargar una configuracion valida para inferencia: "
            f"{error}"
        ) from error


def _required_mapping(
    config: Mapping[str, Any],
    section: str,
) -> Mapping[str, Any]:
    """Obtiene una seccion obligatoria de configuracion."""
    value = config.get(section)
    if not isinstance(value, Mapping):
        raise InferenceConfigurationError(
            f"Falta la seccion de configuracion '{section}'."
        )
    return value


def _validate_inference_config(config: Mapping[str, Any]) -> None:
    """Comprueba el contrato minimo de la solucion operacional."""
    problem = _required_mapping(config, "problem")
    baseline = _required_mapping(config, "baseline")
    target = _required_mapping(config, "target")
    source_dataset = _required_mapping(config, "source_dataset")
    fallback = _required_mapping(config, "fallback")
    selection = _required_mapping(
        fallback,
        "if_no_candidate_beats_baseline",
    )

    try:
        configured_horizon = int(problem["forecast_horizon_months"])
    except (KeyError, TypeError, ValueError) as error:
        raise InferenceConfigurationError(
            "Falta un forecast_horizon_months valido en problem."
        ) from error

    if configured_horizon != SUPPORTED_FORECAST_HORIZON_MONTHS:
        raise InferenceConfigurationError(
            "La configuracion no conserva el horizonte operacional de un mes."
        )

    if baseline.get("name") != MODEL_NAME:
        raise InferenceConfigurationError(
            f"La solucion configurada no es {MODEL_NAME}."
        )

    if baseline.get("prediction_feature") != PREDICTION_FEATURE:
        raise InferenceConfigurationError(
            "La feature del baseline no corresponde al lag natural de 12 meses."
        )

    if selection.get("selected_solution") != MODEL_NAME:
        raise InferenceConfigurationError(
            f"La solucion seleccionada no es {MODEL_NAME}."
        )

    if target.get("source_column") != "overnight_stays_total":
        raise InferenceConfigurationError(
            "La columna fuente del target no es overnight_stays_total."
        )

    if not isinstance(source_dataset.get("path"), str):
        raise InferenceConfigurationError(
            "Falta una ruta valida en source_dataset.path."
        )


def load_inference_dataset(config: Mapping[str, Any]) -> pd.DataFrame:
    """Carga el Gold descriptivo declarado en la configuracion."""
    _validate_inference_config(config)
    source_dataset = _required_mapping(config, "source_dataset")
    dataset_path = resolve_project_path(str(source_dataset["path"]))

    if not dataset_path.exists():
        raise InferenceDataError(
            "No se encontro el dataset de inferencia: "
            f"{_display_path(dataset_path)}."
        )

    try:
        return pd.read_parquet(dataset_path)
    except (OSError, ValueError) as error:
        raise InferenceDataError(
            f"No se pudo leer el dataset de inferencia: {error}"
        ) from error


def _prepare_dataset(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Valida y normaliza solo los campos necesarios del Gold."""
    if not isinstance(dataframe, pd.DataFrame):
        raise InferenceDataError("dataframe debe ser un pandas.DataFrame.")

    if dataframe.empty:
        raise EmptyInferenceDatasetError(
            "El dataset de inferencia esta vacio."
        )

    missing_columns = REQUIRED_SOURCE_COLUMNS.difference(dataframe.columns)
    if missing_columns:
        raise InferenceDataError(
            "Faltan columnas requeridas para inferencia: "
            + ", ".join(sorted(missing_columns))
        )

    result = dataframe.loc[:, sorted(REQUIRED_SOURCE_COLUMNS)].copy()

    try:
        result["date_month"] = pd.to_datetime(
            result["date_month"],
            errors="raise",
        )
    except (TypeError, ValueError) as error:
        raise InferenceDataError(
            "date_month contiene fechas no validas."
        ) from error

    if result["date_month"].isna().any():
        raise InferenceDataError("date_month contiene valores nulos.")

    if not result["date_month"].dt.is_month_start.all():
        raise InferenceDataError(
            "date_month debe representar el primer dia de cada mes."
        )

    result["territory_id"] = result["territory_id"].astype("string")
    result["territory_name"] = result["territory_name"].astype("string")
    result["month_id"] = result["month_id"].astype("string")

    if result[["territory_id", "territory_name", "month_id"]].isna().any().any():
        raise InferenceDataError(
            "Los identificadores territoriales o mensuales contienen nulos."
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
            "El dataset contiene "
            f"{duplicate_count} claves territorio-mes duplicadas."
        )

    provisional_values = result["is_provisional"].dropna()
    valid_provisional = provisional_values.map(
        lambda value: isinstance(value, (bool, int)) and value in (0, 1)
    )
    if result["is_provisional"].isna().any() or not valid_provisional.all():
        raise InferenceDataError(
            "is_provisional debe contener solo booleanos no nulos."
        )
    result["is_provisional"] = result["is_provisional"].astype(bool)

    for column in ("source_snapshot_id", "pipeline_run_id", "data_version"):
        if result[column].isna().any():
            raise InferenceDataError(
                f"{column} debe identificar un unico lineage no nulo."
            )
        lineage_values = result[column].astype(str)
        if lineage_values.str.strip().eq("").any():
            raise InferenceDataError(
                f"{column} no puede contener valores vacios."
            )
        if lineage_values.nunique() != 1:
            raise InferenceDataError(
                f"{column} debe identificar un unico lineage no nulo."
            )
        result[column] = lineage_values.astype("string")

    return result.sort_values(
        ["territory_id", "date_month"]
    ).reset_index(drop=True)


def _local_today() -> date:
    """Obtiene la fecha local del sistema como forecast origin por defecto."""
    return date.today()


def _as_month(as_of_date: AsOfDate | None) -> pd.Period:
    """Normaliza la fecha de referencia a un periodo mensual."""
    if as_of_date is None:
        timestamp = pd.Timestamp(_local_today())
    else:
        try:
            timestamp = pd.Timestamp(as_of_date)
        except (TypeError, ValueError) as error:
            raise InferenceError(
                f"as_of_date no es una fecha valida: {as_of_date!r}."
            ) from error

    if pd.isna(timestamp):
        raise InferenceError("as_of_date no puede ser nulo.")

    return pd.Period(timestamp.strftime("%Y-%m"), freq="M")


def predict_next_month(
    territory_id: str,
    *,
    as_of_date: AsOfDate | None = None,
    forecast_horizon_months: int = SUPPORTED_FORECAST_HORIZON_MONTHS,
    dataframe: pd.DataFrame | None = None,
    config: Mapping[str, Any] | None = None,
    config_path: Path = CONFIG_PATH,
) -> InferenceResult:
    """Predice el siguiente mes con el valor exacto de 12 meses antes.

    Si no se proporciona ``dataframe``, carga el Gold declarado en la
    configuracion. El target se deriva del mes natural de ``as_of_date``;
    el ultimo mes publicado se conserva solo como contexto. Por defecto se
    utiliza la fecha local real del sistema como forecast origin.
    """
    effective_config: Mapping[str, Any]
    if config is None:
        effective_config = _load_inference_config(config_path)
    else:
        effective_config = config

    _validate_inference_config(effective_config)

    if (
        not isinstance(forecast_horizon_months, int)
        or isinstance(forecast_horizon_months, bool)
        or forecast_horizon_months != SUPPORTED_FORECAST_HORIZON_MONTHS
    ):
        raise UnsupportedHorizonError(
            "seasonal_naive_lag_12 solo admite un horizonte de un mes."
        )

    raw_dataframe = (
        load_inference_dataset(effective_config)
        if dataframe is None
        else dataframe
    )
    source = _prepare_dataset(raw_dataframe)

    requested_territory_id = str(territory_id).strip()
    territory_rows = source.loc[
        source["territory_id"].eq(requested_territory_id)
    ]
    if territory_rows.empty:
        raise InvalidTerritoryError(
            f"No existe el territory_id '{requested_territory_id}'."
        )

    territory_names = territory_rows["territory_name"].unique()
    if len(territory_names) != 1:
        raise InferenceDataError(
            "El territorio solicitado tiene mas de un territory_name."
        )

    source_months = source["date_month"].dt.to_period("M")
    latest_available_month = source_months.max()
    as_of_month = _as_month(as_of_date)
    target_month = as_of_month + forecast_horizon_months
    reference_month = target_month - 12

    global_reference_rows = source.loc[
        source_months.eq(reference_month)
    ]
    if global_reference_rows.empty:
        raise GlobalReferenceGapError(
            "No existe el mes de referencia exacto "
            f"{reference_month} en el dataset (gap global)."
        )

    reference_rows = global_reference_rows.loc[
        global_reference_rows["territory_id"].eq(requested_territory_id)
    ]
    if reference_rows.empty:
        raise TerritorialReferenceGapError(
            "No existe la observacion exacta de "
            f"{requested_territory_id} en {reference_month} "
            "(gap territorial)."
        )

    reference_row = reference_rows.iloc[0]
    reference_value = pd.to_numeric(
        pd.Series([reference_row["overnight_stays_total"]]),
        errors="coerce",
    ).iloc[0]
    if pd.isna(reference_value):
        raise InferenceDataError(
            "overnight_stays_total es nulo o no numerico en la "
            "observacion de referencia exacta."
        )
    reference_number = float(reference_value)
    if not isfinite(reference_number):
        raise InferenceDataError(
            "overnight_stays_total debe ser finito en la observacion "
            "de referencia exacta."
        )
    if reference_number < 0:
        raise InferenceDataError(
            "overnight_stays_total es negativo en la observacion de referencia."
        )

    warnings: list[InferenceWarning] = []
    reference_is_provisional = bool(reference_row["is_provisional"])
    if reference_is_provisional:
        warnings.append(
            InferenceWarning(
                code="provisional_reference_data",
                message=(
                    "La observacion de referencia esta marcada como "
                    "provisional."
                ),
            )
        )

    return InferenceResult(
        territory_id=requested_territory_id,
        territory_name=str(territory_names[0]),
        target_month_id=str(target_month),
        forecast_horizon_months=forecast_horizon_months,
        predicted_overnight_stays_total=reference_number,
        reference_month_id=str(reference_month),
        reference_overnight_stays_total=reference_number,
        reference_is_provisional=reference_is_provisional,
        model_name=MODEL_NAME,
        source_snapshot_id=str(reference_row["source_snapshot_id"]),
        pipeline_run_id=str(reference_row["pipeline_run_id"]),
        data_version=str(reference_row["data_version"]),
        latest_available_month_id=str(latest_available_month),
        operational_status="forecast_ready",
        warnings=tuple(warnings),
    )
