"""Datos historicos y metricas reutilizables por el futuro frontal."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.models.modeling_common import (
    CONFIG_PATH,
    PROJECT_ROOT,
    calculate_metrics,
    load_config,
    resolve_project_path,
)


__all__ = [
    "DashboardContext",
    "DashboardDataError",
    "DashboardLineage",
    "InvalidTerritoryError",
    "TerritoryValidationMetrics",
    "build_dashboard_context",
    "calculate_territory_validation_metrics",
    "get_territory_history",
    "get_territory_validation_metrics",
]

OPERATIONAL_BASELINE_ID = "seasonal_naive_lag_12"
PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "data"
    / "model_outputs"
    / "model_selection_validation_predictions.parquet"
)
METRICS_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "model_selection_metrics.csv"
)

BASELINE_OBSERVATION_KEY = (
    "territory_id",
    "target_month_id",
    "evaluation_split",
)
BASELINE_REPLICA_COLUMNS = (
    "territory_name",
    "target_date_month",
    "source_snapshot_id",
    "pipeline_run_id",
    "data_version",
    "created_at",
    "baseline_id",
    "dataset_path",
    "actual",
    "baseline_prediction",
    "validation_start",
    "structural_train_end",
    "availability_train_end",
    "effective_train_end",
)
REQUIRED_PREDICTION_COLUMNS = {
    *BASELINE_OBSERVATION_KEY,
    *BASELINE_REPLICA_COLUMNS,
    "model",
}
HISTORY_COLUMNS = [
    "territory_id",
    "territory_name",
    "month_id",
    "date_month",
    "overnight_stays_total",
    "data_status",
    "is_provisional",
    "coverage_quality",
]
REQUIRED_GOLD_COLUMNS = {
    *HISTORY_COLUMNS,
    "source_snapshot_id",
    "pipeline_run_id",
    "data_version",
}


class DashboardDataError(RuntimeError):
    """Error bloqueante al preparar datos para presentacion."""


class InvalidTerritoryError(DashboardDataError):
    """El territorio solicitado no existe en la fuente requerida."""


class BaselineReplicaError(DashboardDataError):
    """Las replicas del baseline no son coherentes o deduplicables."""


class MetricsReconciliationError(DashboardDataError):
    """Las metricas reconstruidas no coinciden con el artefacto oficial."""


class LineageMismatchError(DashboardDataError):
    """Los artefactos pertenecen a snapshots incompatibles."""


@dataclass(frozen=True)
class TerritoryValidationMetrics:
    """Metricas fuera de muestra del baseline para una provincia."""

    territory_id: str
    territory_name: str
    validation_mae: float
    validation_rmse: float
    validation_wape_pct: float
    validation_bias: float
    validation_rows: int


@dataclass(frozen=True)
class DashboardLineage:
    """Lineage real de las fuentes combinadas por la capa."""

    source_snapshot_id: str
    gold_pipeline_run_id: str
    gold_data_version: str
    validation_pipeline_run_id: str
    validation_data_version: str
    gold_dataset_path: str
    validation_dataset_path: str
    validation_predictions_path: str
    validation_metrics_path: str


@dataclass(frozen=True)
class DashboardContext:
    """Contexto provincial listo para combinar con inferencia en otra capa."""

    territory_id: str
    territory_name: str
    history: pd.DataFrame
    validation_metrics: TerritoryValidationMetrics
    lineage: DashboardLineage

    def to_export_frames(self) -> dict[str, pd.DataFrame]:
        """Convierte el contexto en tablas sencillas para futura descarga."""
        return {
            "history": self.history.copy(),
            "validation_metrics": pd.DataFrame(
                [asdict(self.validation_metrics)]
            ),
            "lineage": pd.DataFrame([asdict(self.lineage)]),
        }


def _effective_config(
    config: Mapping[str, Any] | None,
    config_path: Path,
) -> Mapping[str, Any]:
    """Obtiene configuracion inyectada o carga la vigente del proyecto."""
    return load_config(config_path) if config is None else config


def _required_mapping(
    mapping: Mapping[str, Any],
    key: str,
) -> Mapping[str, Any]:
    """Obtiene una seccion de configuracion obligatoria."""
    value = mapping.get(key)
    if not isinstance(value, Mapping):
        raise DashboardDataError(
            f"Falta la seccion de configuracion '{key}'."
        )
    return value


def _dashboard_contract(
    config: Mapping[str, Any],
) -> tuple[str, tuple[str, ...]]:
    """Valida baseline seleccionado y folds autorizados."""
    baseline = _required_mapping(config, "baseline")
    fallback = _required_mapping(config, "fallback")
    selection = _required_mapping(
        fallback,
        "if_no_candidate_beats_baseline",
    )
    validation = _required_mapping(config, "validation")

    baseline_id = str(baseline.get("name", ""))
    selected_solution = str(selection.get("selected_solution", ""))
    if (
        baseline_id != OPERATIONAL_BASELINE_ID
        or selected_solution != OPERATIONAL_BASELINE_ID
    ):
        raise DashboardDataError(
            "La solucion operacional debe ser seasonal_naive_lag_12."
        )

    folds = validation.get("folds")
    if not isinstance(folds, list) or not folds:
        raise DashboardDataError(
            "La configuracion no contiene folds de validacion."
        )

    split_names = tuple(str(fold.get("name", "")) for fold in folds)
    if any(not name for name in split_names) or len(set(split_names)) != len(
        split_names
    ):
        raise DashboardDataError(
            "Los folds de validacion tienen nombres vacios o duplicados."
        )
    return baseline_id, split_names


def _read_parquet(path: Path, artifact_name: str) -> pd.DataFrame:
    """Carga un Parquet requerido sin regenerar artefactos."""
    if not path.exists():
        raise DashboardDataError(
            f"No se encontro {artifact_name}: {path}."
        )
    try:
        return pd.read_parquet(path)
    except (OSError, ValueError) as error:
        raise DashboardDataError(
            f"No se pudo leer {artifact_name}: {error}"
        ) from error


def load_gold_history(config: Mapping[str, Any]) -> pd.DataFrame:
    """Carga la Gold descriptiva declarada en configuracion."""
    source = _required_mapping(config, "source_dataset")
    path_text = source.get("path")
    if not isinstance(path_text, str) or not path_text.strip():
        raise DashboardDataError("Falta source_dataset.path.")
    return _read_parquet(
        resolve_project_path(path_text),
        "la Gold descriptiva",
    )


def load_validation_predictions() -> pd.DataFrame:
    """Carga las predicciones vigentes de seleccion en validacion."""
    return _read_parquet(
        PREDICTIONS_PATH,
        "las predicciones de validacion",
    )


def load_official_validation_metrics() -> pd.DataFrame:
    """Carga las metricas oficiales vigentes de seleccion."""
    if not METRICS_PATH.exists():
        raise DashboardDataError(
            f"No se encontraron las metricas oficiales: {METRICS_PATH}."
        )
    try:
        return pd.read_csv(METRICS_PATH)
    except (OSError, ValueError) as error:
        raise DashboardDataError(
            f"No se pudieron leer las metricas oficiales: {error}"
        ) from error


def _prepare_gold(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Valida el contrato minimo del historico Gold."""
    if not isinstance(dataframe, pd.DataFrame) or dataframe.empty:
        raise DashboardDataError("La Gold descriptiva esta vacia o no es valida.")

    missing = REQUIRED_GOLD_COLUMNS.difference(dataframe.columns)
    if missing:
        raise DashboardDataError(
            "Faltan columnas en la Gold: " + ", ".join(sorted(missing))
        )

    result = dataframe.loc[:, sorted(REQUIRED_GOLD_COLUMNS)].copy()
    try:
        result["date_month"] = pd.to_datetime(
            result["date_month"],
            errors="raise",
        )
    except (TypeError, ValueError) as error:
        raise DashboardDataError("date_month contiene fechas no validas.") from error

    if result["date_month"].isna().any():
        raise DashboardDataError("date_month contiene valores nulos.")

    result["month_id"] = result["month_id"].astype("string")
    expected_month = result["date_month"].dt.strftime("%Y-%m")
    if not result["month_id"].eq(expected_month).all():
        raise DashboardDataError(
            "month_id y date_month no representan el mismo mes."
        )

    if result.duplicated(["territory_id", "month_id"]).any():
        raise DashboardDataError(
            "La Gold contiene claves territorio-mes duplicadas."
        )
    return result


def get_territory_history(
    territory_id: str,
    *,
    months: int | None = None,
    dataframe: pd.DataFrame | None = None,
    config: Mapping[str, Any] | None = None,
    config_path: Path = CONFIG_PATH,
) -> pd.DataFrame:
    """Obtiene el historico mensual real, ordenado y sin imputaciones.

    ``months`` define una ventana de meses naturales terminada en el ultimo
    mes disponible del territorio. Los gaps permanecen ausentes y, por tanto,
    la ventana puede contener menos filas que meses solicitados.
    """
    effective_config = _effective_config(config, config_path)
    source = _prepare_gold(
        load_gold_history(effective_config)
        if dataframe is None
        else dataframe
    )

    if months is not None and (
        not isinstance(months, int)
        or isinstance(months, bool)
        or months < 1
    ):
        raise ValueError("months debe ser un entero positivo o None.")

    requested_id = str(territory_id).strip()
    history = source.loc[
        source["territory_id"].astype(str).eq(requested_id),
        HISTORY_COLUMNS,
    ].sort_values("date_month")
    if history.empty:
        raise InvalidTerritoryError(
            f"No existe el territory_id '{requested_id}' en la Gold."
        )

    if history["territory_name"].nunique(dropna=False) != 1:
        raise DashboardDataError(
            "El territorio tiene mas de un nombre en la Gold."
        )

    if months is not None:
        last_month = history["date_month"].max().to_period("M")
        first_month = last_month - (months - 1)
        history_months = history["date_month"].dt.to_period("M")
        history = history.loc[history_months.ge(first_month)]
    return history.reset_index(drop=True)


def prepare_baseline_validation_predictions(
    dataframe: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Valida las replicas por candidato y deduplica el baseline.

    La observacion logica se identifica por territorio, mes objetivo y split.
    ``model`` identifica el candidato que origino cada copia del baseline.
    """
    if not isinstance(dataframe, pd.DataFrame) or dataframe.empty:
        raise BaselineReplicaError(
            "Las predicciones de validacion estan vacias o no son validas."
        )

    missing = REQUIRED_PREDICTION_COLUMNS.difference(dataframe.columns)
    if missing:
        raise BaselineReplicaError(
            "Faltan columnas en las predicciones: "
            + ", ".join(sorted(missing))
        )

    baseline_id, expected_splits = _dashboard_contract(config)
    predictions = dataframe.loc[
        :,
        [
            *BASELINE_OBSERVATION_KEY,
            *BASELINE_REPLICA_COLUMNS,
            "model",
        ],
    ].copy()

    actual_splits = set(predictions["evaluation_split"].astype(str).unique())
    if actual_splits != set(expected_splits):
        raise BaselineReplicaError(
            "Las predicciones no contienen exclusivamente los folds vigentes."
        )

    baseline_ids = predictions["baseline_id"].dropna().astype(str).unique()
    if len(baseline_ids) != 1 or baseline_ids[0] != baseline_id:
        raise BaselineReplicaError(
            "baseline_id no coincide con el baseline configurado."
        )

    key = list(BASELINE_OBSERVATION_KEY)
    if predictions[key + ["model"]].isna().any().any():
        raise BaselineReplicaError(
            "La clave logica o el candidato contienen valores nulos."
        )
    if predictions.duplicated(key + ["model"]).any():
        raise BaselineReplicaError(
            "Existe mas de una replica por observacion y candidato."
        )

    candidates = frozenset(predictions["model"].astype(str).unique())
    if not candidates or "" in candidates:
        raise BaselineReplicaError("No existen candidatos validos.")

    candidate_sets = predictions.groupby(
        key,
        dropna=False,
        sort=False,
    )["model"].agg(lambda values: frozenset(values.astype(str)))
    if not candidate_sets.map(lambda values: values == candidates).all():
        raise BaselineReplicaError(
            "Las observaciones no estan replicadas para los mismos candidatos."
        )

    grouped = predictions.groupby(key, dropna=False, sort=False)
    inconsistent_columns = [
        column
        for column in BASELINE_REPLICA_COLUMNS
        if grouped[column].nunique(dropna=False).gt(1).any()
    ]
    if inconsistent_columns:
        raise BaselineReplicaError(
            "Las replicas del baseline difieren en: "
            + ", ".join(inconsistent_columns)
        )

    for column in ("source_snapshot_id", "pipeline_run_id", "data_version"):
        _single_lineage_value(predictions, column)

    unique = (
        predictions
        .drop_duplicates(key, keep="first")
        .drop(columns="model")
        .copy()
    )
    try:
        unique["target_date_month"] = pd.to_datetime(
            unique["target_date_month"],
            errors="raise",
        )
        unique["actual"] = pd.to_numeric(unique["actual"], errors="raise")
        unique["baseline_prediction"] = pd.to_numeric(
            unique["baseline_prediction"],
            errors="raise",
        )
    except (TypeError, ValueError) as error:
        raise BaselineReplicaError(
            "Las predicciones contienen fechas o valores no validos."
        ) from error

    target_months = unique["target_date_month"].dt.strftime("%Y-%m")
    if not unique["target_month_id"].astype(str).eq(target_months).all():
        raise BaselineReplicaError(
            "target_month_id y target_date_month son incompatibles."
        )

    numeric = unique[["actual", "baseline_prediction"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or (numeric < 0).any():
        raise BaselineReplicaError(
            "Actuales y predicciones baseline deben ser finitos y no negativos."
        )

    if unique.groupby("territory_id")["territory_name"].nunique().gt(1).any():
        raise BaselineReplicaError(
            "Un territory_id tiene varios nombres en las predicciones."
        )

    return unique.sort_values(
        ["territory_id", "target_date_month", "evaluation_split"]
    ).reset_index(drop=True)


def _metrics_from_group(group: pd.DataFrame) -> dict[str, float | int]:
    """Calcula metricas con la definicion compartida de modelado."""
    return calculate_metrics(
        group["actual"].to_numpy(dtype=float),
        group["baseline_prediction"].to_numpy(dtype=float),
    )


def _territory_metrics_from_unique(
    unique_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Agrega las observaciones unicas por provincia."""
    records: list[dict[str, Any]] = []
    for (territory_id, territory_name), group in unique_predictions.groupby(
        ["territory_id", "territory_name"],
        observed=True,
        sort=True,
    ):
        metrics = _metrics_from_group(group)
        records.append(
            {
                "territory_id": str(territory_id),
                "territory_name": str(territory_name),
                "validation_mae": float(metrics["MAE"]),
                "validation_rmse": float(metrics["RMSE"]),
                "validation_wape_pct": float(metrics["WAPE_pct"]),
                "validation_bias": float(metrics["mean_bias"]),
                "validation_rows": int(metrics["rows"]),
            }
        )
    return pd.DataFrame.from_records(records)


def calculate_territory_validation_metrics(
    dataframe: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Calcula MAE, RMSE, WAPE, sesgo y n por provincia."""
    unique = prepare_baseline_validation_predictions(dataframe, config)
    return _territory_metrics_from_unique(unique)


def get_territory_validation_metrics(
    territory_id: str,
    *,
    dataframe: pd.DataFrame | None = None,
    config: Mapping[str, Any] | None = None,
    config_path: Path = CONFIG_PATH,
) -> TerritoryValidationMetrics:
    """Obtiene las metricas historicas del baseline para una provincia."""
    effective_config = _effective_config(config, config_path)
    predictions = (
        load_validation_predictions()
        if dataframe is None
        else dataframe
    )
    metrics = calculate_territory_validation_metrics(
        predictions,
        effective_config,
    )
    requested_id = str(territory_id).strip()
    territory = metrics.loc[metrics["territory_id"].eq(requested_id)]
    if territory.empty:
        raise InvalidTerritoryError(
            f"No hay predicciones evaluables para '{requested_id}'."
        )
    return TerritoryValidationMetrics(**territory.iloc[0].to_dict())


def reconcile_official_pooled_metrics(
    unique_predictions: pd.DataFrame,
    official_metrics: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    rtol: float = 1e-9,
    atol: float = 1e-6,
) -> dict[str, float | int]:
    """Comprueba el agregado reconstruido contra el CSV oficial vigente."""
    baseline_id, _ = _dashboard_contract(config)
    required = {
        "evaluation_split",
        "model",
        "rows",
        "MAE",
        "RMSE",
        "WAPE_pct",
        "mean_bias",
    }
    missing = required.difference(official_metrics.columns)
    if missing:
        raise MetricsReconciliationError(
            "Faltan columnas en las metricas oficiales: "
            + ", ".join(sorted(missing))
        )

    official = official_metrics.loc[
        official_metrics["evaluation_split"].eq("validation_pooled")
        & official_metrics["model"].eq(baseline_id)
    ]
    if len(official) != 1:
        raise MetricsReconciliationError(
            "No existe una unica fila pooled oficial para el baseline."
        )

    calculated = _metrics_from_group(unique_predictions)
    official_row = official.iloc[0]
    mapping = {
        "rows": "rows",
        "MAE": "MAE",
        "RMSE": "RMSE",
        "WAPE_pct": "WAPE_pct",
        "mean_bias": "mean_bias",
    }
    differences: list[str] = []
    for calculated_name, official_name in mapping.items():
        actual_value = float(calculated[calculated_name])
        expected_value = float(official_row[official_name])
        if not np.isclose(
            actual_value,
            expected_value,
            rtol=rtol,
            atol=atol,
            equal_nan=True,
        ):
            differences.append(
                f"{calculated_name}: {actual_value} != {expected_value}"
            )
    if differences:
        raise MetricsReconciliationError(
            "Las metricas reconstruidas no coinciden: " + "; ".join(differences)
        )
    return calculated


def _single_lineage_value(dataframe: pd.DataFrame, column: str) -> str:
    """Obtiene un valor unico de lineage, no nulo ni vacio."""
    if column not in dataframe.columns or dataframe[column].isna().any():
        raise LineageMismatchError(
            f"{column} no contiene un lineage completo."
        )
    values = dataframe[column].astype(str)
    if values.str.strip().eq("").any() or values.nunique() != 1:
        raise LineageMismatchError(
            f"{column} no identifica un unico lineage valido."
        )
    return str(values.iloc[0])


def validate_lineage_compatibility(
    gold: pd.DataFrame,
    unique_predictions: pd.DataFrame,
    config: Mapping[str, Any],
) -> DashboardLineage:
    """Valida el snapshot comun y conserva lineage de cada etapa."""
    gold_snapshot = _single_lineage_value(gold, "source_snapshot_id")
    validation_snapshot = _single_lineage_value(
        unique_predictions,
        "source_snapshot_id",
    )
    if gold_snapshot != validation_snapshot:
        raise LineageMismatchError(
            "Gold y predicciones proceden de source_snapshot_id distintos."
        )

    source = _required_mapping(config, "source_dataset")
    modeling = _required_mapping(config, "modeling_dataset")
    gold_path = resolve_project_path(str(source.get("path", "")))
    configured_validation_path = resolve_project_path(
        str(modeling.get("path", ""))
    )
    artifact_dataset_path = _single_lineage_value(
        unique_predictions,
        "dataset_path",
    )
    if resolve_project_path(artifact_dataset_path) != configured_validation_path:
        raise LineageMismatchError(
            "Las predicciones no declaran el modeling_dataset configurado."
        )

    return DashboardLineage(
        source_snapshot_id=gold_snapshot,
        gold_pipeline_run_id=_single_lineage_value(gold, "pipeline_run_id"),
        gold_data_version=_single_lineage_value(gold, "data_version"),
        validation_pipeline_run_id=_single_lineage_value(
            unique_predictions,
            "pipeline_run_id",
        ),
        validation_data_version=_single_lineage_value(
            unique_predictions,
            "data_version",
        ),
        gold_dataset_path=str(gold_path.relative_to(PROJECT_ROOT)),
        validation_dataset_path=str(
            configured_validation_path.relative_to(PROJECT_ROOT)
        ),
        validation_predictions_path=str(
            PREDICTIONS_PATH.relative_to(PROJECT_ROOT)
        ),
        validation_metrics_path=str(METRICS_PATH.relative_to(PROJECT_ROOT)),
    )


def build_dashboard_context(
    territory_id: str,
    *,
    history_months: int | None = None,
    gold: pd.DataFrame | None = None,
    predictions: pd.DataFrame | None = None,
    official_metrics: pd.DataFrame | None = None,
    config: Mapping[str, Any] | None = None,
    config_path: Path = CONFIG_PATH,
) -> DashboardContext:
    """Construye contexto provincial y valida metricas y lineage."""
    effective_config = _effective_config(config, config_path)
    gold_source = (
        load_gold_history(effective_config)
        if gold is None
        else gold
    )
    prepared_gold = _prepare_gold(gold_source)
    prediction_source = (
        load_validation_predictions()
        if predictions is None
        else predictions
    )
    unique_predictions = prepare_baseline_validation_predictions(
        prediction_source,
        effective_config,
    )
    official_source = (
        load_official_validation_metrics()
        if official_metrics is None
        else official_metrics
    )
    reconcile_official_pooled_metrics(
        unique_predictions,
        official_source,
        effective_config,
    )
    lineage = validate_lineage_compatibility(
        prepared_gold,
        unique_predictions,
        effective_config,
    )
    history = get_territory_history(
        territory_id,
        months=history_months,
        dataframe=prepared_gold,
        config=effective_config,
    )
    territory_table = _territory_metrics_from_unique(unique_predictions)
    requested_id = str(territory_id).strip()
    territory = territory_table.loc[
        territory_table["territory_id"].eq(requested_id)
    ]
    if territory.empty:
        raise InvalidTerritoryError(
            f"No hay predicciones evaluables para '{requested_id}'."
        )
    metrics = TerritoryValidationMetrics(**territory.iloc[0].to_dict())
    history_name = str(history["territory_name"].iloc[0])
    if history_name != metrics.territory_name:
        raise DashboardDataError(
            "El nombre territorial no coincide entre Gold y validacion."
        )
    return DashboardContext(
        territory_id=requested_id,
        territory_name=metrics.territory_name,
        history=history,
        validation_metrics=metrics,
        lineage=lineage,
    )
