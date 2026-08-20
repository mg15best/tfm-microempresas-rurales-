"""Datos B5 de operacion y evidencia canonica para el frontal."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml

from src.models.evaluate_v2 import calculate_metrics as calculate_v2_metrics
from src.models.freeze_ets_v2_predictions import (
    ARTIFACT_COLUMNS,
    KEY_COLUMNS,
    logical_prediction_sha256,
)
from src.models.modeling_v2_common import (
    MODELING_V2_CONFIG_PATH,
    PROJECT_ROOT,
    load_modeling_v2_config,
)


__all__ = [
    "CanonicalValidationBundle",
    "DashboardContext",
    "DashboardDataError",
    "DashboardLineage",
    "InvalidTerritoryError",
    "PreparedCanonicalValidation",
    "TerritoryValidationMetrics",
    "build_dashboard_context",
    "calculate_operational_validation_metrics",
    "calculate_territory_validation_metrics",
    "get_territory_history",
    "get_territory_validation_metrics",
    "load_canonical_validation_bundle",
    "load_prepared_canonical_validation",
    "load_gold_history",
    "prepare_canonical_validation",
    "validate_b5_lifecycle",
    "validate_canonical_validation_bundle",
]

SELECTED_MODEL_ID = "holt_winters_additive_damped_v1"
BASELINE_ID = "seasonal_naive_lag_12"
SELECTION_STATUS = "provisional_validation_champion"
EVIDENCE_SCOPE = "canonical_rolling_validation"
PREDICTION_COLUMN = "operational_prediction"
CUTOFF_POLICY_ID = "conservative_target_lag_3_v1"

CANONICAL_ARTIFACT_PATH = (
    "data/model_outputs/ets_v2_rolling_validation_predictions.parquet"
)
CANONICAL_METADATA_PATH = (
    "data/metadata/ets_v2_rolling_validation_predictions.metadata.yml"
)
CANONICAL_ARTIFACT_SHA256 = (
    "7f8a5a44c1204b7c1bfdc5f840e1a58939f8d348ff9e97977dd5d1b0f39fd9c7"
)
CANONICAL_METADATA_SHA256 = (
    "93bf81335aa18d3eed2ba2a0ef1c0d20282d858b0a1e96ec9c6cd5d39a2c9e17"
)
CANONICAL_LOGICAL_SHA256 = (
    "81515245010068b764eb27ba8bc6e32dffcbc5c590b6b823017d98f0121e8f04"
)
CANONICAL_GENERATOR_COMMIT = "3465675af079dc2c9dcfc2f89596c62144e49c76"
CANONICAL_GITHUB_RUN_ID = "32359193026"
CANONICAL_ROWS = 1750
CANONICAL_TERRITORIES = 50
CANONICAL_ORIGINS = 35
CANONICAL_FOLD_IDS = ("validation_1", "validation_2", "validation_3")

# Compatibilidad de import E4 hasta B5-B3; B5 no consume estos artefactos.
LEGACY_PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "data"
    / "model_outputs"
    / "model_selection_validation_predictions.parquet"
)
LEGACY_METRICS_PATH = (
    PROJECT_ROOT / "data" / "metadata" / "model_selection_metrics.csv"
)

HISTORY_COLUMNS = [
    "territory_id",
    "territory_name",
    "month_id",
    "date_month",
    "overnight_stays_total",
    "covid_period",
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
    """Un recurso requerido no cumple el contrato productivo B5."""


class InvalidTerritoryError(DashboardDataError):
    """El territorio solicitado no existe en una fuente requerida."""


class CanonicalArtifactError(DashboardDataError):
    """El bundle de evaluacion no es el artifact oficial congelado."""


class LineageMismatchError(DashboardDataError):
    """Una fuente no declara un lineage completo y coherente."""


@dataclass(frozen=True)
class CanonicalValidationBundle:
    """Artifact ETS oficial validado sin depender del repositorio Git."""

    predictions: pd.DataFrame
    metadata: Mapping[str, Any]
    artifact_path: Path
    metadata_path: Path
    artifact_sha256: str
    metadata_sha256: str
    logical_prediction_sha256: str
    generator_commit_sha: str
    github_run_id: str
    evaluation_config_sha256: str
    evaluation_source_snapshot_ids: tuple[str, ...]
    evaluation_data_versions: tuple[str, ...]
    selected_model_id: str
    baseline_id: str
    cutoff_policy_id: str
    evidence_scope: str


@dataclass(frozen=True)
class TerritoryValidationMetrics:
    """Evidencia rolling canonica del sistema ETS con fallback operativo."""

    territory_id: str
    territory_name: str
    validation_mae: float
    validation_rmse: float
    validation_wape_pct: float
    validation_bias: float
    validation_rows: int
    selected_model_id: str
    selection_status: str
    evidence_scope: str
    prediction_column: str
    candidate_available_rows: int
    availability_fallback_rows: int


@dataclass(frozen=True, init=False)
class PreparedCanonicalValidation:
    """Evidencia validada una vez y agregados territoriales inmutables.

    La construccion directa se bloquea deliberadamente: la unica frontera
    publica es :func:`prepare_canonical_validation`, que verifica el bundle
    completo antes de crear este recurso cacheable.
    """

    bundle: CanonicalValidationBundle
    territory_metrics: tuple[TerritoryValidationMetrics, ...]

    def __init__(self) -> None:
        raise TypeError(
            "Use prepare_canonical_validation() para validar la evidencia."
        )

    @classmethod
    def _from_validated(
        cls,
        bundle: CanonicalValidationBundle,
        metrics: tuple[TerritoryValidationMetrics, ...],
    ) -> PreparedCanonicalValidation:
        instance = object.__new__(cls)
        object.__setattr__(instance, "bundle", bundle)
        object.__setattr__(instance, "territory_metrics", metrics)
        return instance


@dataclass(frozen=True)
class DashboardLineage:
    """Lineages operacional actual y de evaluacion historica separados."""

    operational_source_snapshot_id: str
    operational_pipeline_run_id: str
    operational_data_version: str
    operational_dataset_path: str
    evaluation_artifact_path: str
    evaluation_metadata_path: str
    evaluation_artifact_sha256: str
    evaluation_metadata_sha256: str
    evaluation_logical_prediction_sha256: str
    evaluation_generator_commit_sha: str
    evaluation_github_run_id: str
    evaluation_config_sha256: str
    evaluation_source_snapshot_ids: tuple[str, ...]
    evaluation_data_versions: tuple[str, ...]
    evaluation_scope: str


@dataclass(frozen=True)
class DashboardContext:
    """Historico operacional y evidencia canonica de una provincia."""

    territory_id: str
    territory_name: str
    history: pd.DataFrame
    validation_metrics: TerritoryValidationMetrics
    lineage: DashboardLineage

    def to_export_frames(self) -> dict[str, pd.DataFrame]:
        return {
            "history": self.history.copy(),
            "validation_metrics": pd.DataFrame(
                [asdict(self.validation_metrics)]
            ),
            "lineage": pd.DataFrame([asdict(self.lineage)]),
        }


def _required_mapping(
    mapping: Mapping[str, Any],
    key: str,
) -> Mapping[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, Mapping):
        raise DashboardDataError(f"Falta la seccion de configuracion '{key}'.")
    return value


def _resolve_path(path_value: str, project_root: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else project_root / path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_parquet(path: Path, artifact_name: str) -> pd.DataFrame:
    if not path.exists():
        raise DashboardDataError(f"No se encontro {artifact_name}: {path}.")
    try:
        return pd.read_parquet(path)
    except (OSError, ValueError) as error:
        raise DashboardDataError(
            f"No se pudo leer {artifact_name}: {error}"
        ) from error


def validate_b5_lifecycle(config: Mapping[str, Any]) -> None:
    """Separa baseline historico de champion operacional provisional."""

    methodology = _required_mapping(config, "methodology")
    baseline = _required_mapping(config, "baseline")
    ets = _required_mapping(config, "ets_candidate")
    selection = _required_mapping(config, "operational_selection")
    fallback = _required_mapping(selection, "fallback")
    provenance = _required_mapping(config, "provenance")
    expected = {
        "methodology.version": (methodology.get("version"), "2.0.0-b5"),
        "methodology.status": (
            methodology.get("status"),
            "provisional_validation_champion_selected",
        ),
        "baseline.id": (baseline.get("id"), BASELINE_ID),
        "baseline.role": (
            baseline.get("role"),
            "historical_validation_baseline",
        ),
        "ets.screening_status": (
            ets.get("screening_status"),
            "passed_screening",
        ),
        "selection.selected_model_id": (
            selection.get("selected_model_id"),
            SELECTED_MODEL_ID,
        ),
        "selection.status": (selection.get("status"), SELECTION_STATUS),
        "selection.evidence_scope": (
            selection.get("evidence_scope"),
            EVIDENCE_SCOPE,
        ),
        "selection.independent_test_confirmed": (
            selection.get("independent_test_confirmed"),
            False,
        ),
        "fallback.model_id": (fallback.get("model_id"), BASELINE_ID),
        "fallback.policy": (fallback.get("policy"), "availability_only"),
        "fallback.performance_based": (
            fallback.get("performance_based"),
            False,
        ),
        "provenance.config_version": (
            provenance.get("config_version"),
            "modeling_v2_config_b5_v1",
        ),
        "provenance.code_contract_version": (
            provenance.get("code_contract_version"),
            "point_in_time_v2_b5_v1",
        ),
    }
    mismatches = {
        name: values for name, values in expected.items() if values[0] != values[1]
    }
    if mismatches:
        raise DashboardDataError(f"Lifecycle B5 incoherente: {mismatches}.")


def _canonical_contract(config: Mapping[str, Any]) -> Mapping[str, Any]:
    validate_b5_lifecycle(config)
    selection = _required_mapping(config, "operational_selection")
    contract = _required_mapping(selection, "canonical_validation")
    expected: dict[str, Any] = {
        "artifact_path": CANONICAL_ARTIFACT_PATH,
        "metadata_path": CANONICAL_METADATA_PATH,
        "artifact_sha256": CANONICAL_ARTIFACT_SHA256,
        "metadata_sha256": CANONICAL_METADATA_SHA256,
        "logical_prediction_sha256": CANONICAL_LOGICAL_SHA256,
        "generator_commit_sha": CANONICAL_GENERATOR_COMMIT,
        "github_run_id": CANONICAL_GITHUB_RUN_ID,
        "official_canonical_artifact": True,
        "expected_rows": CANONICAL_ROWS,
        "expected_territories": CANONICAL_TERRITORIES,
        "expected_origins": CANONICAL_ORIGINS,
        "expected_fold_ids": list(CANONICAL_FOLD_IDS),
        "prediction_column": PREDICTION_COLUMN,
    }
    mismatches = {
        key: (contract.get(key), expected_value)
        for key, expected_value in expected.items()
        if contract.get(key) != expected_value
    }
    if selection.get("canonical_artifact_path") != CANONICAL_ARTIFACT_PATH:
        mismatches["operational_selection.canonical_artifact_path"] = (
            selection.get("canonical_artifact_path"),
            CANONICAL_ARTIFACT_PATH,
        )
    if selection.get("canonical_metadata_path") != CANONICAL_METADATA_PATH:
        mismatches["operational_selection.canonical_metadata_path"] = (
            selection.get("canonical_metadata_path"),
            CANONICAL_METADATA_PATH,
        )
    if mismatches:
        raise CanonicalArtifactError(
            f"Anclaje canonico B5 no soportado: {mismatches}."
        )
    return contract


def _load_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise CanonicalArtifactError(f"No se encontro metadata canonica: {path}.")
    try:
        with path.open("r", encoding="utf-8") as stream:
            metadata = yaml.safe_load(stream)
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise CanonicalArtifactError(
            f"No se pudo leer metadata canonica: {error}"
        ) from error
    if not isinstance(metadata, dict):
        raise CanonicalArtifactError("La metadata canonica no contiene un objeto.")
    return metadata


def _metadata_contract(
    metadata: Mapping[str, Any],
    *,
    artifact_sha256: str,
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    artifact = _required_mapping(metadata, "artifact")
    provenance = _required_mapping(metadata, "provenance")
    github = _required_mapping(metadata, "github")
    gold = _required_mapping(provenance, "gold")
    expected = {
        "artifact.logical_path": (
            artifact.get("logical_path"),
            CANONICAL_ARTIFACT_PATH,
        ),
        "artifact.metadata_logical_path": (
            artifact.get("metadata_logical_path"),
            CANONICAL_METADATA_PATH,
        ),
        "artifact.official_canonical_artifact": (
            artifact.get("official_canonical_artifact"),
            True,
        ),
        "artifact.generating_commit_sha": (
            artifact.get("generating_commit_sha"),
            CANONICAL_GENERATOR_COMMIT,
        ),
        "artifact.file_sha256": (
            artifact.get("file_sha256"),
            artifact_sha256,
        ),
        "artifact.logical_prediction_sha256": (
            artifact.get("logical_prediction_sha256"),
            CANONICAL_LOGICAL_SHA256,
        ),
        "artifact.row_count": (artifact.get("row_count"), CANONICAL_ROWS),
        "artifact.territory_count": (
            artifact.get("territory_count"),
            CANONICAL_TERRITORIES,
        ),
        "artifact.origin_count": (
            artifact.get("origin_count"),
            CANONICAL_ORIGINS,
        ),
        "artifact.fold_ids": (
            artifact.get("fold_ids"),
            list(CANONICAL_FOLD_IDS),
        ),
        "github.github_sha": (
            github.get("github_sha"),
            CANONICAL_GENERATOR_COMMIT,
        ),
        "github.github_run_id": (
            str(github.get("github_run_id")),
            CANONICAL_GITHUB_RUN_ID,
        ),
        "provenance.ets_model_id": (
            provenance.get("ets_model_id"),
            SELECTED_MODEL_ID,
        ),
        "provenance.baseline_id": (
            provenance.get("baseline_id"),
            BASELINE_ID,
        ),
        "provenance.cutoff_policy_id": (
            provenance.get("cutoff_policy_id"),
            CUTOFF_POLICY_ID,
        ),
    }
    mismatches = {
        name: values for name, values in expected.items() if values[0] != values[1]
    }
    for name, values in (
        ("gold.source_snapshot_ids", gold.get("source_snapshot_ids")),
        ("gold.data_versions", gold.get("data_versions")),
    ):
        if not isinstance(values, list) or not values or any(
            not str(value).strip() for value in values
        ):
            mismatches[name] = (values, "non-empty list")
    config_sha = str(provenance.get("config_sha256", ""))
    if len(config_sha) != 64 or any(
        character not in "0123456789abcdef" for character in config_sha
    ):
        mismatches["provenance.config_sha256"] = (config_sha, "sha256")
    if mismatches:
        raise CanonicalArtifactError(
            f"Metadata canonica incompatible: {mismatches}."
        )
    return artifact, provenance, gold


def calculate_operational_validation_metrics(
    predictions: pd.DataFrame,
) -> dict[str, float | int]:
    """Reconstruye metricas V2 sobre operational_prediction."""

    required = {"actual", PREDICTION_COLUMN}
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise DashboardDataError(
            "Artifact sin columnas de metricas: " + ", ".join(missing)
        )
    view = predictions[["actual", PREDICTION_COLUMN]].rename(
        columns={PREDICTION_COLUMN: "prediction"}
    )
    try:
        return calculate_v2_metrics(view)
    except (TypeError, ValueError) as error:
        raise DashboardDataError(
            f"No se pudieron reconstruir metricas V2: {error}"
        ) from error


def _validate_predictions(
    predictions: pd.DataFrame,
    metadata: Mapping[str, Any],
) -> str:
    if list(predictions.columns) != ARTIFACT_COLUMNS:
        raise CanonicalArtifactError("El Parquet no tiene el schema B4C exacto.")
    if len(predictions) != CANONICAL_ROWS:
        raise CanonicalArtifactError("El Parquet no contiene 1.750 filas.")
    if predictions.duplicated(KEY_COLUMNS).any():
        raise CanonicalArtifactError("El Parquet contiene keys duplicadas.")
    if predictions[KEY_COLUMNS].isna().any().any():
        raise CanonicalArtifactError("El Parquet contiene keys nulas.")
    if predictions["territory_id"].nunique() != CANONICAL_TERRITORIES:
        raise CanonicalArtifactError("El Parquet no contiene 50 territorios.")
    if predictions["target_month_id"].nunique() != CANONICAL_ORIGINS:
        raise CanonicalArtifactError("El Parquet no contiene 35 origins.")
    if set(predictions["fold_id"].astype(str)) != set(CANONICAL_FOLD_IDS):
        raise CanonicalArtifactError("Los folds del Parquet no son canonicos.")
    if set(predictions["model_id"].astype(str)) != {SELECTED_MODEL_ID}:
        raise CanonicalArtifactError("El Parquet no identifica el ETS seleccionado.")
    if set(predictions["cutoff_policy_id"].astype(str)) != {CUTOFF_POLICY_ID}:
        raise CanonicalArtifactError("El Parquet no usa el cutoff B5.")
    if predictions.groupby("territory_id")["territory_name"].nunique().gt(1).any():
        raise CanonicalArtifactError("Un territorio tiene varios nombres.")

    numeric = predictions[
        ["actual", "baseline_prediction", PREDICTION_COLUMN]
    ].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(numeric).all().all():
        raise CanonicalArtifactError("El Parquet contiene valores no finitos.")
    if numeric.lt(0).any().any():
        raise CanonicalArtifactError("El Parquet contiene valores negativos.")

    available = predictions["candidate_available"].astype(bool)
    fallback = predictions["availability_fallback_used"].astype(bool)
    if not fallback.eq(~available).all():
        raise CanonicalArtifactError("Candidate y fallback no son complementarios.")
    if not np.allclose(
        predictions.loc[fallback, PREDICTION_COLUMN],
        predictions.loc[fallback, "baseline_prediction"],
        rtol=0,
        atol=0,
    ):
        raise CanonicalArtifactError("El fallback no usa baseline_prediction.")
    if not np.allclose(
        predictions.loc[available, PREDICTION_COLUMN],
        predictions.loc[available, "ets_candidate_prediction"],
        rtol=0,
        atol=0,
    ):
        raise CanonicalArtifactError("El point disponible no usa el ETS.")

    invariants = _required_mapping(metadata, "invariants")
    expected_invariants = {
        "rows": len(predictions),
        "unique_keys": len(predictions.drop_duplicates(KEY_COLUMNS)),
        "territories": predictions["territory_id"].nunique(),
        "origins": predictions["target_month_id"].nunique(),
        "candidate_available_rows": int(available.sum()),
        "availability_fallback_rows": int(fallback.sum()),
    }
    mismatches = {
        key: (invariants.get(key), expected)
        for key, expected in expected_invariants.items()
        if invariants.get(key) != expected
    }
    if mismatches:
        raise CanonicalArtifactError(
            f"Metadata y Parquet no comparten invariantes: {mismatches}."
        )

    try:
        logical = logical_prediction_sha256(predictions)
    except (AssertionError, TypeError, ValueError) as error:
        raise CanonicalArtifactError(
            f"No se pudo calcular el hash logico: {error}"
        ) from error
    if logical != CANONICAL_LOGICAL_SHA256:
        raise CanonicalArtifactError(f"Hash logico no canonico: {logical}.")

    metrics = calculate_operational_validation_metrics(predictions)
    evaluation = _required_mapping(metadata, "evaluation")
    pooled = _required_mapping(evaluation, "pooled")
    expected_metrics = {
        "n": pooled.get("n"),
        "MAE": pooled.get("candidate_MAE"),
        "RMSE": pooled.get("candidate_RMSE"),
        "WAPE_pct": pooled.get("candidate_WAPE_pct"),
        "bias": pooled.get("candidate_bias"),
    }
    metric_mismatches = {
        key: (metrics[key], expected)
        for key, expected in expected_metrics.items()
        if not np.isclose(
            float(metrics[key]),
            float(expected),
            rtol=1e-12,
            atol=1e-9,
        )
    }
    if metric_mismatches:
        raise CanonicalArtifactError(
            f"Metricas metadata/Parquet incompatibles: {metric_mismatches}."
        )
    return logical


def load_canonical_validation_bundle(
    config: Mapping[str, Any] | None = None,
    *,
    config_path: Path = MODELING_V2_CONFIG_PATH,
    project_root: Path = PROJECT_ROOT,
) -> CanonicalValidationBundle:
    """Carga y valida el bundle oficial sin refit, freeze ni acceso a Git."""

    effective = (
        load_modeling_v2_config(config_path) if config is None else config
    )
    contract = _canonical_contract(effective)
    artifact_path = _resolve_path(str(contract["artifact_path"]), project_root)
    metadata_path = _resolve_path(str(contract["metadata_path"]), project_root)
    if not artifact_path.exists():
        raise CanonicalArtifactError(
            f"No se encontro artifact canonico: {artifact_path}."
        )
    if not metadata_path.exists():
        raise CanonicalArtifactError(
            f"No se encontro metadata canonica: {metadata_path}."
        )

    artifact_sha = _sha256_file(artifact_path)
    metadata_sha = _sha256_file(metadata_path)
    if artifact_sha != CANONICAL_ARTIFACT_SHA256:
        raise CanonicalArtifactError(
            f"SHA-256 del artifact no canonico: {artifact_sha}."
        )
    if metadata_sha != CANONICAL_METADATA_SHA256:
        raise CanonicalArtifactError(
            f"SHA-256 de metadata no canonico: {metadata_sha}."
        )

    metadata = _load_metadata(metadata_path)
    _, provenance, gold = _metadata_contract(
        metadata,
        artifact_sha256=artifact_sha,
    )
    predictions = _read_parquet(artifact_path, "el artifact canonico")
    logical = _validate_predictions(predictions, metadata)
    return CanonicalValidationBundle(
        predictions=predictions,
        metadata=metadata,
        artifact_path=artifact_path,
        metadata_path=metadata_path,
        artifact_sha256=artifact_sha,
        metadata_sha256=metadata_sha,
        logical_prediction_sha256=logical,
        generator_commit_sha=CANONICAL_GENERATOR_COMMIT,
        github_run_id=CANONICAL_GITHUB_RUN_ID,
        evaluation_config_sha256=str(provenance["config_sha256"]),
        evaluation_source_snapshot_ids=tuple(
            str(value) for value in gold["source_snapshot_ids"]
        ),
        evaluation_data_versions=tuple(
            str(value) for value in gold["data_versions"]
        ),
        selected_model_id=SELECTED_MODEL_ID,
        baseline_id=BASELINE_ID,
        cutoff_policy_id=CUTOFF_POLICY_ID,
        evidence_scope=EVIDENCE_SCOPE,
    )


def validate_canonical_validation_bundle(
    bundle: CanonicalValidationBundle,
) -> None:
    """Revalida en memoria un bundle inyectado sin I/O, Git ni refit."""

    if not isinstance(bundle, CanonicalValidationBundle):
        raise CanonicalArtifactError(
            "Se requiere CanonicalValidationBundle; un DataFrame no es evidencia."
        )
    expected = {
        "artifact_sha256": CANONICAL_ARTIFACT_SHA256,
        "metadata_sha256": CANONICAL_METADATA_SHA256,
        "logical_prediction_sha256": CANONICAL_LOGICAL_SHA256,
        "generator_commit_sha": CANONICAL_GENERATOR_COMMIT,
        "github_run_id": CANONICAL_GITHUB_RUN_ID,
        "selected_model_id": SELECTED_MODEL_ID,
        "baseline_id": BASELINE_ID,
        "cutoff_policy_id": CUTOFF_POLICY_ID,
        "evidence_scope": EVIDENCE_SCOPE,
    }
    mismatches = {
        field: (getattr(bundle, field), value)
        for field, value in expected.items()
        if getattr(bundle, field) != value
    }
    if mismatches:
        raise CanonicalArtifactError(
            f"Identidad del bundle inyectado incompatible: {mismatches}."
        )
    _, provenance, gold = _metadata_contract(
        bundle.metadata,
        artifact_sha256=bundle.artifact_sha256,
    )
    logical = _validate_predictions(bundle.predictions, bundle.metadata)
    lineage = (
        str(provenance["config_sha256"]),
        tuple(str(value) for value in gold["source_snapshot_ids"]),
        tuple(str(value) for value in gold["data_versions"]),
    )
    expected_lineage = (
        bundle.evaluation_config_sha256,
        bundle.evaluation_source_snapshot_ids,
        bundle.evaluation_data_versions,
    )
    if logical != bundle.logical_prediction_sha256 or lineage != expected_lineage:
        raise CanonicalArtifactError(
            "Contenido y provenance del bundle inyectado son incompatibles."
        )


def load_gold_history(config: Mapping[str, Any]) -> pd.DataFrame:
    """Carga la Gold operacional declarada por modeling_v2_config.yml."""

    source = _required_mapping(config, "source")
    path_text = source.get("path")
    if not isinstance(path_text, str) or not path_text.strip():
        raise DashboardDataError("Falta source.path en config V2.")
    return _read_parquet(_resolve_path(path_text, PROJECT_ROOT), "la Gold")


def load_validation_predictions() -> pd.DataFrame:
    """Compatibilidad E4 temporal; no es evidencia vigente B5."""

    return _read_parquet(
        LEGACY_PREDICTIONS_PATH,
        "las predicciones historicas E4",
    )


def load_official_validation_metrics() -> pd.DataFrame:
    """Compatibilidad E4 temporal; no es evidencia vigente B5."""

    if not LEGACY_METRICS_PATH.exists():
        raise DashboardDataError(
            f"No se encontraron metricas historicas: {LEGACY_METRICS_PATH}."
        )
    try:
        return pd.read_csv(LEGACY_METRICS_PATH)
    except (OSError, ValueError) as error:
        raise DashboardDataError(
            f"No se pudieron leer metricas historicas: {error}"
        ) from error


def _prepare_gold(dataframe: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(dataframe, pd.DataFrame) or dataframe.empty:
        raise DashboardDataError("La Gold esta vacia o no es valida.")
    missing = sorted(REQUIRED_GOLD_COLUMNS.difference(dataframe.columns))
    if missing:
        raise DashboardDataError("Faltan columnas Gold: " + ", ".join(missing))
    result = dataframe.loc[:, sorted(REQUIRED_GOLD_COLUMNS)].copy()
    try:
        result["date_month"] = pd.to_datetime(
            result["date_month"],
            errors="raise",
        )
    except (TypeError, ValueError) as error:
        raise DashboardDataError("date_month contiene fechas invalidas.") from error
    if result["date_month"].isna().any():
        raise DashboardDataError("date_month contiene valores nulos.")
    result["month_id"] = result["month_id"].astype("string")
    if not result["month_id"].eq(
        result["date_month"].dt.strftime("%Y-%m")
    ).all():
        raise DashboardDataError("month_id/date_month son incompatibles.")
    if result.duplicated(["territory_id", "month_id"]).any():
        raise DashboardDataError("La Gold contiene keys duplicadas.")
    return result


def _single_lineage_value(dataframe: pd.DataFrame, column: str) -> str:
    if column not in dataframe.columns or dataframe[column].isna().any():
        raise LineageMismatchError(f"{column} no contiene lineage completo.")
    values = dataframe[column].astype(str)
    if values.str.strip().eq("").any() or values.nunique() != 1:
        raise LineageMismatchError(f"{column} no identifica un lineage unico.")
    return str(values.iloc[0])


def get_territory_history(
    territory_id: str,
    *,
    months: int | None = None,
    latest_available_month_id: str | None = None,
    dataframe: pd.DataFrame | None = None,
    config: Mapping[str, Any] | None = None,
    config_path: Path = MODELING_V2_CONFIG_PATH,
) -> pd.DataFrame:
    """Obtiene historia territorial hasta un cutoff operacional explicito.

    Cuando ``months`` limita la salida, la ventana natural termina en el
    cutoff, no en el ultimo mes presente en la Gold. Los gaps se conservan.
    """

    effective = (
        load_modeling_v2_config(config_path) if config is None else config
    )
    source = _prepare_gold(
        load_gold_history(effective) if dataframe is None else dataframe
    )
    if months is not None and (
        not isinstance(months, int) or isinstance(months, bool) or months < 1
    ):
        raise ValueError("months debe ser un entero positivo o None.")
    cutoff: pd.Period | None = None
    if latest_available_month_id is not None:
        try:
            cutoff = pd.Period(latest_available_month_id, freq="M")
        except (TypeError, ValueError) as error:
            raise ValueError(
                "latest_available_month_id debe identificar un mes valido."
            ) from error
        if pd.isna(cutoff):
            raise ValueError(
                "latest_available_month_id debe identificar un mes valido."
            )
    requested = str(territory_id).strip()
    history = source.loc[
        source["territory_id"].astype(str).eq(requested),
        HISTORY_COLUMNS,
    ].sort_values("date_month")
    if history.empty:
        raise InvalidTerritoryError(f"No existe territory_id '{requested}'.")
    history_periods = history["date_month"].dt.to_period("M")
    if cutoff is not None:
        history = history.loc[history_periods.le(cutoff)]
        if history.empty:
            raise DashboardDataError(
                f"No hay historico para '{requested}' hasta {cutoff}."
            )
    if history["territory_name"].nunique(dropna=False) != 1:
        raise DashboardDataError("El territorio tiene varios nombres en Gold.")
    if months is not None:
        last = (
            cutoff
            if cutoff is not None
            else history["date_month"].max().to_period("M")
        )
        first = last - (months - 1)
        history = history.loc[history["date_month"].dt.to_period("M").ge(first)]
    return history.reset_index(drop=True)


def calculate_territory_validation_metrics(
    predictions: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Agrega operational_prediction por provincia con definiciones V2."""

    validate_b5_lifecycle(config)
    records: list[dict[str, Any]] = []
    for (territory_id, territory_name), group in predictions.groupby(
        ["territory_id", "territory_name"],
        observed=True,
        sort=True,
    ):
        metrics = calculate_operational_validation_metrics(group)
        records.append(
            {
                "territory_id": str(territory_id),
                "territory_name": str(territory_name),
                "validation_mae": float(metrics["MAE"]),
                "validation_rmse": float(metrics["RMSE"]),
                "validation_wape_pct": float(metrics["WAPE_pct"]),
                "validation_bias": float(metrics["bias"]),
                "validation_rows": int(metrics["n"]),
                "selected_model_id": SELECTED_MODEL_ID,
                "selection_status": SELECTION_STATUS,
                "evidence_scope": EVIDENCE_SCOPE,
                "prediction_column": PREDICTION_COLUMN,
                "candidate_available_rows": int(
                    group["candidate_available"].sum()
                ),
                "availability_fallback_rows": int(
                    group["availability_fallback_used"].sum()
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def _freeze_validated_canonical_validation(
    bundle: CanonicalValidationBundle,
    config: Mapping[str, Any],
) -> PreparedCanonicalValidation:
    metrics = calculate_territory_validation_metrics(
        bundle.predictions,
        config,
    )
    frozen_metrics = tuple(
        TerritoryValidationMetrics(**row._asdict())
        for row in metrics.itertuples(index=False)
    )
    if len(frozen_metrics) != CANONICAL_TERRITORIES:
        raise CanonicalArtifactError(
            "La evidencia preparada no contiene 50 territorios."
        )
    return PreparedCanonicalValidation._from_validated(
        bundle,
        frozen_metrics,
    )


def prepare_canonical_validation(
    bundle: CanonicalValidationBundle,
    config: Mapping[str, Any],
) -> PreparedCanonicalValidation:
    """Valida un bundle inyectado y congela sus métricas territoriales."""

    validate_b5_lifecycle(config)
    validate_canonical_validation_bundle(bundle)
    return _freeze_validated_canonical_validation(bundle, config)


def load_prepared_canonical_validation(
    config: Mapping[str, Any],
) -> PreparedCanonicalValidation:
    """Carga/valida físicamente una vez y prepara evidencia cacheable."""

    validate_b5_lifecycle(config)
    bundle = load_canonical_validation_bundle(config)
    return _freeze_validated_canonical_validation(bundle, config)


def get_territory_validation_metrics(
    territory_id: str,
    *,
    evidence: PreparedCanonicalValidation | None = None,
    config: Mapping[str, Any] | None = None,
    config_path: Path = MODELING_V2_CONFIG_PATH,
) -> TerritoryValidationMetrics:
    effective = (
        load_modeling_v2_config(config_path) if config is None else config
    )
    prepared = evidence
    if prepared is None:
        prepared = load_prepared_canonical_validation(effective)
    if not isinstance(prepared, PreparedCanonicalValidation):
        raise CanonicalArtifactError(
            "Se requiere evidencia canonica preparada y validada."
        )
    requested = str(territory_id).strip()
    for metrics in prepared.territory_metrics:
        if metrics.territory_id == requested:
            return metrics
    raise InvalidTerritoryError(
        f"No hay evidencia canonica para '{requested}'."
    )


def _dashboard_lineage(
    gold: pd.DataFrame,
    bundle: CanonicalValidationBundle,
    config: Mapping[str, Any],
) -> DashboardLineage:
    source = _required_mapping(config, "source")
    return DashboardLineage(
        operational_source_snapshot_id=_single_lineage_value(
            gold,
            "source_snapshot_id",
        ),
        operational_pipeline_run_id=_single_lineage_value(
            gold,
            "pipeline_run_id",
        ),
        operational_data_version=_single_lineage_value(gold, "data_version"),
        operational_dataset_path=str(source["path"]),
        evaluation_artifact_path=CANONICAL_ARTIFACT_PATH,
        evaluation_metadata_path=CANONICAL_METADATA_PATH,
        evaluation_artifact_sha256=bundle.artifact_sha256,
        evaluation_metadata_sha256=bundle.metadata_sha256,
        evaluation_logical_prediction_sha256=(
            bundle.logical_prediction_sha256
        ),
        evaluation_generator_commit_sha=bundle.generator_commit_sha,
        evaluation_github_run_id=bundle.github_run_id,
        evaluation_config_sha256=bundle.evaluation_config_sha256,
        evaluation_source_snapshot_ids=bundle.evaluation_source_snapshot_ids,
        evaluation_data_versions=bundle.evaluation_data_versions,
        evaluation_scope=bundle.evidence_scope,
    )


def build_dashboard_context(
    territory_id: str,
    *,
    history_months: int | None = None,
    latest_available_month_id: str | None = None,
    gold: pd.DataFrame | None = None,
    canonical_bundle: CanonicalValidationBundle | None = None,
    prepared_validation: PreparedCanonicalValidation | None = None,
    config: Mapping[str, Any] | None = None,
    config_path: Path = MODELING_V2_CONFIG_PATH,
) -> DashboardContext:
    """Construye dashboard desde Gold actual y evidencia ETS canonica."""

    effective = (
        load_modeling_v2_config(config_path) if config is None else config
    )
    validate_b5_lifecycle(effective)
    gold_source = load_gold_history(effective) if gold is None else gold
    prepared_gold = _prepare_gold(gold_source)
    if canonical_bundle is not None and prepared_validation is not None:
        raise DashboardDataError(
            "No se puede inyectar bundle y evidencia preparada a la vez."
        )
    prepared = prepared_validation
    if prepared is None:
        prepared = (
            load_prepared_canonical_validation(effective)
            if canonical_bundle is None
            else prepare_canonical_validation(canonical_bundle, effective)
        )
    elif not isinstance(prepared, PreparedCanonicalValidation):
        raise CanonicalArtifactError(
            "Se requiere evidencia canonica preparada y validada."
        )
    bundle = prepared.bundle
    history = get_territory_history(
        territory_id,
        months=history_months,
        latest_available_month_id=latest_available_month_id,
        dataframe=prepared_gold,
        config=effective,
    )
    metrics = get_territory_validation_metrics(
        territory_id,
        evidence=prepared,
        config=effective,
    )
    history_name = str(history["territory_name"].iloc[0])
    if history_name != metrics.territory_name:
        raise DashboardDataError(
            "El nombre territorial difiere entre Gold y evidencia canonica."
        )
    return DashboardContext(
        territory_id=str(territory_id).strip(),
        territory_name=history_name,
        history=history,
        validation_metrics=metrics,
        lineage=_dashboard_lineage(prepared_gold, bundle, effective),
    )
