"""Congela las predicciones rolling-validation ETS B4 en Ubuntu canonico.

El artefacto separa el forecast ETS puro de la prediccion operacional. La
ruta de lectura/reconstruccion no ajusta modelos y es la que debe reutilizar
B5 para trabajar siempre sobre las mismas 1.750 predicciones.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import struct
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import patsy
import pyarrow
import scipy
import statsmodels
import yaml

try:
    from src.models.ets_v2 import build_ets_predictions
    from src.models.evaluate_v2 import (
        calculate_candidate_fold_metrics,
        calculate_candidate_origin_metrics,
        calculate_candidate_territory_metrics,
        calculate_paired_metrics,
        load_gold_history,
        reproduce_baseline_in_memory,
        screen_ets_candidate,
    )
    from src.models.modeling_v2_common import (
        PROJECT_ROOT,
    )
except ModuleNotFoundError:
    from ets_v2 import build_ets_predictions
    from evaluate_v2 import (
        calculate_candidate_fold_metrics,
        calculate_candidate_origin_metrics,
        calculate_candidate_territory_metrics,
        calculate_paired_metrics,
        load_gold_history,
        reproduce_baseline_in_memory,
        screen_ets_candidate,
    )
    from modeling_v2_common import PROJECT_ROOT


ARTIFACT_VERSION = "1.0.0-b4c"
ARTIFACT_FILENAME = "ets_v2_rolling_validation_predictions.parquet"
METADATA_FILENAME = "ets_v2_rolling_validation_predictions.metadata.yml"
DEFAULT_ARTIFACT_PATH = PROJECT_ROOT / "data" / "model_outputs" / ARTIFACT_FILENAME
DEFAULT_METADATA_PATH = PROJECT_ROOT / "data" / "metadata" / METADATA_FILENAME
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "data" / "metadata" / "modeling_v2_config.yml"
DEFAULT_REQUIREMENTS_PATH = PROJECT_ROOT / "requirements.txt"
CONFIG_REPOSITORY_PATH = "data/metadata/modeling_v2_config.yml"
REQUIREMENTS_REPOSITORY_PATH = "requirements.txt"

KEY_COLUMNS = ["fold_id", "territory_id", "target_month_id"]
SORT_COLUMNS = ["fold_id", "target_month_id", "territory_id"]
ARTIFACT_COLUMNS = [
    "fold_id",
    "territory_id",
    "territory_name",
    "target_month_id",
    "business_origin_month_id",
    "latest_available_month_id",
    "actual",
    "baseline_prediction",
    "ets_candidate_prediction",
    "ets_raw_prediction",
    "operational_prediction",
    "candidate_available",
    "availability_fallback_used",
    "fallback_reason",
    "clipping_applied",
    "training_start",
    "training_end",
    "training_rows",
    "observed_training_rows",
    "imputed_months_n",
    "model_id",
    "cutoff_policy_id",
]
STRING_COLUMNS = [
    "fold_id",
    "territory_id",
    "territory_name",
    "target_month_id",
    "business_origin_month_id",
    "latest_available_month_id",
    "fallback_reason",
    "training_start",
    "training_end",
    "model_id",
    "cutoff_policy_id",
]
FLOAT_COLUMNS = [
    "actual",
    "baseline_prediction",
    "ets_candidate_prediction",
    "ets_raw_prediction",
    "operational_prediction",
]
BOOL_COLUMNS = [
    "candidate_available",
    "availability_fallback_used",
    "clipping_applied",
]
INT_COLUMNS = [
    "training_rows",
    "observed_training_rows",
    "imputed_months_n",
]
LOGICAL_HASH_ALGORITHM = "sha256_typed_rows_v1"
LOGICAL_ARTIFACT_PATH = f"data/model_outputs/{ARTIFACT_FILENAME}"
LOGICAL_METADATA_PATH = f"data/metadata/{METADATA_FILENAME}"
EXPECTED_GITHUB_REPOSITORY = "mg15best/tfm-microempresas-rurales-"
EXPECTED_GITHUB_REF = "refs/heads/main"
EXPECTED_GITHUB_REF_NAME = "main"
EXPECTED_GITHUB_WORKFLOW = "Freeze ETS v2 canonical predictions"
EXPECTED_GITHUB_EVENT = "workflow_dispatch"
EXPECTED_GITHUB_SERVER_URL = "https://github.com"
SEMANTIC_ABSOLUTE_TOLERANCE = 1e-12
AVAILABLE_FALLBACK_REASON = "not_used"
BADAJOZ_EXPECTED_FALLBACK_REASON = "training_gap_unsupported"
ALLOWED_UNAVAILABLE_FALLBACK_REASONS = {
    "insufficient_history",
    "training_gap_unsupported",
    "fit_failure",
    "invalid_forecast",
}
EXPECTED_PANDAS_DTYPES = {
    **{column: "string" for column in STRING_COLUMNS},
    **{column: "float64" for column in FLOAT_COLUMNS},
    **{column: "bool" for column in BOOL_COLUMNS},
    **{column: "int64" for column in INT_COLUMNS},
}
VINTAGE_LIMITATION = (
    "Availability-correct point-in-time reconstruction using the latest "
    "revised Gold vintage; historical release vintages are unavailable."
)
CROSS_PLATFORM_NOTE = (
    "ETS is deterministic within the pinned environment, but optimized "
    "floating-point results vary slightly between Windows and Linux. The "
    "Ubuntu 24.04 artifact is the canonical source of truth."
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _validate_repository_path(path: str) -> str:
    normalized = str(path)
    parts = normalized.split("/")
    if (
        not normalized
        or normalized.startswith("/")
        or "\\" in normalized
        or ":" in normalized
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise AssertionError(f"Ruta Git versionada no valida: {path!r}.")
    return normalized


def _validate_git_commit(
    commit_sha: str,
    *,
    repository: Path | None = None,
) -> str:
    sha = str(commit_sha)
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise AssertionError(f"Generator commit SHA no valido: {sha!r}.")
    root = PROJECT_ROOT if repository is None else Path(repository)
    result = subprocess.run(
        ["git", "rev-parse", "--verify", f"{sha}^{{commit}}"],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        raise AssertionError(f"Generator commit Git no accesible: {sha}.")
    resolved = result.stdout.decode("ascii", errors="strict").strip()
    if resolved != sha:
        raise AssertionError(
            f"Generator commit Git no coincide: {resolved!r} != {sha!r}."
        )
    return sha


def _git_blob_bytes(
    commit_sha: str,
    repository_path: str,
    *,
    repository: Path | None = None,
) -> bytes:
    """Lee un blob Git sin pasar sus bytes por conversion de texto."""

    root = PROJECT_ROOT if repository is None else Path(repository)
    sha = _validate_git_commit(commit_sha, repository=root)
    path = _validate_repository_path(repository_path)
    result = subprocess.run(
        ["git", "cat-file", "blob", f"{sha}:{path}"],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"Blob Git no accesible en generator commit: {sha}:{path}."
        )
    return bytes(result.stdout)


def _load_modeling_config_bytes(content: bytes) -> dict[str, Any]:
    try:
        decoded = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise AssertionError("Config Git no es UTF-8 valido.") from exc
    config = yaml.safe_load(decoded)
    if not isinstance(config, dict):
        raise AssertionError("Config Git no contiene un mapping YAML.")
    if config.get("methodology", {}).get("id") != "point_in_time_v2":
        raise AssertionError("Config Git no declara point_in_time_v2.")
    return config


def _plain(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return [_plain(record) for record in value.to_dict(orient="records")]
    if isinstance(value, pd.Series):
        return [_plain(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if pd.isna(value):
        return None
    return value


def _requirement_pins_from_bytes(content: bytes) -> dict[str, str]:
    try:
        decoded = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise AssertionError("Requirements Git no es UTF-8 valido.") from exc
    pins: dict[str, str] = {}
    for raw_line in decoded.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "==" not in line:
            continue
        package, version = line.split("==", maxsplit=1)
        pins[package.strip().lower()] = version.strip()
    return pins


def _read_requirement_pins(path: Path = DEFAULT_REQUIREMENTS_PATH) -> dict[str, str]:
    return _requirement_pins_from_bytes(path.read_bytes())


def validate_canonical_contract(
    config: Mapping[str, Any],
    requirements_path: Path = DEFAULT_REQUIREMENTS_PATH,
    *,
    requirement_pins: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Comprueba que config y requirements declaran el mismo stack B4R."""

    canonical = {
        str(key): str(value)
        for key, value in config["numerical_reproducibility"][
            "canonical_environment"
        ].items()
    }
    expected = {
        "runner": "ubuntu-24.04",
        "python": "3.14.7",
        "numpy": "2.5.2",
        "scipy": "1.18.0",
        "pandas": "3.0.3",
        "patsy": "1.0.2",
        "statsmodels": "0.14.6",
    }
    if canonical != expected:
        raise RuntimeError(f"Entorno canonico B4R inesperado: {canonical}")
    pins = (
        _read_requirement_pins(requirements_path)
        if requirement_pins is None
        else {str(key): str(value) for key, value in requirement_pins.items()}
    )
    mismatches = {
        package: (pins.get(package), version)
        for package, version in expected.items()
        if package not in {"runner", "python"}
        and pins.get(package) != version
    }
    if mismatches:
        raise RuntimeError(f"Requirements no coincide con B4R: {mismatches}")
    return canonical


def _linux_distribution() -> tuple[str, str]:
    release = Path("/etc/os-release")
    if not release.exists():
        return "", ""
    values: dict[str, str] = {}
    for line in release.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", maxsplit=1)
        values[key] = value.strip().strip('"')
    return values.get("ID", ""), values.get("VERSION_ID", "")


def runtime_environment() -> dict[str, str]:
    distribution_id, distribution_version = _linux_distribution()
    runner = (
        "ubuntu-24.04"
        if distribution_id == "ubuntu" and distribution_version == "24.04"
        else f"{platform.system().lower()}-{platform.release()}"
    )
    return {
        "runner": runner,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
        "patsy": patsy.__version__,
        "statsmodels": statsmodels.__version__,
        "pyarrow": pyarrow.__version__,
    }


def validate_canonical_environment(
    expected: Mapping[str, str],
    actual: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Falla antes del fit si OS, Python o stack no son los canonicos."""

    observed = dict(actual or runtime_environment())
    mismatches = {
        key: (observed.get(key), str(expected[key]))
        for key in expected
        if observed.get(key) != str(expected[key])
    }
    if mismatches:
        raise RuntimeError(
            "El freeze oficial exige el entorno canonico B4R: "
            f"{mismatches}"
        )
    return observed


def validate_original_artifact_schema(frame: pd.DataFrame) -> None:
    """Rechaza schema drift antes de proyectar o convertir el parquet."""

    actual_columns = list(frame.columns)
    if actual_columns != ARTIFACT_COLUMNS:
        missing = [column for column in ARTIFACT_COLUMNS if column not in frame]
        extra = [column for column in actual_columns if column not in ARTIFACT_COLUMNS]
        raise AssertionError(
            "Schema fisico B4C inesperado: "
            f"missing={missing}, extra={extra}, order={actual_columns}."
        )
    actual_dtypes = {column: str(frame[column].dtype) for column in frame.columns}
    if actual_dtypes != EXPECTED_PANDAS_DTYPES:
        mismatches = {
            column: (actual_dtypes.get(column), expected)
            for column, expected in EXPECTED_PANDAS_DTYPES.items()
            if actual_dtypes.get(column) != expected
        }
        raise AssertionError(f"Dtypes fisicos B4C inesperados: {mismatches}")


def resolve_generation_context(
    *,
    expected_runtime: Mapping[str, str],
    actual_runtime: Mapping[str, str] | None = None,
    environment: Mapping[str, str] | None = None,
    git_head: str | None = None,
    allow_noncanonical_dry_run: bool = False,
) -> tuple[bool, dict[str, str] | None]:
    """Distingue official Actions/main de dry-run local, siempre fail-closed."""

    actual = dict(actual_runtime or runtime_environment())
    env = dict(os.environ if environment is None else environment)
    head = str(git_head or _git_head())
    in_actions = env.get("GITHUB_ACTIONS") == "true"
    if not in_actions:
        if not allow_noncanonical_dry_run:
            raise RuntimeError(
                "La generacion oficial solo puede ejecutarse en GitHub Actions/main; "
                "use --allow-noncanonical-dry-run para una salida local temporal."
            )
        return False, None

    validate_canonical_environment(expected_runtime, actual)
    expected_context = {
        "GITHUB_REPOSITORY": EXPECTED_GITHUB_REPOSITORY,
        "GITHUB_REF": EXPECTED_GITHUB_REF,
        "GITHUB_REF_NAME": EXPECTED_GITHUB_REF_NAME,
        "GITHUB_SHA": head,
        "GITHUB_WORKFLOW": EXPECTED_GITHUB_WORKFLOW,
        "GITHUB_EVENT_NAME": EXPECTED_GITHUB_EVENT,
        "GITHUB_SERVER_URL": EXPECTED_GITHUB_SERVER_URL,
    }
    mismatches = {
        key: (env.get(key), expected)
        for key, expected in expected_context.items()
        if env.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"Contexto GitHub canonico invalido: {mismatches}")
    run_id = str(env.get("GITHUB_RUN_ID", ""))
    run_number = str(env.get("GITHUB_RUN_NUMBER", ""))
    if not run_id.isdigit() or not run_number.isdigit():
        raise RuntimeError("GitHub run id/number ausente o no numerico.")
    server_url = str(env["GITHUB_SERVER_URL"]).rstrip("/")
    run_url = f"{server_url}/{EXPECTED_GITHUB_REPOSITORY}/actions/runs/{run_id}"
    return True, {
        "github_repository": str(env["GITHUB_REPOSITORY"]),
        "github_ref": str(env["GITHUB_REF"]),
        "github_ref_name": str(env["GITHUB_REF_NAME"]),
        "github_sha": str(env["GITHUB_SHA"]),
        "github_run_id": run_id,
        "github_run_number": run_number,
        "github_workflow": str(env["GITHUB_WORKFLOW"]),
        "github_event_name": str(env["GITHUB_EVENT_NAME"]),
        "github_server_url": server_url,
        "github_run_url": run_url,
    }


def _coerce_artifact_dtypes(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in STRING_COLUMNS:
        result[column] = result[column].astype("string")
    for column in FLOAT_COLUMNS:
        result[column] = pd.to_numeric(result[column], errors="raise").astype(
            "float64"
        )
    for column in BOOL_COLUMNS:
        if result[column].isna().any():
            raise AssertionError(f"Booleano ausente en {column}.")
        result[column] = result[column].astype(bool)
    for column in INT_COLUMNS:
        result[column] = pd.to_numeric(result[column], errors="raise").astype(
            "int64"
        )
    return result[ARTIFACT_COLUMNS]


def build_frozen_artifact(candidate_predictions: pd.DataFrame) -> pd.DataFrame:
    """Reduce la salida de fits a un contrato analitico sin timings ni logs."""

    renames = {
        "candidate_prediction": "ets_candidate_prediction",
        "raw_prediction": "ets_raw_prediction",
        "operational_prediction": "operational_prediction",
        "fallback_used": "availability_fallback_used",
        "training_observed_rows": "observed_training_rows",
    }
    required_source = (set(ARTIFACT_COLUMNS) - set(renames.values())) | set(
        renames
    )
    missing = sorted(required_source.difference(candidate_predictions.columns))
    if missing:
        raise ValueError("Salida ETS sin columnas para freeze: " + ", ".join(missing))
    artifact = candidate_predictions.rename(columns=renames)
    artifact = _coerce_artifact_dtypes(artifact)
    return artifact.sort_values(SORT_COLUMNS, ignore_index=True)


def _baseline_key_view(baseline_predictions: pd.DataFrame) -> pd.DataFrame:
    return baseline_predictions[[*KEY_COLUMNS, "actual", "prediction"]].rename(
        columns={"prediction": "baseline_prediction"}
    )


def validate_frozen_artifact(
    artifact: pd.DataFrame,
    baseline_predictions: pd.DataFrame,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Valida identidad, disponibilidad, cutoff y ausencia de B2/B3."""

    validate_original_artifact_schema(artifact)
    if bool(config["ets_candidate"]["fallback"].get("performance_based")):
        raise AssertionError("B4C prohibe fallback basado en performance.")
    expected_rows = sum(
        int(fold["expected_evaluable_rows"]) for fold in config["folds"]
    )
    if len(artifact) != expected_rows or expected_rows != 1750:
        raise AssertionError(f"B4C requiere 1.750 filas, obtuvo {len(artifact)}.")
    if artifact.duplicated(KEY_COLUMNS).any():
        raise AssertionError("El artefacto B4C contiene keys duplicadas.")
    expected_sort = artifact.sort_values(SORT_COLUMNS, ignore_index=True)
    pd.testing.assert_frame_equal(artifact, expected_sort)

    baseline = _baseline_key_view(baseline_predictions).sort_values(
        KEY_COLUMNS, ignore_index=True
    )
    frozen = artifact[[*KEY_COLUMNS, "actual", "baseline_prediction"]].sort_values(
        KEY_COLUMNS, ignore_index=True
    )
    pd.testing.assert_frame_equal(
        frozen[KEY_COLUMNS].astype("string"),
        baseline[KEY_COLUMNS].astype("string"),
        check_names=True,
    )
    np.testing.assert_allclose(frozen["actual"], baseline["actual"], rtol=0, atol=0)
    np.testing.assert_allclose(
        frozen["baseline_prediction"], baseline["baseline_prediction"], rtol=0, atol=0
    )

    if artifact["territory_id"].nunique() != 50:
        raise AssertionError("B4C requiere exactamente 50 provincias.")
    if artifact["target_month_id"].nunique() != 35:
        raise AssertionError("B4C requiere exactamente 35 origins evaluables.")
    expected_fold_rows = {
        str(fold["id"]): int(fold["expected_evaluable_rows"])
        for fold in config["folds"]
    }
    actual_fold_rows = artifact["fold_id"].value_counts().sort_index().to_dict()
    if actual_fold_rows != expected_fold_rows:
        raise AssertionError(f"Filas por fold inesperadas: {actual_fold_rows}")

    required_strings = [
        column for column in STRING_COLUMNS if column not in {"training_start", "training_end"}
    ]
    if artifact[required_strings].isna().any().any():
        missing_strings = artifact[required_strings].columns[
            artifact[required_strings].isna().any()
        ].tolist()
        raise AssertionError(f"Strings obligatorios ausentes: {missing_strings}")
    if artifact[["training_start", "training_end"]].isna().any().any():
        raise AssertionError("Training start/end debe existir en las 1.750 filas.")

    required_finite = ["actual", "baseline_prediction", "operational_prediction"]
    for column in required_finite:
        values = pd.to_numeric(artifact[column], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise AssertionError(f"{column} debe ser finita y no negativa.")

    target = pd.PeriodIndex(artifact["target_month_id"], freq="M")
    latest = pd.PeriodIndex(artifact["latest_available_month_id"], freq="M")
    business_origin = pd.PeriodIndex(
        artifact["business_origin_month_id"], freq="M"
    )
    if not (latest < target).all() or not ((target.asi8 - latest.asi8) == 3).all():
        raise AssertionError("Cutoff ETS futuro o distinto de target-3.")
    if not ((target.asi8 - business_origin.asi8) == 1).all():
        raise AssertionError("Business origin distinto de target-1.")
    training_start = pd.PeriodIndex(artifact["training_start"], freq="M")
    training_end = pd.PeriodIndex(artifact["training_end"], freq="M")
    if not (training_start <= training_end).all():
        raise AssertionError("Training start posterior a training end.")
    if not (training_end <= latest).all():
        raise AssertionError("Training end posterior al cutoff.")
    if (artifact[INT_COLUMNS] < 0).any().any():
        raise AssertionError("Conteos de training negativos.")

    available = artifact["candidate_available"]
    fallback = artifact["availability_fallback_used"]
    if not fallback.eq(~available).all():
        raise AssertionError("Availability fallback no complementa disponibilidad ETS.")
    if artifact.loc[available, "ets_candidate_prediction"].isna().any():
        raise AssertionError("ETS disponible sin candidate prediction.")
    if artifact.loc[available, "ets_raw_prediction"].isna().any():
        raise AssertionError("ETS disponible sin raw prediction.")
    if artifact.loc[~available, "ets_candidate_prediction"].notna().any():
        raise AssertionError("Se invento un forecast ETS no disponible.")
    if artifact.loc[~available, "ets_raw_prediction"].notna().any():
        raise AssertionError("Se invento un raw forecast ETS no disponible.")
    raw_available = artifact.loc[available, "ets_raw_prediction"].to_numpy(dtype=float)
    candidate_available = artifact.loc[
        available, "ets_candidate_prediction"
    ].to_numpy(dtype=float)
    if not np.isfinite(raw_available).all():
        raise AssertionError("Raw ETS disponible debe ser finita.")
    if not np.isfinite(candidate_available).all() or (candidate_available < 0).any():
        raise AssertionError("Candidate ETS disponible debe ser finita y no negativa.")
    expected_candidate = np.maximum(raw_available, 0.0)
    if not np.allclose(
        candidate_available,
        expected_candidate,
        rtol=0,
        atol=SEMANTIC_ABSOLUTE_TOLERANCE,
    ):
        raise AssertionError(
            "Candidate ETS no coincide con max(raw, 0) dentro de atol=1e-12."
        )
    expected_clipping = raw_available < 0
    actual_clipping = artifact.loc[available, "clipping_applied"].to_numpy(dtype=bool)
    if not np.array_equal(actual_clipping, expected_clipping):
        raise AssertionError("clipping_applied no coincide con raw < 0.")
    if artifact.loc[~available, "clipping_applied"].any():
        raise AssertionError("ETS no disponible no puede declarar clipping.")
    if fallback.loc[available].any():
        raise AssertionError("ETS disponible no puede usar fallback.")
    if not artifact.loc[available, "fallback_reason"].eq(
        AVAILABLE_FALLBACK_REASON
    ).all():
        raise AssertionError("ETS disponible requiere fallback_reason=not_used.")
    if not np.allclose(
        artifact.loc[available, "operational_prediction"].to_numpy(dtype=float),
        candidate_available,
        rtol=0,
        atol=SEMANTIC_ABSOLUTE_TOLERANCE,
    ):
        raise AssertionError("Operational ETS no coincide con candidate ETS.")

    unavailable_reasons = set(artifact.loc[~available, "fallback_reason"].astype(str))
    invalid_reasons = sorted(
        unavailable_reasons.difference(ALLOWED_UNAVAILABLE_FALLBACK_REASONS)
    )
    if invalid_reasons:
        raise AssertionError(f"Fallback reasons no permitidos: {invalid_reasons}")
    np.testing.assert_allclose(
        artifact.loc[fallback, "operational_prediction"],
        artifact.loc[fallback, "baseline_prediction"],
        rtol=0,
        atol=0,
    )

    badajoz = artifact["territory_id"].eq("ES-PROV-06")
    if int(badajoz.sum()) != 35 or int(fallback.sum()) != 35:
        raise AssertionError("Badajoz/fallback debe ocupar exactamente 35 filas.")
    if not artifact.loc[badajoz, "territory_name"].str.casefold().eq("badajoz").all():
        raise AssertionError("ES-PROV-06 debe conservar el nombre Badajoz.")
    if not fallback.eq(badajoz).all() or artifact.loc[badajoz, "candidate_available"].any():
        raise AssertionError("El fallback operativo debe pertenecer solo a Badajoz.")
    if not artifact.loc[badajoz, "fallback_reason"].eq(
        BADAJOZ_EXPECTED_FALLBACK_REASON
    ).all():
        raise AssertionError(
            "Badajoz requiere fallback_reason=training_gap_unsupported."
        )

    minimum_training = int(config["ets_candidate"]["minimum_training_months"])
    available_rows = artifact.loc[available]
    if (available_rows["training_rows"] < minimum_training).any() or (
        available_rows["observed_training_rows"] < minimum_training
    ).any():
        raise AssertionError("ETS disponible sin historia minima.")
    if (
        available_rows["observed_training_rows"] > available_rows["training_rows"]
    ).any() or (available_rows["imputed_months_n"] > available_rows["training_rows"]).any():
        raise AssertionError("Conteos de training ETS incoherentes.")

    if artifact["model_id"].nunique() != 1 or artifact["model_id"].iloc[0] != str(
        config["ets_candidate"]["id"]
    ):
        raise AssertionError("Model provenance ETS inesperada.")
    if artifact["cutoff_policy_id"].nunique() != 1 or artifact[
        "cutoff_policy_id"
    ].iloc[0] != str(config["cutoff_policy"]["id"]):
        raise AssertionError("Cutoff provenance inesperada.")
    forbidden = {
        "lower",
        "upper",
        "interval_lower",
        "interval_upper",
        "trend_factor",
        "performance_fallback",
    }
    contamination = sorted(forbidden.intersection(artifact.columns))
    if contamination:
        raise AssertionError(f"Contaminacion B2/B3/router: {contamination}")
    return {
        "rows": int(len(artifact)),
        "unique_keys": int(len(artifact.drop_duplicates(KEY_COLUMNS))),
        "territories": int(artifact["territory_id"].nunique()),
        "origins": int(artifact["target_month_id"].nunique()),
        "candidate_available_rows": int(available.sum()),
        "availability_fallback_rows": int(fallback.sum()),
        "fold_rows": actual_fold_rows,
    }


def _artifact_for_metrics(artifact: pd.DataFrame) -> pd.DataFrame:
    return artifact.rename(
        columns={
            "ets_candidate_prediction": "candidate_prediction",
            "availability_fallback_used": "fallback_used",
        }
    ).copy()


def reconstruct_artifact_evaluation(
    artifact: pd.DataFrame,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Recalcula metricas y screening desde filas congeladas, sin fit ETS."""

    view = _artifact_for_metrics(artifact)
    pooled = calculate_paired_metrics(
        view, candidate_prediction_column="operational_prediction"
    )
    folds = calculate_candidate_fold_metrics(view, config)
    territories = calculate_candidate_territory_metrics(view)
    origins = calculate_candidate_origin_metrics(view)
    screening = screen_ets_candidate(pooled, folds, territories, config)
    checks = {key: value for key, value in screening.items() if key[:2] in {
        "A_", "B_", "C_", "D_", "E_", "F_"
    }}
    if not checks or not all(checks.values()):
        raise AssertionError(f"El artefacto no supera screening A-F: {checks}")
    origin_skills = pd.to_numeric(origins["mae_skill_pct"], errors="raise")
    return {
        "pooled_metrics": pooled,
        "fold_metrics": folds,
        "territory_metrics": territories,
        "origin_metrics": origins,
        "screening": screening,
        "summary": {
            "pooled": pooled,
            "folds": folds.to_dict(orient="records"),
            "territories": {
                "count": int(len(territories)),
                "wins": int(territories["outcome"].eq("win").sum()),
                "losses": int(territories["outcome"].eq("loss").sum()),
                "ties": int(territories["outcome"].eq("tie").sum()),
                "median_mae_skill_pct": float(
                    pd.to_numeric(territories["mae_skill_pct"]).median()
                ),
            },
            "origins": {
                "count": int(len(origins)),
                "wins": int(origins["outcome"].eq("win").sum()),
                "losses": int(origins["outcome"].eq("loss").sum()),
                "ties": int(origins["outcome"].eq("tie").sum()),
                "median_mae_skill_pct": float(origin_skills.median()),
            },
            "screening": screening,
        },
    }


def _update_typed_value(digest: Any, column: str, value: Any) -> None:
    if pd.isna(value):
        digest.update(b"\x00")
        return
    digest.update(b"\x01")
    if column in STRING_COLUMNS:
        encoded = str(value).encode("utf-8")
        digest.update(struct.pack("<I", len(encoded)))
        digest.update(encoded)
    elif column in FLOAT_COLUMNS:
        digest.update(struct.pack("<d", float(value)))
    elif column in BOOL_COLUMNS:
        digest.update(b"\x01" if bool(value) else b"\x00")
    elif column in INT_COLUMNS:
        digest.update(struct.pack("<q", int(value)))
    else:
        raise AssertionError(f"Columna sin serializador logico: {column}")


def logical_prediction_sha256(artifact: pd.DataFrame) -> str:
    """Hash estable: schema + filas ordenadas + valores tipados little-endian."""

    if list(artifact.columns) != ARTIFACT_COLUMNS:
        raise ValueError("El hash logico exige el schema B4C exacto.")
    ordered = artifact.sort_values(SORT_COLUMNS, ignore_index=True)
    digest = hashlib.sha256()
    header = json.dumps(
        {
            "algorithm": LOGICAL_HASH_ALGORITHM,
            "columns": ARTIFACT_COLUMNS,
            "sort": SORT_COLUMNS,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    digest.update(struct.pack("<I", len(header)))
    digest.update(header)
    digest.update(struct.pack("<q", len(ordered)))
    for row in ordered.itertuples(index=False, name=None):
        for column, value in zip(ARTIFACT_COLUMNS, row, strict=True):
            _update_typed_value(digest, column, value)
    return digest.hexdigest()


def _git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _resolve_artifact_generator_commit(metadata: Mapping[str, Any]) -> str:
    artifact_meta = metadata.get("artifact")
    if not isinstance(artifact_meta, Mapping):
        raise AssertionError("Metadata artifact ausente.")
    generator_sha = _validate_git_commit(
        str(artifact_meta.get("generating_commit_sha", ""))
    )
    official = artifact_meta.get("official_canonical_artifact")
    if not isinstance(official, bool):
        raise AssertionError("official_canonical_artifact debe ser booleano.")
    github_meta = metadata.get("github")
    if official:
        if not isinstance(github_meta, Mapping):
            raise AssertionError("Artifact oficial sin contexto GitHub.")
        github_sha = str(github_meta.get("github_sha", ""))
        if github_sha != generator_sha:
            raise AssertionError(
                "Metadata github.github_sha no coincide con "
                "artifact.generating_commit_sha."
            )
    elif github_meta is not None:
        raise AssertionError(
            "Dry-run local no debe declarar contexto GitHub oficial."
        )
    return generator_sha


def _repository_worktree_path(repository_path: str) -> Path:
    path = _validate_repository_path(repository_path)
    resolved_root = PROJECT_ROOT.resolve()
    resolved_path = (PROJECT_ROOT / Path(*path.split("/"))).resolve()
    if not resolved_path.is_relative_to(resolved_root):
        raise AssertionError(f"Ruta versionada fuera del repositorio: {path}.")
    return resolved_path


def _require_worktree_file_matches_sha256(
    path: Path,
    expected_sha256: str,
    *,
    label: str,
) -> None:
    if not path.is_file():
        raise AssertionError(f"{label} actual no existe: {path}.")
    actual_sha256 = _sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise AssertionError(
            f"{label} actual no coincide con el source del generator commit; "
            "se requiere el source historico para reconstruir el artifact. "
            f"actual={actual_sha256}, historico={expected_sha256}."
        )


def _versioned_source_context(generator_sha: str) -> dict[str, Any]:
    """Resuelve config/requirements/Gold desde el commit que genero el artifact."""

    config_bytes = _git_blob_bytes(generator_sha, CONFIG_REPOSITORY_PATH)
    config_sha256 = _sha256_bytes(config_bytes)
    config = _load_modeling_config_bytes(config_bytes)
    requirements_bytes = _git_blob_bytes(
        generator_sha, REQUIREMENTS_REPOSITORY_PATH
    )
    requirement_pins = _requirement_pins_from_bytes(requirements_bytes)
    gold_repository_path = _validate_repository_path(
        str(config["source"]["path"])
    )
    gold_bytes = _git_blob_bytes(generator_sha, gold_repository_path)
    gold_sha256 = _sha256_bytes(gold_bytes)
    gold_path = _repository_worktree_path(gold_repository_path)
    _require_worktree_file_matches_sha256(
        gold_path,
        gold_sha256,
        label="Gold working tree",
    )
    gold = load_gold_history(gold_path)
    return {
        "config": config,
        "config_sha256": config_sha256,
        "requirement_pins": requirement_pins,
        "gold": gold,
        "gold_path": gold_path,
        "gold_repository_path": gold_repository_path,
        "gold_sha256": gold_sha256,
    }


def _validate_historical_source_metadata(
    metadata: Mapping[str, Any],
    sources: Mapping[str, Any],
) -> None:
    provenance = metadata.get("provenance")
    if not isinstance(provenance, Mapping):
        raise AssertionError("Metadata provenance ausente.")
    if provenance.get("config_path") != CONFIG_REPOSITORY_PATH:
        raise AssertionError("Metadata provenance.config_path no coincide.")
    if provenance.get("config_sha256") != sources["config_sha256"]:
        raise AssertionError(
            "Metadata provenance.config_sha256 no coincide con el blob Git "
            "del generator commit."
        )
    gold_meta = provenance.get("gold")
    if not isinstance(gold_meta, Mapping):
        raise AssertionError("Metadata provenance.gold ausente.")
    if gold_meta.get("path") != sources["gold_repository_path"]:
        raise AssertionError("Metadata provenance.gold.path no coincide.")
    if gold_meta.get("file_sha256") != sources["gold_sha256"]:
        raise AssertionError(
            "Metadata provenance.gold.file_sha256 no coincide con el blob Git "
            "del generator commit."
        )


def _gold_provenance(
    gold: pd.DataFrame,
    gold_path: Path,
    *,
    file_sha256: str | None = None,
) -> dict[str, Any]:
    snapshots = sorted(
        str(value) for value in gold["source_snapshot_id"].dropna().unique()
    )
    versions = sorted(str(value) for value in gold["data_version"].dropna().unique())
    return {
        "path": str(gold_path.relative_to(PROJECT_ROOT)).replace(os.sep, "/"),
        "file_sha256": file_sha256 or _sha256_file(gold_path),
        "source_snapshot_ids": snapshots,
        "data_versions": versions,
        "vintage_limitation": VINTAGE_LIMITATION,
    }


def build_metadata(
    *,
    artifact: pd.DataFrame,
    artifact_path: Path,
    config: Mapping[str, Any],
    config_path: Path,
    config_sha256: str,
    gold: pd.DataFrame,
    gold_path: Path,
    gold_sha256: str,
    canonical_environment: Mapping[str, str],
    actual_environment: Mapping[str, str],
    official_canonical_artifact: bool,
    generating_commit_sha: str,
    github_context: Mapping[str, str] | None,
    invariants: Mapping[str, Any],
    evaluation: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "artifact": {
            "name": ARTIFACT_FILENAME,
            "logical_path": LOGICAL_ARTIFACT_PATH,
            "metadata_logical_path": LOGICAL_METADATA_PATH,
            "version": ARTIFACT_VERSION,
            "created_at_utc": datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z"),
            "official_canonical_artifact": official_canonical_artifact,
            "generating_commit_sha": generating_commit_sha,
            "row_count": int(len(artifact)),
            "territory_count": int(artifact["territory_id"].nunique()),
            "origin_count": int(artifact["target_month_id"].nunique()),
            "fold_ids": sorted(str(value) for value in artifact["fold_id"].unique()),
            "file_sha256": _sha256_file(artifact_path),
            "logical_prediction_sha256": logical_prediction_sha256(artifact),
        },
        "logical_hash_contract": {
            "algorithm": LOGICAL_HASH_ALGORITHM,
            "sort_columns": SORT_COLUMNS,
            "columns": ARTIFACT_COLUMNS,
            "serialization": (
                "SHA-256 over a JSON schema header and typed rows; strings are "
                "UTF-8 length-prefixed, floats are IEEE-754 float64 little-endian, "
                "integers are int64 little-endian, booleans are one byte, and "
                "nulls use an explicit marker. created_at is excluded."
            ),
        },
        "provenance": {
            "gold": _gold_provenance(
                gold,
                gold_path,
                file_sha256=gold_sha256,
            ),
            "baseline_id": str(config["baseline"]["id"]),
            "ets_model_id": str(config["ets_candidate"]["id"]),
            "ets_model_version": str(config["ets_candidate"]["candidate_version"]),
            "ets_library": str(config["ets_candidate"]["library"]),
            "ets_library_version": str(config["ets_candidate"]["library_version"]),
            "cutoff_policy_id": str(config["cutoff_policy"]["id"]),
            "config_path": str(config_path.relative_to(PROJECT_ROOT)).replace(
                os.sep, "/"
            ),
            "config_sha256": config_sha256,
            "config_version": str(config["provenance"]["config_version"]),
            "code_contract_version": str(
                config["provenance"]["code_contract_version"]
            ),
        },
        "folds": [
            {
                "id": str(fold["id"]),
                "start": str(fold["start"]),
                "end": str(fold["end"]),
                "expected_evaluable_rows": int(fold["expected_evaluable_rows"]),
            }
            for fold in config["folds"]
        ],
        "environment": {
            "canonical": dict(canonical_environment),
            "actual": dict(actual_environment),
            "cross_platform_note": CROSS_PLATFORM_NOTE,
        },
        "storage": {
            "format": "parquet",
            "engine": "pyarrow",
            "engine_version": str(actual_environment["pyarrow"]),
            "compression": "zstd",
        },
        "github": dict(github_context) if github_context is not None else None,
        "schema": {
            "version": "ets_v2_rolling_validation_predictions_schema_v1",
            "columns": ARTIFACT_COLUMNS,
            "pandas_dtypes": EXPECTED_PANDAS_DTYPES,
        },
        "invariants": _plain(invariants),
        "evaluation": _plain(evaluation["summary"]),
    }


def _assert_metadata_equal(label: str, actual: Any, expected: Any) -> None:
    """Comparacion recursiva estricta con tolerancia solo para floats."""

    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            raise AssertionError(f"Metadata {label} no es un mapping.")
        if set(actual) != set(expected):
            raise AssertionError(
                f"Metadata {label} keys no coinciden: "
                f"actual={sorted(actual)}, expected={sorted(expected)}"
            )
        for key, expected_value in expected.items():
            _assert_metadata_equal(
                f"{label}.{key}", actual[key], expected_value
            )
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise AssertionError(f"Metadata {label} lista no coincide.")
        for index, (actual_value, expected_value) in enumerate(
            zip(actual, expected, strict=True)
        ):
            _assert_metadata_equal(
                f"{label}[{index}]", actual_value, expected_value
            )
        return
    if isinstance(expected, float):
        try:
            actual_float = float(actual)
        except (TypeError, ValueError) as exc:
            raise AssertionError(f"Metadata {label} no es float.") from exc
        if not np.isfinite(actual_float) or not np.isclose(
            actual_float,
            expected,
            rtol=0,
            atol=SEMANTIC_ABSOLUTE_TOLERANCE,
        ):
            raise AssertionError(
                f"Metadata {label} no coincide: {actual!r} != {expected!r}."
            )
        return
    if actual != expected:
        raise AssertionError(
            f"Metadata {label} no coincide: {actual!r} != {expected!r}."
        )


def _validate_created_at_utc(value: Any) -> None:
    text = str(value)
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", text):
        raise AssertionError("created_at_utc no usa el formato UTC canonico.")
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.tzinfo != timezone.utc:
        raise AssertionError("created_at_utc no esta en UTC.")


def _expected_fold_metadata(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "id": str(fold["id"]),
            "start": str(fold["start"]),
            "end": str(fold["end"]),
            "expected_evaluable_rows": int(fold["expected_evaluable_rows"]),
        }
        for fold in config["folds"]
    ]


def verify_metadata_contract(
    *,
    metadata: Mapping[str, Any],
    artifact: pd.DataFrame,
    artifact_path: Path,
    config: Mapping[str, Any],
    gold: pd.DataFrame,
    gold_path: Path,
    generator_commit_sha: str,
    config_sha256: str,
    gold_sha256: str,
    requirement_pins: Mapping[str, str],
    invariants: Mapping[str, Any],
    evaluation: Mapping[str, Any],
) -> tuple[str, str]:
    """Vincula YAML, parquet, commit, inputs, entorno y evidencia analitica."""

    expected_top_level = {
        "artifact",
        "logical_hash_contract",
        "provenance",
        "folds",
        "environment",
        "storage",
        "github",
        "schema",
        "invariants",
        "evaluation",
    }
    if set(metadata) != expected_top_level:
        raise AssertionError(
            "Metadata top-level inesperada: "
            f"actual={sorted(metadata)}, expected={sorted(expected_top_level)}"
        )

    artifact_meta = metadata.get("artifact")
    if not isinstance(artifact_meta, Mapping):
        raise AssertionError("Metadata artifact ausente.")
    _validate_created_at_utc(artifact_meta.get("created_at_utc"))
    actual_file_hash = _sha256_file(artifact_path)
    actual_logical_hash = logical_prediction_sha256(artifact)
    official = artifact_meta.get("official_canonical_artifact")
    if not isinstance(official, bool):
        raise AssertionError("official_canonical_artifact debe ser booleano.")
    expected_artifact = {
        "name": ARTIFACT_FILENAME,
        "logical_path": LOGICAL_ARTIFACT_PATH,
        "metadata_logical_path": LOGICAL_METADATA_PATH,
        "version": ARTIFACT_VERSION,
        "created_at_utc": artifact_meta["created_at_utc"],
        "official_canonical_artifact": official,
        "generating_commit_sha": generator_commit_sha,
        "row_count": int(len(artifact)),
        "territory_count": int(artifact["territory_id"].nunique()),
        "origin_count": int(artifact["target_month_id"].nunique()),
        "fold_ids": sorted(str(value) for value in artifact["fold_id"].unique()),
        "file_sha256": actual_file_hash,
        "logical_prediction_sha256": actual_logical_hash,
    }
    _assert_metadata_equal("artifact", artifact_meta, expected_artifact)

    expected_logical_contract = {
        "algorithm": LOGICAL_HASH_ALGORITHM,
        "sort_columns": SORT_COLUMNS,
        "columns": ARTIFACT_COLUMNS,
        "serialization": (
            "SHA-256 over a JSON schema header and typed rows; strings are "
            "UTF-8 length-prefixed, floats are IEEE-754 float64 little-endian, "
            "integers are int64 little-endian, booleans are one byte, and "
            "nulls use an explicit marker. created_at is excluded."
        ),
    }
    _assert_metadata_equal(
        "logical_hash_contract",
        metadata["logical_hash_contract"],
        expected_logical_contract,
    )

    expected_provenance = {
        "gold": _gold_provenance(
            gold,
            gold_path,
            file_sha256=gold_sha256,
        ),
        "baseline_id": str(config["baseline"]["id"]),
        "ets_model_id": str(config["ets_candidate"]["id"]),
        "ets_model_version": str(config["ets_candidate"]["candidate_version"]),
        "ets_library": str(config["ets_candidate"]["library"]),
        "ets_library_version": str(config["ets_candidate"]["library_version"]),
        "cutoff_policy_id": str(config["cutoff_policy"]["id"]),
        "config_path": CONFIG_REPOSITORY_PATH,
        "config_sha256": config_sha256,
        "config_version": str(config["provenance"]["config_version"]),
        "code_contract_version": str(
            config["provenance"]["code_contract_version"]
        ),
    }
    _assert_metadata_equal("provenance", metadata["provenance"], expected_provenance)
    _assert_metadata_equal("folds", metadata["folds"], _expected_fold_metadata(config))

    canonical = validate_canonical_contract(
        config,
        requirement_pins=requirement_pins,
    )
    expected_runtime = {**canonical, "pyarrow": requirement_pins["pyarrow"]}
    environment_meta = metadata.get("environment")
    if not isinstance(environment_meta, Mapping):
        raise AssertionError("Metadata environment ausente.")
    _assert_metadata_equal(
        "environment.canonical", environment_meta.get("canonical"), expected_runtime
    )
    expected_environment_keys = {*expected_runtime, "platform"}
    actual_environment_meta = environment_meta.get("actual")
    if not isinstance(actual_environment_meta, Mapping) or set(
        actual_environment_meta
    ) != expected_environment_keys:
        raise AssertionError("Metadata environment.actual tiene keys inesperadas.")
    if official:
        for key, expected_value in expected_runtime.items():
            _assert_metadata_equal(
                f"environment.actual.{key}",
                actual_environment_meta.get(key),
                expected_value,
            )
        if not str(actual_environment_meta.get("platform", "")).strip():
            raise AssertionError("Platform oficial ausente.")
    else:
        _assert_metadata_equal(
            "environment.actual", actual_environment_meta, runtime_environment()
        )
    _assert_metadata_equal(
        "environment.cross_platform_note",
        environment_meta.get("cross_platform_note"),
        CROSS_PLATFORM_NOTE,
    )
    _assert_metadata_equal(
        "storage",
        metadata["storage"],
        {
            "format": "parquet",
            "engine": "pyarrow",
            "engine_version": expected_runtime["pyarrow"],
            "compression": "zstd",
        },
    )

    github_meta = metadata.get("github")
    if official:
        if not isinstance(github_meta, Mapping):
            raise AssertionError("Artifact oficial sin contexto GitHub.")
        run_id = str(github_meta.get("github_run_id", ""))
        run_number = str(github_meta.get("github_run_number", ""))
        if not run_id.isdigit() or not run_number.isdigit():
            raise AssertionError("GitHub run id/number invalido.")
        expected_github = {
            "github_repository": EXPECTED_GITHUB_REPOSITORY,
            "github_ref": EXPECTED_GITHUB_REF,
            "github_ref_name": EXPECTED_GITHUB_REF_NAME,
            "github_sha": generator_commit_sha,
            "github_run_id": run_id,
            "github_run_number": run_number,
            "github_workflow": EXPECTED_GITHUB_WORKFLOW,
            "github_event_name": EXPECTED_GITHUB_EVENT,
            "github_server_url": EXPECTED_GITHUB_SERVER_URL,
            "github_run_url": (
                f"{EXPECTED_GITHUB_SERVER_URL}/{EXPECTED_GITHUB_REPOSITORY}"
                f"/actions/runs/{run_id}"
            ),
        }
        _assert_metadata_equal("github", github_meta, expected_github)
    elif github_meta is not None:
        raise AssertionError("Dry-run local no debe declarar contexto GitHub oficial.")

    _assert_metadata_equal(
        "schema",
        metadata["schema"],
        {
            "version": "ets_v2_rolling_validation_predictions_schema_v1",
            "columns": ARTIFACT_COLUMNS,
            "pandas_dtypes": EXPECTED_PANDAS_DTYPES,
        },
    )
    _assert_metadata_equal("invariants", metadata["invariants"], _plain(invariants))
    _assert_metadata_equal(
        "evaluation", metadata["evaluation"], _plain(evaluation["summary"])
    )
    return actual_file_hash, actual_logical_hash


def generate_frozen_artifact_in_memory(
    gold: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Ejecuta 1.715 fits una vez y devuelve artefacto, baseline y evidencia."""

    baseline_result = reproduce_baseline_in_memory(gold, config)
    baseline = baseline_result["comparable_predictions"]
    candidate = build_ets_predictions(baseline, gold, config["ets_candidate"])
    artifact = build_frozen_artifact(candidate)
    invariants = validate_frozen_artifact(artifact, baseline, config)
    evaluation = reconstruct_artifact_evaluation(artifact, config)
    return artifact, baseline, invariants, evaluation


def freeze_predictions(
    artifact_path: Path,
    metadata_path: Path,
    *,
    allow_noncanonical_dry_run: bool = False,
) -> dict[str, Any]:
    """Genera parquet + YAML; fuera de Ubuntu solo permite tmp explicito."""

    git_head = _git_head()
    sources = _versioned_source_context(git_head)
    config = sources["config"]
    requirement_pins = sources["requirement_pins"]
    canonical = validate_canonical_contract(
        config,
        requirement_pins=requirement_pins,
    )
    actual = runtime_environment()
    pyarrow_pin = requirement_pins["pyarrow"]
    expected_runtime = {**canonical, "pyarrow": pyarrow_pin}
    is_official, github_context = resolve_generation_context(
        expected_runtime=expected_runtime,
        actual_runtime=actual,
        environment=os.environ,
        git_head=git_head,
        allow_noncanonical_dry_run=allow_noncanonical_dry_run,
    )
    if not is_official:
        for output in (artifact_path, metadata_path):
            if output.resolve().is_relative_to(PROJECT_ROOT.resolve()):
                raise RuntimeError(
                    "Un dry-run no canonico solo puede escribir fuera del repositorio."
                )

    gold_path = sources["gold_path"]
    gold = sources["gold"]
    artifact, _, invariants, evaluation = generate_frozen_artifact_in_memory(
        gold, config
    )
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    artifact.to_parquet(
        artifact_path,
        index=False,
        engine="pyarrow",
        compression="zstd",
    )
    metadata = build_metadata(
        artifact=artifact,
        artifact_path=artifact_path,
        config=config,
        config_path=DEFAULT_CONFIG_PATH,
        config_sha256=sources["config_sha256"],
        gold=gold,
        gold_path=gold_path,
        gold_sha256=sources["gold_sha256"],
        canonical_environment=expected_runtime,
        actual_environment=actual,
        official_canonical_artifact=is_official,
        generating_commit_sha=git_head,
        github_context=github_context,
        invariants=invariants,
        evaluation=evaluation,
    )
    metadata_path.write_text(
        yaml.safe_dump(metadata, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    verify_frozen_artifact(artifact_path, metadata_path)
    return metadata


def verify_frozen_artifact(
    artifact_path: Path,
    metadata_path: Path,
) -> dict[str, Any]:
    """Verifica hashes, metadata e invariantes sin volver a ajustar ETS."""

    metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, Mapping):
        raise AssertionError("Metadata YAML no contiene un mapping.")
    generator_commit_sha = _resolve_artifact_generator_commit(metadata)
    sources = _versioned_source_context(generator_commit_sha)
    _validate_historical_source_metadata(metadata, sources)
    config = sources["config"]
    artifact = pd.read_parquet(artifact_path)
    validate_original_artifact_schema(artifact)
    gold_path = sources["gold_path"]
    gold = sources["gold"]
    baseline = reproduce_baseline_in_memory(gold, config)["comparable_predictions"]
    invariants = validate_frozen_artifact(artifact, baseline, config)
    evaluation = reconstruct_artifact_evaluation(artifact, config)
    actual_file, actual_logical = verify_metadata_contract(
        metadata=metadata,
        artifact=artifact,
        artifact_path=artifact_path,
        config=config,
        gold=gold,
        gold_path=gold_path,
        generator_commit_sha=generator_commit_sha,
        config_sha256=sources["config_sha256"],
        gold_sha256=sources["gold_sha256"],
        requirement_pins=sources["requirement_pins"],
        invariants=invariants,
        evaluation=evaluation,
    )
    return {
        "file_sha256": actual_file,
        "logical_prediction_sha256": actual_logical,
        "generator_commit_sha": generator_commit_sha,
        "invariants": invariants,
        "evaluation": evaluation["summary"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-path", type=Path, default=DEFAULT_ARTIFACT_PATH)
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument(
        "--allow-noncanonical-dry-run",
        action="store_true",
        help="Permite Windows solo con salidas temporales fuera del repositorio.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.verify_only:
        result = verify_frozen_artifact(args.artifact_path, args.metadata_path)
    else:
        result = freeze_predictions(
            args.artifact_path,
            args.metadata_path,
            allow_noncanonical_dry_run=args.allow_noncanonical_dry_run,
        )
    print(yaml.safe_dump(_plain(result), sort_keys=False, allow_unicode=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
