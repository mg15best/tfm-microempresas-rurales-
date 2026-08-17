from __future__ import annotations

import copy
import hashlib
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import yaml

import src.models.ets_v2 as ets_module
import src.models.freeze_ets_v2_predictions as freeze_module
from src.models.evaluate_v2 import load_gold_history, reproduce_baseline_in_memory
from src.models.freeze_ets_v2_predictions import (
    ARTIFACT_COLUMNS,
    EXPECTED_GITHUB_EVENT,
    EXPECTED_GITHUB_REF,
    EXPECTED_GITHUB_REF_NAME,
    EXPECTED_GITHUB_REPOSITORY,
    EXPECTED_GITHUB_SERVER_URL,
    EXPECTED_GITHUB_WORKFLOW,
    LOGICAL_HASH_ALGORITHM,
    PROJECT_ROOT,
    SORT_COLUMNS,
    freeze_predictions,
    logical_prediction_sha256,
    reconstruct_artifact_evaluation,
    resolve_generation_context,
    validate_canonical_contract,
    validate_frozen_artifact,
    verify_frozen_artifact,
)
from src.models.modeling_v2_common import load_modeling_v2_config


@pytest.fixture(scope="module")
def frozen_case(tmp_path_factory):
    output = tmp_path_factory.mktemp("ets_b4c")
    artifact_path = output / "ets_v2_rolling_validation_predictions.parquet"
    metadata_path = output / "ets_v2_rolling_validation_predictions.metadata.yml"
    # Los tests siempre producen un dry-run, incluso dentro de otro workflow CI.
    with patch.dict(os.environ, {"GITHUB_ACTIONS": "false"}):
        metadata = freeze_predictions(
            artifact_path,
            metadata_path,
            allow_noncanonical_dry_run=True,
        )
    artifact = pd.read_parquet(artifact_path)
    config = load_modeling_v2_config()
    gold = load_gold_history()
    baseline = reproduce_baseline_in_memory(gold, config)["comparable_predictions"]
    return {
        "artifact": artifact,
        "artifact_path": artifact_path,
        "metadata": metadata,
        "metadata_path": metadata_path,
        "config": config,
        "baseline": baseline,
    }


def _write_pair(
    tmp_path: Path,
    frozen_case,
    *,
    artifact: pd.DataFrame | None = None,
    metadata: dict | None = None,
) -> tuple[Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    artifact_path = tmp_path / "ets_v2_rolling_validation_predictions.parquet"
    metadata_path = tmp_path / "ets_v2_rolling_validation_predictions.metadata.yml"
    if artifact is None:
        artifact_path.write_bytes(frozen_case["artifact_path"].read_bytes())
    else:
        artifact.to_parquet(
            artifact_path, index=False, engine="pyarrow", compression="zstd"
        )
    resolved_metadata = copy.deepcopy(
        frozen_case["metadata"] if metadata is None else metadata
    )
    metadata_path.write_text(
        yaml.safe_dump(resolved_metadata, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return artifact_path, metadata_path


def _set_nested(container: dict, path: tuple[object, ...], value: object) -> None:
    current = container
    for key in path[:-1]:
        current = current[key]
    current[path[-1]] = value


def _canonical_runtime() -> dict[str, str]:
    canonical = validate_canonical_contract(load_modeling_v2_config())
    return {**canonical, "pyarrow": "24.0.0", "platform": "Linux-test"}


def _expected_runtime() -> dict[str, str]:
    return {
        key: value for key, value in _canonical_runtime().items() if key != "platform"
    }


def _github_environment(head: str) -> dict[str, str]:
    return {
        "GITHUB_ACTIONS": "true",
        "GITHUB_REPOSITORY": EXPECTED_GITHUB_REPOSITORY,
        "GITHUB_REF": EXPECTED_GITHUB_REF,
        "GITHUB_REF_NAME": EXPECTED_GITHUB_REF_NAME,
        "GITHUB_SHA": head,
        "GITHUB_RUN_ID": "123456789",
        "GITHUB_RUN_NUMBER": "42",
        "GITHUB_WORKFLOW": EXPECTED_GITHUB_WORKFLOW,
        "GITHUB_EVENT_NAME": EXPECTED_GITHUB_EVENT,
        "GITHUB_SERVER_URL": EXPECTED_GITHUB_SERVER_URL,
    }


def test_exact_rows_unique_keys_and_same_baseline_keys(frozen_case) -> None:
    artifact = frozen_case["artifact"]
    baseline = frozen_case["baseline"]
    assert len(artifact) == 1750
    assert not artifact.duplicated(
        ["fold_id", "territory_id", "target_month_id"]
    ).any()
    expected = set(
        baseline[["fold_id", "territory_id", "target_month_id"]]
        .itertuples(index=False, name=None)
    )
    actual = set(
        artifact[["fold_id", "territory_id", "target_month_id"]]
        .itertuples(index=False, name=None)
    )
    assert actual == expected


def test_panel_shape_and_deterministic_sort(frozen_case) -> None:
    artifact = frozen_case["artifact"]
    assert artifact["territory_id"].nunique() == 50
    assert artifact["target_month_id"].nunique() == 35
    assert artifact["fold_id"].value_counts().sort_index().to_dict() == {
        "validation_1": 550,
        "validation_2": 600,
        "validation_3": 600,
    }
    pd.testing.assert_frame_equal(
        artifact, artifact.sort_values(SORT_COLUMNS, ignore_index=True)
    )


def test_available_ets_raw_clipping_candidate_and_operational_contract(
    frozen_case,
) -> None:
    available = frozen_case["artifact"].loc[
        frozen_case["artifact"]["candidate_available"]
    ]
    expected_candidate = np.maximum(available["ets_raw_prediction"], 0.0)
    np.testing.assert_allclose(
        available["ets_candidate_prediction"], expected_candidate, rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(
        available["operational_prediction"],
        available["ets_candidate_prediction"],
        rtol=0,
        atol=1e-12,
    )
    assert np.array_equal(
        available["clipping_applied"].to_numpy(),
        available["ets_raw_prediction"].lt(0).to_numpy(),
    )
    assert not available["availability_fallback_used"].any()
    assert available["fallback_reason"].eq("not_used").all()


def test_negative_raw_is_clipped_to_zero_and_nonnegative_raw_is_not(
    frozen_case,
) -> None:
    available = frozen_case["artifact"].loc[
        frozen_case["artifact"]["candidate_available"]
    ]
    clipped = available.loc[available["ets_raw_prediction"] < 0]
    unclipped = available.loc[available["ets_raw_prediction"] >= 0]
    assert not clipped.empty and not unclipped.empty
    assert clipped["ets_candidate_prediction"].eq(0).all()
    assert clipped["clipping_applied"].all()
    np.testing.assert_array_equal(
        unclipped["ets_candidate_prediction"], unclipped["ets_raw_prediction"]
    )
    assert not unclipped["clipping_applied"].any()


def test_badajoz_id_has_complete_availability_fallback_contract(frozen_case) -> None:
    artifact = frozen_case["artifact"]
    badajoz = artifact.loc[artifact["territory_id"].eq("ES-PROV-06")]
    assert len(badajoz) == 35
    assert set(badajoz["territory_name"]) == {"Badajoz"}
    assert not badajoz["candidate_available"].any()
    assert badajoz["availability_fallback_used"].all()
    assert badajoz["fallback_reason"].eq("training_gap_unsupported").all()
    assert badajoz["ets_candidate_prediction"].isna().all()
    assert badajoz["ets_raw_prediction"].isna().all()
    np.testing.assert_array_equal(
        badajoz["operational_prediction"], badajoz["baseline_prediction"]
    )
    assert set(artifact.loc[artifact["availability_fallback_used"], "territory_id"]) == {
        "ES-PROV-06"
    }


def test_available_candidate_null_is_rejected(frozen_case) -> None:
    artifact = frozen_case["artifact"].copy()
    index = artifact.index[artifact["candidate_available"]][0]
    artifact.loc[index, "ets_candidate_prediction"] = np.nan
    with pytest.raises(AssertionError, match="sin candidate prediction"):
        validate_frozen_artifact(
            artifact, frozen_case["baseline"], frozen_case["config"]
        )


def test_available_operational_mismatch_is_rejected(frozen_case) -> None:
    artifact = frozen_case["artifact"].copy()
    index = artifact.index[artifact["candidate_available"]][0]
    artifact.loc[index, "operational_prediction"] += 1.0
    with pytest.raises(AssertionError, match="Operational ETS"):
        validate_frozen_artifact(
            artifact, frozen_case["baseline"], frozen_case["config"]
        )


@pytest.mark.parametrize(
    "column",
    [
        "actual",
        "baseline_prediction",
        "ets_raw_prediction",
        "ets_candidate_prediction",
        "operational_prediction",
    ],
)
@pytest.mark.parametrize("invalid", [np.inf, -np.inf])
def test_infinite_analytical_values_are_rejected(
    frozen_case, column: str, invalid: float
) -> None:
    artifact = frozen_case["artifact"].copy()
    index = artifact.index[artifact["candidate_available"]][0]
    artifact.loc[index, column] = invalid
    with pytest.raises(AssertionError):
        validate_frozen_artifact(
            artifact, frozen_case["baseline"], frozen_case["config"]
        )


def test_invalid_fallback_reason_is_rejected(frozen_case) -> None:
    artifact = frozen_case["artifact"].copy()
    index = artifact.index[~artifact["candidate_available"]][0]
    artifact.loc[index, "fallback_reason"] = "performance_loss"
    with pytest.raises(AssertionError, match="Fallback reasons"):
        validate_frozen_artifact(
            artifact, frozen_case["baseline"], frozen_case["config"]
        )


def test_missing_training_dates_are_rejected(frozen_case) -> None:
    artifact = frozen_case["artifact"].copy()
    artifact.loc[0, "training_end"] = pd.NA
    with pytest.raises(AssertionError, match="Training start/end"):
        validate_frozen_artifact(
            artifact, frozen_case["baseline"], frozen_case["config"]
        )


def test_cutoff_contract(frozen_case) -> None:
    artifact = frozen_case["artifact"]
    target = pd.PeriodIndex(artifact["target_month_id"], freq="M")
    latest = pd.PeriodIndex(artifact["latest_available_month_id"], freq="M")
    business = pd.PeriodIndex(artifact["business_origin_month_id"], freq="M")
    training_end = pd.PeriodIndex(artifact["training_end"], freq="M")
    assert ((target.asi8 - latest.asi8) == 3).all()
    assert ((target.asi8 - business.asi8) == 1).all()
    assert (training_end <= latest).all()


def test_schema_has_no_b2_b3_interval_or_router_contamination(frozen_case) -> None:
    artifact = frozen_case["artifact"]
    assert list(artifact.columns) == ARTIFACT_COLUMNS
    forbidden = {
        "trend_factor",
        "lower",
        "upper",
        "interval_lower",
        "interval_upper",
        "performance_fallback",
    }
    assert forbidden.isdisjoint(artifact.columns)
    config = frozen_case["config"]
    assert config["candidate"]["screening_status"] == "rejected_screening"
    assert config["ets_candidate"]["screening_status"] == "passed_screening"
    assert config["ets_candidate"]["fallback"]["performance_based"] is False


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("extra", "Schema fisico"),
        ("missing", "Schema fisico"),
        ("dtype", "Dtypes fisicos"),
    ],
)
def test_verify_rejects_physical_schema_drift(
    frozen_case, tmp_path: Path, mutation: str, message: str
) -> None:
    artifact = frozen_case["artifact"].copy()
    if mutation == "extra":
        artifact["unexpected"] = 1
    elif mutation == "missing":
        artifact = artifact.drop(columns=["imputed_months_n"])
    else:
        artifact["training_rows"] = artifact["training_rows"].astype("float64")
    artifact_path, metadata_path = _write_pair(tmp_path, frozen_case, artifact=artifact)
    with pytest.raises(AssertionError, match=message):
        verify_frozen_artifact(artifact_path, metadata_path)


def test_logical_hash_is_order_independent_and_deterministic(frozen_case) -> None:
    artifact = frozen_case["artifact"]
    first = logical_prediction_sha256(artifact)
    second = logical_prediction_sha256(
        artifact.sample(frac=1, random_state=20260817).reset_index(drop=True)
    )
    assert first == second
    assert len(first) == 64


def test_metadata_has_complete_provenance(frozen_case) -> None:
    metadata = frozen_case["metadata"]
    artifact_meta = metadata["artifact"]
    assert artifact_meta["version"] == "1.0.0-b4c"
    assert artifact_meta["row_count"] == 1750
    assert artifact_meta["logical_path"].startswith("data/model_outputs/")
    assert artifact_meta["official_canonical_artifact"] is False
    assert metadata["github"] is None
    assert metadata["logical_hash_contract"]["algorithm"] == LOGICAL_HASH_ALGORITHM
    assert metadata["provenance"]["ets_library_version"] == "0.14.6"
    assert metadata["environment"]["canonical"]["pyarrow"] == "24.0.0"
    assert metadata["storage"]["engine_version"] == "24.0.0"
    assert metadata["schema"]["columns"] == ARTIFACT_COLUMNS
    assert len(metadata["folds"]) == 3


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("artifact", "file_sha256"), "0" * 64),
        (("artifact", "logical_prediction_sha256"), "1" * 64),
        (("artifact", "row_count"), 1749),
        (("artifact", "generating_commit_sha"), "2" * 40),
        (("provenance", "gold", "file_sha256"), "3" * 64),
        (("provenance", "config_sha256"), "4" * 64),
        (("provenance", "ets_model_version"), "9.9.9"),
        (("provenance", "ets_library_version"), "9.9.9"),
        (("environment", "canonical", "python"), "3.14.8"),
        (("folds", 0, "expected_evaluable_rows"), 549),
        (("evaluation", "pooled", "candidate_MAE"), 0.0),
        (("evaluation", "folds", 0, "mae_skill_pct"), 0.0),
        (("evaluation", "screening", "A_pooled_mae_skill_positive"), False),
    ],
)
def test_verify_rejects_stale_or_altered_metadata(
    frozen_case, tmp_path: Path, path: tuple[object, ...], replacement: object
) -> None:
    metadata = copy.deepcopy(frozen_case["metadata"])
    _set_nested(metadata, path, replacement)
    artifact_path, metadata_path = _write_pair(tmp_path, frozen_case, metadata=metadata)
    with pytest.raises(AssertionError, match="Metadata"):
        verify_frozen_artifact(artifact_path, metadata_path)


def test_metadata_from_another_compatible_parquet_is_rejected(
    frozen_case, tmp_path: Path
) -> None:
    other_path = tmp_path / "other.parquet"
    frozen_case["artifact"].to_parquet(
        other_path, index=False, engine="pyarrow", compression="gzip"
    )
    metadata = copy.deepcopy(frozen_case["metadata"])
    metadata["artifact"]["file_sha256"] = hashlib.sha256(
        other_path.read_bytes()
    ).hexdigest()
    artifact_path, metadata_path = _write_pair(
        tmp_path / "pair", frozen_case, metadata=metadata
    )
    with pytest.raises(AssertionError, match="file_sha256"):
        verify_frozen_artifact(artifact_path, metadata_path)


def test_verify_and_reconstruction_do_not_refit(
    frozen_case, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden_fit(*args, **kwargs):
        raise AssertionError("ETS refit forbidden")

    monkeypatch.setattr(ets_module, "fit_ets_forecast", forbidden_fit)
    monkeypatch.setattr(freeze_module, "build_ets_predictions", forbidden_fit)
    verification = verify_frozen_artifact(
        frozen_case["artifact_path"], frozen_case["metadata_path"]
    )
    evaluation = reconstruct_artifact_evaluation(
        frozen_case["artifact"], frozen_case["config"]
    )
    assert verification["invariants"]["rows"] == 1750
    assert evaluation["pooled_metrics"]["n"] == 1750


def test_windows_without_dry_run_fails_before_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    windows = {**_canonical_runtime(), "runner": "windows-11", "platform": "Windows"}
    monkeypatch.setattr(freeze_module, "runtime_environment", lambda: windows)
    monkeypatch.setattr(freeze_module, "_git_head", lambda: "a" * 40)
    artifact_path = tmp_path / "must-not-exist.parquet"
    metadata_path = tmp_path / "must-not-exist.yml"
    with patch.dict(os.environ, {"GITHUB_ACTIONS": "false"}):
        with pytest.raises(RuntimeError, match="solo puede ejecutarse"):
            freeze_predictions(artifact_path, metadata_path)
    assert not artifact_path.exists() and not metadata_path.exists()


def test_local_ubuntu_can_only_be_nonofficial_dry_run() -> None:
    official, context = resolve_generation_context(
        expected_runtime=_expected_runtime(),
        actual_runtime=_canonical_runtime(),
        environment={"GITHUB_ACTIONS": "false"},
        git_head="a" * 40,
        allow_noncanonical_dry_run=True,
    )
    assert official is False and context is None
    with pytest.raises(RuntimeError, match="GitHub Actions/main"):
        resolve_generation_context(
            expected_runtime=_expected_runtime(),
            actual_runtime=_canonical_runtime(),
            environment={"GITHUB_ACTIONS": "false"},
            git_head="a" * 40,
        )


def test_github_actions_non_main_fails_closed() -> None:
    head = "a" * 40
    environment = _github_environment(head)
    environment["GITHUB_REF"] = "refs/heads/feature"
    environment["GITHUB_REF_NAME"] = "feature"
    with pytest.raises(RuntimeError, match="Contexto GitHub"):
        resolve_generation_context(
            expected_runtime=_expected_runtime(),
            actual_runtime=_canonical_runtime(),
            environment=environment,
            git_head=head,
            allow_noncanonical_dry_run=True,
        )


def test_github_actions_main_context_allows_official_generation() -> None:
    head = "a" * 40
    official, context = resolve_generation_context(
        expected_runtime=_expected_runtime(),
        actual_runtime=_canonical_runtime(),
        environment=_github_environment(head),
        git_head=head,
    )
    assert official is True
    assert context is not None
    assert context["github_sha"] == head
    assert context["github_ref_name"] == "main"
    assert context["github_run_url"].endswith("/actions/runs/123456789")


def test_verify_only_cli_is_read_only_and_returns_zero(frozen_case) -> None:
    paths = (frozen_case["artifact_path"], frozen_case["metadata_path"])
    before = {
        path: (path.stat().st_mtime_ns, hashlib.sha256(path.read_bytes()).hexdigest())
        for path in paths
    }
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.models.freeze_ets_v2_predictions",
            "--verify-only",
            "--artifact-path",
            str(frozen_case["artifact_path"]),
            "--metadata-path",
            str(frozen_case["metadata_path"]),
        ],
        cwd=PROJECT_ROOT,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    after = {
        path: (path.stat().st_mtime_ns, hashlib.sha256(path.read_bytes()).hexdigest())
        for path in paths
    }
    assert after == before


def test_verify_only_cli_corrupt_metadata_returns_nonzero(
    frozen_case, tmp_path: Path
) -> None:
    metadata = copy.deepcopy(frozen_case["metadata"])
    metadata["artifact"]["row_count"] = 1
    artifact_path, metadata_path = _write_pair(tmp_path, frozen_case, metadata=metadata)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.models.freeze_ets_v2_predictions",
            "--verify-only",
            "--artifact-path",
            str(artifact_path),
            "--metadata-path",
            str(metadata_path),
        ],
        cwd=PROJECT_ROOT,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0


def test_metrics_and_screening_are_reconstructed_from_artifact(frozen_case) -> None:
    evaluation = reconstruct_artifact_evaluation(
        frozen_case["artifact"], frozen_case["config"]
    )
    assert evaluation["pooled_metrics"]["n"] == 1750
    assert len(evaluation["fold_metrics"]) == 3
    assert len(evaluation["territory_metrics"]) == 50
    assert len(evaluation["origin_metrics"]) == 35
    checks = {
        key: value
        for key, value in evaluation["screening"].items()
        if key.startswith(("A_", "B_", "C_", "D_", "E_", "F_"))
    }
    assert len(checks) == 6 and all(checks.values())
