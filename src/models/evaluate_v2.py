"""Reproduccion en memoria del baseline bajo el contrato point-in-time V2.

El modulo implementa rolling validation, no abre una ventana test ni persiste
artefactos de evaluacion. La disponibilidad historica es correcta respecto al
cutoff mensual, pero usa el ultimo vintage revisado disponible.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

try:
    from src.models.modeling_v2_common import (
        PROJECT_ROOT,
        build_backtest_origins,
        cutoff_policy_from_config,
        load_modeling_v2_config,
        resolve_information_cutoff,
    )
    from src.models.seasonal_trend_v2 import (
        build_current_seasonal_trend_forecasts,
        build_seasonal_trend_predictions,
    )
except ModuleNotFoundError:
    from modeling_v2_common import (
        PROJECT_ROOT,
        build_backtest_origins,
        cutoff_policy_from_config,
        load_modeling_v2_config,
        resolve_information_cutoff,
    )
    from seasonal_trend_v2 import (
        build_current_seasonal_trend_forecasts,
        build_seasonal_trend_predictions,
    )


DEFAULT_GOLD_PATH = (
    PROJECT_ROOT / "data" / "gold" / "gold_tourism_demand_monthly.parquet"
)
FROZEN_V1_PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "data"
    / "model_outputs"
    / "model_selection_validation_predictions.parquet"
)


@dataclass(frozen=True)
class BaselineComparison:
    """Resumen de equivalencia entre baseline V2 y replicas congeladas V1."""

    v2_rows: int
    v1_comparable_rows: int
    common_rows: int
    missing_in_v1: int
    extra_in_v1: int
    prediction_mismatches: int
    actual_mismatches: int

    @property
    def exact_match(self) -> bool:
        return not any(
            (
                self.missing_in_v1,
                self.extra_in_v1,
                self.prediction_mismatches,
                self.actual_mismatches,
            )
        )


def load_gold_history(path: Path = DEFAULT_GOLD_PATH) -> pd.DataFrame:
    """Lee la Gold autorizada sin modificar ni materializar datasets V2."""

    if not path.exists():
        raise FileNotFoundError(f"No se encontro la Gold autorizada: {path}")
    return pd.read_parquet(path)


def _validate_gold_contract(dataframe: pd.DataFrame) -> pd.DataFrame:
    required = {
        "territory_id",
        "territory_name",
        "territory_level",
        "month_id",
        "overnight_stays_total",
        "complete_month_available",
        "data_status",
        "is_provisional",
        "source_snapshot_id",
        "data_version",
    }
    missing = sorted(required.difference(dataframe.columns))
    if missing:
        raise ValueError("Gold sin columnas V2: " + ", ".join(missing))

    history = dataframe.loc[
        dataframe["territory_level"].astype(str).eq("province")
    ].copy()
    history["_month"] = pd.PeriodIndex(history["month_id"], freq="M")
    duplicated = history.duplicated(["territory_id", "month_id"], keep=False)
    if duplicated.any():
        raise ValueError("Gold duplica territory_id/month_id.")
    return history


def build_baseline_predictions(
    gold_history: pd.DataFrame,
    config: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Reconstruye seasonal naive lag-12 para todos los origins configurados.

    Se conserva una fila de disponibilidad por territorio/target. Cuando falta
    la referencia estacional no se imputa: prediction queda ausente y el motivo
    es ``missing_seasonal_reference``.
    """

    resolved_config = config or load_modeling_v2_config()
    history = _validate_gold_contract(gold_history)
    origins = build_backtest_origins(resolved_config)
    baseline = resolved_config["baseline"]
    lag_months = int(baseline["target_lag_months"])
    missing_reason = str(baseline["missing_reference_reason"])
    policy = cutoff_policy_from_config(resolved_config)
    records: list[pd.DataFrame] = []

    for origin in origins.itertuples(index=False):
        target_month = pd.Period(origin.target_month_id, freq="M")
        reference_month = target_month - lag_months
        latest_available = pd.Period(
            origin.latest_available_month_id, freq="M"
        )
        if reference_month > latest_available:
            raise AssertionError(
                "El baseline intento usar una referencia posterior al cutoff."
            )

        targets = history.loc[history["_month"].eq(target_month)].copy()
        if targets.empty:
            raise ValueError(f"Gold sin target mensual {target_month}.")

        target_view = targets[
            [
                "territory_id",
                "territory_name",
                "overnight_stays_total",
                "complete_month_available",
                "data_status",
                "is_provisional",
                "source_snapshot_id",
                "data_version",
            ]
        ].rename(
            columns={
                "overnight_stays_total": "actual",
                "complete_month_available": "target_complete",
                "data_status": "target_data_status",
                "is_provisional": "target_is_provisional",
                "source_snapshot_id": "target_source_snapshot_id",
                "data_version": "target_data_version",
            }
        )

        references = history.loc[history["_month"].eq(reference_month)]
        reference_view = references[
            [
                "territory_id",
                "overnight_stays_total",
                "data_status",
                "is_provisional",
                "source_snapshot_id",
            ]
        ].rename(
            columns={
                "overnight_stays_total": "reference",
                "data_status": "reference_data_status",
                "is_provisional": "reference_is_provisional",
                "source_snapshot_id": "reference_source_snapshot_id",
            }
        )

        joined = target_view.merge(
            reference_view,
            on="territory_id",
            how="left",
            validate="one_to_one",
        )
        joined["fold_id"] = origin.fold_id
        joined["period"] = origin.period
        joined["target_month_id"] = origin.target_month_id
        joined["business_origin_month_id"] = (
            origin.business_origin_month_id
        )
        joined["latest_available_month_id"] = (
            origin.latest_available_month_id
        )
        joined["max_training_target_month_id"] = (
            origin.max_training_target_month_id
        )
        joined["cutoff_policy_id"] = origin.cutoff_policy_id
        joined["seasonal_reference_month_id"] = str(reference_month)
        joined["baseline_id"] = str(baseline["id"])
        joined["baseline_role"] = str(baseline["role"])
        joined["methodology_id"] = str(
            resolved_config["methodology"]["id"]
        )
        joined["vintage_policy_id"] = str(
            resolved_config["methodology"]["information_reconstruction"]
        )

        joined["actual"] = pd.to_numeric(joined["actual"], errors="coerce")
        joined["reference"] = pd.to_numeric(
            joined["reference"], errors="coerce"
        )
        joined["prediction"] = joined["reference"]

        reasons = pd.Series("available", index=joined.index, dtype="string")
        reasons.loc[joined["actual"].isna()] = "missing_target"
        reasons.loc[~joined["target_complete"].fillna(False)] = (
            "incomplete_target"
        )
        reasons.loc[joined["target_is_provisional"].fillna(True)] = (
            "provisional_target"
        )
        reasons.loc[
            joined["reference"].isna()
            & reasons.eq("available")
        ] = missing_reason
        joined["availability_reason"] = reasons
        joined["availability_status"] = np.where(
            reasons.eq("available"), "available", "unavailable"
        )
        joined["prediction_available"] = reasons.eq("available")
        records.append(joined)

    predictions = pd.concat(records, ignore_index=True)
    predictions = predictions.sort_values(
        ["target_month_id", "territory_id"], ignore_index=True
    )
    if predictions.duplicated(["territory_id", "target_month_id"]).any():
        raise AssertionError("Predicciones V2 duplicadas por territorio/target.")
    if policy.max_training_target_lag_months == policy.latest_available_lag_months:
        unequal = predictions["max_training_target_month_id"].ne(
            predictions["latest_available_month_id"]
        )
        if unequal.any():
            raise AssertionError("El cutoff de labels V2 no coincide con latest.")
    return predictions


def comparable_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Devuelve solo predicciones y targets finales realmente comparables."""

    mask = (
        predictions["prediction_available"].fillna(False)
        & predictions["actual"].notna()
        & predictions["prediction"].notna()
        & ~predictions["target_is_provisional"].fillna(True)
    )
    return predictions.loc[mask].copy().sort_values(
        ["target_month_id", "territory_id"], ignore_index=True
    )


def assert_validation_targets_are_final(predictions: pd.DataFrame) -> None:
    """Protege model selection frente a targets provisionales."""

    provisional = predictions["target_is_provisional"].fillna(True)
    if provisional.any():
        months = sorted(
            predictions.loc[provisional, "target_month_id"].unique().tolist()
        )
        raise AssertionError(
            "Rolling validation contiene targets provisionales: "
            + ", ".join(months)
        )


def validate_expected_baseline_rows(
    predictions: pd.DataFrame,
    config: Mapping[str, Any] | None = None,
) -> dict[str, int]:
    """Valida los recuentos de aceptacion sin convertirlos en predicciones."""

    resolved_config = config or load_modeling_v2_config()
    comparable = comparable_predictions(predictions)
    actual_counts = comparable.groupby("fold_id").size().to_dict()
    expected_counts = {
        str(fold["id"]): int(fold["expected_evaluable_rows"])
        for fold in resolved_config["folds"]
    }
    if actual_counts != expected_counts:
        raise AssertionError(
            f"Recuentos baseline V2 inesperados: {actual_counts}; "
            f"esperados: {expected_counts}."
        )
    return {str(key): int(value) for key, value in actual_counts.items()}


def calculate_metrics(dataframe: pd.DataFrame) -> dict[str, float | int]:
    """Calcula n, MAE, RMSE, WAPE y bias (prediction - actual)."""

    if dataframe.empty:
        return {
            "n": 0,
            "MAE": np.nan,
            "RMSE": np.nan,
            "WAPE_pct": np.nan,
            "bias": np.nan,
        }
    actual = pd.to_numeric(dataframe["actual"], errors="raise").to_numpy(
        dtype=float
    )
    prediction = pd.to_numeric(
        dataframe["prediction"], errors="raise"
    ).to_numpy(dtype=float)
    if not np.isfinite(actual).all() or not np.isfinite(prediction).all():
        raise ValueError("Las metricas requieren actual/prediction finitos.")
    error = prediction - actual
    absolute_error = np.abs(error)
    denominator = float(np.abs(actual).sum())
    return {
        "n": int(len(dataframe)),
        "MAE": float(absolute_error.mean()),
        "RMSE": float(np.sqrt(np.square(error).mean())),
        "WAPE_pct": (
            float(absolute_error.sum() / denominator * 100)
            if denominator != 0
            else np.nan
        ),
        "bias": float(error.mean()),
    }


def calculate_skill_mae_pct(
    baseline_mae: float,
    candidate_mae: float,
) -> float:
    """Calcula skill MAE; positivo mejora y negativo empeora."""

    baseline = float(baseline_mae)
    candidate = float(candidate_mae)
    if baseline < 0 or candidate < 0:
        raise ValueError("MAE no puede ser negativo.")
    if baseline == 0:
        return 0.0 if candidate == 0 else np.nan
    return float(100 * (baseline - candidate) / baseline)


def _with_self_skill(metrics: dict[str, float | int]) -> dict[str, float | int]:
    return {
        **metrics,
        "skill_mae_pct": calculate_skill_mae_pct(
            float(metrics["MAE"]), float(metrics["MAE"])
        ),
    }


def calculate_fold_metrics(
    predictions: pd.DataFrame,
    config: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Calcula metricas por rolling validation fold."""

    resolved_config = config or load_modeling_v2_config()
    comparable = comparable_predictions(predictions)
    rows: list[dict[str, Any]] = []
    for fold in resolved_config["folds"]:
        fold_id = str(fold["id"])
        group = comparable.loc[comparable["fold_id"].eq(fold_id)]
        metrics = _with_self_skill(calculate_metrics(group))
        rows.append(
            {
                "fold_id": fold_id,
                "period": f"{fold['start']}/{fold['end']}",
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def calculate_territory_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Calcula metricas diagnosticas por territorio."""

    comparable = comparable_predictions(predictions)
    rows: list[dict[str, Any]] = []
    for territory_id, group in comparable.groupby("territory_id", sort=True):
        names = group["territory_name"].dropna().astype(str).unique()
        if len(names) != 1:
            raise ValueError(f"Nombre territorial inconsistente: {territory_id}.")
        rows.append(
            {
                "territory_id": str(territory_id),
                "territory_name": names[0],
                **_with_self_skill(calculate_metrics(group)),
            }
        )
    return pd.DataFrame(rows)


def calculate_month_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Calcula metricas por mes calendario, sin crear reglas de fallback."""

    comparable = comparable_predictions(predictions)
    month_number = pd.PeriodIndex(
        comparable["target_month_id"], freq="M"
    ).month
    comparable = comparable.assign(month_number=month_number)
    rows = [
        {
            "month_number": int(month),
            **_with_self_skill(calculate_metrics(group)),
        }
        for month, group in comparable.groupby("month_number", sort=True)
    ]
    return pd.DataFrame(rows)


def calculate_origin_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Calcula metricas para cada origin, incluidos origins sin prediccion."""

    rows: list[dict[str, Any]] = []
    for target_month, group in predictions.groupby(
        "target_month_id", sort=True
    ):
        comparable = comparable_predictions(group)
        metrics = _with_self_skill(calculate_metrics(comparable))
        rows.append(
            {
                "fold_id": str(group["fold_id"].iloc[0]),
                "target_month_id": str(target_month),
                "business_origin_month_id": str(
                    group["business_origin_month_id"].iloc[0]
                ),
                "latest_available_month_id": str(
                    group["latest_available_month_id"].iloc[0]
                ),
                "n_territories": int(metrics.pop("n")),
                **metrics,
            }
        )
    return pd.DataFrame(rows).sort_values(
        "target_month_id", ignore_index=True
    )


def calculate_pooled_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Calcula metricas pooled comparables para los tres folds."""

    metrics = _with_self_skill(
        calculate_metrics(comparable_predictions(predictions))
    )
    return pd.DataFrame([{"fold_id": "validation_pooled", **metrics}])


def compare_with_frozen_v1(
    predictions: pd.DataFrame,
    frozen_predictions: pd.DataFrame | None = None,
    *,
    frozen_path: Path = FROZEN_V1_PREDICTIONS_PATH,
) -> BaselineComparison:
    """Compara keys, actual y baseline con las replicas congeladas V1."""

    frozen = (
        frozen_predictions.copy()
        if frozen_predictions is not None
        else pd.read_parquet(frozen_path)
    )
    fold_ids = sorted(predictions["fold_id"].astype(str).unique())
    frozen = frozen.loc[frozen["evaluation_split"].isin(fold_ids)].copy()
    keys_v1 = ["territory_id", "target_month_id", "evaluation_split"]

    consistency = frozen.groupby(keys_v1, observed=True).agg(
        actual_values=("actual", lambda values: values.nunique(dropna=False)),
        prediction_values=(
            "baseline_prediction",
            lambda values: values.nunique(dropna=False),
        ),
    )
    if (
        consistency["actual_values"].gt(1).any()
        or consistency["prediction_values"].gt(1).any()
    ):
        raise ValueError("Las replicas baseline V1 no son consistentes.")

    v1 = frozen.drop_duplicates(keys_v1)[
        [*keys_v1, "actual", "baseline_prediction"]
    ].rename(
        columns={
            "evaluation_split": "fold_id",
            "actual": "actual_v1",
            "baseline_prediction": "prediction_v1",
        }
    )
    v2 = comparable_predictions(predictions)[
        ["territory_id", "target_month_id", "fold_id", "actual", "prediction"]
    ]
    keys = ["territory_id", "target_month_id", "fold_id"]
    merged = v2.merge(v1, on=keys, how="outer", indicator=True)
    common = merged.loc[merged["_merge"].eq("both")]
    prediction_match = np.isclose(
        common["prediction"],
        common["prediction_v1"],
        rtol=0,
        atol=1e-9,
        equal_nan=True,
    )
    actual_match = np.isclose(
        common["actual"],
        common["actual_v1"],
        rtol=0,
        atol=1e-9,
        equal_nan=True,
    )
    return BaselineComparison(
        v2_rows=int(len(v2)),
        v1_comparable_rows=int(len(v1)),
        common_rows=int(len(common)),
        missing_in_v1=int(merged["_merge"].eq("left_only").sum()),
        extra_in_v1=int(merged["_merge"].eq("right_only").sum()),
        prediction_mismatches=int((~prediction_match).sum()),
        actual_mismatches=int((~actual_match).sum()),
    )


def first_fold_stress_evidence(
    predictions: pd.DataFrame,
    *,
    fold_id: str = "validation_1",
) -> dict[str, Any]:
    """Resume el primer fold como stress period, sin atribucion causal."""

    fold = predictions.loc[predictions["fold_id"].eq(fold_id)].copy()
    comparable = comparable_predictions(fold)
    actual_volume = float(comparable["actual"].sum())
    reference_volume = float(comparable["reference"].sum())
    missing = sorted(
        fold.loc[
            ~fold["prediction_available"].fillna(False), "target_month_id"
        ].unique()
    )
    return {
        "fold_id": fold_id,
        "interpretation": "stress_period",
        "n": int(len(comparable)),
        "actual_volume": actual_volume,
        "reference_volume": reference_volume,
        "absolute_error": float(
            (comparable["prediction"] - comparable["actual"]).abs().sum()
        ),
        "actual_reference_ratio": (
            actual_volume / reference_volume
            if reference_volume != 0
            else np.nan
        ),
        "missing_origin_months": missing,
        "missing_prediction_rows": int(
            (~fold["prediction_available"].fillna(False)).sum()
        ),
    }


def reproduce_baseline_in_memory(
    gold_history: pd.DataFrame | None = None,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Ejecuta B1 en memoria y devuelve tablas preparadas para fases futuras."""

    resolved_config = config or load_modeling_v2_config()
    source_path = Path(str(resolved_config["source"]["path"]))
    if not source_path.is_absolute():
        source_path = PROJECT_ROOT / source_path
    predictions = build_baseline_predictions(
        (
            gold_history
            if gold_history is not None
            else load_gold_history(source_path)
        ),
        resolved_config,
    )
    assert_validation_targets_are_final(predictions)
    validate_expected_baseline_rows(predictions, resolved_config)
    return {
        "predictions": predictions,
        "comparable_predictions": comparable_predictions(predictions),
        "pooled_metrics": calculate_pooled_metrics(predictions),
        "fold_metrics": calculate_fold_metrics(predictions, resolved_config),
        "territory_metrics": calculate_territory_metrics(predictions),
        "month_metrics": calculate_month_metrics(predictions),
        "origin_metrics": calculate_origin_metrics(predictions),
        "first_fold_stress": first_fold_stress_evidence(predictions),
        "v1_comparison": asdict(compare_with_frozen_v1(predictions)),
    }


def calculate_skill_wape_pct(
    baseline_wape: float,
    candidate_wape: float,
) -> float:
    """Calcula skill WAPE con el mismo signo positivo de mejora."""

    baseline = float(baseline_wape)
    candidate = float(candidate_wape)
    if baseline < 0 or candidate < 0:
        raise ValueError("WAPE no puede ser negativo.")
    if baseline == 0:
        return 0.0 if candidate == 0 else np.nan
    return float(100 * (baseline - candidate) / baseline)


def _metrics_for_prediction(
    dataframe: pd.DataFrame,
    prediction_column: str,
) -> dict[str, float | int]:
    return calculate_metrics(
        dataframe[["actual", prediction_column]].rename(
            columns={prediction_column: "prediction"}
        )
    )


def assert_candidate_rows_are_paired(
    baseline_predictions: pd.DataFrame,
    candidate_predictions: pd.DataFrame,
) -> dict[str, int]:
    """Exige identidad exacta de keys, actual y baseline en vista operativa."""

    keys = ["fold_id", "territory_id", "target_month_id"]
    baseline = baseline_predictions[
        [*keys, "actual", "prediction"]
    ].rename(
        columns={
            "actual": "actual_expected",
            "prediction": "baseline_expected",
        }
    )
    candidate = candidate_predictions[
        [*keys, "actual", "baseline_prediction"]
    ]
    if baseline.duplicated(keys).any() or candidate.duplicated(keys).any():
        raise AssertionError("La comparacion emparejada contiene keys duplicadas.")
    merged = baseline.merge(candidate, on=keys, how="outer", indicator=True)
    missing = int(merged["_merge"].eq("left_only").sum())
    extra = int(merged["_merge"].eq("right_only").sum())
    if missing or extra:
        raise AssertionError(
            f"Keys operativas no coinciden: missing={missing}, extra={extra}."
        )
    common = merged.loc[merged["_merge"].eq("both")]
    actual_mismatch = int(
        (~np.isclose(
            common["actual_expected"], common["actual"],
            rtol=0, atol=1e-9, equal_nan=True,
        )).sum()
    )
    baseline_mismatch = int(
        (~np.isclose(
            common["baseline_expected"], common["baseline_prediction"],
            rtol=0, atol=1e-9, equal_nan=True,
        )).sum()
    )
    if actual_mismatch or baseline_mismatch:
        raise AssertionError(
            "Candidate y baseline no comparten actual/prediccion baseline."
        )
    return {
        "baseline_rows": int(len(baseline)),
        "candidate_rows": int(len(candidate)),
        "common_rows": int(len(common)),
        "missing_keys": missing,
        "extra_keys": extra,
        "actual_mismatches": actual_mismatch,
        "baseline_mismatches": baseline_mismatch,
    }


def calculate_paired_metrics(
    dataframe: pd.DataFrame,
    *,
    candidate_prediction_column: str,
) -> dict[str, float | int]:
    """Compara errores baseline/candidato sobre las mismas filas."""

    required = {"actual", "baseline_prediction", candidate_prediction_column}
    missing = sorted(required.difference(dataframe.columns))
    if missing:
        raise ValueError("Comparacion sin columnas: " + ", ".join(missing))
    paired = dataframe.loc[
        dataframe[list(required)].notna().all(axis=1)
    ].copy()
    if len(paired) != len(dataframe):
        raise AssertionError("La comparacion contiene filas no emparejadas.")
    baseline = _metrics_for_prediction(paired, "baseline_prediction")
    candidate = _metrics_for_prediction(paired, candidate_prediction_column)
    return {
        "n": int(len(paired)),
        "baseline_MAE": float(baseline["MAE"]),
        "candidate_MAE": float(candidate["MAE"]),
        "baseline_RMSE": float(baseline["RMSE"]),
        "candidate_RMSE": float(candidate["RMSE"]),
        "baseline_WAPE_pct": float(baseline["WAPE_pct"]),
        "candidate_WAPE_pct": float(candidate["WAPE_pct"]),
        "baseline_bias": float(baseline["bias"]),
        "candidate_bias": float(candidate["bias"]),
        "mae_skill_pct": calculate_skill_mae_pct(
            float(baseline["MAE"]), float(candidate["MAE"])
        ),
        "wape_skill_pct": calculate_skill_wape_pct(
            float(baseline["WAPE_pct"]), float(candidate["WAPE_pct"])
        ),
    }


def _comparison_outcome(skill: float, *, tolerance: float = 1e-12) -> str:
    if skill > tolerance:
        return "win"
    if skill < -tolerance:
        return "loss"
    return "tie"


def _group_candidate_metrics(
    candidate_predictions: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_arg: str | list[str] = (
        group_columns[0] if len(group_columns) == 1 else group_columns
    )
    for group_key, group in candidate_predictions.groupby(
        group_arg, sort=True, observed=True
    ):
        key_values = (
            (group_key,) if len(group_columns) == 1 else tuple(group_key)
        )
        metrics = calculate_paired_metrics(
            group, candidate_prediction_column="operational_prediction"
        )
        available = int(group["candidate_available"].sum())
        fallback_rows = int(group["fallback_used"].sum())
        rows.append(
            {
                **dict(zip(group_columns, key_values)),
                "candidate_available_rows": available,
                "fallback_rows": fallback_rows,
                "candidate_coverage_pct": available / len(group) * 100,
                **metrics,
                "outcome": _comparison_outcome(
                    float(metrics["mae_skill_pct"])
                ),
            }
        )
    return pd.DataFrame(rows)


def calculate_candidate_fold_metrics(
    candidate_predictions: pd.DataFrame,
    config: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Compara baseline y candidato operativo por rolling validation fold."""

    resolved_config = config or load_modeling_v2_config()
    metrics = _group_candidate_metrics(candidate_predictions, ["fold_id"])
    periods = {
        str(fold["id"]): f"{fold['start']}/{fold['end']}"
        for fold in resolved_config["folds"]
    }
    metrics.insert(
        1, "period", metrics["fold_id"].map(periods)
    )
    return metrics


def calculate_candidate_territory_metrics(
    candidate_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Compara baseline y candidato operativo por provincia."""

    metrics = _group_candidate_metrics(
        candidate_predictions, ["territory_id"]
    )
    names = (
        candidate_predictions[["territory_id", "territory_name"]]
        .drop_duplicates()
        .groupby("territory_id")["territory_name"]
        .agg(lambda values: values.astype(str).unique().tolist())
    )
    if names.map(len).ne(1).any():
        raise ValueError("Nombres territoriales inconsistentes.")
    metrics.insert(
        1,
        "territory_name",
        metrics["territory_id"].map(names.map(lambda values: values[0])),
    )
    return metrics


def calculate_candidate_month_metrics(
    candidate_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Compara por mes calendario sin crear reglas mensuales."""

    data = candidate_predictions.copy()
    data["month_number"] = pd.PeriodIndex(
        data["target_month_id"], freq="M"
    ).month
    return _group_candidate_metrics(data, ["month_number"])


def calculate_candidate_origin_metrics(
    candidate_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Compara por target/origin sobre todos los territorios disponibles."""

    metrics = _group_candidate_metrics(
        candidate_predictions, ["fold_id", "target_month_id"]
    )
    metrics = metrics.rename(columns={"n": "n_territories"})
    return metrics.sort_values("target_month_id", ignore_index=True)


def calculate_trend_factor_distribution(
    candidate_predictions: pd.DataFrame,
    *,
    by_fold: bool,
) -> pd.DataFrame:
    """Resume factores raw disponibles pooled o por fold."""

    available = candidate_predictions.loc[
        candidate_predictions["candidate_available"]
        & candidate_predictions["trend_factor"].notna()
    ].copy()
    groups = (
        available.groupby("fold_id", sort=True)
        if by_fold
        else [("validation_pooled", available)]
    )
    rows: list[dict[str, Any]] = []
    for fold_id, group in groups:
        factors = pd.to_numeric(group["trend_factor"], errors="raise")
        rows.append(
            {
                "fold_id": str(fold_id),
                "n": int(len(factors)),
                "min": float(factors.min()),
                "P01": float(factors.quantile(0.01)),
                "P05": float(factors.quantile(0.05)),
                "P25": float(factors.quantile(0.25)),
                "median": float(factors.median()),
                "P75": float(factors.quantile(0.75)),
                "P95": float(factors.quantile(0.95)),
                "P99": float(factors.quantile(0.99)),
                "max": float(factors.max()),
            }
        )
    return pd.DataFrame(rows)


def largest_candidate_adjustments(
    candidate_predictions: pd.DataFrame,
    *,
    limit: int = 20,
) -> pd.DataFrame:
    """Devuelve los mayores cambios relativos raw frente al seasonal naive."""

    available = candidate_predictions.loc[
        candidate_predictions["candidate_available"]
    ].copy()
    denominator = available["baseline_prediction"].abs()
    available["relative_adjustment"] = np.where(
        denominator.gt(0),
        (available["candidate_prediction"] - available["baseline_prediction"])
        .abs()
        / denominator,
        np.nan,
    )
    available["candidate_error"] = (
        available["candidate_prediction"] - available["actual"]
    )
    available["baseline_error"] = (
        available["baseline_prediction"] - available["actual"]
    )
    columns = [
        "fold_id",
        "territory_id",
        "territory_name",
        "target_month_id",
        "baseline_prediction",
        "trend_factor",
        "candidate_prediction",
        "actual",
        "candidate_error",
        "baseline_error",
        "relative_adjustment",
    ]
    return available.sort_values(
        "relative_adjustment", ascending=False
    )[columns].head(limit).reset_index(drop=True)


def screen_seasonal_trend_candidate(
    pooled_metrics: Mapping[str, Any],
    fold_metrics: pd.DataFrame,
    territory_metrics: pd.DataFrame,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Aplica el screening determinista B2, no una gate estadistica final."""

    resolved_config = config or load_modeling_v2_config()
    screening = resolved_config["screening"]
    if (
        screening.get("bias_guardrail")
        != "absolute_candidate_bias_not_greater_than_baseline"
    ):
        raise ValueError("Guardrail de bias B2 no soportado.")
    fold_skills = pd.to_numeric(fold_metrics["mae_skill_pct"])
    territory_skills = pd.to_numeric(territory_metrics["mae_skill_pct"])
    fold_wins = int(fold_skills.gt(0).sum())
    territory_wins = int(territory_skills.gt(0).sum())
    if territory_skills.empty:
        raise ValueError("El screening requiere metricas territoriales.")
    territory_win_fraction = territory_wins / len(territory_skills)

    checks = {
        "A_pooled_mae_skill_positive": (
            float(pooled_metrics["mae_skill_pct"]) > 0
        ),
        "B_at_least_two_folds_positive": (
            fold_wins >= int(screening["minimum_positive_fold_count"])
        ),
        "C_median_fold_skill_positive": float(fold_skills.median()) > 0,
        "D_more_than_half_provinces_positive": (
            territory_win_fraction
            > float(screening["province_win_fraction_strictly_greater_than"])
        ),
        "E_median_territorial_skill_positive": (
            float(territory_skills.median()) > 0
        ),
        "F_absolute_bias_not_worse": (
            abs(float(pooled_metrics["candidate_bias"]))
            <= abs(float(pooled_metrics["baseline_bias"]))
        ),
    }
    if all(checks.values()):
        conclusion = "SEASONAL TREND CANDIDATE PASSES SCREENING"
    elif checks["A_pooled_mae_skill_positive"]:
        conclusion = "SEASONAL TREND CANDIDATE UNSTABLE"
    else:
        conclusion = "SEASONAL TREND CANDIDATE FAILS SCREENING"
    return {
        **checks,
        "fold_wins": fold_wins,
        "territory_wins": territory_wins,
        "territory_win_fraction": territory_win_fraction,
        "median_fold_skill_pct": float(fold_skills.median()),
        "median_territorial_skill_pct": float(territory_skills.median()),
        "conclusion": conclusion,
    }


def build_current_candidate_illustration(
    gold_history: pd.DataFrame,
    *,
    target_month_id: str,
    config: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Genera una ilustracion read-only para un target mensual futuro."""

    resolved_config = config or load_modeling_v2_config()
    policy = cutoff_policy_from_config(resolved_config)
    origin = resolve_information_cutoff(target_month_id, policy)
    return build_current_seasonal_trend_forecasts(
        gold_history, origin, resolved_config["candidate"]
    )


def reproduce_seasonal_trend_in_memory(
    gold_history: pd.DataFrame | None = None,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Ejecuta el screening experimental B2 completamente en memoria."""

    resolved_config = config or load_modeling_v2_config()
    source_path = Path(str(resolved_config["source"]["path"]))
    if not source_path.is_absolute():
        source_path = PROJECT_ROOT / source_path
    gold = (
        gold_history
        if gold_history is not None
        else load_gold_history(source_path)
    )
    baseline_result = reproduce_baseline_in_memory(gold, resolved_config)
    baseline = baseline_result["comparable_predictions"]
    candidate = build_seasonal_trend_predictions(
        baseline, gold, resolved_config["candidate"]
    )
    paired_invariant = assert_candidate_rows_are_paired(baseline, candidate)

    pure = candidate.loc[candidate["candidate_available"]].copy()
    pure_metrics = calculate_paired_metrics(
        pure, candidate_prediction_column="candidate_prediction"
    )
    operational_metrics = calculate_paired_metrics(
        candidate, candidate_prediction_column="operational_prediction"
    )
    fold_metrics = calculate_candidate_fold_metrics(candidate, resolved_config)
    territory_metrics = calculate_candidate_territory_metrics(candidate)
    month_metrics = calculate_candidate_month_metrics(candidate)
    origin_metrics = calculate_candidate_origin_metrics(candidate)
    post_stress = candidate.loc[
        candidate["fold_id"].isin(["validation_2", "validation_3"])
    ]
    post_stress_metrics = calculate_paired_metrics(
        post_stress, candidate_prediction_column="operational_prediction"
    )
    screening = screen_seasonal_trend_candidate(
        operational_metrics,
        fold_metrics,
        territory_metrics,
        resolved_config,
    )
    return {
        "baseline": baseline_result,
        "candidate_predictions": candidate,
        "pure_candidate_predictions": pure,
        "paired_invariant": paired_invariant,
        "pure_metrics": pure_metrics,
        "operational_metrics": operational_metrics,
        "fold_metrics": fold_metrics,
        "territory_metrics": territory_metrics,
        "month_metrics": month_metrics,
        "origin_metrics": origin_metrics,
        "trend_factor_pooled": calculate_trend_factor_distribution(
            candidate, by_fold=False
        ),
        "trend_factor_by_fold": calculate_trend_factor_distribution(
            candidate, by_fold=True
        ),
        "extreme_adjustments": largest_candidate_adjustments(candidate),
        "fallback_reasons": (
            candidate.loc[candidate["fallback_used"], "fallback_reason"]
            .value_counts()
            .rename_axis("fallback_reason")
            .reset_index(name="rows")
        ),
        "post_stress_metrics": post_stress_metrics,
        "screening": screening,
    }
