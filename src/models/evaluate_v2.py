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
    )
except ModuleNotFoundError:
    from modeling_v2_common import (
        PROJECT_ROOT,
        build_backtest_origins,
        cutoff_policy_from_config,
        load_modeling_v2_config,
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
