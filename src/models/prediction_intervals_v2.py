"""Intervalos de prediccion prequential point-in-time para el baseline V2.

El metodo usa errores historicos escalados y un cuantil finito conservador.
La calibracion pooled reconoce dependencia transversal entre provincias y no
se presenta como una garantia exacta bajo independencia.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil
from typing import Any, Mapping

import numpy as np
import pandas as pd

try:
    from src.models.modeling_v2_common import TemporalOrigin
except ModuleNotFoundError:
    from modeling_v2_common import TemporalOrigin


@dataclass(frozen=True)
class PredictionIntervalResult:
    """Contrato inmutable de un rango de prediccion empirico."""

    territory_id: str
    target_month_id: str
    point_prediction: float
    lower: float | None
    upper: float | None
    nominal_level: float
    method_id: str
    calibration_scores_n: int
    calibration_origins_n: int
    calibration_max_target_month_id: str | None
    calibration_quantile: float | None
    interval_available: bool
    unavailable_reason: str | None


@dataclass(frozen=True)
class _CalibrationSummary:
    scores_n: int
    origins_n: int
    maximum_target_month_id: str | None
    quantile: float | None
    unavailable_reason: str | None


def calculate_scaled_absolute_residual(
    actual: float,
    prediction: float,
    seasonal_reference: float,
) -> float:
    """Calcula abs(actual - prediction) / max(seasonal_reference, 1)."""

    values = np.asarray([actual, prediction, seasonal_reference], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("El score requiere valores finitos.")
    if values[2] < 0:
        raise ValueError("La referencia estacional no puede ser negativa.")
    scale = max(float(values[2]), 1.0)
    return float(abs(values[0] - values[1]) / scale)


def finite_sample_order_quantile(
    scores: np.ndarray | list[float] | pd.Series,
    nominal_level: float,
) -> tuple[float, int]:
    """Devuelve el k-esimo menor con k=ceil((n+1)*level), indexado desde 1."""

    level = float(nominal_level)
    if not 0 < level < 1:
        raise ValueError("nominal_level debe estar entre cero y uno.")
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("El cuantil requiere al menos un score.")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Los scores deben ser finitos y no negativos.")
    n = int(len(values))
    k = min(int(ceil((n + 1) * level)), n)
    quantile = float(np.partition(values, k - 1)[k - 1])
    return quantile, k


def build_historical_baseline_score_bank(
    gold_history: pd.DataFrame,
    *,
    baseline_id: str,
    seasonal_reference_lag_months: int = 12,
) -> pd.DataFrame:
    """Reconstruye scores baseline calendario exactos desde toda la Gold."""

    required = {
        "territory_id",
        "territory_level",
        "month_id",
        "overnight_stays_total",
        "complete_month_available",
        "is_provisional",
    }
    missing = sorted(required.difference(gold_history.columns))
    if missing:
        raise ValueError("Gold sin columnas de calibracion: " + ", ".join(missing))
    if int(seasonal_reference_lag_months) != 12:
        raise ValueError("B3 calibra exclusivamente seasonal naive lag-12.")

    history = gold_history.loc[
        gold_history["territory_level"].astype(str).eq("province")
    ].copy()
    if history.duplicated(["territory_id", "month_id"]).any():
        raise ValueError("Gold duplica territory_id/month_id.")
    history["_target_month"] = pd.PeriodIndex(history["month_id"], freq="M")
    history["seasonal_reference_month_id"] = (
        history["_target_month"] - seasonal_reference_lag_months
    ).astype(str)

    targets = history[
        [
            "territory_id",
            "month_id",
            "overnight_stays_total",
            "complete_month_available",
            "is_provisional",
            "seasonal_reference_month_id",
        ]
    ].rename(
        columns={
            "month_id": "target_month_id",
            "overnight_stays_total": "actual",
            "complete_month_available": "target_complete",
            "is_provisional": "target_is_provisional",
        }
    )
    references = history[
        ["territory_id", "month_id", "overnight_stays_total"]
    ].rename(
        columns={
            "month_id": "seasonal_reference_month_id",
            "overnight_stays_total": "seasonal_reference",
        }
    )
    bank = targets.merge(
        references,
        on=["territory_id", "seasonal_reference_month_id"],
        how="inner",
        validate="one_to_one",
    )
    bank["actual"] = pd.to_numeric(bank["actual"], errors="coerce")
    bank["seasonal_reference"] = pd.to_numeric(
        bank["seasonal_reference"], errors="coerce"
    )
    valid = (
        bank["target_complete"].fillna(False)
        & bank["actual"].notna()
        & bank["seasonal_reference"].notna()
        & np.isfinite(bank["actual"])
        & np.isfinite(bank["seasonal_reference"])
        & bank["actual"].ge(0)
        & bank["seasonal_reference"].ge(0)
    )
    bank = bank.loc[valid].copy()
    bank["prediction"] = bank["seasonal_reference"].astype(float)
    bank["score"] = (
        (bank["actual"] - bank["prediction"]).abs()
        / bank["seasonal_reference"].clip(lower=1)
    )
    bank["baseline_id"] = str(baseline_id)
    bank["method_source"] = "point_in_time_v2_baseline_reconstruction"
    return bank.sort_values(
        ["target_month_id", "territory_id"], ignore_index=True
    )


def eligible_calibration_scores(
    score_bank: pd.DataFrame,
    *,
    latest_available_month_id: str,
    baseline_id: str,
) -> pd.DataFrame:
    """Filtra scores cuyo actual ya seria conocido en el origin actual."""

    required = {"target_month_id", "score", "baseline_id"}
    missing = sorted(required.difference(score_bank.columns))
    if missing:
        raise ValueError("Banco sin columnas: " + ", ".join(missing))
    if set(score_bank["baseline_id"].astype(str).unique()) != {baseline_id}:
        raise ValueError("El banco contiene un modelo distinto del baseline.")
    cutoff = pd.Period(latest_available_month_id, freq="M")
    targets = pd.PeriodIndex(score_bank["target_month_id"], freq="M")
    scores = pd.to_numeric(score_bank["score"], errors="coerce")
    eligible = score_bank.loc[
        (targets <= cutoff) & scores.notna() & np.isfinite(scores) & scores.ge(0)
    ].copy()
    if not eligible.empty:
        maximum = pd.Period(eligible["target_month_id"].max(), freq="M")
        if maximum > cutoff:
            raise AssertionError("La calibracion accedio despues del cutoff.")
    return eligible.sort_values(
        ["target_month_id", "territory_id"], ignore_index=True
    )


def _summarize_calibration(
    score_bank: pd.DataFrame,
    *,
    latest_available_month_id: str,
    interval_config: Mapping[str, Any],
) -> _CalibrationSummary:
    eligible = eligible_calibration_scores(
        score_bank,
        latest_available_month_id=latest_available_month_id,
        baseline_id=str(interval_config["baseline_id"]),
    )
    origins_n = int(eligible["target_month_id"].nunique())
    scores_n = int(len(eligible))
    maximum = (
        str(eligible["target_month_id"].max())
        if not eligible.empty
        else None
    )
    if origins_n < int(interval_config["minimum_calibration_origins"]):
        return _CalibrationSummary(
            scores_n,
            origins_n,
            maximum,
            None,
            "insufficient_calibration_origins",
        )
    quantile, _ = finite_sample_order_quantile(
        eligible["score"], float(interval_config["nominal_level"])
    )
    return _CalibrationSummary(
        scores_n, origins_n, maximum, quantile, None
    )


def _interval_from_calibration_summary(
    *,
    territory_id: str,
    target_month_id: str,
    point_prediction: float,
    seasonal_reference: float,
    interval_config: Mapping[str, Any],
    calibration: _CalibrationSummary,
) -> PredictionIntervalResult:
    point = float(point_prediction)
    reference = float(seasonal_reference)
    level = float(interval_config["nominal_level"])
    method_id = str(interval_config["method_id"])
    if not np.isfinite(point) or not np.isfinite(reference):
        return PredictionIntervalResult(
            str(territory_id), str(target_month_id), point,
            None, None, level, method_id, 0, 0, None, None,
            False, "invalid_point_input",
        )
    if point < 0 or reference < 0:
        raise ValueError("Point y referencia deben ser no negativos.")
    if calibration.unavailable_reason is not None:
        return PredictionIntervalResult(
            str(territory_id), str(target_month_id), point,
            None, None, level, method_id,
            calibration.scores_n, calibration.origins_n,
            calibration.maximum_target_month_id, None,
            False, calibration.unavailable_reason,
        )

    if calibration.quantile is None:
        raise AssertionError("Calibracion disponible sin cuantil.")
    margin = calibration.quantile * max(reference, 1.0)
    lower = max(float(interval_config["lower_bound"]), point - margin)
    upper = point + margin
    if not lower <= point <= upper:
        raise AssertionError("El point forecast quedo fuera de su intervalo.")
    return PredictionIntervalResult(
        str(territory_id), str(target_month_id), point,
        float(lower), float(upper), level, method_id,
        calibration.scores_n, calibration.origins_n,
        calibration.maximum_target_month_id, calibration.quantile,
        True, None,
    )


def calculate_prediction_interval(
    *,
    territory_id: str,
    target_month_id: str,
    point_prediction: float,
    seasonal_reference: float,
    origin: TemporalOrigin,
    score_bank: pd.DataFrame,
    interval_config: Mapping[str, Any],
) -> PredictionIntervalResult:
    """Calibra y construye un intervalo simetrico para un point forecast."""

    if str(target_month_id) != origin.target_month_id:
        raise ValueError("El target no coincide con el origin temporal.")
    point = float(point_prediction)
    reference = float(seasonal_reference)
    if not np.isfinite(point) or not np.isfinite(reference):
        return _interval_from_calibration_summary(
            territory_id=territory_id,
            target_month_id=target_month_id,
            point_prediction=point,
            seasonal_reference=reference,
            interval_config=interval_config,
            calibration=_CalibrationSummary(0, 0, None, None, None),
        )
    if point < 0 or reference < 0:
        raise ValueError("Point y referencia deben ser no negativos.")
    calibration = _summarize_calibration(
        score_bank,
        latest_available_month_id=origin.latest_available_month_id,
        interval_config=interval_config,
    )
    return _interval_from_calibration_summary(
        territory_id=territory_id,
        target_month_id=target_month_id,
        point_prediction=point,
        seasonal_reference=reference,
        interval_config=interval_config,
        calibration=calibration,
    )


def calculate_interval_score(
    actual: float,
    lower: float,
    upper: float,
    alpha: float,
) -> float:
    """Calcula width mas penalizaciones de misses para un intervalo central."""

    actual_value = float(actual)
    lower_value = float(lower)
    upper_value = float(upper)
    alpha_value = float(alpha)
    if not 0 < alpha_value < 1:
        raise ValueError("alpha debe estar entre cero y uno.")
    if lower_value > upper_value:
        raise ValueError("lower no puede superar upper.")
    width = upper_value - lower_value
    below_penalty = (
        (2 / alpha_value) * (lower_value - actual_value)
        if actual_value < lower_value
        else 0.0
    )
    above_penalty = (
        (2 / alpha_value) * (actual_value - upper_value)
        if actual_value > upper_value
        else 0.0
    )
    return float(width + below_penalty + above_penalty)


def apply_prequential_intervals(
    baseline_predictions: pd.DataFrame,
    score_bank: pd.DataFrame,
    interval_config: Mapping[str, Any],
) -> pd.DataFrame:
    """Aplica un banco expanding distinto a cada target/origin baseline."""

    required = {
        "fold_id",
        "territory_id",
        "target_month_id",
        "business_origin_month_id",
        "latest_available_month_id",
        "max_training_target_month_id",
        "cutoff_policy_id",
        "actual",
        "prediction",
        "reference",
    }
    missing = sorted(required.difference(baseline_predictions.columns))
    if missing:
        raise ValueError("Baseline sin columnas de intervalo: " + ", ".join(missing))
    if baseline_predictions.duplicated(
        ["fold_id", "territory_id", "target_month_id"]
    ).any():
        raise ValueError("Baseline contiene keys duplicadas.")

    calibration_cache: dict[str, _CalibrationSummary] = {}
    records: list[dict[str, Any]] = []
    for row in baseline_predictions.itertuples(index=False):
        latest_available = str(row.latest_available_month_id)
        if latest_available not in calibration_cache:
            calibration_cache[latest_available] = _summarize_calibration(
                score_bank,
                latest_available_month_id=latest_available,
                interval_config=interval_config,
            )
        result = _interval_from_calibration_summary(
            territory_id=str(row.territory_id),
            target_month_id=str(row.target_month_id),
            point_prediction=float(row.prediction),
            seasonal_reference=float(row.reference),
            interval_config=interval_config,
            calibration=calibration_cache[latest_available],
        )
        records.append(asdict(result))

    intervals = pd.DataFrame(records)
    keys = ["territory_id", "target_month_id"]
    enriched = baseline_predictions.merge(
        intervals,
        on=keys,
        how="left",
        validate="one_to_one",
    )
    if not np.allclose(
        enriched["prediction"], enriched["point_prediction"],
        rtol=0, atol=1e-9,
    ):
        raise AssertionError("B3 modifico una point prediction baseline.")

    available = enriched["interval_available"].fillna(False)
    enriched["covered"] = False
    enriched["miss_below"] = False
    enriched["miss_above"] = False
    enriched["width"] = np.nan
    enriched["normalized_width"] = np.nan
    enriched["interval_score"] = np.nan
    if available.any():
        selected = enriched.loc[available]
        enriched.loc[available, "covered"] = (
            selected["actual"].ge(selected["lower"])
            & selected["actual"].le(selected["upper"])
        )
        enriched.loc[available, "miss_below"] = selected["actual"].lt(
            selected["lower"]
        )
        enriched.loc[available, "miss_above"] = selected["actual"].gt(
            selected["upper"]
        )
        enriched.loc[available, "width"] = (
            selected["upper"] - selected["lower"]
        )
        enriched.loc[available, "normalized_width"] = (
            enriched.loc[available, "width"]
            / selected["point_prediction"].clip(lower=1)
        )
        alpha = float(interval_config["alpha"])
        enriched.loc[available, "interval_score"] = [
            calculate_interval_score(actual, lower, upper, alpha)
            for actual, lower, upper in zip(
                selected["actual"], selected["lower"], selected["upper"]
            )
        ]
    return enriched.sort_values(
        ["target_month_id", "territory_id"], ignore_index=True
    )


def first_interval_eligible_target(
    score_bank: pd.DataFrame,
    *,
    minimum_calibration_origins: int,
    availability_lag_months: int,
) -> str:
    """Deriva el primer target con un ciclo completo de origins observables."""

    if score_bank.empty:
        raise ValueError("Banco de calibracion vacio.")
    first_score = pd.Period(score_bank["target_month_id"].min(), freq="M")
    last_score = pd.Period(score_bank["target_month_id"].max(), freq="M")
    for target in pd.period_range(
        first_score, last_score + availability_lag_months + 1, freq="M"
    ):
        cutoff = target - availability_lag_months
        eligible_origins = pd.PeriodIndex(
            score_bank["target_month_id"], freq="M"
        ).to_series().loc[lambda values: values.le(cutoff)].nunique()
        if int(eligible_origins) >= int(minimum_calibration_origins):
            return str(target)
    raise ValueError("No existe un target con calibracion suficiente.")


def build_current_baseline_intervals(
    gold_history: pd.DataFrame,
    *,
    origin: TemporalOrigin,
    score_bank: pd.DataFrame,
    interval_config: Mapping[str, Any],
    seasonal_reference_lag_months: int = 12,
) -> pd.DataFrame:
    """Calcula intervalos read-only actuales para todas las provincias."""

    history = gold_history.loc[
        gold_history["territory_level"].astype(str).eq("province")
    ].copy()
    names = (
        history[["territory_id", "territory_name"]]
        .drop_duplicates("territory_id")
        .set_index("territory_id")["territory_name"]
        .astype(str)
        .to_dict()
    )
    reference_month = str(
        pd.Period(origin.target_month_id, freq="M")
        - int(seasonal_reference_lag_months)
    )
    references = history.loc[
        history["month_id"].astype(str).eq(reference_month),
        ["territory_id", "overnight_stays_total"],
    ]
    rows: list[dict[str, Any]] = []
    for row in references.itertuples(index=False):
        point = float(row.overnight_stays_total)
        interval = calculate_prediction_interval(
            territory_id=str(row.territory_id),
            target_month_id=origin.target_month_id,
            point_prediction=point,
            seasonal_reference=point,
            origin=origin,
            score_bank=score_bank,
            interval_config=interval_config,
        )
        record = asdict(interval)
        record["territory_name"] = names[str(row.territory_id)]
        if interval.interval_available:
            record["width"] = float(interval.upper - interval.lower)
            record["normalized_width"] = float(
                record["width"] / max(point, 1.0)
            )
        else:
            record["width"] = np.nan
            record["normalized_width"] = np.nan
        rows.append(record)
    return pd.DataFrame(rows).sort_values("territory_id", ignore_index=True)


# ---------------------------------------------------------------------------
# Intervalo operacional B5. Estas funciones son deliberadamente separadas de
# B3: el point es operational_prediction y el baseline es solo una escala.
# ---------------------------------------------------------------------------

OPERATIONAL_INTERVAL_METHOD_ID = (
    "operational_prequential_scaled_absolute_residual_interval_v1"
)
OPERATIONAL_ARTIFACT_PATH = (
    "data/model_outputs/ets_v2_rolling_validation_predictions.parquet"
)


def validate_operational_interval_config(
    interval_config: Mapping[str, Any],
) -> None:
    """Congela la unica especificacion de intervalo operacional B5."""

    expected = {
        "method_id": OPERATIONAL_INTERVAL_METHOD_ID,
        "nominal_level": 0.80,
        "alpha": 0.20,
        "minimum_calibration_origins": 12,
        "calibration_scope": "pooled_territories_expanding_origins",
        "score_prediction_column": "operational_prediction",
        "scale_column": "baseline_prediction",
        "source_artifact": OPERATIONAL_ARTIFACT_PATH,
        "quantile_method": "finite_sample_order_statistic",
        "lower_bound": 0,
        "transversal_dependence_limitation": True,
        "exact_iid_coverage_guarantee": False,
    }
    mismatches = {
        key: (interval_config.get(key), expected_value)
        for key, expected_value in expected.items()
        if interval_config.get(key) != expected_value
    }
    if mismatches:
        raise ValueError(
            f"Configuracion de intervalo operacional no soportada: {mismatches}"
        )


def build_operational_score_bank(
    canonical_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Construye el banco B5 solo desde predicciones operacionales canónicas."""

    required = {
        "territory_id",
        "target_month_id",
        "latest_available_month_id",
        "actual",
        "baseline_prediction",
        "operational_prediction",
    }
    missing = sorted(required.difference(canonical_predictions.columns))
    if missing:
        raise ValueError(
            "Artifact operacional sin columnas: " + ", ".join(missing)
        )
    bank = canonical_predictions.copy()
    if bank.empty:
        raise ValueError("Artifact operacional vacio.")
    keys = ["territory_id", "target_month_id"]
    if bank.duplicated(keys).any():
        raise ValueError("Artifact operacional contiene keys duplicadas.")

    try:
        targets = pd.PeriodIndex(bank["target_month_id"], freq="M")
        cutoffs = pd.PeriodIndex(
            bank["latest_available_month_id"], freq="M"
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Artifact operacional contiene meses no validos."
        ) from error
    if (cutoffs >= targets).any():
        raise ValueError(
            "latest_available_month_id debe preceder al target operacional."
        )

    numeric_columns = [
        "actual",
        "baseline_prediction",
        "operational_prediction",
    ]
    for column in numeric_columns:
        bank[column] = pd.to_numeric(bank[column], errors="coerce")
        values = bank[column]
        if (
            values.isna().any()
            or not np.isfinite(values).all()
            or values.lt(0).any()
        ):
            raise ValueError(
                f"Artifact operacional contiene {column} invalido."
            )

    bank["score"] = (
        (bank["actual"] - bank["operational_prediction"]).abs()
        / bank["baseline_prediction"].clip(lower=1)
    )
    if not np.isfinite(bank["score"]).all() or bank["score"].lt(0).any():
        raise AssertionError("El banco operacional genero scores invalidos.")
    bank["score_prediction_column"] = "operational_prediction"
    bank["scale_column"] = "baseline_prediction"
    bank["interval_method_id"] = OPERATIONAL_INTERVAL_METHOD_ID
    return bank.sort_values(keys[::-1], ignore_index=True)


def eligible_operational_calibration_scores(
    score_bank: pd.DataFrame,
    *,
    latest_available_month_id: str,
) -> pd.DataFrame:
    """Excluye todo residual cuyo actual aun no seria conocido en el cutoff."""

    required = {
        "territory_id",
        "target_month_id",
        "score",
        "score_prediction_column",
        "scale_column",
        "interval_method_id",
    }
    missing = sorted(required.difference(score_bank.columns))
    if missing:
        raise ValueError("Banco operacional sin columnas: " + ", ".join(missing))
    if set(score_bank["score_prediction_column"].astype(str)) != {
        "operational_prediction"
    }:
        raise ValueError("El score operacional no usa operational_prediction.")
    if set(score_bank["scale_column"].astype(str)) != {
        "baseline_prediction"
    }:
        raise ValueError("La escala operacional no usa baseline_prediction.")
    if set(score_bank["interval_method_id"].astype(str)) != {
        OPERATIONAL_INTERVAL_METHOD_ID
    }:
        raise ValueError("El banco no pertenece al metodo operacional B5.")

    cutoff = pd.Period(latest_available_month_id, freq="M")
    targets = pd.PeriodIndex(score_bank["target_month_id"], freq="M")
    scores = pd.to_numeric(score_bank["score"], errors="coerce")
    eligible = score_bank.loc[
        (targets <= cutoff)
        & scores.notna()
        & np.isfinite(scores)
        & scores.ge(0)
    ].copy()
    if not eligible.empty:
        maximum = pd.Period(
            eligible["target_month_id"].max(), freq="M"
        )
        if maximum > cutoff:
            raise AssertionError(
                "La calibracion operacional accedio a un residual futuro."
            )
    return eligible.sort_values(
        ["target_month_id", "territory_id"], ignore_index=True
    )


def _summarize_operational_calibration(
    score_bank: pd.DataFrame,
    *,
    latest_available_month_id: str,
    interval_config: Mapping[str, Any],
) -> _CalibrationSummary:
    validate_operational_interval_config(interval_config)
    eligible = eligible_operational_calibration_scores(
        score_bank,
        latest_available_month_id=latest_available_month_id,
    )
    origins_n = int(eligible["target_month_id"].nunique())
    scores_n = int(len(eligible))
    maximum = (
        str(eligible["target_month_id"].max())
        if not eligible.empty
        else None
    )
    if origins_n < int(interval_config["minimum_calibration_origins"]):
        return _CalibrationSummary(
            scores_n,
            origins_n,
            maximum,
            None,
            "insufficient_calibration_origins",
        )
    quantile, _ = finite_sample_order_quantile(
        eligible["score"], float(interval_config["nominal_level"])
    )
    return _CalibrationSummary(
        scores_n, origins_n, maximum, quantile, None
    )


def _operational_interval_from_summary(
    *,
    territory_id: str,
    target_month_id: str,
    point_prediction: float,
    baseline_prediction: float | None,
    interval_config: Mapping[str, Any],
    calibration: _CalibrationSummary,
) -> PredictionIntervalResult:
    point = float(point_prediction)
    level = float(interval_config["nominal_level"])
    method_id = str(interval_config["method_id"])
    if not np.isfinite(point) or point < 0:
        raise ValueError("El point operacional debe ser finito y no negativo.")
    if baseline_prediction is None or pd.isna(baseline_prediction):
        return PredictionIntervalResult(
            str(territory_id), str(target_month_id), point,
            None, None, level, method_id,
            calibration.scores_n, calibration.origins_n,
            calibration.maximum_target_month_id, calibration.quantile,
            False, "missing_interval_scale",
        )
    scale = float(baseline_prediction)
    if not np.isfinite(scale) or scale < 0:
        raise ValueError(
            "baseline_prediction debe ser finita y no negativa cuando existe."
        )
    if calibration.unavailable_reason is not None:
        return PredictionIntervalResult(
            str(territory_id), str(target_month_id), point,
            None, None, level, method_id,
            calibration.scores_n, calibration.origins_n,
            calibration.maximum_target_month_id, None,
            False, calibration.unavailable_reason,
        )
    if calibration.quantile is None:
        raise AssertionError("Calibracion operacional disponible sin cuantil.")

    margin = calibration.quantile * max(scale, 1.0)
    lower = max(float(interval_config["lower_bound"]), point - margin)
    upper = point + margin
    if not lower <= point <= upper or lower < 0:
        raise AssertionError("Intervalo operacional invalido.")
    return PredictionIntervalResult(
        str(territory_id), str(target_month_id), point,
        float(lower), float(upper), level, method_id,
        calibration.scores_n, calibration.origins_n,
        calibration.maximum_target_month_id, calibration.quantile,
        True, None,
    )


def calculate_current_operational_interval(
    *,
    territory_id: str,
    target_month_id: str,
    point_prediction: float,
    baseline_prediction: float | None,
    origin: TemporalOrigin,
    score_bank: pd.DataFrame,
    interval_config: Mapping[str, Any],
) -> PredictionIntervalResult:
    """Calcula el intervalo B5 actual sin invalidar un point ETS valido."""

    validate_operational_interval_config(interval_config)
    if str(target_month_id) != origin.target_month_id:
        raise ValueError("El target operacional no coincide con el origin.")
    calibration = _summarize_operational_calibration(
        score_bank,
        latest_available_month_id=origin.latest_available_month_id,
        interval_config=interval_config,
    )
    return _operational_interval_from_summary(
        territory_id=territory_id,
        target_month_id=target_month_id,
        point_prediction=point_prediction,
        baseline_prediction=baseline_prediction,
        interval_config=interval_config,
        calibration=calibration,
    )


def apply_operational_prequential_intervals(
    canonical_predictions: pd.DataFrame,
    interval_config: Mapping[str, Any],
) -> pd.DataFrame:
    """Aplica read-only la calibracion expanding al artifact canonico B5."""

    validate_operational_interval_config(interval_config)
    score_bank = build_operational_score_bank(canonical_predictions)
    calibration_cache: dict[str, _CalibrationSummary] = {}
    records: list[dict[str, Any]] = []
    for row in score_bank.itertuples(index=False):
        latest_available = str(row.latest_available_month_id)
        if latest_available not in calibration_cache:
            calibration_cache[latest_available] = (
                _summarize_operational_calibration(
                    score_bank,
                    latest_available_month_id=latest_available,
                    interval_config=interval_config,
                )
            )
        interval = _operational_interval_from_summary(
            territory_id=str(row.territory_id),
            target_month_id=str(row.target_month_id),
            point_prediction=float(row.operational_prediction),
            baseline_prediction=float(row.baseline_prediction),
            interval_config=interval_config,
            calibration=calibration_cache[latest_available],
        )
        records.append(asdict(interval))

    intervals = pd.DataFrame.from_records(records)
    keys = ["territory_id", "target_month_id"]
    enriched = canonical_predictions.merge(
        intervals,
        on=keys,
        how="left",
        validate="one_to_one",
    )
    if not np.allclose(
        enriched["operational_prediction"],
        enriched["point_prediction"],
        rtol=0,
        atol=1e-12,
    ):
        raise AssertionError("El intervalo B5 modifico un point operacional.")

    available = enriched["interval_available"].fillna(False).astype(bool)
    enriched["covered"] = False
    enriched["miss_below"] = False
    enriched["miss_above"] = False
    enriched["width"] = np.nan
    enriched["normalized_width"] = np.nan
    enriched["interval_score"] = np.nan
    if available.any():
        actual = enriched.loc[available, "actual"]
        lower = enriched.loc[available, "lower"]
        upper = enriched.loc[available, "upper"]
        enriched.loc[available, "covered"] = actual.ge(lower) & actual.le(upper)
        enriched.loc[available, "miss_below"] = actual.lt(lower)
        enriched.loc[available, "miss_above"] = actual.gt(upper)
        enriched.loc[available, "width"] = upper - lower
        enriched.loc[available, "normalized_width"] = (
            enriched.loc[available, "width"]
            / enriched.loc[available, "point_prediction"].clip(lower=1)
        )
        alpha = float(interval_config["alpha"])
        enriched.loc[available, "interval_score"] = [
            calculate_interval_score(actual_value, lower_value, upper_value, alpha)
            for actual_value, lower_value, upper_value in zip(actual, lower, upper)
        ]

    calibration_max = enriched.loc[
        available, "calibration_max_target_month_id"
    ]
    if not calibration_max.empty:
        maximum = pd.PeriodIndex(calibration_max, freq="M")
        cutoff = pd.PeriodIndex(
            enriched.loc[available, "latest_available_month_id"], freq="M"
        )
        if (maximum > cutoff).any():
            raise AssertionError("La evaluacion operacional uso residuos futuros.")
    return enriched.sort_values(
        ["target_month_id", "territory_id"], ignore_index=True
    )


def evaluate_operational_prequential_intervals(
    canonical_predictions: pd.DataFrame,
    interval_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Resume sin escribir la validacion prequential del intervalo B5."""

    intervals = apply_operational_prequential_intervals(
        canonical_predictions, interval_config
    )
    available = intervals.loc[
        intervals["interval_available"].fillna(False)
    ].copy()
    fallback_available = available.loc[
        available.get(
            "availability_fallback_used",
            pd.Series(False, index=available.index),
        ).fillna(False)
    ]
    metrics: dict[str, Any] = {
        "total_rows": int(len(intervals)),
        "interval_evaluable_rows": int(len(available)),
        "non_evaluable_rows": int(len(intervals) - len(available)),
        "evaluable_origins": int(available["target_month_id"].nunique()),
        "first_evaluable_target": (
            str(available["target_month_id"].min())
            if not available.empty
            else None
        ),
        "observed_coverage_pct": (
            float(100 * available["covered"].mean())
            if not available.empty
            else float("nan")
        ),
        "median_width": (
            float(available["width"].median())
            if not available.empty
            else float("nan")
        ),
        "mean_width": (
            float(available["width"].mean())
            if not available.empty
            else float("nan")
        ),
        "misses_below": int(available["miss_below"].sum()),
        "misses_above": int(available["miss_above"].sum()),
        "fallback_evaluable_rows": int(len(fallback_available)),
        "fallback_coverage_pct": (
            float(100 * fallback_available["covered"].mean())
            if not fallback_available.empty
            else float("nan")
        ),
        "interval_predictions": intervals,
    }
    return metrics
