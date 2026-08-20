"""Holt-Winters local y point-in-time para el ultimo candidato V2.

La especificacion es unica y preespecificada: nivel, tendencia aditiva
amortiguada y estacionalidad aditiva mensual de periodo 12. Cada ajuste usa
solo la serie de una provincia hasta el cutoff de informacion del origin.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any, Mapping
import warnings

import numpy as np
import pandas as pd
import statsmodels
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.holtwinters import ExponentialSmoothing

try:
    from src.models.modeling_v2_common import TemporalOrigin
except ModuleNotFoundError:
    from modeling_v2_common import TemporalOrigin


# En este punto config, shape e historia ya han superado sus contratos. Estas
# son las unicas excepciones de ajuste/numericas que se convierten en una
# indisponibilidad operacional; defectos de programacion deben propagarse.
EXPECTED_ETS_FIT_EXCEPTIONS = (
    FloatingPointError,
    OverflowError,
    np.linalg.LinAlgError,
    ValueError,
)


@dataclass(frozen=True)
class ETSTrainingSeries:
    """Serie mensual preparada sin acceder a observaciones futuras."""

    available: bool
    values: tuple[float, ...]
    training_rows: int
    observed_rows: int
    training_start: str | None
    training_end: str | None
    imputed_month_ids: tuple[str, ...]
    unavailable_reason: str | None


@dataclass(frozen=True)
class ETSForecastResult:
    """Resultado inmutable de un fit Holt-Winters para un origin."""

    territory_id: str
    target_month_id: str
    latest_available_month_id: str
    effective_horizon_steps: int
    prediction: float | None
    raw_prediction: float | None
    candidate_available: bool
    unavailable_reason: str | None
    training_rows: int
    training_observed_rows: int
    training_start: str | None
    training_end: str | None
    model_id: str
    clipping_applied: bool
    fit_attempted: bool
    fit_seconds: float
    imputed_months_n: int
    imputed_month_ids: tuple[str, ...]
    fit_warning_count: int
    fit_warning_messages: tuple[str, ...]


def validate_ets_config(config: Mapping[str, Any]) -> None:
    """Impide convertir B4 en una busqueda de hiperparametros."""

    expected = {
        "id": "holt_winters_additive_damped_v1",
        "library": "statsmodels",
        "library_version": "0.14.6",
        "scope": "local_per_territory",
        "frequency": "monthly",
        "trend": "add",
        "damped_trend": True,
        "seasonal": "add",
        "seasonal_periods": 12,
        "initialization_method": "estimated",
        "optimized": True,
        "optimizer": "L-BFGS-B",
        "use_brute": False,
        "remove_bias": False,
        "use_boxcox": False,
        "minimum_training_months": 60,
        "business_horizon_months": 1,
        "effective_horizon_steps": 3,
        "gap_policy": "causal_observed_seasonal_lag_12",
        "negative_strategy": "clip_zero",
    }
    mismatches = {
        key: (config.get(key), expected_value)
        for key, expected_value in expected.items()
        if config.get(key) != expected_value
    }
    if mismatches:
        raise ValueError(f"Configuracion ETS B4 no soportada: {mismatches}")
    if statsmodels.__version__ != str(config["library_version"]):
        raise RuntimeError(
            "Version statsmodels distinta de la congelada: "
            f"{statsmodels.__version__}."
        )
    if bool(config["fallback"].get("performance_based")):
        raise ValueError("B4 prohibe fallback basado en performance.")


def resolve_effective_horizon(
    origin: TemporalOrigin,
    config: Mapping[str, Any],
) -> int:
    """Valida horizontes de negocio y efectivo sin llamarlo one-step."""

    target = pd.Period(origin.target_month_id, freq="M")
    business_origin = pd.Period(origin.business_origin_month_id, freq="M")
    latest = pd.Period(origin.latest_available_month_id, freq="M")
    business_horizon = target.ordinal - business_origin.ordinal
    effective_horizon = target.ordinal - latest.ordinal
    if business_horizon != int(config["business_horizon_months"]):
        raise ValueError("Horizonte de negocio distinto del congelado.")
    if effective_horizon != int(config["effective_horizon_steps"]):
        raise ValueError("Horizonte efectivo distinto del congelado.")
    if latest >= target:
        raise AssertionError("El cutoff ETS no puede alcanzar el target.")
    return effective_horizon


def prepare_ets_training_series(
    territory_history: pd.DataFrame,
    origin: TemporalOrigin,
    config: Mapping[str, Any],
) -> ETSTrainingSeries:
    """Construye un calendario mensual causal con imputacion lag-12 observada.

    Los gaps nunca se eliminan ni se interpolan. Un mes ausente solo puede
    recibir el valor realmente observado del mismo mes del ano anterior. Una
    imputacion previa no puede alimentar otra imputacion.
    """

    required = {
        "territory_id",
        "month_id",
        "overnight_stays_total",
        "complete_month_available",
    }
    missing = sorted(required.difference(territory_history.columns))
    if missing:
        raise ValueError("Historia ETS sin columnas: " + ", ".join(missing))
    if territory_history["territory_id"].astype(str).nunique() > 1:
        raise ValueError("Un fit ETS no puede mezclar territorios.")
    if territory_history.duplicated("month_id").any():
        raise ValueError("Historia ETS duplica month_id.")

    cutoff = pd.Period(origin.latest_available_month_id, freq="M")
    history = territory_history.copy()
    history["_month"] = pd.PeriodIndex(history["month_id"], freq="M")
    history = history.loc[history["_month"] <= cutoff].copy()
    if history.empty:
        return ETSTrainingSeries(
            False, (), 0, 0, None, None, (), "insufficient_history"
        )
    if (history["_month"] > cutoff).any():
        raise AssertionError("ETS accedio a una observacion futura.")

    values = pd.to_numeric(
        history["overnight_stays_total"], errors="coerce"
    )
    valid = (
        history["complete_month_available"].fillna(False)
        & values.notna()
        & np.isfinite(values)
        & values.ge(0)
    )
    observed = {
        period: float(value)
        for period, value in zip(history.loc[valid, "_month"], values[valid])
    }
    minimum = int(config["minimum_training_months"])
    if len(observed) < minimum:
        return ETSTrainingSeries(
            False,
            (),
            0,
            len(observed),
            str(history["_month"].min()),
            str(cutoff),
            (),
            "insufficient_history",
        )

    start = history["_month"].min()
    expected = pd.period_range(start, cutoff, freq="M")
    prepared: list[float] = []
    imputed: list[str] = []
    for month in expected:
        if month in observed:
            prepared.append(observed[month])
            continue
        seasonal_source = month - 12
        if seasonal_source not in observed:
            return ETSTrainingSeries(
                False,
                (),
                len(expected),
                len(observed),
                str(start),
                str(cutoff),
                tuple(imputed),
                "training_gap_unsupported",
            )
        prepared.append(observed[seasonal_source])
        imputed.append(str(month))

    array = np.asarray(prepared, dtype=float)
    if len(array) < minimum:
        return ETSTrainingSeries(
            False,
            (),
            len(array),
            len(observed),
            str(start),
            str(cutoff),
            tuple(imputed),
            "insufficient_history",
        )
    if not np.isfinite(array).all() or (array < 0).any():
        return ETSTrainingSeries(
            False,
            (),
            len(array),
            len(observed),
            str(start),
            str(cutoff),
            tuple(imputed),
            "training_gap_unsupported",
        )
    return ETSTrainingSeries(
        True,
        tuple(float(value) for value in array),
        len(array),
        len(observed),
        str(start),
        str(cutoff),
        tuple(imputed),
        None,
    )


def _unavailable_forecast(
    *,
    territory_id: str,
    origin: TemporalOrigin,
    config: Mapping[str, Any],
    training: ETSTrainingSeries,
    reason: str,
    fit_attempted: bool = False,
    fit_seconds: float = 0.0,
    warning_messages: tuple[str, ...] = (),
) -> ETSForecastResult:
    return ETSForecastResult(
        territory_id=str(territory_id),
        target_month_id=origin.target_month_id,
        latest_available_month_id=origin.latest_available_month_id,
        effective_horizon_steps=int(config["effective_horizon_steps"]),
        prediction=None,
        raw_prediction=None,
        candidate_available=False,
        unavailable_reason=reason,
        training_rows=training.training_rows,
        training_observed_rows=training.observed_rows,
        training_start=training.training_start,
        training_end=training.training_end,
        model_id=str(config["id"]),
        clipping_applied=False,
        fit_attempted=fit_attempted,
        fit_seconds=float(fit_seconds),
        imputed_months_n=len(training.imputed_month_ids),
        imputed_month_ids=training.imputed_month_ids,
        fit_warning_count=len(warning_messages),
        fit_warning_messages=warning_messages,
    )


def fit_ets_forecast(
    territory_history: pd.DataFrame,
    origin: TemporalOrigin,
    config: Mapping[str, Any],
) -> ETSForecastResult:
    """Ajusta una unica ETS local y selecciona el tercer forecast step."""

    validate_ets_config(config)
    horizon = resolve_effective_horizon(origin, config)
    territory_values = territory_history["territory_id"].astype(str).unique()
    if len(territory_values) != 1:
        raise ValueError("El fit ETS requiere exactamente un territorio.")
    territory_id = str(territory_values[0])
    training = prepare_ets_training_series(territory_history, origin, config)
    if not training.available:
        return _unavailable_forecast(
            territory_id=territory_id,
            origin=origin,
            config=config,
            training=training,
            reason=str(training.unavailable_reason),
        )

    started = perf_counter()
    caught_messages: tuple[str, ...] = ()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = ExponentialSmoothing(
                np.asarray(training.values, dtype=float),
                trend=str(config["trend"]),
                damped_trend=bool(config["damped_trend"]),
                seasonal=str(config["seasonal"]),
                seasonal_periods=int(config["seasonal_periods"]),
                initialization_method=str(config["initialization_method"]),
                use_boxcox=bool(config["use_boxcox"]),
                missing="raise",
            )
            fitted = model.fit(
                optimized=bool(config["optimized"]),
                remove_bias=bool(config["remove_bias"]),
                use_brute=bool(config["use_brute"]),
                method=str(config["optimizer"]),
            )
            forecast = np.asarray(fitted.forecast(horizon), dtype=float)
        caught_messages = tuple(
            f"{item.category.__name__}: {item.message}" for item in caught
        )
    except EXPECTED_ETS_FIT_EXCEPTIONS as exc:
        elapsed = perf_counter() - started
        return _unavailable_forecast(
            territory_id=territory_id,
            origin=origin,
            config=config,
            training=training,
            reason="fit_failure",
            fit_attempted=True,
            fit_seconds=elapsed,
            warning_messages=(f"{type(exc).__name__}: {exc}",),
        )
    elapsed = perf_counter() - started

    convergence_warning = any(
        item.category is ConvergenceWarning for item in caught
    )
    mle_retvals = getattr(fitted, "mle_retvals", None)
    optimizer_success = True
    if isinstance(mle_retvals, Mapping):
        optimizer_success = bool(mle_retvals.get("success", True))
    elif hasattr(mle_retvals, "success"):
        optimizer_success = bool(mle_retvals.success)
    if convergence_warning or not optimizer_success:
        return _unavailable_forecast(
            territory_id=territory_id,
            origin=origin,
            config=config,
            training=training,
            reason="fit_failure",
            fit_attempted=True,
            fit_seconds=elapsed,
            warning_messages=caught_messages,
        )
    if len(forecast) != horizon or not np.isfinite(forecast).all():
        return _unavailable_forecast(
            territory_id=territory_id,
            origin=origin,
            config=config,
            training=training,
            reason="invalid_forecast",
            fit_attempted=True,
            fit_seconds=elapsed,
            warning_messages=caught_messages,
        )

    raw = float(forecast[horizon - 1])
    if not np.isfinite(raw):
        return _unavailable_forecast(
            territory_id=territory_id,
            origin=origin,
            config=config,
            training=training,
            reason="invalid_forecast",
            fit_attempted=True,
            fit_seconds=elapsed,
            warning_messages=caught_messages,
        )
    prediction = max(raw, 0.0)
    return ETSForecastResult(
        territory_id=territory_id,
        target_month_id=origin.target_month_id,
        latest_available_month_id=origin.latest_available_month_id,
        effective_horizon_steps=horizon,
        prediction=float(prediction),
        raw_prediction=raw,
        candidate_available=True,
        unavailable_reason=None,
        training_rows=training.training_rows,
        training_observed_rows=training.observed_rows,
        training_start=training.training_start,
        training_end=training.training_end,
        model_id=str(config["id"]),
        clipping_applied=raw < 0,
        fit_attempted=True,
        fit_seconds=float(elapsed),
        imputed_months_n=len(training.imputed_month_ids),
        imputed_month_ids=training.imputed_month_ids,
        fit_warning_count=len(caught_messages),
        fit_warning_messages=caught_messages,
    )


def _province_histories(
    gold_history: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    required = {
        "territory_id",
        "territory_name",
        "territory_level",
        "month_id",
        "overnight_stays_total",
        "complete_month_available",
    }
    missing = sorted(required.difference(gold_history.columns))
    if missing:
        raise ValueError("Gold sin columnas ETS: " + ", ".join(missing))
    history = gold_history.loc[
        gold_history["territory_level"].astype(str).eq("province")
    ].copy()
    if history.duplicated(["territory_id", "month_id"]).any():
        raise ValueError("Gold duplica territory_id/month_id.")
    return {
        str(territory_id): group.copy()
        for territory_id, group in history.groupby("territory_id", sort=True)
    }


def build_ets_predictions(
    baseline_predictions: pd.DataFrame,
    gold_history: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Construye vistas pure y operational sobre las mismas keys baseline."""

    validate_ets_config(config)
    required = {
        "fold_id",
        "territory_id",
        "territory_name",
        "target_month_id",
        "business_origin_month_id",
        "latest_available_month_id",
        "max_training_target_month_id",
        "cutoff_policy_id",
        "actual",
        "prediction",
    }
    missing = sorted(required.difference(baseline_predictions.columns))
    if missing:
        raise ValueError("Baseline V2 sin columnas ETS: " + ", ".join(missing))
    keys = ["fold_id", "territory_id", "target_month_id"]
    if baseline_predictions.duplicated(keys).any():
        raise ValueError("Baseline V2 contiene keys duplicadas.")

    histories = _province_histories(gold_history)
    result = baseline_predictions.copy().rename(
        columns={"prediction": "baseline_prediction"}
    )
    records: list[dict[str, Any]] = []
    for row in result.itertuples(index=False):
        origin = TemporalOrigin(
            target_month_id=str(row.target_month_id),
            business_origin_month_id=str(row.business_origin_month_id),
            latest_available_month_id=str(row.latest_available_month_id),
            max_training_target_month_id=str(row.max_training_target_month_id),
            cutoff_policy_id=str(row.cutoff_policy_id),
        )
        forecast = fit_ets_forecast(
            histories[str(row.territory_id)], origin, config
        )
        record = asdict(forecast)
        if forecast.candidate_available:
            operational = float(forecast.prediction)
            fallback_used = False
            fallback_reason = "not_used"
        else:
            operational = float(row.baseline_prediction)
            fallback_used = True
            fallback_reason = str(forecast.unavailable_reason)
        record.update(
            {
                "candidate_id": str(config["id"]),
                "candidate_version": str(config["candidate_version"]),
                "fallback_policy_id": str(config["fallback"]["id"]),
                "candidate_prediction": forecast.prediction,
                "fallback_used": fallback_used,
                "fallback_reason": fallback_reason,
                "operational_prediction": operational,
            }
        )
        records.append(record)

    calculated = pd.DataFrame(records, index=result.index)
    identity_columns = [
        "territory_id",
        "target_month_id",
        "latest_available_month_id",
    ]
    for column in identity_columns:
        if not calculated[column].astype(str).eq(
            result[column].astype(str)
        ).all():
            raise AssertionError(f"Contrato ETS no coincide en {column}.")
    duplicate_columns = set(result.columns).intersection(calculated.columns)
    duplicate_columns.difference_update(identity_columns)
    if duplicate_columns:
        raise AssertionError(f"Columnas ETS duplicadas: {duplicate_columns}")
    calculated = calculated.drop(columns=identity_columns)
    result = pd.concat([result, calculated], axis=1)
    if result["operational_prediction"].isna().any():
        raise AssertionError("La vista ETS operativa contiene ausencias.")
    if (result["operational_prediction"] < 0).any():
        raise AssertionError("La vista ETS operativa contiene negativos.")
    return result.sort_values(
        ["target_month_id", "territory_id"], ignore_index=True
    )


def build_current_ets_forecasts(
    gold_history: pd.DataFrame,
    origin: TemporalOrigin,
    config: Mapping[str, Any],
    *,
    baseline_lag_months: int = 12,
) -> pd.DataFrame:
    """Calcula read-only ETS y availability fallback para 50 provincias."""

    validate_ets_config(config)
    histories = _province_histories(gold_history)
    reference_month = str(
        pd.Period(origin.target_month_id, freq="M") - baseline_lag_months
    )
    rows: list[dict[str, Any]] = []
    for territory_id, history in histories.items():
        name = str(history["territory_name"].iloc[0])
        reference = history.loc[
            history["month_id"].astype(str).eq(reference_month),
            "overnight_stays_total",
        ]
        baseline = (
            float(reference.iloc[0])
            if len(reference) == 1 and pd.notna(reference.iloc[0])
            else np.nan
        )
        forecast = fit_ets_forecast(history, origin, config)
        record = asdict(forecast)
        record["territory_name"] = name
        record["baseline_prediction"] = baseline
        if forecast.candidate_available:
            record["fallback_used"] = False
            record["fallback_reason"] = "not_used"
            record["operational_prediction"] = forecast.prediction
        elif np.isfinite(baseline) and baseline >= 0:
            record["fallback_used"] = True
            record["fallback_reason"] = forecast.unavailable_reason
            record["operational_prediction"] = baseline
        else:
            record["fallback_used"] = False
            record["fallback_reason"] = "missing_seasonal_reference"
            record["operational_prediction"] = np.nan
        rows.append(record)
    return pd.DataFrame(rows).sort_values("territory_id", ignore_index=True)
