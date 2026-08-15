"""Formula pura seasonal trend-adjusted para el screening point-in-time B2.

La ventana esta preespecificada en tres meses y usa un ratio interanual raw.
No hay clipping, ajuste de parametros ni reglas de fallback por performance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

try:
    from src.models.modeling_v2_common import TemporalOrigin
except ModuleNotFoundError:
    from modeling_v2_common import TemporalOrigin


@dataclass(frozen=True)
class SeasonalTrendWindow:
    """Meses exactos requeridos por un forecast seasonal trend-adjusted."""

    target_month_id: str
    cutoff_month_id: str
    seasonal_reference_month_id: str
    recent_window_month_ids: tuple[str, str, str]
    prior_year_window_month_ids: tuple[str, str, str]


@dataclass(frozen=True)
class SeasonalTrendForecast:
    """Resultado puro de la formula, antes del fallback de disponibilidad."""

    candidate_available: bool
    seasonal_reference: float | None
    recent_sum: float | None
    prior_year_sum: float | None
    trend_factor: float | None
    candidate_prediction: float | None
    fallback_reason: str | None


@dataclass(frozen=True)
class AvailabilityFallbackForecast:
    """Aplicacion tecnica del baseline cuando la formula no esta disponible."""

    operational_prediction: float
    fallback_used: bool
    fallback_reason: str


def resolve_candidate_window(
    origin: TemporalOrigin,
    *,
    window_months: int,
    seasonal_reference_lag_months: int,
) -> SeasonalTrendWindow:
    """Resuelve joins calendario exactos sin usar posiciones de fila."""

    if int(window_months) != 3:
        raise ValueError("B2 fija una unica ventana de tres meses.")
    if int(seasonal_reference_lag_months) != 12:
        raise ValueError("B2 fija el ancla estacional en target menos 12.")

    target = pd.Period(origin.target_month_id, freq="M")
    cutoff = pd.Period(origin.latest_available_month_id, freq="M")
    seasonal_reference = target - seasonal_reference_lag_months
    recent = tuple(
        str(cutoff - offset)
        for offset in range(window_months - 1, -1, -1)
    )
    prior_year = tuple(
        str(pd.Period(month, freq="M") - 12)
        for month in recent
    )

    recent_periods = pd.PeriodIndex(recent, freq="M")
    if recent_periods.max() != cutoff:
        raise AssertionError("La ventana reciente no termina en el cutoff.")
    if (recent_periods > cutoff).any():
        raise AssertionError("La ventana reciente accede despues del cutoff.")
    forbidden = {target - 1, target - 2}
    if forbidden.intersection(set(recent_periods)):
        raise AssertionError("La formula intento acceder a t-1 o t-2.")

    return SeasonalTrendWindow(
        target_month_id=str(target),
        cutoff_month_id=str(cutoff),
        seasonal_reference_month_id=str(seasonal_reference),
        recent_window_month_ids=recent,
        prior_year_window_month_ids=prior_year,
    )


def _missing(value: Any) -> bool:
    return value is None or bool(pd.isna(value))


def _values_for_months(
    monthly_values: Mapping[str, Any],
    months: tuple[str, str, str],
) -> list[Any] | None:
    values = [monthly_values.get(month) for month in months]
    return None if any(_missing(value) for value in values) else values


def calculate_seasonal_trend_forecast(
    monthly_values: Mapping[str, Any],
    window: SeasonalTrendWindow,
) -> SeasonalTrendForecast:
    """Calcula anchor * raw recent/prior con entradas calendario exactas."""

    seasonal_reference_raw = monthly_values.get(
        window.seasonal_reference_month_id
    )
    if _missing(seasonal_reference_raw):
        return SeasonalTrendForecast(
            False, None, None, None, None, None,
            "missing_seasonal_reference",
        )

    recent_raw = _values_for_months(
        monthly_values, window.recent_window_month_ids
    )
    if recent_raw is None:
        return SeasonalTrendForecast(
            False, float(seasonal_reference_raw), None, None, None, None,
            "missing_recent_window",
        )

    prior_raw = _values_for_months(
        monthly_values, window.prior_year_window_month_ids
    )
    if prior_raw is None:
        return SeasonalTrendForecast(
            False,
            float(seasonal_reference_raw),
            float(np.sum(recent_raw)),
            None,
            None,
            None,
            "missing_prior_year_window",
        )

    inputs = np.asarray(
        [seasonal_reference_raw, *recent_raw, *prior_raw], dtype=float
    )
    if not np.isfinite(inputs).all() or (inputs < 0).any():
        return SeasonalTrendForecast(
            False, None, None, None, None, None, "invalid_trend_input"
        )

    seasonal_reference = float(inputs[0])
    recent_sum = float(inputs[1:4].sum())
    prior_year_sum = float(inputs[4:7].sum())
    if prior_year_sum <= 0:
        return SeasonalTrendForecast(
            False,
            seasonal_reference,
            recent_sum,
            prior_year_sum,
            None,
            None,
            "non_positive_prior_year_sum",
        )

    trend_factor = recent_sum / prior_year_sum
    candidate_prediction = seasonal_reference * trend_factor
    if not np.isfinite(trend_factor) or not np.isfinite(candidate_prediction):
        return SeasonalTrendForecast(
            False,
            seasonal_reference,
            recent_sum,
            prior_year_sum,
            None,
            None,
            "invalid_trend_input",
        )
    if candidate_prediction < 0:
        raise AssertionError("La formula produjo una prediccion negativa.")

    return SeasonalTrendForecast(
        True,
        seasonal_reference,
        recent_sum,
        prior_year_sum,
        trend_factor,
        candidate_prediction,
        None,
    )


def apply_availability_fallback(
    forecast: SeasonalTrendForecast,
    baseline_prediction: float,
) -> AvailabilityFallbackForecast:
    """Usa baseline solo cuando el candidato no esta disponible."""

    baseline = float(baseline_prediction)
    if not np.isfinite(baseline) or baseline < 0:
        raise ValueError("El availability fallback requiere baseline valido.")
    if forecast.candidate_available:
        if forecast.candidate_prediction is None:
            raise AssertionError("Candidato disponible sin prediccion.")
        return AvailabilityFallbackForecast(
            float(forecast.candidate_prediction), False, "not_used"
        )
    if forecast.fallback_reason == "missing_seasonal_reference":
        raise ValueError("No existe baseline para aplicar availability fallback.")
    return AvailabilityFallbackForecast(
        baseline,
        True,
        str(forecast.fallback_reason),
    )


def _territory_monthly_values(
    gold_history: pd.DataFrame,
) -> dict[str, dict[str, float | None]]:
    required = {"territory_id", "territory_level", "month_id", "overnight_stays_total"}
    missing = sorted(required.difference(gold_history.columns))
    if missing:
        raise ValueError("Gold sin columnas del candidato: " + ", ".join(missing))
    history = gold_history.loc[
        gold_history["territory_level"].astype(str).eq("province")
    ].copy()
    if history.duplicated(["territory_id", "month_id"]).any():
        raise ValueError("Gold duplica territory_id/month_id.")
    result: dict[str, dict[str, float | None]] = {}
    for territory_id, group in history.groupby("territory_id", sort=False):
        values: dict[str, float | None] = {}
        for row in group[["month_id", "overnight_stays_total"]].itertuples(
            index=False
        ):
            value = row.overnight_stays_total
            values[str(row.month_id)] = None if _missing(value) else float(value)
        result[str(territory_id)] = values
    return result


def build_seasonal_trend_predictions(
    baseline_predictions: pd.DataFrame,
    gold_history: pd.DataFrame,
    candidate_config: Mapping[str, Any],
) -> pd.DataFrame:
    """Calcula candidato puro y operativo sobre las keys baseline recibidas."""

    if bool(candidate_config.get("clipping")):
        raise ValueError("B2 prohibe clipping del trend factor raw.")
    if bool(candidate_config.get("fallback", {}).get("performance_based")):
        raise ValueError("B2 prohibe fallback basado en performance.")

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
        raise ValueError("Baseline V2 sin columnas: " + ", ".join(missing))
    if baseline_predictions.duplicated(
        ["fold_id", "territory_id", "target_month_id"]
    ).any():
        raise ValueError("Baseline V2 contiene keys duplicadas.")

    histories = _territory_monthly_values(gold_history)
    result = baseline_predictions.copy().rename(
        columns={"prediction": "baseline_prediction"}
    )
    calculated: list[dict[str, Any]] = []

    for row in result.itertuples(index=False):
        origin = TemporalOrigin(
            target_month_id=str(row.target_month_id),
            business_origin_month_id=str(row.business_origin_month_id),
            latest_available_month_id=str(row.latest_available_month_id),
            max_training_target_month_id=str(row.max_training_target_month_id),
            cutoff_policy_id=str(row.cutoff_policy_id),
        )
        window = resolve_candidate_window(
            origin,
            window_months=int(candidate_config["window_months"]),
            seasonal_reference_lag_months=int(
                candidate_config["seasonal_reference_lag_months"]
            ),
        )
        forecast = calculate_seasonal_trend_forecast(
            histories[str(row.territory_id)], window
        )
        fallback = apply_availability_fallback(
            forecast, float(row.baseline_prediction)
        )
        calculated.append(
            {
                "candidate_id": str(candidate_config["id"]),
                "candidate_version": str(
                    candidate_config["candidate_version"]
                ),
                "formula_version": str(candidate_config["formula_version"]),
                "fallback_policy_id": str(
                    candidate_config["fallback"]["id"]
                ),
                "seasonal_reference_month_id_candidate": (
                    window.seasonal_reference_month_id
                ),
                "recent_window_month_ids": window.recent_window_month_ids,
                "prior_year_window_month_ids": (
                    window.prior_year_window_month_ids
                ),
                "seasonal_reference": forecast.seasonal_reference,
                "recent_sum": forecast.recent_sum,
                "prior_year_sum": forecast.prior_year_sum,
                "trend_factor": forecast.trend_factor,
                "candidate_prediction": forecast.candidate_prediction,
                "candidate_available": forecast.candidate_available,
                "fallback_used": fallback.fallback_used,
                "fallback_reason": fallback.fallback_reason,
                "operational_prediction": fallback.operational_prediction,
            }
        )

    calculated_frame = pd.DataFrame(calculated, index=result.index)
    result = pd.concat([result, calculated_frame], axis=1)
    if (result["operational_prediction"] < 0).any():
        raise AssertionError("El candidato operativo produjo negativos.")
    if result["operational_prediction"].isna().any():
        raise AssertionError("El candidato operativo contiene ausencias.")
    return result.sort_values(
        ["target_month_id", "territory_id"], ignore_index=True
    )


def build_current_seasonal_trend_forecasts(
    gold_history: pd.DataFrame,
    origin: TemporalOrigin,
    candidate_config: Mapping[str, Any],
) -> pd.DataFrame:
    """Calcula una ilustracion futura read-only sin targets observados."""

    if bool(candidate_config.get("clipping")):
        raise ValueError("B2 prohibe clipping del trend factor raw.")
    if bool(candidate_config.get("fallback", {}).get("performance_based")):
        raise ValueError("B2 prohibe fallback basado en performance.")

    histories = _territory_monthly_values(gold_history)
    names = (
        gold_history.loc[
            gold_history["territory_level"].astype(str).eq("province"),
            ["territory_id", "territory_name"],
        ]
        .drop_duplicates("territory_id")
        .set_index("territory_id")["territory_name"]
        .astype(str)
        .to_dict()
    )
    window = resolve_candidate_window(
        origin,
        window_months=int(candidate_config["window_months"]),
        seasonal_reference_lag_months=int(
            candidate_config["seasonal_reference_lag_months"]
        ),
    )
    rows: list[dict[str, Any]] = []
    for territory_id in sorted(histories):
        forecast = calculate_seasonal_trend_forecast(
            histories[territory_id], window
        )
        baseline = histories[territory_id].get(
            window.seasonal_reference_month_id
        )
        if _missing(baseline):
            rows.append(
                {
                    "territory_id": territory_id,
                    "territory_name": names[territory_id],
                    "target_month_id": origin.target_month_id,
                    "candidate_available": False,
                    "fallback_used": False,
                    "fallback_reason": "missing_seasonal_reference",
                    "baseline_prediction": np.nan,
                    "recent_sum": forecast.recent_sum,
                    "prior_year_sum": forecast.prior_year_sum,
                    "trend_factor": forecast.trend_factor,
                    "candidate_prediction": forecast.candidate_prediction,
                    "operational_prediction": np.nan,
                }
            )
            continue
        fallback = apply_availability_fallback(forecast, float(baseline))
        rows.append(
            {
                "territory_id": territory_id,
                "territory_name": names[territory_id],
                "target_month_id": origin.target_month_id,
                "candidate_available": forecast.candidate_available,
                "fallback_used": fallback.fallback_used,
                "fallback_reason": fallback.fallback_reason,
                "baseline_prediction": float(baseline),
                "recent_sum": forecast.recent_sum,
                "prior_year_sum": forecast.prior_year_sum,
                "trend_factor": forecast.trend_factor,
                "candidate_prediction": forecast.candidate_prediction,
                "operational_prediction": fallback.operational_prediction,
            }
        )
    return pd.DataFrame(rows)
