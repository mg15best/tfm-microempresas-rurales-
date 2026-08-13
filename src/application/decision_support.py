"""Interpretacion operativa transparente de un forecast provincial."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Literal

import numpy as np
import pandas as pd

from src.application.forecast_service import ForecastProductContext


__all__ = [
    "ActivityLevel",
    "DecisionSupport",
    "DecisionSupportError",
    "DecisionSupportWarning",
    "build_decision_support",
]

ActivityLevel = Literal["low", "usual", "high", "insufficient"]

RULE_ID = "seasonal_q25_q75_last_10_final_non_covid_v1"
MAX_REFERENCE_OBSERVATIONS = 10
MIN_REFERENCE_OBSERVATIONS = 5
LOW_QUANTILE = 0.25
HIGH_QUANTILE = 0.75

REQUIRED_HISTORY_COLUMNS = {
    "territory_id",
    "month_id",
    "date_month",
    "overnight_stays_total",
    "covid_period",
    "is_provisional",
}

GUIDANCE = {
    "low": (
        "Contrasta la señal provincial con reservas y eventos locales antes "
        "de reducir recursos o actividad comercial."
    ),
    "usual": (
        "Mantén la planificación de referencia y contrástala con reservas, "
        "capacidad y eventos locales."
    ),
    "high": (
        "Revisa con antelación capacidad, personal y aprovisionamiento; ajusta "
        "solo si las reservas y el contexto local confirman la señal."
    ),
}

LIMITATIONS = (
    "Señal provincial de pernoctaciones; no estima la demanda de un "
    "establecimiento concreto.",
    "El nivel expresa posición histórica, no confianza estadística, "
    "causalidad ni retorno económico.",
)


class DecisionSupportError(RuntimeError):
    """El contexto de producto no permite una interpretacion reproducible."""


@dataclass(frozen=True)
class DecisionSupportWarning:
    """Advertencia breve y determinista para consumo por el frontal."""

    code: str
    message: str


@dataclass(frozen=True)
class DecisionSupport:
    """Señal estacional provincial con regla y muestra auditables."""

    activity_level: ActivityLevel
    historical_percentile_pct: float | None
    historical_median: float | None
    historical_q25: float | None
    historical_q75: float | None
    historical_minimum: float | None
    historical_maximum: float | None
    historical_sample_size: int
    comparison_calendar_month: int
    comparison_month_ids: tuple[str, ...]
    excluded_covid_month_ids: tuple[str, ...]
    excluded_provisional_month_ids: tuple[str, ...]
    excluded_invalid_month_ids: tuple[str, ...]
    omitted_older_month_ids: tuple[str, ...]
    missing_comparison_years: tuple[int, ...]
    rule_id: str
    action_guidance: str | None
    limitations: tuple[str, ...]
    warnings: tuple[DecisionSupportWarning, ...]


def _validate_flag(history: pd.DataFrame, column: str) -> None:
    """Exige booleanos no nulos para que las exclusiones sean explicables."""
    values = history[column]
    valid = values.map(lambda value: isinstance(value, (bool, np.bool_)))
    if values.isna().any() or not valid.all():
        raise DecisionSupportError(
            f"{column} debe contener solo booleanos no nulos."
        )


def _prepare_history(product_context: ForecastProductContext) -> pd.DataFrame:
    """Valida y normaliza solo el historico ya incluido en el producto."""
    history = product_context.dashboard.history
    if not isinstance(history, pd.DataFrame) or history.empty:
        raise DecisionSupportError("El contexto no contiene historico provincial.")

    missing = REQUIRED_HISTORY_COLUMNS.difference(history.columns)
    if missing:
        raise DecisionSupportError(
            "Faltan columnas en el historico de producto: "
            + ", ".join(sorted(missing))
        )

    result = history.loc[:, sorted(REQUIRED_HISTORY_COLUMNS)].copy()
    requested_id = str(product_context.forecast.territory_id)
    territory_ids = result["territory_id"].astype(str)
    if not territory_ids.eq(requested_id).all():
        raise DecisionSupportError(
            "El historico no pertenece exclusivamente al territorio previsto."
        )

    try:
        result["date_month"] = pd.to_datetime(
            result["date_month"],
            errors="raise",
        )
        result["overnight_stays_total"] = pd.to_numeric(
            result["overnight_stays_total"],
            errors="coerce",
        )
    except (TypeError, ValueError) as error:
        raise DecisionSupportError(
            "El historico contiene fechas no validas."
        ) from error

    if result["date_month"].isna().any():
        raise DecisionSupportError("El historico contiene fechas nulas.")
    expected_month_ids = result["date_month"].dt.strftime("%Y-%m")
    if not result["month_id"].astype(str).eq(expected_month_ids).all():
        raise DecisionSupportError(
            "month_id y date_month no representan el mismo mes."
        )
    if result.duplicated(["territory_id", "month_id"]).any():
        raise DecisionSupportError(
            "El historico contiene claves territorio-mes duplicadas."
        )

    _validate_flag(result, "covid_period")
    _validate_flag(result, "is_provisional")
    return result.sort_values("date_month").reset_index(drop=True)


def _target_period(product_context: ForecastProductContext) -> pd.Period:
    """Obtiene el mes objetivo operacional y valida el forecast."""
    try:
        target = pd.Period(
            product_context.forecast.target_month_id,
            freq="M",
        )
    except (TypeError, ValueError) as error:
        raise DecisionSupportError("El mes objetivo no es valido.") from error

    forecast = float(
        product_context.forecast.predicted_overnight_stays_total
    )
    if not isfinite(forecast) or forecast < 0:
        raise DecisionSupportError(
            "El forecast debe ser finito y no negativo."
        )
    return target


def _warning(
    warnings: list[DecisionSupportWarning],
    code: str,
    message: str,
) -> None:
    """Añade una advertencia una sola vez por codigo."""
    if code not in {warning.code for warning in warnings}:
        warnings.append(DecisionSupportWarning(code=code, message=message))


def _midrank_percentile(values: np.ndarray, forecast: float) -> float:
    """Calcula percentil empirico con rango medio para gestionar empates."""
    below = int(np.count_nonzero(values < forecast))
    equal = int(np.count_nonzero(values == forecast))
    return 100.0 * (below + 0.5 * equal) / len(values)


def _month_ids(dataframe: pd.DataFrame) -> tuple[str, ...]:
    """Serializa la muestra y las exclusiones sin perder su orden temporal."""
    return tuple(dataframe["month_id"].astype(str).tolist())


def build_decision_support(
    product_context: ForecastProductContext,
) -> DecisionSupport:
    """Compara el forecast con el mismo mes de hasta diez años comparables.

    La referencia conserva las diez observaciones finales y no-COVID más
    recientes anteriores al objetivo. Con cinco o más observaciones, un valor
    bajo Q25 es ``low``, uno sobre Q75 es ``high`` y los límites incluidos son
    ``usual``. La fiabilidad histórica del modelo no altera esta clasificación.
    """
    history = _prepare_history(product_context)
    target = _target_period(product_context)
    target_date = target.to_timestamp()
    forecast = float(
        product_context.forecast.predicted_overnight_stays_total
    )

    same_month = history.loc[
        history["date_month"].dt.month.eq(target.month)
        & history["date_month"].lt(target_date)
    ].copy()
    excluded_covid = same_month.loc[same_month["covid_period"]]
    excluded_provisional = same_month.loc[same_month["is_provisional"]]
    numeric = same_month["overnight_stays_total"].to_numpy(dtype=float)
    invalid_mask = ~np.isfinite(numeric) | (numeric < 0)
    excluded_invalid = same_month.loc[invalid_mask]

    comparable = same_month.loc[
        ~same_month["covid_period"]
        & ~same_month["is_provisional"]
        & ~invalid_mask
    ].sort_values("date_month")
    selected = comparable.tail(MAX_REFERENCE_OBSERVATIONS)
    omitted_older = comparable.iloc[:-MAX_REFERENCE_OBSERVATIONS]

    observed_years = set(same_month["date_month"].dt.year.astype(int))
    if observed_years:
        expected_years = set(
            range(min(observed_years), int(target.year))
        )
        missing_years = tuple(sorted(expected_years.difference(observed_years)))
    else:
        missing_years = ()

    warnings = [
        DecisionSupportWarning(code=item.code, message=item.message)
        for item in product_context.forecast.warnings
    ]
    if missing_years:
        _warning(
            warnings,
            "historical_comparison_gaps",
            "Faltan observaciones del mes comparable en algunos años.",
        )
    if not excluded_invalid.empty:
        _warning(
            warnings,
            "invalid_historical_observations_excluded",
            "Se excluyeron valores históricos nulos, no finitos o negativos.",
        )

    values = selected["overnight_stays_total"].to_numpy(dtype=float)
    sample_size = len(values)
    if sample_size:
        median = float(np.median(values))
        minimum = float(np.min(values))
        maximum = float(np.max(values))
        percentile = float(_midrank_percentile(values, forecast))
    else:
        median = minimum = maximum = percentile = None

    q25: float | None = None
    q75: float | None = None
    activity_level: ActivityLevel = "insufficient"
    action_guidance: str | None = None
    if sample_size < MIN_REFERENCE_OBSERVATIONS:
        _warning(
            warnings,
            "insufficient_seasonal_history",
            "No hay al menos cinco observaciones finales y no-COVID del mes.",
        )
    else:
        q25, q75 = (
            float(value)
            for value in np.quantile(
                values,
                [LOW_QUANTILE, HIGH_QUANTILE],
                method="linear",
            )
        )
        if forecast < q25:
            activity_level = "low"
        elif forecast > q75:
            activity_level = "high"
        else:
            activity_level = "usual"
        action_guidance = GUIDANCE[activity_level]

        if minimum == maximum:
            _warning(
                warnings,
                "flat_comparison_history",
                "Todas las observaciones de la referencia estacional son iguales.",
            )
        if forecast < minimum or forecast > maximum:
            _warning(
                warnings,
                "forecast_outside_historical_range",
                "El forecast queda fuera del rango de la referencia estacional.",
            )

    return DecisionSupport(
        activity_level=activity_level,
        historical_percentile_pct=percentile,
        historical_median=median,
        historical_q25=q25,
        historical_q75=q75,
        historical_minimum=minimum,
        historical_maximum=maximum,
        historical_sample_size=sample_size,
        comparison_calendar_month=int(target.month),
        comparison_month_ids=_month_ids(selected),
        excluded_covid_month_ids=_month_ids(excluded_covid),
        excluded_provisional_month_ids=_month_ids(excluded_provisional),
        excluded_invalid_month_ids=_month_ids(excluded_invalid),
        omitted_older_month_ids=_month_ids(omitted_older),
        missing_comparison_years=missing_years,
        rule_id=RULE_ID,
        action_guidance=action_guidance,
        limitations=LIMITATIONS,
        warnings=tuple(warnings),
    )
