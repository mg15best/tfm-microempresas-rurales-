"""Contrato temporal mensual point-in-time para el backtest V2.

La disponibilidad de publicacion se reconstruye con un cutoff conservador.
Los vintages historicos exactos no estan disponibles, por lo que V2 usa la
historia actualmente revisada (vintage limitation).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELING_V2_CONFIG_PATH = (
    PROJECT_ROOT / "data" / "metadata" / "modeling_v2_config.yml"
)
FEATURE_AVAILABILITY_V2_PATH = (
    PROJECT_ROOT / "data" / "metadata" / "feature_availability_v2.yml"
)


@dataclass(frozen=True)
class CutoffPolicy:
    """Politica versionada de disponibilidad mensual."""

    policy_id: str
    business_origin_lag_months: int
    latest_available_lag_months: int
    max_training_target_lag_months: int

    def __post_init__(self) -> None:
        if not self.policy_id:
            raise ValueError("cutoff policy id no puede estar vacio.")
        if self.business_origin_lag_months < 1:
            raise ValueError("business origin debe preceder al target.")
        if (
            self.latest_available_lag_months
            <= self.business_origin_lag_months
        ):
            raise ValueError(
                "latest available debe preceder al business origin."
            )
        if (
            self.max_training_target_lag_months
            < self.latest_available_lag_months
        ):
            raise ValueError(
                "El cutoff de training labels no puede ser posterior a "
                "latest available."
            )


@dataclass(frozen=True)
class TemporalOrigin:
    """Origin mensual inmutable de una rolling validation point-in-time."""

    target_month_id: str
    business_origin_month_id: str
    latest_available_month_id: str
    max_training_target_month_id: str
    cutoff_policy_id: str


def _load_yaml(path: Path, *, expected_name: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"No se encontro {expected_name}: {path}")
    with path.open("r", encoding="utf-8") as file:
        content = yaml.safe_load(file)
    if not isinstance(content, dict):
        raise ValueError(f"{expected_name} no contiene un objeto YAML.")
    return content


def load_modeling_v2_config(
    path: Path = MODELING_V2_CONFIG_PATH,
) -> dict[str, Any]:
    """Carga la configuracion minima y versionada de modelado V2."""

    config = _load_yaml(path, expected_name="modeling_v2_config.yml")
    if config.get("methodology", {}).get("id") != "point_in_time_v2":
        raise ValueError("La configuracion no declara point_in_time_v2.")
    return config


def load_feature_availability_v2(
    path: Path = FEATURE_AVAILABILITY_V2_PATH,
) -> dict[str, Any]:
    """Carga la especificacion contractual de disponibilidad de features."""

    spec = _load_yaml(path, expected_name="feature_availability_v2.yml")
    if spec.get("specification", {}).get("id") != "feature_availability_v2":
        raise ValueError("La especificacion de disponibilidad V2 no es valida.")
    return spec


def cutoff_policy_from_config(config: Mapping[str, Any]) -> CutoffPolicy:
    """Construye una politica tipada desde una configuracion V2."""

    raw = config.get("cutoff_policy")
    if not isinstance(raw, Mapping):
        raise ValueError("Falta cutoff_policy en la configuracion V2.")
    return CutoffPolicy(
        policy_id=str(raw["id"]),
        business_origin_lag_months=int(
            raw["business_origin_lag_months"]
        ),
        latest_available_lag_months=int(
            raw["latest_available_lag_months"]
        ),
        max_training_target_lag_months=int(
            raw["max_training_target_lag_months"]
        ),
    )


def _as_month(value: str | pd.Timestamp | pd.Period) -> pd.Period:
    try:
        return pd.Period(value, freq="M")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Mes mensual no valido: {value!r}.") from exc


def resolve_information_cutoff(
    target_month: str | pd.Timestamp | pd.Period,
    policy: CutoffPolicy,
) -> TemporalOrigin:
    """Resuelve origin, cutoff de observaciones y purge de labels para target."""

    target = _as_month(target_month)
    business_origin = target - policy.business_origin_lag_months
    latest_available = target - policy.latest_available_lag_months
    max_training_target = target - policy.max_training_target_lag_months

    if latest_available >= business_origin:
        raise AssertionError(
            "latest available no puede alcanzar el business origin."
        )
    if max_training_target > latest_available:
        raise AssertionError(
            "training labels posteriores a latest available no son conocidas."
        )

    return TemporalOrigin(
        target_month_id=str(target),
        business_origin_month_id=str(business_origin),
        latest_available_month_id=str(latest_available),
        max_training_target_month_id=str(max_training_target),
        cutoff_policy_id=policy.policy_id,
    )


def build_backtest_origins(
    config: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Construye un origin comun por target mensual de rolling validation."""

    resolved_config = config or load_modeling_v2_config()
    policy = cutoff_policy_from_config(resolved_config)
    records: list[dict[str, Any]] = []
    seen_targets: set[str] = set()

    for fold in resolved_config["folds"]:
        fold_id = str(fold["id"])
        start = _as_month(str(fold["start"]))
        end = _as_month(str(fold["end"]))
        if end < start:
            raise ValueError(f"Fold V2 invertido: {fold_id}.")
        period = f"{start}/{end}"

        for target in pd.period_range(start, end, freq="M"):
            origin = resolve_information_cutoff(target, policy)
            if origin.target_month_id in seen_targets:
                raise ValueError(
                    "Los folds V2 se solapan en "
                    f"{origin.target_month_id}."
                )
            seen_targets.add(origin.target_month_id)
            records.append(
                {
                    "fold_id": fold_id,
                    "period": period,
                    **origin.__dict__,
                }
            )

    return pd.DataFrame.from_records(records).sort_values(
        "target_month_id", ignore_index=True
    )


def assert_training_labels_within_cutoff(
    labels: pd.DataFrame,
    origin: TemporalOrigin,
    *,
    target_month_column: str = "month_id",
) -> None:
    """Falla si un conjunto de labels contiene meses posteriores al cutoff."""

    if labels.empty:
        return
    if target_month_column not in labels.columns:
        raise ValueError(f"Falta la columna {target_month_column}.")
    periods = pd.PeriodIndex(labels[target_month_column], freq="M")
    cutoff = _as_month(origin.max_training_target_month_id)
    if (periods > cutoff).any():
        offending = sorted({str(value) for value in periods[periods > cutoff]})
        raise AssertionError(
            "Training labels posteriores al cutoff point-in-time: "
            + ", ".join(offending)
        )


def purge_training_labels(
    labels: pd.DataFrame,
    origin: TemporalOrigin,
    *,
    target_month_column: str = "month_id",
) -> pd.DataFrame:
    """Aplica el purge mensual y devuelve solo labels conocidos en el origin."""

    if target_month_column not in labels.columns:
        raise ValueError(f"Falta la columna {target_month_column}.")
    periods = pd.PeriodIndex(labels[target_month_column], freq="M")
    cutoff = _as_month(origin.max_training_target_month_id)
    purged = labels.loc[periods <= cutoff].copy()
    assert_training_labels_within_cutoff(
        purged,
        origin,
        target_month_column=target_month_column,
    )
    return purged


def filter_history_to_information_cutoff(
    history: pd.DataFrame,
    origin: TemporalOrigin,
    *,
    observation_month_column: str = "month_id",
) -> pd.DataFrame:
    """Impide que una rolling futura lea observaciones posteriores al cutoff."""

    if observation_month_column not in history.columns:
        raise ValueError(f"Falta la columna {observation_month_column}.")
    periods = pd.PeriodIndex(history[observation_month_column], freq="M")
    cutoff = _as_month(origin.latest_available_month_id)
    filtered = history.loc[periods <= cutoff].copy()
    if not filtered.empty:
        kept = pd.PeriodIndex(filtered[observation_month_column], freq="M")
        if (kept > cutoff).any():
            raise AssertionError(
                "Una observacion posterior al cutoff alcanzo la rolling."
            )
    return filtered

