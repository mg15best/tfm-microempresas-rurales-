"""
Utilidades compartidas para la evaluacion y seleccion reproducible de modelos.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "modeling_config.yml"
)


def load_config(
    config_path: Path = CONFIG_PATH,
) -> dict[str, Any]:
    """Carga y valida la configuracion de modelado."""
    if not config_path.exists():
        raise FileNotFoundError(
            "No se encontro la configuracion: "
            f"{config_path.relative_to(PROJECT_ROOT)}"
        )

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            "modeling_config.yml no contiene un objeto YAML valido."
        )

    return config


def resolve_project_path(path_text: str) -> Path:
    """Resuelve una ruta relativa respecto a la raiz del proyecto."""
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def validate_model_input_availability(
    config: dict[str, Any],
    predictors: list[str],
) -> None:
    """Valida disponibilidad operacional de predictores en el forecast origin."""
    availability = config.get("point_in_time_availability")

    if not isinstance(availability, dict):
        raise ValueError(
            "Falta la seccion point_in_time_availability en la configuracion."
        )

    configured_horizon = int(availability["forecast_horizon_months"])
    problem_horizon = int(config["problem"]["forecast_horizon_months"])

    if configured_horizon != problem_horizon:
        raise ValueError(
            "El horizonte de disponibilidad point-in-time no coincide con "
            "el horizonte del problema."
        )

    forecast_origin = str(availability["forecast_origin"])
    canonical_forecast_origin = str(
        availability["canonical_forecast_origin"]
    )

    if forecast_origin != canonical_forecast_origin:
        raise ValueError(
            "El forecast origin configurado no coincide con el valor "
            "canonico soportado para la Entrega 4: "
            f"{canonical_forecast_origin}."
        )

    minimum_lag = int(availability["minimum_safe_eotr_lag_months"])
    known_in_advance = {
        str(value)
        for value in availability.get("known_in_advance_predictors", [])
    }
    eotr_lags = {
        str(name): int(lag)
        for name, lag in availability.get("eotr_predictor_lags", {}).items()
    }
    unavailable = {
        str(value)
        for value in availability.get("unavailable_at_forecast_origin", [])
    }
    predictor_set = set(predictors)

    conflicting = sorted(
        (known_in_advance | set(eotr_lags)).intersection(unavailable)
    )
    if conflicting:
        raise ValueError(
            "Predictores clasificados simultaneamente como disponibles y "
            "no disponibles: " + ", ".join(conflicting)
        )

    unavailable_used = sorted(predictor_set.intersection(unavailable))
    if unavailable_used:
        raise ValueError(
            "model_inputs contiene predictores no disponibles en el "
            "forecast origin: " + ", ".join(unavailable_used)
        )

    classified = known_in_advance | set(eotr_lags)
    unclassified = sorted(predictor_set.difference(classified))
    if unclassified:
        raise ValueError(
            "model_inputs contiene predictores sin clasificacion de "
            "disponibilidad point-in-time: " + ", ".join(unclassified)
        )

    unsafe_eotr = sorted(
        name
        for name in predictor_set.intersection(eotr_lags)
        if eotr_lags[name] < minimum_lag
    )
    if unsafe_eotr:
        raise ValueError(
            "model_inputs contiene predictores EOTR con un desfase inferior "
            f"al minimo seguro de {minimum_lag} meses: "
            + ", ".join(unsafe_eotr)
        )


def get_model_inputs(
    config: dict[str, Any],
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Obtiene las variables numericas, categoricas y booleanas."""
    model_inputs = config["model_inputs"]

    numeric_features = [
        str(value)
        for value in model_inputs["numeric_features"]
    ]

    categorical_features = [
        str(value)
        for value in model_inputs["categorical_features"]
    ]

    boolean_features = [
        str(value)
        for value in model_inputs["boolean_features"]
    ]

    feature_columns = (
        numeric_features
        + categorical_features
    )

    if len(feature_columns) != len(set(feature_columns)):
        raise ValueError(
            "model_inputs contiene variables predictoras duplicadas."
        )

    configured_predictors = list(
        dict.fromkeys(feature_columns + boolean_features)
    )
    validate_model_input_availability(
        config,
        configured_predictors,
    )

    return (
        numeric_features,
        categorical_features,
        boolean_features,
        feature_columns,
    )


def get_validation_folds(
    config: dict[str, Any],
    *,
    include_test: bool = False,
) -> list[dict[str, str]]:
    """Construye los folds expansivos definidos en la configuracion."""
    folds: list[dict[str, str]] = []

    for fold in config["validation"]["folds"]:
        train_end = pd.Period(
            str(fold["train_end"]),
            freq="M",
        ).to_timestamp(how="start")

        folds.append(
            {
                "name": str(fold["name"]),
                "train_end": train_end.strftime("%Y-%m-%d"),
            }
        )

    if include_test:
        test_start = pd.Period(
            str(config["validation"]["final_test"]["start"]),
            freq="M",
        )

        test_train_end = (
            test_start - 1
        ).to_timestamp(how="start")

        folds.append(
            {
                "name": "test",
                "train_end": test_train_end.strftime("%Y-%m-%d"),
            }
        )

    return folds


def load_modeling_dataset(
    config: dict[str, Any],
) -> pd.DataFrame:
    """Carga, valida y normaliza el dataset de modelado."""
    dataset_path = resolve_project_path(
        str(config["modeling_dataset"]["path"])
    )

    if not dataset_path.exists():
        raise FileNotFoundError(
            "No se encontro el dataset de modelado: "
            f"{dataset_path.relative_to(PROJECT_ROOT)}"
        )

    dataframe = pd.read_parquet(dataset_path).copy()

    if dataframe.empty:
        raise ValueError(
            "El dataset de modelado esta vacio."
        )

    (
        numeric_features,
        categorical_features,
        boolean_features,
        feature_columns,
    ) = get_model_inputs(config)

    target_column = str(config["target"]["column"])
    baseline_column = str(
        config["baseline"]["prediction_feature"]
    )

    required_columns = {
        "territory_id",
        "territory_name",
        "target_month_id",
        "target_date_month",
        "month",
        "quarter",
        "evaluation_split",
        "is_provisional",
        target_column,
        baseline_column,
        *feature_columns,
    }

    missing_columns = required_columns.difference(
        dataframe.columns
    )

    if missing_columns:
        raise ValueError(
            "Faltan columnas requeridas: "
            + ", ".join(sorted(missing_columns))
        )

    dataframe["target_date_month"] = pd.to_datetime(
        dataframe["target_date_month"],
        errors="raise",
    )

    dataframe["is_provisional"] = (
        dataframe["is_provisional"]
        .fillna(False)
        .astype(bool)
    )

    for column in boolean_features:
        dataframe[column] = (
            dataframe[column]
            .fillna(False)
            .astype("int8")
        )

    for column in numeric_features:
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        ).astype(float)

    for column in categorical_features:
        dataframe[column] = (
            dataframe[column]
            .astype("string")
            .fillna("missing")
        )

    return dataframe


def build_preprocessor(
    config: dict[str, Any],
) -> ColumnTransformer:
    """Construye el preprocesamiento comun de variables."""
    (
        numeric_features,
        categorical_features,
        _,
        _,
    ) = get_model_inputs(config)

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                "passthrough",
                numeric_features,
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
        sparse_threshold=0,
    )


def common_evaluable_mask(
    dataframe: pd.DataFrame,
    config: dict[str, Any],
) -> pd.Series:
    """Selecciona filas comparables entre modelos y baseline."""
    target_column = str(config["target"]["column"])
    baseline_column = str(
        config["baseline"]["prediction_feature"]
    )

    return (
        dataframe[target_column].notna()
        & dataframe[baseline_column].notna()
        & ~dataframe["is_provisional"].fillna(False)
    )


def calculate_metrics(
    actual: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, float | int]:
    """Calcula MAE, RMSE, WAPE y sesgo medio."""
    actual = np.asarray(actual, dtype=float)
    prediction = np.asarray(prediction, dtype=float)

    if actual.shape != prediction.shape:
        raise ValueError(
            "actual y prediction deben tener la misma forma."
        )

    if actual.size == 0:
        raise ValueError(
            "No se pueden calcular metricas sin observaciones."
        )

    error = prediction - actual
    absolute_error = np.abs(error)
    actual_sum = np.abs(actual).sum()

    return {
        "rows": int(actual.size),
        "MAE": float(absolute_error.mean()),
        "RMSE": float(np.sqrt(np.mean(error ** 2))),
        "WAPE_pct": float(
            absolute_error.sum() / actual_sum * 100
            if actual_sum != 0
            else np.nan
        ),
        "mean_bias": float(error.mean()),
    }


def calculate_improvement_pct(
    baseline_value: float,
    model_value: float,
) -> float:
    """Calcula la mejora porcentual frente al baseline."""
    if baseline_value == 0:
        return np.nan

    return float(
        (baseline_value - model_value)
        / baseline_value
        * 100
    )
