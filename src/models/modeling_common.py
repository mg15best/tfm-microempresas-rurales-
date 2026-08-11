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
VALIDATION_RULES_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "modeling_validation_rules.yml"
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


def load_validation_rules(
    rules_path: Path = VALIDATION_RULES_PATH,
) -> dict[str, Any]:
    """Carga las reglas normativas de validacion de modelado."""
    if not rules_path.exists():
        raise FileNotFoundError(
            "No se encontraron las reglas de validacion: "
            f"{rules_path.relative_to(PROJECT_ROOT)}"
        )

    with rules_path.open("r", encoding="utf-8") as file:
        rules = yaml.safe_load(file)

    if not isinstance(rules, dict):
        raise ValueError(
            "modeling_validation_rules.yml no contiene un objeto YAML valido."
        )

    return rules


def resolve_project_path(path_text: str) -> Path:
    """Resuelve una ruta relativa respecto a la raiz del proyecto."""
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def validate_model_input_availability(
    config: dict[str, Any],
    predictors: list[str],
    validation_rules: dict[str, Any] | None = None,
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

    rules = validation_rules or load_validation_rules()
    point_in_time_rules = rules["validation"][
        "point_in_time_availability"
    ]
    forecast_origin = str(availability["forecast_origin"])
    supported_forecast_origin = str(
        point_in_time_rules["supported_forecast_origin"]
    )

    if forecast_origin != supported_forecast_origin:
        raise ValueError(
            "El forecast origin configurado no coincide con el valor "
            "canonico soportado para la Entrega 4: "
            f"{supported_forecast_origin}."
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

    lineage_offsets = {
        str(item["feature"]): int(item["offset_months"])
        for item in rules["validation"]["temporal_integrity"]["lag_rules"]
    }
    missing_lineage = sorted(
        predictor_set.intersection(eotr_lags).difference(lineage_offsets)
    )
    mismatched_lineage = sorted(
        name
        for name in predictor_set.intersection(eotr_lags, lineage_offsets)
        if eotr_lags[name] != lineage_offsets[name]
    )
    if missing_lineage or mismatched_lineage:
        details: list[str] = []
        if missing_lineage:
            details.append("sin regla temporal: " + ", ".join(missing_lineage))
        if mismatched_lineage:
            details.append(
                "offset incoherente: " + ", ".join(mismatched_lineage)
            )
        raise ValueError(
            "La clasificacion EOTR no coincide con temporal_integrity: "
            + "; ".join(details)
        )


def get_minimum_safe_training_label_lag(
    config: dict[str, Any],
) -> int:
    """Obtiene el lag de publicacion que tambien rige las etiquetas train."""
    availability = config.get("point_in_time_availability")
    if not isinstance(availability, dict):
        raise ValueError(
            "Falta la seccion point_in_time_availability en la configuracion."
        )

    training_labels = availability.get("training_labels")
    if not isinstance(training_labels, dict):
        raise ValueError(
            "Falta declarar la politica point-in-time de training_labels."
        )

    if not bool(
        training_labels.get("require_published_before_forecast_origin")
    ):
        raise ValueError(
            "Las etiquetas de entrenamiento EOTR deben estar publicadas "
            "antes del forecast origin."
        )

    if not bool(
        training_labels.get("cutoff_uses_minimum_safe_eotr_lag_months")
    ):
        raise ValueError(
            "El cutoff de etiquetas debe derivar de "
            "minimum_safe_eotr_lag_months."
        )

    minimum_lag = int(availability["minimum_safe_eotr_lag_months"])
    if minimum_lag < 1:
        raise ValueError(
            "minimum_safe_eotr_lag_months debe ser al menos 1."
        )
    return minimum_lag


def calculate_training_label_cutoffs(
    validation_start: str | pd.Timestamp | pd.Period,
    structural_train_end: str | pd.Timestamp | pd.Period,
    minimum_safe_target_lag_months: int,
) -> dict[str, pd.Period]:
    """Calcula cutoffs mensuales estructural, de publicacion y efectivo."""
    minimum_lag = int(minimum_safe_target_lag_months)
    if minimum_lag < 1:
        raise ValueError(
            "minimum_safe_target_lag_months debe ser al menos 1."
        )

    validation_period = pd.Period(validation_start, freq="M")
    structural_period = pd.Period(structural_train_end, freq="M")
    availability_period = validation_period - minimum_lag
    effective_period = min(structural_period, availability_period)

    return {
        "validation_start": validation_period,
        "structural_train_end": structural_period,
        "availability_train_end": availability_period,
        "effective_train_end": effective_period,
    }


def training_label_masks(
    dataframe: pd.DataFrame,
    evaluable: pd.Series,
    fold: dict[str, str],
) -> tuple[pd.Series, pd.Series]:
    """Crea mascaras train antes y despues del purge de disponibilidad."""
    structural_end = pd.Period(fold["structural_train_end"], freq="M")
    availability_end = pd.Period(fold["availability_train_end"], freq="M")
    effective_end = pd.Period(fold["effective_train_end"], freq="M")

    if effective_end > availability_end:
        raise ValueError(
            f"Cutoff efectivo posterior a disponibilidad en {fold['name']}."
        )

    target_periods = dataframe["target_date_month"].dt.to_period("M")
    before_purge = target_periods.le(structural_end) & evaluable
    after_purge = target_periods.le(effective_end) & evaluable

    if after_purge.any():
        max_label = target_periods.loc[after_purge].max()
        if max_label > effective_end:
            raise AssertionError(
                f"Etiqueta train posterior al cutoff efectivo en {fold['name']}."
            )

    return before_purge, after_purge


def ensure_test_window_is_untouched(
    config: dict[str, Any],
    *,
    purpose: str,
) -> None:
    """Impide reutilizar una ventana test ya abierta."""
    final_test = config["validation"]["final_test"]
    if str(final_test.get("test_status", "")) == "already_opened":
        raise RuntimeError(
            "Final test window is already opened and cannot be reused for "
            f"{purpose}. Use a future untouched evaluation window."
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
    minimum_lag = get_minimum_safe_training_label_lag(config)

    for fold in config["validation"]["folds"]:
        cutoffs = calculate_training_label_cutoffs(
            str(fold["validation_start"]),
            str(fold["train_end"]),
            minimum_lag,
        )

        folds.append(
            {
                "name": str(fold["name"]),
                **{
                    name: period.strftime("%Y-%m")
                    for name, period in cutoffs.items()
                },
                "train_end": cutoffs["effective_train_end"].to_timestamp(
                    how="start"
                ).strftime("%Y-%m-%d"),
            }
        )

    if include_test:
        test_start = pd.Period(
            str(config["validation"]["final_test"]["start"]),
            freq="M",
        )

        cutoffs = calculate_training_label_cutoffs(
            test_start,
            test_start - 1,
            minimum_lag,
        )

        folds.append(
            {
                "name": "test",
                **{
                    name: period.strftime("%Y-%m")
                    for name, period in cutoffs.items()
                },
                "train_end": cutoffs["effective_train_end"].to_timestamp(
                    how="start"
                ).strftime("%Y-%m-%d"),
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
