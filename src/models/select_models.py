"""
Reproduce la selección de modelos usando exclusivamente las tres
ventanas de validación temporal.

Evalúa:
- baseline estacional lag-12;
- Ridge con la rejilla de alpha documentada;
- configuración congelada hgb_raw_02.

El test final no se carga como ventana de selección ni interviene en
la elección de modelos.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from src.models.modeling_common import (
        PROJECT_ROOT,
        build_preprocessor,
        calculate_improvement_pct,
        calculate_metrics,
        common_evaluable_mask,
        get_model_inputs,
        get_validation_folds,
        load_config,
        load_modeling_dataset,
        training_label_masks,
    )
except ModuleNotFoundError:
    from modeling_common import (
        PROJECT_ROOT,
        build_preprocessor,
        calculate_improvement_pct,
        calculate_metrics,
        common_evaluable_mask,
        get_model_inputs,
        get_validation_folds,
        load_config,
        load_modeling_dataset,
        training_label_masks,
    )


PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "data"
    / "model_outputs"
    / "model_selection_validation_predictions.parquet"
)

METRICS_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "model_selection_metrics.csv"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "model_selection_validation_report.md"
)


def build_ridge_pipeline(
    config: dict[str, Any],
    alpha: float,
) -> Pipeline:
    """Construye el pipeline de Ridge documentado."""
    (
        numeric_features,
        categorical_features,
        _,
        _,
    ) = get_model_inputs(config)

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
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

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", Ridge(alpha=alpha)),
        ]
    )


def build_hgb_pipeline(
    config: dict[str, Any],
) -> Pipeline:
    """Construye la configuración congelada hgb_raw_02."""
    selection = config["model_selection"]
    hgb_config = selection["hist_gradient_boosting"]

    parameters = dict(hgb_config["parameters"])
    parameters["random_state"] = int(
        config["reproducibility"]["random_state"]
    )

    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(config)),
            (
                "model",
                HistGradientBoostingRegressor(**parameters),
            ),
        ]
    )


def alpha_model_id(alpha: float) -> str:
    """Genera un identificador estable para cada Ridge."""
    text = f"{alpha:g}".replace(".", "_")
    return f"ridge_alpha_{text}"


def select_solution_by_validation(
    config: dict[str, Any],
    candidate_model: str,
    baseline_mae: float,
    candidate_mae: float,
) -> tuple[str, float]:
    """Aplica el umbral configurado usando solo resultados de validacion."""
    improvement = calculate_improvement_pct(
        baseline_mae,
        candidate_mae,
    )
    minimum_improvement = float(
        config["model_selection"]["minimum_mae_improvement_pct"]
    )
    fallback = str(
        config["fallback"]["if_no_candidate_beats_baseline"][
            "selected_solution"
        ]
    )
    selected = (
        candidate_model
        if improvement >= minimum_improvement
        else fallback
    )
    return selected, improvement


def evaluate_model_fold(
    dataframe: pd.DataFrame,
    evaluable: pd.Series,
    config: dict[str, Any],
    fold: dict[str, str],
    model_id: str,
    pipeline_factory: Callable[[], Pipeline],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Entrena un modelo en un fold expansivo y calcula métricas."""
    fold_name = fold["name"]
    effective_train_end = pd.Period(
        fold["effective_train_end"],
        freq="M",
    )

    target_column = str(config["target"]["column"])

    baseline_column = str(
        config["baseline"]["prediction_feature"]
    )

    baseline_id = str(config["baseline"]["name"])
    dataset_path = str(config["modeling_dataset"]["path"])

    _, _, _, feature_columns = get_model_inputs(config)

    train_before_purge_mask, train_mask = training_label_masks(
        dataframe,
        evaluable,
        fold,
    )

    validation_mask = (
        dataframe["evaluation_split"].eq(fold_name)
        & evaluable
    )

    train = dataframe.loc[train_mask].copy()
    validation = dataframe.loc[validation_mask].copy()

    if train.empty:
        raise ValueError(
            f"Entrenamiento vacío para {fold_name}."
        )

    if validation.empty:
        raise ValueError(
            f"Validación vacía para {fold_name}."
        )

    pipeline = pipeline_factory()

    pipeline.fit(
        train[feature_columns],
        train[target_column].to_numpy(dtype=float),
    )

    prediction_raw = pipeline.predict(
        validation[feature_columns]
    )

    clip_predictions = bool(
        config["model_selection"][
            "clip_negative_predictions_to_zero"
        ]
    )

    prediction = (
        np.maximum(prediction_raw, 0)
        if clip_predictions
        else prediction_raw
    )

    actual = validation[target_column].to_numpy(dtype=float)
    baseline_prediction = validation[
        baseline_column
    ].to_numpy(dtype=float)

    metrics = calculate_metrics(actual, prediction)
    max_training_target = (
        train["target_date_month"].dt.to_period("M").max()
    )
    if max_training_target > effective_train_end:
        raise AssertionError(
            f"Etiqueta train posterior al cutoff efectivo en {fold_name}."
        )

    rows_before_purge = int(train_before_purge_mask.sum())
    rows_after_purge = int(train_mask.sum())

    metrics_row: dict[str, Any] = {
        "evaluation_split": fold_name,
        "model": model_id,
        "validation_start": fold["validation_start"],
        "structural_train_end": fold["structural_train_end"],
        "availability_train_end": fold["availability_train_end"],
        "effective_train_end": fold["effective_train_end"],
        "max_training_target": max_training_target.strftime("%Y-%m"),
        "train_end": fold["effective_train_end"],
        "train_rows_before_purge": rows_before_purge,
        "train_rows_after_purge": rows_after_purge,
        "train_rows_purged": rows_before_purge - rows_after_purge,
        "train_rows": rows_after_purge,
        **metrics,
        "negative_raw_predictions": int(
            (prediction_raw < 0).sum()
        ),
    }

    predictions = validation[
        [
            "territory_id",
            "territory_name",
            "target_month_id",
            "target_date_month",
            "evaluation_split",
            "source_snapshot_id",
            "pipeline_run_id",
            "data_version",
            "created_at",
        ]
    ].copy()

    predictions["model"] = model_id
    predictions["baseline_id"] = baseline_id
    predictions["dataset_path"] = dataset_path
    predictions["actual"] = actual
    predictions["baseline_prediction"] = baseline_prediction
    predictions["model_prediction_raw"] = prediction_raw
    predictions["model_prediction"] = prediction
    predictions["validation_start"] = fold["validation_start"]
    predictions["structural_train_end"] = fold["structural_train_end"]
    predictions["availability_train_end"] = fold["availability_train_end"]
    predictions["effective_train_end"] = fold["effective_train_end"]

    return predictions, metrics_row


def add_baseline_metrics(
    predictions: pd.DataFrame,
    metrics_rows: list[dict[str, Any]],
    baseline_id: str,
) -> None:
    """Añade una sola fila de baseline por validación."""
    baseline_data = (
        predictions.drop_duplicates(
            subset=[
                "territory_id",
                "target_month_id",
                "evaluation_split",
            ]
        )
    )

    for split_name, group in baseline_data.groupby(
        "evaluation_split",
        observed=True,
        sort=False,
    ):
        metrics_rows.append(
            {
                "evaluation_split": split_name,
                "model": baseline_id,
                "validation_start": "not_applicable",
                "structural_train_end": "not_applicable",
                "availability_train_end": "not_applicable",
                "effective_train_end": "not_applicable",
                "max_training_target": "not_applicable",
                "train_end": "not_applicable",
                "train_rows_before_purge": np.nan,
                "train_rows_after_purge": np.nan,
                "train_rows_purged": np.nan,
                "train_rows": np.nan,
                **calculate_metrics(
                    group["actual"].to_numpy(dtype=float),
                    group["baseline_prediction"].to_numpy(
                        dtype=float
                    ),
                ),
                "negative_raw_predictions": 0,
            }
        )


def add_pooled_metrics(
    predictions: pd.DataFrame,
    metrics_rows: list[dict[str, Any]],
    baseline_id: str,
) -> None:
    """Calcula métricas agregadas sobre los tres folds."""
    baseline_data = predictions.drop_duplicates(
        subset=[
            "territory_id",
            "target_month_id",
            "evaluation_split",
        ]
    )

    metrics_rows.append(
        {
            "evaluation_split": "validation_pooled",
            "model": baseline_id,
            "validation_start": "not_applicable",
            "structural_train_end": "not_applicable",
            "availability_train_end": "not_applicable",
            "effective_train_end": "not_applicable",
            "max_training_target": "not_applicable",
            "train_end": "expanding_folds",
            "train_rows_before_purge": np.nan,
            "train_rows_after_purge": np.nan,
            "train_rows_purged": np.nan,
            "train_rows": np.nan,
            **calculate_metrics(
                baseline_data["actual"].to_numpy(dtype=float),
                baseline_data[
                    "baseline_prediction"
                ].to_numpy(dtype=float),
            ),
            "negative_raw_predictions": 0,
        }
    )

    for model_id, group in predictions.groupby(
        "model",
        observed=True,
        sort=False,
    ):
        metrics_rows.append(
            {
                "evaluation_split": "validation_pooled",
                "model": model_id,
                "validation_start": "not_applicable",
                "structural_train_end": "expanding_folds",
                "availability_train_end": "expanding_folds",
                "effective_train_end": "expanding_folds",
                "max_training_target": "expanding_folds",
                "train_end": "expanding_folds",
                "train_rows_before_purge": np.nan,
                "train_rows_after_purge": np.nan,
                "train_rows_purged": np.nan,
                "train_rows": np.nan,
                **calculate_metrics(
                    group["actual"].to_numpy(dtype=float),
                    group["model_prediction"].to_numpy(
                        dtype=float
                    ),
                ),
                "negative_raw_predictions": int(
                    (
                        group["model_prediction_raw"] < 0
                    ).sum()
                ),
            }
        )


def format_number(value: float) -> str:
    """Formatea números con convención española."""
    return (
        f"{value:,.2f}"
        .replace(",", "X")
        .replace(".", ",")
        .replace("X", ".")
    )


def write_report(
    config: dict[str, Any],
    metrics: pd.DataFrame,
    ridge_best: pd.Series,
    hgb_result: pd.Series,
    baseline_result: pd.Series,
) -> None:
    """Genera el informe reproducible de selección."""
    pooled = metrics[
        metrics["evaluation_split"].eq(
            "validation_pooled"
        )
    ].copy()

    ridge_table = pooled[
        pooled["model"].str.startswith(
            "ridge_alpha_"
        )
    ].sort_values("MAE")

    ridge_rows = "\n".join(
        "| "
        + str(row.model)
        + " | "
        + format_number(float(row.MAE))
        + " | "
        + format_number(float(row.RMSE))
        + " | "
        + format_number(float(row.WAPE_pct))
        + " % |"
        for row in ridge_table.itertuples()
    )

    baseline_mae = float(baseline_result["MAE"])
    ridge_improvement = calculate_improvement_pct(
        baseline_mae,
        float(ridge_best["MAE"]),
    )
    hgb_improvement = calculate_improvement_pct(
        baseline_mae,
        float(hgb_result["MAE"]),
    )

    best_candidate = min(
        [ridge_best, hgb_result],
        key=lambda row: float(row["MAE"]),
    )

    selected_solution, best_candidate_improvement = (
        select_solution_by_validation(
            config,
            str(best_candidate["model"]),
            baseline_mae,
            float(best_candidate["MAE"]),
        )
    )

    minimum_improvement = float(
        config["model_selection"]["minimum_mae_improvement_pct"]
    )

    generated_at = datetime.now(
        timezone.utc
    ).isoformat()
    _, _, _, feature_columns = get_model_inputs(config)
    availability = config["point_in_time_availability"]
    test_status = str(
        config["validation"]["final_test"]["test_status"]
    )
    operational_inputs = ", ".join(
        f"`{feature}`" for feature in feature_columns
    )
    cutoff_metrics = (
        metrics[
            metrics["model"].eq(str(hgb_result["model"]))
            & metrics["evaluation_split"].isin(
                config["model_selection"]["selection_splits"]
            )
        ]
        .sort_values("evaluation_split")
    )
    cutoff_rows = "\n".join(
        "| "
        + " | ".join(
            [
                str(row.evaluation_split),
                str(row.validation_start),
                str(row.structural_train_end),
                str(row.availability_train_end),
                str(row.effective_train_end),
                str(row.max_training_target),
                str(int(row.train_rows_before_purge)),
                str(int(row.train_rows_after_purge)),
                str(int(row.train_rows_purged)),
            ]
        )
        + " |"
        for row in cutoff_metrics.itertuples()
    )

    report = f"""# Selección reproducible de modelos mediante validación temporal

- **Generado en UTC:** `{generated_at}`
- **Dataset:** `{config["modeling_dataset"]["path"]}`
- **Estrategia:** validación temporal expansiva
- **Splits de selección:** validation_1, validation_2 y validation_3
- **Test final utilizado en selección:** no
- **Estado de la ventana de test:** `{test_status}`
- **Filas comparables agregadas:** {int(baseline_result["rows"])}
- **Predicciones negativas:** recortadas a cero
- **Forecast origin:** `{availability["forecast_origin"]}`
- **Desfase EOTR mínimo seguro:** {int(availability["minimum_safe_eotr_lag_months"])} meses
- **Predictores operacionales:** {operational_inputs}

## Disponibilidad point-in-time de etiquetas de entrenamiento

Ridge y HGB aplican el mismo purge mensual. El límite estructural del fold
no presupone que una etiqueta EOTR ya esté publicada: el límite de
disponibilidad es `validation_start - minimum_safe_eotr_lag_months` y el
límite efectivo es el más restrictivo de ambos. El baseline lag-12 no se
entrena y no depende de este purge.

| Fold | Inicio validación | Fin estructural train | Fin por disponibilidad | Fin efectivo train | Máxima etiqueta usada | Filas antes | Filas después | Filas purgadas |
|---|---|---|---|---|---|---:|---:|---:|
{cutoff_rows}

## Alcance reproducido

La selección ejecutable compara el baseline estacional, la rejilla
documentada de Ridge y la configuración congelada `hgb_raw_02`.

El repositorio histórico no conserva la rejilla completa de
configuraciones HGB mencionada en la memoria. Por ello este script no
inventa configuraciones adicionales ni afirma reproducir una búsqueda
HGB que no quedó registrada.

## Baseline estacional lag-12

| Métrica | Resultado |
|---|---:|
| MAE | {format_number(baseline_mae)} |
| RMSE | {format_number(float(baseline_result["RMSE"]))} |
| WAPE | {format_number(float(baseline_result["WAPE_pct"]))} % |
| Sesgo medio | {format_number(float(baseline_result["mean_bias"]))} |

## Búsqueda de Ridge

| Configuración | MAE | RMSE | WAPE |
|---|---:|---:|---:|
{ridge_rows}

Mejor Ridge: `{ridge_best["model"]}`.

- MAE: {format_number(float(ridge_best["MAE"]))}
- Mejora frente al baseline: {format_number(ridge_improvement)} %

## HistGradientBoosting congelado

Configuración evaluada: `hgb_raw_02`.

- MAE: {format_number(float(hgb_result["MAE"]))}
- RMSE: {format_number(float(hgb_result["RMSE"]))}
- WAPE: {format_number(float(hgb_result["WAPE_pct"]))} %
- Sesgo medio: {format_number(float(hgb_result["mean_bias"]))}
- Mejora frente al baseline: {format_number(hgb_improvement)} %

## Decisión de validación

- Mejor candidato de machine learning: `{best_candidate["model"]}`
- Mejora agregada del mejor candidato: {format_number(best_candidate_improvement)} %
- Umbral mínimo configurado: {format_number(minimum_improvement)} %
- Solución seleccionada tras validación: `{selected_solution}`

El umbral pooled de mejora igual o superior al {format_number(minimum_improvement)} %
es la gate automática. La estabilidad entre folds se interpreta como
diagnóstico: ante evidencia inestable o insuficiente se prefiere el baseline,
sin consultar el test. El conjunto de test permanece excluido de este proceso.
"""

    REPORT_PATH.write_text(
        report,
        encoding="utf-8",
    )


def main() -> int:
    """Ejecuta la selección reproducible."""
    config = load_config()
    dataframe = load_modeling_dataset(config)
    evaluable = common_evaluable_mask(
        dataframe,
        config,
    )

    selection_config = config["model_selection"]
    selection_splits = [
        str(value)
        for value in selection_config["selection_splits"]
    ]

    folds = [
        fold
        for fold in get_validation_folds(
            config,
            include_test=False,
        )
        if fold["name"] in selection_splits
    ]

    if [fold["name"] for fold in folds] != selection_splits:
        raise ValueError(
            "Los folds configurados no coinciden con selection_splits."
        )

    ridge_alphas = [
        float(value)
        for value in selection_config[
            "ridge"
        ]["alpha_values"]
    ]

    hgb_model_id = str(
        selection_config[
            "hist_gradient_boosting"
        ]["model_id"]
    )

    model_factories: list[
        tuple[str, Callable[[], Pipeline]]
    ] = []

    for alpha in ridge_alphas:
        model_factories.append(
            (
                alpha_model_id(alpha),
                lambda alpha=alpha: build_ridge_pipeline(
                    config,
                    alpha,
                ),
            )
        )

    model_factories.append(
        (
            hgb_model_id,
            lambda: build_hgb_pipeline(config),
        )
    )

    predictions_parts: list[pd.DataFrame] = []
    metrics_rows: list[dict[str, Any]] = []

    for model_id, pipeline_factory in model_factories:
        for fold in folds:
            fold_predictions, fold_metrics = (
                evaluate_model_fold(
                    dataframe,
                    evaluable,
                    config,
                    fold,
                    model_id,
                    pipeline_factory,
                )
            )

            predictions_parts.append(
                fold_predictions
            )
            metrics_rows.append(
                fold_metrics
            )

    predictions = pd.concat(
        predictions_parts,
        ignore_index=True,
    )

    if predictions["evaluation_split"].eq("test").any():
        raise AssertionError(
            "El test no puede intervenir en la selección."
        )

    baseline_id = str(config["baseline"]["name"])

    add_baseline_metrics(
        predictions,
        metrics_rows,
        baseline_id,
    )

    add_pooled_metrics(
        predictions,
        metrics_rows,
        baseline_id,
    )

    metrics = pd.DataFrame(metrics_rows)

    pooled = metrics[
        metrics["evaluation_split"].eq(
            "validation_pooled"
        )
    ].copy()

    ridge_best = (
        pooled[
            pooled["model"].str.startswith(
                "ridge_alpha_"
            )
        ]
        .sort_values("MAE")
        .iloc[0]
    )

    hgb_result = pooled[
        pooled["model"].eq(hgb_model_id)
    ].iloc[0]

    baseline_result = pooled[
        pooled["model"].eq(baseline_id)
    ].iloc[0]

    PREDICTIONS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    METRICS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions.to_parquet(
        PREDICTIONS_PATH,
        index=False,
    )

    metrics.to_csv(
        METRICS_PATH,
        index=False,
        encoding="utf-8",
        float_format="%.6f",
    )

    write_report(
        config,
        metrics,
        ridge_best,
        hgb_result,
        baseline_result,
    )

    print("select_models.py: OK")
    print(
        "Mejor Ridge: "
        f"{ridge_best['model']} | "
        f"MAE {ridge_best['MAE']:.2f}"
    )
    print(
        "HGB: "
        f"{hgb_model_id} | "
        f"MAE {hgb_result['MAE']:.2f}"
    )
    print(
        "Baseline: "
        f"{baseline_id} | "
        f"MAE {baseline_result['MAE']:.2f}"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
