"""
Evalúa de forma reproducible el candidato final HistGradientBoosting
frente al baseline estacional lag-12.

La configuración hgb_raw_02 fue seleccionada únicamente mediante las
tres ventanas de validación temporal. El test final se evalúa una sola vez
y no se utiliza para modificar variables, transformaciones o hiperparámetros.

Entradas
--------
- data/gold/gold_modeling_dataset_monthly.parquet

Salidas
-------
- data/model_outputs/final_candidate_predictions.parquet
- data/metadata/final_candidate_metrics_by_split.csv
- data/metadata/final_candidate_test_by_territory.csv
- data/metadata/final_candidate_test_by_month.csv
- data/metadata/final_candidate_evaluation_report.md
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "gold"
    / "gold_modeling_dataset_monthly.parquet"
)

PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "data"
    / "model_outputs"
    / "final_candidate_predictions.parquet"
)

METRICS_BY_SPLIT_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "final_candidate_metrics_by_split.csv"
)

TEST_BY_TERRITORY_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "final_candidate_test_by_territory.csv"
)

TEST_BY_MONTH_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "final_candidate_test_by_month.csv"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "final_candidate_evaluation_report.md"
)

MODEL_ID = "hgb_raw_02"
BASELINE_ID = "seasonal_naive_lag_12"
RANDOM_STATE = 42

NUMERIC_FEATURES = [
    "year",
    "is_summer",
    "is_christmas_period",
    "covid_period",
    "lag_1_overnight_stays",
    "lag_3_overnight_stays",
    "lag_12_overnight_stays",
    "rolling_mean_3m_overnight_stays",
    "rolling_mean_12m_overnight_stays",
    "yoy_change_overnight_stays",
    "lag_1_occupancy_rate_pct",
    "lag_12_occupancy_rate_pct",
    "lag_1_weekend_occupancy_rate_pct",
    "lag_1_average_stay",
    "lag_12_average_stay",
    "lag_1_domestic_overnight_stays_share",
    "lag_1_foreign_overnight_stays_share",
    "lag_1_places_estimated",
    "lag_1_establishments_estimated",
    "lag_1_staff_employed",
]

CATEGORICAL_FEATURES = [
    "territory_id",
    "month",
    "quarter",
]

BOOLEAN_FEATURES = [
    "is_summer",
    "is_christmas_period",
    "covid_period",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

FOLDS = [
    {
        "name": "validation_1",
        "train_end": "2021-05-01",
    },
    {
        "name": "validation_2",
        "train_end": "2022-05-01",
    },
    {
        "name": "validation_3",
        "train_end": "2023-05-01",
    },
    {
        "name": "test",
        "train_end": "2024-05-01",
    },
]

MODEL_PARAMETERS: dict[str, Any] = {
    "learning_rate": 0.05,
    "max_iter": 300,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 20,
    "l2_regularization": 1.0,
    "early_stopping": False,
    "random_state": RANDOM_STATE,
}


def load_dataset() -> pd.DataFrame:
    """Carga y normaliza el dataset de modelado."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            "No se encontró el dataset de modelado: "
            f"{DATASET_PATH.relative_to(PROJECT_ROOT)}"
        )

    dataframe = pd.read_parquet(DATASET_PATH).copy()

    if dataframe.empty:
        raise ValueError("El dataset de modelado está vacío.")

    required_columns = {
        "territory_id",
        "territory_name",
        "target_month_id",
        "target_date_month",
        "month",
        "quarter",
        "evaluation_split",
        "is_provisional",
        "target_overnight_stays_total",
        "lag_12_overnight_stays",
        *FEATURE_COLUMNS,
    }

    missing_columns = required_columns.difference(dataframe.columns)

    if missing_columns:
        raise ValueError(
            "Faltan columnas requeridas: "
            + ", ".join(sorted(missing_columns))
        )

    dataframe["target_date_month"] = pd.to_datetime(
        dataframe["target_date_month"],
        errors="raise",
    )

    for column in BOOLEAN_FEATURES:
        dataframe[column] = (
            dataframe[column]
            .fillna(False)
            .astype("int8")
        )

    for column in NUMERIC_FEATURES:
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        ).astype(float)

    for column in CATEGORICAL_FEATURES:
        dataframe[column] = (
            dataframe[column]
            .astype("string")
            .fillna("missing")
        )

    return dataframe


def build_pipeline() -> Pipeline:
    """Construye la configuración congelada hgb_raw_02."""
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                "passthrough",
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        sparse_threshold=0,
    )

    model = HistGradientBoostingRegressor(**MODEL_PARAMETERS)

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def calculate_metrics(
    actual: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, float | int]:
    """Calcula MAE, RMSE, WAPE y sesgo medio."""
    actual = np.asarray(actual, dtype=float)
    prediction = np.asarray(prediction, dtype=float)

    error = prediction - actual
    absolute_error = np.abs(error)
    actual_sum = np.abs(actual).sum()

    return {
        "rows": int(len(actual)),
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
    """Calcula la mejora porcentual; positivo significa mejora."""
    if baseline_value == 0:
        return np.nan

    return float(
        (baseline_value - model_value)
        / baseline_value
        * 100
    )


def evaluate_fold(
    dataframe: pd.DataFrame,
    common_evaluable: pd.Series,
    fold: dict[str, str],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Entrena hasta el corte del fold y genera predicciones comparables."""
    fold_name = fold["name"]
    train_end = pd.Timestamp(fold["train_end"])

    train_mask = (
        dataframe["target_date_month"].le(train_end)
        & common_evaluable
    )

    evaluation_mask = (
        dataframe["evaluation_split"].eq(fold_name)
        & common_evaluable
    )

    train = dataframe.loc[train_mask].copy()
    evaluation = dataframe.loc[evaluation_mask].copy()

    if train.empty:
        raise ValueError(
            f"Entrenamiento vacío para {fold_name}."
        )

    if evaluation.empty:
        raise ValueError(
            f"Evaluación vacía para {fold_name}."
        )

    X_train = train[FEATURE_COLUMNS]

    y_train = pd.to_numeric(
        train["target_overnight_stays_total"],
        errors="raise",
    ).to_numpy(dtype=float)

    X_evaluation = evaluation[FEATURE_COLUMNS]

    actual = pd.to_numeric(
        evaluation["target_overnight_stays_total"],
        errors="raise",
    ).to_numpy(dtype=float)

    baseline_prediction = pd.to_numeric(
        evaluation["lag_12_overnight_stays"],
        errors="raise",
    ).to_numpy(dtype=float)

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    model_prediction_raw = pipeline.predict(X_evaluation)
    model_prediction = np.maximum(model_prediction_raw, 0)

    baseline_metrics = calculate_metrics(
        actual,
        baseline_prediction,
    )

    model_metrics = calculate_metrics(
        actual,
        model_prediction,
    )

    metrics_rows = [
        {
            "evaluation_split": fold_name,
            "model": BASELINE_ID,
            "train_end": train_end.strftime("%Y-%m"),
            "train_rows": int(len(train)),
            **baseline_metrics,
            "negative_raw_predictions": 0,
        },
        {
            "evaluation_split": fold_name,
            "model": MODEL_ID,
            "train_end": train_end.strftime("%Y-%m"),
            "train_rows": int(len(train)),
            **model_metrics,
            "negative_raw_predictions": int(
                (model_prediction_raw < 0).sum()
            ),
        },
    ]

    predictions = evaluation[
        [
            "territory_id",
            "territory_name",
            "target_month_id",
            "target_date_month",
            "month",
            "quarter",
            "evaluation_split",
        ]
    ].copy()

    predictions["actual"] = actual
    predictions["baseline_prediction"] = baseline_prediction
    predictions["model_prediction_raw"] = model_prediction_raw
    predictions["model_prediction"] = model_prediction

    predictions["baseline_error"] = (
        predictions["baseline_prediction"]
        - predictions["actual"]
    )

    predictions["model_error"] = (
        predictions["model_prediction"]
        - predictions["actual"]
    )

    predictions["baseline_absolute_error"] = (
        predictions["baseline_error"].abs()
    )

    predictions["model_absolute_error"] = (
        predictions["model_error"].abs()
    )

    predictions["model_improves_row"] = (
        predictions["model_absolute_error"]
        < predictions["baseline_absolute_error"]
    )

    return predictions, metrics_rows


def add_pooled_validation_metrics(
    predictions: pd.DataFrame,
    metrics_rows: list[dict[str, Any]],
) -> None:
    """Añade las métricas agregadas de las tres validaciones."""
    validation = predictions[
        predictions["evaluation_split"].isin(
            [
                "validation_1",
                "validation_2",
                "validation_3",
            ]
        )
    ].copy()

    for model_id, prediction_column in [
        (
            BASELINE_ID,
            "baseline_prediction",
        ),
        (
            MODEL_ID,
            "model_prediction",
        ),
    ]:
        metrics = calculate_metrics(
            validation["actual"].to_numpy(dtype=float),
            validation[prediction_column].to_numpy(dtype=float),
        )

        metrics_rows.append(
            {
                "evaluation_split": "validation_pooled",
                "model": model_id,
                "train_end": "expanding_folds",
                "train_rows": np.nan,
                **metrics,
                "negative_raw_predictions": (
                    int(
                        (
                            validation["model_prediction_raw"]
                            < 0
                        ).sum()
                    )
                    if model_id == MODEL_ID
                    else 0
                ),
            }
        )


def build_comparison_table(
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Añade la mejora del candidato frente al baseline por split."""
    baseline = (
        metrics[
            metrics["model"].eq(BASELINE_ID)
        ]
        .set_index("evaluation_split")
    )

    model = (
        metrics[
            metrics["model"].eq(MODEL_ID)
        ]
        .set_index("evaluation_split")
    )

    rows: list[dict[str, Any]] = []

    for split_name in model.index:
        baseline_row = baseline.loc[split_name]
        model_row = model.loc[split_name]

        rows.append(
            {
                "evaluation_split": split_name,
                "baseline_MAE": float(
                    baseline_row["MAE"]
                ),
                "model_MAE": float(
                    model_row["MAE"]
                ),
                "mae_improvement_pct": (
                    calculate_improvement_pct(
                        float(baseline_row["MAE"]),
                        float(model_row["MAE"]),
                    )
                ),
                "baseline_RMSE": float(
                    baseline_row["RMSE"]
                ),
                "model_RMSE": float(
                    model_row["RMSE"]
                ),
                "rmse_improvement_pct": (
                    calculate_improvement_pct(
                        float(baseline_row["RMSE"]),
                        float(model_row["RMSE"]),
                    )
                ),
                "baseline_WAPE_pct": float(
                    baseline_row["WAPE_pct"]
                ),
                "model_WAPE_pct": float(
                    model_row["WAPE_pct"]
                ),
                "baseline_mean_bias": float(
                    baseline_row["mean_bias"]
                ),
                "model_mean_bias": float(
                    model_row["mean_bias"]
                ),
            }
        )

    return pd.DataFrame(rows)


def grouped_test_metrics(
    test_predictions: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    """Calcula la comparación de test para un nivel de agrupación."""
    rows: list[dict[str, Any]] = []

    grouped = test_predictions.groupby(
        group_columns,
        observed=True,
        sort=True,
    )

    for group_key, group in grouped:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        baseline_metrics = calculate_metrics(
            group["actual"].to_numpy(dtype=float),
            group["baseline_prediction"].to_numpy(dtype=float),
        )

        model_metrics = calculate_metrics(
            group["actual"].to_numpy(dtype=float),
            group["model_prediction"].to_numpy(dtype=float),
        )

        row = {
            column: value
            for column, value in zip(
                group_columns,
                group_key,
                strict=True,
            )
        }

        row.update(
            {
                "rows": int(len(group)),
                "baseline_MAE": baseline_metrics["MAE"],
                "model_MAE": model_metrics["MAE"],
                "mae_improvement_pct": (
                    calculate_improvement_pct(
                        float(baseline_metrics["MAE"]),
                        float(model_metrics["MAE"]),
                    )
                ),
                "baseline_RMSE": baseline_metrics["RMSE"],
                "model_RMSE": model_metrics["RMSE"],
                "baseline_WAPE_pct": (
                    baseline_metrics["WAPE_pct"]
                ),
                "model_WAPE_pct": model_metrics["WAPE_pct"],
                "baseline_mean_bias": (
                    baseline_metrics["mean_bias"]
                ),
                "model_mean_bias": (
                    model_metrics["mean_bias"]
                ),
                "model_improves": bool(
                    model_metrics["MAE"]
                    < baseline_metrics["MAE"]
                ),
            }
        )

        rows.append(row)

    return pd.DataFrame(rows)


def markdown_table(
    dataframe: pd.DataFrame,
    *,
    decimals: int = 2,
) -> str:
    """Convierte un dataframe pequeño en una tabla Markdown."""
    headers = [
        str(column)
        for column in dataframe.columns
    ]

    table_rows: list[list[str]] = []

    for _, row in dataframe.iterrows():
        formatted: list[str] = []

        for value in row:
            if isinstance(
                value,
                (float, np.floating),
            ):
                if np.isnan(value):
                    formatted.append("")
                else:
                    formatted.append(
                        f"{float(value):,.{decimals}f}"
                    )
            else:
                formatted.append(str(value))

        table_rows.append(formatted)

    header = "| " + " | ".join(headers) + " |"
    separator = (
        "| "
        + " | ".join(["---"] * len(headers))
        + " |"
    )

    body = [
        "| " + " | ".join(row) + " |"
        for row in table_rows
    ]

    return "\n".join(
        [header, separator, *body]
    )


def write_report(
    comparison: pd.DataFrame,
    territory_table: pd.DataFrame,
    month_table: pd.DataFrame,
    test_predictions: pd.DataFrame,
) -> None:
    """Genera el informe reproducible de evaluación final."""
    generated_at = datetime.now(
        timezone.utc
    ).isoformat()

    test_row = comparison[
        comparison["evaluation_split"].eq("test")
    ].iloc[0]

    validation_row = comparison[
        comparison["evaluation_split"].eq(
            "validation_pooled"
        )
    ].iloc[0]

    territories_improved = int(
        territory_table["model_improves"].sum()
    )

    territory_count = int(len(territory_table))

    months_improved = int(
        month_table["model_improves"].sum()
    )

    month_count = int(len(month_table))

    validation_threshold_pass = bool(
        validation_row["mae_improvement_pct"] >= 5
    )

    test_threshold_pass = bool(
        test_row["mae_improvement_pct"] >= 5
    )

    territory_majority_pass = bool(
        territories_improved > territory_count / 2
    )

    full_promotion_pass = bool(
        validation_threshold_pass
        and test_threshold_pass
        and territory_majority_pass
    )

    decision = (
        "El candidato satisface todos los criterios de promoción."
        if full_promotion_pass
        else (
            "El candidato mejora claramente el test final, pero no "
            "satisface el criterio de mejora agregada en validación. "
            "Se documenta como mejor modelo de machine learning en test, "
            "mientras el baseline se conserva como referencia y fallback "
            "operativo por su mayor estabilidad temporal."
        )
    )

    comparison_report = comparison[
        [
            "evaluation_split",
            "baseline_MAE",
            "model_MAE",
            "mae_improvement_pct",
            "baseline_RMSE",
            "model_RMSE",
            "baseline_WAPE_pct",
            "model_WAPE_pct",
            "baseline_mean_bias",
            "model_mean_bias",
        ]
    ].copy()

    month_report = (
        month_table[
            [
                "month",
                "rows",
                "baseline_MAE",
                "model_MAE",
                "mae_improvement_pct",
                "model_improves",
            ]
        ]
        .sort_values("month")
        .copy()
    )

    territory_best = (
        territory_table
        .sort_values(
            "mae_improvement_pct",
            ascending=False,
        )
        .head(5)
        [
            [
                "territory_id",
                "territory_name",
                "baseline_MAE",
                "model_MAE",
                "mae_improvement_pct",
            ]
        ]
    )

    territory_worst = (
        territory_table
        .sort_values(
            "mae_improvement_pct",
            ascending=True,
        )
        .head(5)
        [
            [
                "territory_id",
                "territory_name",
                "baseline_MAE",
                "model_MAE",
                "mae_improvement_pct",
            ]
        ]
    )

    report = f"""# Evaluación final del candidato HistGradientBoosting

- **Dataset:** `{DATASET_PATH.relative_to(PROJECT_ROOT)}`
- **Candidato:** `{MODEL_ID}`
- **Baseline:** `{BASELINE_ID}`
- **Generado en UTC:** `{generated_at}`
- **Filas de test:** {len(test_predictions):,}
- **Periodo de test:** {test_predictions["target_date_month"].min():%Y-%m} → {test_predictions["target_date_month"].max():%Y-%m}
- **Territorios de test:** {test_predictions["territory_id"].nunique()}

## Configuración congelada

| Parámetro | Valor |
|---|---:|
| target_transform | raw |
| learning_rate | 0,05 |
| max_iter | 300 |
| max_leaf_nodes | 31 |
| min_samples_leaf | 20 |
| l2_regularization | 1,0 |
| early_stopping | False |
| random_state | 42 |

La configuración fue seleccionada exclusivamente con las tres ventanas de
validación. El test final no se utilizó para modificar hiperparámetros,
variables ni transformaciones.

## Comparación temporal

{markdown_table(comparison_report)}

## Resultado principal de test

El candidato obtiene un MAE de **{float(test_row["model_MAE"]):,.2f}**,
frente a **{float(test_row["baseline_MAE"]):,.2f}** del baseline. La mejora
relativa en MAE es del **{float(test_row["mae_improvement_pct"]):,.2f} %**.

El RMSE mejora un **{float(test_row["rmse_improvement_pct"]):,.2f} %**.
El WAPE pasa de **{float(test_row["baseline_WAPE_pct"]):,.2f} %** a
**{float(test_row["model_WAPE_pct"]):,.2f} %**.

El sesgo medio del candidato es **{float(test_row["model_mean_bias"]):,.2f}**,
por lo que presenta una ligera tendencia agregada a sobreestimar. El baseline
presenta un sesgo de **{float(test_row["baseline_mean_bias"]):,.2f}**.

## Consistencia territorial y mensual

El candidato mejora el MAE en **{territories_improved} de {territory_count}
territorios ({territories_improved / territory_count * 100:.2f} %)**.

También mejora en **{months_improved} de {month_count} meses calendario**.

{markdown_table(month_report)}

## Territorios con mayor mejora

{markdown_table(territory_best)}

## Territorios con mayor deterioro

{markdown_table(territory_worst)}

## Criterios de promoción

| Criterio | Resultado |
|---|---|
| Mejora MAE agregada en validación ≥ 5 % | {"PASS" if validation_threshold_pass else "FAIL"} |
| Mejora MAE en test ≥ 5 % | {"PASS" if test_threshold_pass else "FAIL"} |
| Mejora en la mayoría de territorios | {"PASS" if territory_majority_pass else "FAIL"} |
| Promoción completa | {"PASS" if full_promotion_pass else "FAIL"} |

## Decisión

{decision}

No se realizarán ajustes posteriores basados en el test final. Cualquier
mejora futura deberá definirse como un nuevo experimento y volver a validarse
sin reutilizar este test para seleccionar hiperparámetros.
"""

    REPORT_PATH.write_text(
        report,
        encoding="utf-8",
    )


def main() -> int:
    """Ejecuta validaciones, test y generación de artefactos."""
    dataframe = load_dataset()

    common_evaluable = (
        dataframe["target_overnight_stays_total"].notna()
        & dataframe["lag_12_overnight_stays"].notna()
        & ~dataframe["is_provisional"].fillna(False)
    )

    predictions_parts: list[pd.DataFrame] = []
    metrics_rows: list[dict[str, Any]] = []

    for fold in FOLDS:
        fold_predictions, fold_metrics = evaluate_fold(
            dataframe,
            common_evaluable,
            fold,
        )

        predictions_parts.append(fold_predictions)
        metrics_rows.extend(fold_metrics)

    predictions = pd.concat(
        predictions_parts,
        ignore_index=True,
    )

    add_pooled_validation_metrics(
        predictions,
        metrics_rows,
    )

    metrics = pd.DataFrame(metrics_rows)
    comparison = build_comparison_table(metrics)

    test_predictions = predictions[
        predictions["evaluation_split"].eq("test")
    ].copy()

    test_predictions["month"] = pd.to_numeric(
        test_predictions["month"],
        errors="raise",
    ).astype(int)

    territory_table = grouped_test_metrics(
        test_predictions,
        [
            "territory_id",
            "territory_name",
        ],
    )

    month_table = (
        grouped_test_metrics(
            test_predictions,
            ["month"],
        )
        .sort_values("month")
    )

    PREDICTIONS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    METRICS_BY_SPLIT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions.to_parquet(
        PREDICTIONS_PATH,
        index=False,
    )

    metrics.to_csv(
        METRICS_BY_SPLIT_PATH,
        index=False,
        encoding="utf-8",
        float_format="%.6f",
    )

    territory_table.to_csv(
        TEST_BY_TERRITORY_PATH,
        index=False,
        encoding="utf-8",
        float_format="%.6f",
    )

    month_table.to_csv(
        TEST_BY_MONTH_PATH,
        index=False,
        encoding="utf-8",
        float_format="%.6f",
    )

    write_report(
        comparison,
        territory_table,
        month_table,
        test_predictions,
    )

    test_comparison = comparison[
        comparison["evaluation_split"].eq("test")
    ].iloc[0]

    validation_comparison = comparison[
        comparison["evaluation_split"].eq(
            "validation_pooled"
        )
    ].iloc[0]

    territories_improved = int(
        territory_table["model_improves"].sum()
    )

    print(
        "Evaluación final reproducida correctamente."
    )

    print("\nValidación agregada:")
    print(
        "Baseline MAE: "
        f"{validation_comparison['baseline_MAE']:,.2f}"
    )
    print(
        "HGB MAE: "
        f"{validation_comparison['model_MAE']:,.2f}"
    )
    print(
        "Mejora MAE: "
        f"{validation_comparison['mae_improvement_pct']:,.2f} %"
    )

    print("\nTest final:")
    print(
        "Baseline MAE: "
        f"{test_comparison['baseline_MAE']:,.2f}"
    )
    print(
        "HGB MAE: "
        f"{test_comparison['model_MAE']:,.2f}"
    )
    print(
        "Mejora MAE: "
        f"{test_comparison['mae_improvement_pct']:,.2f} %"
    )
    print(
        "HGB RMSE: "
        f"{test_comparison['model_RMSE']:,.2f}"
    )
    print(
        "HGB WAPE: "
        f"{test_comparison['model_WAPE_pct']:,.2f} %"
    )
    print(
        "HGB sesgo medio: "
        f"{test_comparison['model_mean_bias']:,.2f}"
    )

    print(
        "\nTerritorios mejorados: "
        f"{territories_improved}"
        f" / {len(territory_table)}"
    )

    print("\nSalidas:")

    for path in [
        PREDICTIONS_PATH,
        METRICS_BY_SPLIT_PATH,
        TEST_BY_TERRITORY_PATH,
        TEST_BY_MONTH_PATH,
        REPORT_PATH,
    ]:
        print(
            "- "
            + str(
                path.relative_to(PROJECT_ROOT)
            )
        )

    expected_rows = {
        "validation_1": 550,
        "validation_2": 600,
        "validation_3": 600,
        "test": 600,
    }

    actual_rows = (
        predictions.groupby(
            "evaluation_split",
            observed=True,
        )
        .size()
        .to_dict()
    )

    assert actual_rows == expected_rows
    assert len(test_predictions) == 600
    assert (
        test_predictions["territory_id"].nunique()
        == 50
    )

    assert np.isclose(
        float(
            test_comparison["baseline_MAE"]
        ),
        3045.00,
        atol=0.01,
        rtol=0,
    )

    assert np.isclose(
        float(
            test_comparison["model_MAE"]
        ),
        2760.59,
        atol=0.01,
        rtol=0,
    )

    assert np.isclose(
        float(
            test_comparison["mae_improvement_pct"]
        ),
        9.34,
        atol=0.01,
        rtol=0,
    )

    assert territories_improved == 41

    print("\nFinal candidate evaluation: OK")

    return 0


if __name__ == "__main__":
    sys.exit(main())
