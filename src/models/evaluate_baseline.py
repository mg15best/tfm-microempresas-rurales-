"""
Evalúa de forma reproducible el baseline estacional lag-12.

Entrada:
- data/gold/gold_modeling_dataset_monthly.parquet
- data/metadata/modeling_config.yml

Salidas:
- data/metadata/baseline_metrics_summary.csv
- data/metadata/baseline_metrics_by_territory.csv
- data/metadata/baseline_metrics_by_month.csv
- data/metadata/baseline_metrics_by_season.csv
- data/metadata/baseline_evaluation_report.md
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "data" / "metadata" / "modeling_config.yml"
SUMMARY_PATH = PROJECT_ROOT / "data" / "metadata" / "baseline_metrics_summary.csv"
BY_TERRITORY_PATH = PROJECT_ROOT / "data" / "metadata" / "baseline_metrics_by_territory.csv"
BY_MONTH_PATH = PROJECT_ROOT / "data" / "metadata" / "baseline_metrics_by_month.csv"
BY_SEASON_PATH = PROJECT_ROOT / "data" / "metadata" / "baseline_metrics_by_season.csv"
REPORT_PATH = PROJECT_ROOT / "data" / "metadata" / "baseline_evaluation_report.md"


def load_config() -> dict[str, Any]:
    """Carga la configuración reproducible de modelado."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            "No se encontró la configuración: "
            f"{CONFIG_PATH.relative_to(PROJECT_ROOT)}"
        )

    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("modeling_config.yml no contiene un objeto YAML válido.")

    return config


def resolve_project_path(path_text: str) -> Path:
    """Resuelve una ruta relativa respecto a la raíz del proyecto."""
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def evaluation_split_names(config: dict[str, Any]) -> list[str]:
    """Devuelve las validaciones temporales y el test final."""
    folds = config["validation"]["folds"]
    return [*[str(fold["name"]) for fold in folds], "test"]


def prepare_evaluation_rows(
    dataframe: pd.DataFrame,
    *,
    split_names: list[str],
    target_column: str,
    prediction_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Selecciona filas evaluables y resume las exclusiones."""
    selected = dataframe[
        dataframe["evaluation_split"].isin(split_names)
    ].copy()

    selected["target_available"] = selected[target_column].notna()
    selected["prediction_available"] = selected[prediction_column].notna()
    selected["not_provisional"] = ~selected["is_provisional"].fillna(False)
    selected["baseline_evaluable"] = (
        selected["target_available"]
        & selected["prediction_available"]
        & selected["not_provisional"]
    )

    coverage = (
        selected.groupby("evaluation_split", observed=True)
        .agg(
            total_rows=("territory_id", "size"),
            evaluable_rows=("baseline_evaluable", "sum"),
            target_missing=("target_available", lambda values: int((~values).sum())),
            baseline_missing=("prediction_available", lambda values: int((~values).sum())),
            provisional_rows=("not_provisional", lambda values: int((~values).sum())),
        )
        .reindex(split_names)
        .reset_index()
    )

    coverage["excluded_rows"] = coverage["total_rows"] - coverage["evaluable_rows"]

    evaluation = selected.loc[selected["baseline_evaluable"]].copy()
    evaluation["actual"] = pd.to_numeric(
        evaluation[target_column], errors="raise"
    ).astype(float)
    evaluation["prediction"] = pd.to_numeric(
        evaluation[prediction_column], errors="raise"
    ).astype(float)
    evaluation["error"] = evaluation["prediction"] - evaluation["actual"]
    evaluation["absolute_error"] = evaluation["error"].abs()
    evaluation["squared_error"] = evaluation["error"] ** 2

    return evaluation, coverage


def calculate_metrics(group: pd.DataFrame) -> pd.Series:
    """Calcula MAE, RMSE, WAPE y sesgo medio."""
    actual_sum = float(group["actual"].abs().sum())
    absolute_error_sum = float(group["absolute_error"].sum())
    wape = absolute_error_sum / actual_sum * 100 if actual_sum != 0 else np.nan

    return pd.Series(
        {
            "rows": int(len(group)),
            "MAE": float(group["absolute_error"].mean()),
            "RMSE": float(np.sqrt(group["squared_error"].mean())),
            "WAPE_pct": float(wape),
            "mean_bias": float(group["error"].mean()),
        }
    )


def calculate_summary_metrics(
    evaluation: pd.DataFrame,
    split_names: list[str],
) -> pd.DataFrame:
    """Calcula métricas por split y globales."""
    by_split = (
        evaluation.groupby("evaluation_split", observed=True, sort=False)
        .apply(calculate_metrics, include_groups=False)
        .reindex(split_names)
        .reset_index()
        .rename(columns={"evaluation_split": "split"})
    )

    overall = calculate_metrics(evaluation).to_frame().T
    overall.insert(0, "split", "overall")

    summary = pd.concat([by_split, overall], ignore_index=True)
    summary.insert(0, "model", "seasonal_naive_lag_12")
    summary["rows"] = summary["rows"].round().astype("int64")
    return summary


def add_season(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Añade una estación a partir del mes calendario."""
    result = dataframe.copy()
    season_by_month = {
        1: "winter",
        2: "winter",
        3: "spring",
        4: "spring",
        5: "spring",
        6: "summer",
        7: "summer",
        8: "summer",
        9: "autumn",
        10: "autumn",
        11: "autumn",
        12: "winter",
    }
    result["season"] = result["month"].astype(int).map(season_by_month).astype("string")
    return result


def grouped_metrics(
    evaluation: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    """Calcula métricas para una agrupación."""
    metrics = (
        evaluation.groupby(group_columns, observed=True, sort=True)
        .apply(calculate_metrics, include_groups=False)
        .reset_index()
    )
    metrics.insert(0, "model", "seasonal_naive_lag_12")
    metrics["rows"] = metrics["rows"].round().astype("int64")
    return metrics


def markdown_table(dataframe: pd.DataFrame, *, float_decimals: int = 2) -> str:
    """Convierte un dataframe pequeño en tabla Markdown."""
    headers = [str(column) for column in dataframe.columns]
    rows: list[list[str]] = []

    for _, row in dataframe.iterrows():
        formatted: list[str] = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                formatted.append(f"{float(value):,.{float_decimals}f}")
            else:
                formatted.append(str(value))
        rows.append(formatted)

    header_line = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_line, separator, *body])


def write_report(
    *,
    dataset_path: Path,
    summary: pd.DataFrame,
    coverage: pd.DataFrame,
    by_month: pd.DataFrame,
) -> None:
    """Genera el informe Markdown del baseline."""
    generated_at = datetime.now(timezone.utc).isoformat()
    split_summary = summary[summary["split"].ne("overall")].copy()
    overall = summary[summary["split"].eq("overall")].iloc[0]
    hardest_split = split_summary.sort_values("MAE", ascending=False).iloc[0]
    best_split = split_summary.sort_values("MAE", ascending=True).iloc[0]
    hardest_month = by_month.sort_values("MAE", ascending=False).iloc[0]

    coverage_table = coverage[
        [
            "evaluation_split",
            "total_rows",
            "evaluable_rows",
            "excluded_rows",
            "baseline_missing",
            "provisional_rows",
        ]
    ].copy()

    metrics_table = summary[
        ["split", "rows", "MAE", "RMSE", "WAPE_pct", "mean_bias"]
    ].copy()

    report = f"""# Evaluación del baseline estacional lag-12

- **Dataset:** `{dataset_path.relative_to(PROJECT_ROOT)}`
- **Modelo:** `seasonal_naive_lag_12`
- **Predicción:** pernoctaciones del mismo mes del año anterior
- **Generado en UTC:** `{generated_at}`
- **Filas evaluadas:** {int(overall['rows']):,}
- **MAE global:** {float(overall['MAE']):,.2f}
- **RMSE global:** {float(overall['RMSE']):,.2f}
- **WAPE global:** {float(overall['WAPE_pct']):,.2f} %
- **Sesgo medio global:** {float(overall['mean_bias']):,.2f}

## Cobertura de evaluación

{markdown_table(coverage_table, float_decimals=0)}

Las filas sin `lag_12_overnight_stays` se excluyen porque el baseline no puede
generar una predicción válida. No se imputan como cero ni se sustituyen por
otra observación disponible.

## Métricas por partición temporal

{markdown_table(metrics_table)}

El split con mayor MAE es **{hardest_split['split']}**
({float(hardest_split['MAE']):,.2f}), mientras que el menor MAE se observa
en **{best_split['split']}** ({float(best_split['MAE']):,.2f}).

El sesgo se define como `predicción - valor real`. Un valor negativo indica
infraestimación de la demanda.

## Diagnóstico temporal básico

El mes calendario con mayor MAE agregado es el **{int(hardest_month['month'])}**,
con un MAE de **{float(hardest_month['MAE']):,.2f}**.

## Criterio de comparación posterior

Los modelos candidatos deberán compararse con este baseline sobre las mismas
filas evaluables y las mismas particiones temporales.
"""

    REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> int:
    """Ejecuta la evaluación reproducible del baseline."""
    config = load_config()
    dataset_path = resolve_project_path(config["modeling_dataset"]["path"])

    if not dataset_path.exists():
        raise FileNotFoundError(
            "No se encontró el dataset de modelado: "
            f"{dataset_path.relative_to(PROJECT_ROOT)}"
        )

    dataframe = pd.read_parquet(dataset_path)
    if dataframe.empty:
        raise ValueError("El dataset de modelado está vacío.")

    required_columns = {
        "territory_id",
        "territory_name",
        "target_month_id",
        "target_date_month",
        "month",
        "evaluation_split",
        "is_provisional",
        "target_overnight_stays_total",
        "lag_12_overnight_stays",
    }
    missing_columns = required_columns.difference(dataframe.columns)
    if missing_columns:
        raise ValueError(
            "Faltan columnas requeridas: " + ", ".join(sorted(missing_columns))
        )

    split_names = evaluation_split_names(config)
    target_column = str(config["target"]["column"])
    prediction_column = str(config["baseline"]["prediction_feature"])

    evaluation, coverage = prepare_evaluation_rows(
        dataframe,
        split_names=split_names,
        target_column=target_column,
        prediction_column=prediction_column,
    )

    if evaluation.empty:
        raise ValueError("No existen filas evaluables para el baseline.")
    if evaluation["is_provisional"].fillna(False).any():
        raise ValueError("La evaluación contiene filas provisionales.")

    summary = calculate_summary_metrics(evaluation, split_names)
    evaluation = add_season(evaluation)
    by_territory = grouped_metrics(evaluation, ["territory_id", "territory_name"])
    by_month = grouped_metrics(evaluation, ["month"])
    by_season = grouped_metrics(evaluation, ["season"])

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_PATH, index=False, encoding="utf-8", float_format="%.6f")
    by_territory.to_csv(
        BY_TERRITORY_PATH, index=False, encoding="utf-8", float_format="%.6f"
    )
    by_month.to_csv(BY_MONTH_PATH, index=False, encoding="utf-8", float_format="%.6f")
    by_season.to_csv(BY_SEASON_PATH, index=False, encoding="utf-8", float_format="%.6f")
    write_report(
        dataset_path=dataset_path,
        summary=summary,
        coverage=coverage,
        by_month=by_month,
    )

    print("Baseline estacional evaluado correctamente.")
    print(f"Filas evaluadas: {len(evaluation):,}")
    print("\nMétricas por split y global:")
    print(
        summary[
            ["split", "rows", "MAE", "RMSE", "WAPE_pct", "mean_bias"]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:,.2f}",
        )
    )
    print("\nSalidas:")
    for path in [
        SUMMARY_PATH,
        BY_TERRITORY_PATH,
        BY_MONTH_PATH,
        BY_SEASON_PATH,
        REPORT_PATH,
    ]:
        print("- " + str(path.relative_to(PROJECT_ROOT)))

    return 0


if __name__ == "__main__":
    sys.exit(main())