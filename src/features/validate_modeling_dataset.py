"""
Validación reproducible del dataset gold de modelado temporal.

Entradas:
- data/gold/gold_modeling_dataset_monthly.parquet
- data/gold/gold_tourism_demand_monthly.parquet
- data/metadata/modeling_validation_rules.yml

Salida:
- data/metadata/modeling_data_quality_report.md

Devuelve 0 sin fallos críticos y 1 cuando existe algún FAIL.
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
RULES_PATH = PROJECT_ROOT / "data" / "metadata" / "modeling_validation_rules.yml"
SOURCE_PATH = PROJECT_ROOT / "data" / "gold" / "gold_tourism_demand_monthly.parquet"
REPORT_PATH = PROJECT_ROOT / "data" / "metadata" / "modeling_data_quality_report.md"


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró: {path.relative_to(PROJECT_ROOT)}"
        )

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ValueError(
            f"{path.name} no contiene una estructura YAML válida."
        )

    return data


def resolve_project_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def add_result(
    results: list[dict[str, str]],
    check: str,
    passed: bool,
    details: str,
    severity: str = "error",
) -> None:
    status = "PASS" if passed else ("WARN" if severity == "warning" else "FAIL")
    results.append(
        {
            "check": check,
            "status": status,
            "severity": severity,
            "details": details,
        }
    )


def numeric_array(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(
        series,
        errors="coerce",
    ).to_numpy(dtype=float, na_value=np.nan)


def count_differences(
    actual: pd.Series,
    expected: pd.Series,
    tolerance: float,
) -> int:
    actual_values = numeric_array(actual)
    expected_values = numeric_array(expected)

    equal = (
        (np.isnan(actual_values) & np.isnan(expected_values))
        | np.isclose(
            actual_values,
            expected_values,
            rtol=0,
            atol=tolerance,
            equal_nan=False,
        )
    )
    return int((~equal).sum())


def calendar_lookup(
    source: pd.DataFrame,
    source_column: str,
    offset_months: int,
    expected_column: str,
) -> pd.DataFrame:
    lookup = source[
        ["territory_id", "date_month", source_column]
    ].copy()

    lookup["target_date_month"] = (
        lookup["date_month"]
        + pd.DateOffset(months=offset_months)
    )

    return lookup[
        ["territory_id", "target_date_month", source_column]
    ].rename(columns={source_column: expected_column})


def expected_rolling(
    modeling: pd.DataFrame,
    source: pd.DataFrame,
    source_column: str,
    window_months: int,
) -> pd.Series:
    check = modeling[
        ["territory_id", "target_date_month"]
    ].copy()

    expected_columns: list[str] = []

    for offset in range(1, window_months + 1):
        expected_column = f"__expected_{offset}"
        expected_columns.append(expected_column)

        check = check.merge(
            calendar_lookup(
                source,
                source_column,
                offset,
                expected_column,
            ),
            on=["territory_id", "target_date_month"],
            how="left",
            validate="one_to_one",
        )

    complete = check[expected_columns].notna().all(axis=1)

    return (
        check[expected_columns]
        .mean(axis=1, skipna=False)
        .where(complete)
        .astype("Float64")
    )


def validate_structure(
    modeling: pd.DataFrame,
    rules: dict[str, Any],
    results: list[dict[str, str]],
) -> bool:
    dataset_rules = rules["dataset"]
    required = dataset_rules["required_columns"]
    missing = [
        column
        for column in required
        if column not in modeling.columns
    ]

    add_result(
        results,
        "required_columns",
        not missing,
        (
            "Todas las columnas obligatorias están presentes."
            if not missing
            else "Faltan: " + ", ".join(missing)
        ),
    )

    if missing:
        return False

    key = dataset_rules["key_columns"]
    null_keys = int(modeling[key].isna().any(axis=1).sum())
    duplicates = int(modeling.duplicated(subset=key).sum())

    add_result(
        results,
        "modeling_key_not_null",
        null_keys == 0,
        f"Filas con clave nula: {null_keys}.",
    )
    add_result(
        results,
        "unique_modeling_key",
        duplicates == 0,
        f"Claves duplicadas: {duplicates}.",
    )

    expected_horizon = int(
        dataset_rules["expected_forecast_horizon"]
    )
    invalid_horizon = int(
        modeling["forecast_horizon"].ne(expected_horizon).sum()
    )
    add_result(
        results,
        "forecast_horizon",
        invalid_horizon == 0,
        f"Filas con horizonte distinto de {expected_horizon}: {invalid_horizon}.",
    )

    observed_territories = int(
        modeling["territory_id"].nunique()
    )
    expected_territories = int(
        dataset_rules["expected_territory_count"]
    )
    add_result(
        results,
        "territory_count",
        observed_territories == expected_territories,
        (
            f"Territorios observados: {observed_territories}; "
            f"esperados: {expected_territories}."
        ),
    )

    expected_level = dataset_rules["expected_territory_level"]
    invalid_levels = int(
        modeling["territory_level"]
        .dropna()
        .ne(expected_level)
        .sum()
    )
    add_result(
        results,
        "territory_level",
        invalid_levels == 0,
        f"Filas con nivel distinto de {expected_level}: {invalid_levels}.",
    )

    return True


def validate_ranges(
    modeling: pd.DataFrame,
    rules: dict[str, Any],
    results: list[dict[str, str]],
) -> None:
    validation = rules["validation"]

    for column in validation["non_negative_columns"]:
        values = pd.to_numeric(
            modeling[column],
            errors="coerce",
        ).dropna()
        invalid = int(values.lt(0).sum())
        add_result(
            results,
            f"non_negative::{column}",
            invalid == 0,
            f"Valores negativos: {invalid}.",
        )

    for column in validation["percentage_columns_0_100"]:
        values = pd.to_numeric(
            modeling[column],
            errors="coerce",
        ).dropna()
        invalid = int((~values.between(0, 100)).sum())
        add_result(
            results,
            f"range_0_100::{column}",
            invalid == 0,
            f"Valores fuera de 0-100: {invalid}.",
        )

    for column in validation["share_columns_0_1"]:
        values = pd.to_numeric(
            modeling[column],
            errors="coerce",
        ).dropna()
        invalid = int((~values.between(0, 1)).sum())
        add_result(
            results,
            f"range_0_1::{column}",
            invalid == 0,
            f"Valores fuera de 0-1: {invalid}.",
        )

    tolerance = float(validation["numeric_tolerance"])

    for pair in validation["share_pairs"]:
        first = pair["first"]
        second = pair["second"]
        expected_sum = float(pair["expected_sum"])
        complete = modeling[first].notna() & modeling[second].notna()

        values = (
            pd.to_numeric(modeling.loc[complete, first], errors="coerce")
            + pd.to_numeric(modeling.loc[complete, second], errors="coerce")
        ).to_numpy(dtype=float)

        invalid = int(
            (~np.isclose(
                values,
                expected_sum,
                rtol=0,
                atol=tolerance,
            )).sum()
        )

        add_result(
            results,
            f"share_pair::{pair['name']}",
            invalid == 0,
            f"Filas completas que no suman {expected_sum}: {invalid}.",
        )

    allowed_splits = set(
        validation["allowed_evaluation_splits"]
    )
    observed_splits = set(
        modeling["evaluation_split"]
        .dropna()
        .astype(str)
        .unique()
    )
    unexpected_splits = sorted(
        observed_splits.difference(allowed_splits)
    )
    add_result(
        results,
        "allowed_evaluation_splits",
        not unexpected_splits,
        (
            "Todos los splits están permitidos."
            if not unexpected_splits
            else "No permitidos: " + ", ".join(unexpected_splits)
        ),
    )

    allowed_flags = set(
        validation["allowed_data_quality_flags"]
    )
    observed_flags = set(
        modeling["data_quality_flag"]
        .dropna()
        .astype(str)
        .unique()
    )
    unexpected_flags = sorted(
        observed_flags.difference(allowed_flags)
    )
    add_result(
        results,
        "allowed_data_quality_flags",
        not unexpected_flags,
        (
            "Todos los indicadores de calidad están permitidos."
            if not unexpected_flags
            else "No permitidos: " + ", ".join(unexpected_flags)
        ),
    )


def validate_dates_and_splits(
    modeling: pd.DataFrame,
    rules: dict[str, Any],
    results: list[dict[str, str]],
) -> None:
    expected_month_id = modeling[
        "target_date_month"
    ].dt.strftime("%Y-%m")

    mismatched_months = int(
        modeling["target_month_id"]
        .astype("string")
        .ne(expected_month_id)
        .sum()
    )
    add_result(
        results,
        "target_month_matches_date",
        mismatched_months == 0,
        f"Filas incoherentes: {mismatched_months}.",
    )

    non_first_day = int(
        modeling["target_date_month"].dt.day.ne(1).sum()
    )
    add_result(
        results,
        "target_date_first_day",
        non_first_day == 0,
        f"Fechas que no son día 1: {non_first_day}.",
    )

    split_rules = rules["validation"]["split_rules"]

    for split_name, split_rule in split_rules.items():
        start = pd.Timestamp(f"{split_rule['start']}-01")
        end = (
            pd.Timestamp(f"{split_rule['end']}-01")
            + pd.offsets.MonthEnd(0)
        )

        actual = modeling["evaluation_split"].eq(split_name)
        expected = modeling["target_date_month"].between(start, end)
        mismatch = int(actual.ne(expected).sum())

        add_result(
            results,
            f"split_dates::{split_name}",
            mismatch == 0,
            f"Filas asignadas incorrectamente: {mismatch}.",
        )

    provisional_monitoring = modeling[
        "evaluation_split"
    ].eq("provisional_monitoring")

    invalid_provisional = int(
        (
            provisional_monitoring
            & ~modeling["is_provisional"].fillna(False)
        ).sum()
    )
    add_result(
        results,
        "provisional_monitoring_rows",
        invalid_provisional == 0,
        (
            "Filas de seguimiento no marcadas como provisionales: "
            f"{invalid_provisional}."
        ),
    )

    provisional_in_selection = int(
        (
            modeling["is_provisional"].fillna(False)
            & modeling["evaluation_split"].isin(
                [
                    "validation_1",
                    "validation_2",
                    "validation_3",
                    "test",
                ]
            )
        ).sum()
    )
    add_result(
        results,
        "provisional_excluded_from_selection",
        provisional_in_selection == 0,
        (
            "Filas provisionales en validación o test: "
            f"{provisional_in_selection}."
        ),
    )


def validate_temporal_features(
    modeling: pd.DataFrame,
    source: pd.DataFrame,
    rules: dict[str, Any],
    results: list[dict[str, str]],
) -> None:
    validation = rules["validation"]
    tolerance = float(validation["numeric_tolerance"])

    for lag_rule in validation["temporal_integrity"]["lag_rules"]:
        feature = lag_rule["feature"]
        lookup = calendar_lookup(
            source,
            lag_rule["source"],
            int(lag_rule["offset_months"]),
            "__expected",
        )

        check = modeling[
            ["territory_id", "target_date_month", feature]
        ].merge(
            lookup,
            on=["territory_id", "target_date_month"],
            how="left",
            validate="one_to_one",
        )

        differences = count_differences(
            check[feature],
            check["__expected"],
            tolerance,
        )
        add_result(
            results,
            f"calendar_lag::{feature}",
            differences == 0,
            f"Diferencias respecto al mes exacto: {differences}.",
        )

    for rolling_rule in validation["rolling_rules"]:
        feature = rolling_rule["feature"]
        expected = expected_rolling(
            modeling,
            source,
            rolling_rule["source"],
            int(rolling_rule["window_months"]),
        )
        differences = count_differences(
            modeling[feature],
            expected,
            tolerance,
        )
        add_result(
            results,
            f"calendar_rolling::{feature}",
            differences == 0,
            (
                "Diferencias respecto a la ventana "
                f"calendárica completa: {differences}."
            ),
        )

    yoy_check = modeling[
        [
            "territory_id",
            "target_date_month",
            "yoy_change_overnight_stays",
        ]
    ].merge(
        calendar_lookup(
            source,
            "overnight_stays_total",
            1,
            "__lag_1",
        ),
        on=["territory_id", "target_date_month"],
        how="left",
        validate="one_to_one",
    ).merge(
        calendar_lookup(
            source,
            "overnight_stays_total",
            13,
            "__lag_13",
        ),
        on=["territory_id", "target_date_month"],
        how="left",
        validate="one_to_one",
    )

    valid = (
        yoy_check["__lag_1"].notna()
        & yoy_check["__lag_13"].notna()
        & yoy_check["__lag_13"].ne(0)
    )

    expected_yoy = (
        (
            yoy_check["__lag_1"]
            - yoy_check["__lag_13"]
        )
        / yoy_check["__lag_13"]
        * 100
    ).where(valid)

    differences = count_differences(
        yoy_check["yoy_change_overnight_stays"],
        expected_yoy,
        tolerance,
    )
    add_result(
        results,
        "historical_yoy_change",
        differences == 0,
        f"Diferencias respecto a t-1 frente a t-13: {differences}.",
    )


def validate_traceability(
    modeling: pd.DataFrame,
    rules: dict[str, Any],
    results: list[dict[str, str]],
) -> None:
    required = rules["validation"]["traceability"][
        "required_columns"
    ]

    for column in required:
        null_count = int(modeling[column].isna().sum())
        add_result(
            results,
            f"traceability_not_null::{column}",
            null_count == 0,
            f"Valores nulos: {null_count}.",
        )

    for column in [
        "source_snapshot_id",
        "pipeline_run_id",
        "data_version",
        "created_at",
    ]:
        unique_count = int(
            modeling[column].dropna().nunique()
        )
        add_result(
            results,
            f"single_value::{column}",
            unique_count == 1,
            f"Valores únicos observados: {unique_count}.",
        )


def markdown_table(
    headers: list[str],
    rows: list[list[object]],
) -> str:
    header = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join(
        ["---"] * len(headers)
    ) + " |"
    body = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def write_report(
    modeling: pd.DataFrame,
    results: list[dict[str, str]],
    dataset_path: Path,
) -> None:
    pass_count = sum(
        item["status"] == "PASS"
        for item in results
    )
    warn_count = sum(
        item["status"] == "WARN"
        for item in results
    )
    fail_count = sum(
        item["status"] == "FAIL"
        for item in results
    )

    split_rows = [
        [name, int(count)]
        for name, count in (
            modeling["evaluation_split"]
            .value_counts(dropna=False)
            .sort_index()
            .items()
        )
    ]

    quality_rows = [
        [name, int(count)]
        for name, count in (
            modeling["data_quality_flag"]
            .value_counts(dropna=False)
            .sort_index()
            .items()
        )
    ]

    result_rows = [
        [
            item["check"],
            item["status"],
            item["severity"],
            item["details"],
        ]
        for item in results
    ]

    report = f"""# Informe de calidad del dataset de modelado

- **Dataset:** `{dataset_path.relative_to(PROJECT_ROOT)}`
- **Generado en UTC:** `{datetime.now(timezone.utc).isoformat()}`
- **Filas:** {len(modeling):,}
- **Columnas:** {len(modeling.columns)}
- **Territorios:** {modeling["territory_id"].nunique()}
- **Periodo:** {modeling["target_month_id"].min()} → {modeling["target_month_id"].max()}
- **Resultado:** {pass_count} PASS / {warn_count} WARN / {fail_count} FAIL

## Distribución temporal

{markdown_table(["Split", "Filas"], split_rows)}

## Indicadores de calidad

{markdown_table(["Indicador", "Filas"], quality_rows)}

## Validaciones

{markdown_table(
    ["Control", "Estado", "Severidad", "Detalle"],
    result_rows,
)}
"""

    REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> int:
    rules = load_yaml(RULES_PATH)
    dataset_path = resolve_project_path(
        rules["dataset"]["path"]
    )

    if not dataset_path.exists():
        raise FileNotFoundError(
            "No se encontró el dataset de modelado: "
            f"{dataset_path.relative_to(PROJECT_ROOT)}"
        )

    if not SOURCE_PATH.exists():
        raise FileNotFoundError(
            "No se encontró la gold descriptiva: "
            f"{SOURCE_PATH.relative_to(PROJECT_ROOT)}"
        )

    modeling = pd.read_parquet(dataset_path)
    source = pd.read_parquet(SOURCE_PATH)

    if modeling.empty or source.empty:
        raise ValueError(
            "El dataset de modelado o la gold descriptiva están vacíos."
        )

    modeling = modeling.copy()
    source = source.copy()

    modeling["target_date_month"] = pd.to_datetime(
        modeling["target_date_month"]
    )
    source["date_month"] = pd.to_datetime(
        source["date_month"]
    )

    results: list[dict[str, str]] = []

    complete_structure = validate_structure(
        modeling,
        rules,
        results,
    )

    if complete_structure:
        validate_ranges(
            modeling,
            rules,
            results,
        )
        validate_dates_and_splits(
            modeling,
            rules,
            results,
        )
        validate_temporal_features(
            modeling,
            source,
            rules,
            results,
        )
        validate_traceability(
            modeling,
            rules,
            results,
        )

    write_report(
        modeling,
        results,
        dataset_path,
    )

    pass_count = sum(
        item["status"] == "PASS"
        for item in results
    )
    warn_count = sum(
        item["status"] == "WARN"
        for item in results
    )
    fail_count = sum(
        item["status"] == "FAIL"
        for item in results
    )

    print("Validación del dataset de modelado completada.")
    print(
        f"Resultado: {pass_count} PASS / "
        f"{warn_count} WARN / {fail_count} FAIL"
    )
    print(
        "Informe: "
        f"{REPORT_PATH.relative_to(PROJECT_ROOT)}"
    )

    return 1 if fail_count else 0


if __name__ == "__main__":
    sys.exit(main())