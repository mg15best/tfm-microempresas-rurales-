"""
Validación reproducible de la capa gold.

Entradas:
- data/gold/gold_tourism_demand_monthly.parquet
- data/metadata/validation_rules.yml
- dim_territory.parquet
- dim_calendar_month.parquet
- data/metadata/schema_gold.yml: contrato formal de columnas, tipos, nulabilidad, valores permitidos, rangos, orden y clave primaria.

Salida:
- data/metadata/data_quality_report.md
- data/metadata/missing_territory_months.csv

El script devuelve:
- código 0 cuando no existen errores;
- código 1 cuando alguna validación crítica falla.
"""

from __future__ import annotations

import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


# ---------------------------------------------------------------------
# Rutas generales
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RULES_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "validation_rules.yml"
)

SCHEMA_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "schema_gold.yml"
)

DOWNLOAD_LOG_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "download_log.csv"
)

MISSING_TERRITORY_MONTHS_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "missing_territory_months.csv"
)

# ---------------------------------------------------------------------
# Carga de configuración
# ---------------------------------------------------------------------

def load_rules() -> dict[str, Any]:
    """Carga las reglas YAML."""

    if not RULES_PATH.exists():
        raise FileNotFoundError(
            "No se encontró el archivo de reglas: "
            f"{RULES_PATH.relative_to(PROJECT_ROOT)}"
        )

    with RULES_PATH.open(
        mode="r",
        encoding="utf-8",
    ) as file:
        rules = yaml.safe_load(file)

    if not isinstance(rules, dict):
        raise ValueError(
            "validation_rules.yml no contiene "
            "una estructura YAML válida."
        )

    return rules

def load_schema() -> dict[str, Any]:
    """Carga el contrato formal de la tabla gold."""

    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(
            "No se encontró el contrato gold: "
            f"{SCHEMA_PATH.relative_to(PROJECT_ROOT)}"
        )

    with SCHEMA_PATH.open(
        mode="r",
        encoding="utf-8",
    ) as file:
        schema = yaml.safe_load(file)

    if not isinstance(schema, dict):
        raise ValueError(
            "schema_gold.yml no contiene una estructura YAML válida."
        )

    if "dataset" not in schema or "columns" not in schema:
        raise ValueError(
            "schema_gold.yml debe contener los bloques "
            "'dataset' y 'columns'."
        )

    return schema

def normalize_dtype_name(dtype: object) -> str:
    """
    Normaliza la representación de tipos de pandas.

    Pandas y PyArrow pueden representar el mismo tipo lógico con
    nombres diferentes según la versión instalada. Por ejemplo:

    - string[python], string[pyarrow] o str -> string
    - datetime64[us] o datetime64[ms] -> datetime64[ns]
    - datetime64[us, UTC] -> datetime64[ns, UTC]
    """

    dtype_name = str(dtype)

    if (
        dtype_name == "str"
        or dtype_name == "object"
        or dtype_name.startswith("string")
    ):
        return "string"

    if dtype_name.startswith("datetime64["):
        if "," in dtype_name:
            timezone = (
                dtype_name
                .split(",", maxsplit=1)[1]
                .rstrip("]")
                .strip()
            )
            return f"datetime64[ns, {timezone}]"

        return "datetime64[ns]"

    return dtype_name

def resolve_project_path(relative_path: str) -> Path:
    """Convierte una ruta del YAML en ruta absoluta."""

    return PROJECT_ROOT / Path(relative_path)


# ---------------------------------------------------------------------
# Registro de resultados
# ---------------------------------------------------------------------

def add_result(
    results: list[dict[str, str]],
    *,
    check_name: str,
    passed: bool,
    details: str,
    severity: str = "error",
) -> None:
    """Añade el resultado de una validación."""

    if passed:
        status = "PASS"
    elif severity == "warning":
        status = "WARN"
    else:
        status = "FAIL"

    results.append(
        {
            "check": check_name,
            "status": status,
            "severity": severity,
            "details": details,
        }
    )


# ---------------------------------------------------------------------
# Utilidades numéricas
# ---------------------------------------------------------------------

def count_values_outside_range(
    series: pd.Series,
    minimum: float,
    maximum: float,
) -> int:
    """Cuenta valores no nulos fuera de un rango."""

    values = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    return int(
        (~values.between(minimum, maximum)).sum()
    )


def count_negative_values(
    series: pd.Series,
) -> int:
    """Cuenta valores negativos no nulos."""

    values = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    return int(values.lt(0).sum())


# ---------------------------------------------------------------------
# Generación de tablas Markdown
# ---------------------------------------------------------------------

def markdown_table(
    headers: list[str],
    rows: list[list[object]],
) -> str:
    """Convierte una lista de filas en tabla Markdown."""

    header_line = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join(
        ["---"] * len(headers)
    ) + " |"

    body_lines = []

    for row in rows:
        clean_values = [
            str(value).replace("|", "\\|")
            for value in row
        ]

        body_lines.append(
            "| " + " | ".join(clean_values) + " |"
        )

    return "\n".join(
        [
            header_line,
            separator,
            *body_lines,
        ]
    )

def find_missing_territory_months(
    dataframe: pd.DataFrame,
    allowed_missing_global_months: list[str],
) -> pd.DataFrame:
    """
    Identifica las combinaciones provincia-mes ausentes.

    Los meses globalmente ausentes y documentados en
    validation_rules.yml se excluyen de la cuadrícula esperada.
    """

    output_columns = [
        "territory_id",
        "territory_name",
        "month_id",
        "missing_reason",
    ]

    if dataframe.empty:
        return pd.DataFrame(columns=output_columns)

    required_columns = {
        "territory_id",
        "territory_name",
        "month_id",
        "date_month",
    }

    missing_columns = sorted(
        required_columns - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            "No se puede comprobar la cobertura provincia-mes. "
            "Faltan las columnas: "
            + ", ".join(missing_columns)
        )

    dates = pd.to_datetime(
        dataframe["date_month"],
        errors="coerce",
    )

    if dates.isna().any():
        raise ValueError(
            "No se puede comprobar la cobertura provincia-mes "
            "porque existen fechas no válidas."
        )

    expected_months = (
        pd.date_range(
            start=dates.min(),
            end=dates.max(),
            freq="MS",
        )
        .strftime("%Y-%m")
        .tolist()
    )

    allowed_missing = {
        str(month)
        for month in allowed_missing_global_months
    }

    expected_months = [
        month
        for month in expected_months
        if month not in allowed_missing
    ]

    territory_ids = sorted(
        dataframe["territory_id"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    expected_grid = (
        pd.MultiIndex.from_product(
            [
                territory_ids,
                expected_months,
            ],
            names=[
                "territory_id",
                "month_id",
            ],
        )
        .to_frame(index=False)
        .astype("string")
    )

    actual_keys = (
        dataframe[
            [
                "territory_id",
                "month_id",
            ]
        ]
        .astype("string")
        .drop_duplicates()
    )

    missing_combinations = (
        expected_grid.merge(
            actual_keys,
            on=[
                "territory_id",
                "month_id",
            ],
            how="left",
            indicator=True,
        )
        .loc[
            lambda frame: frame["_merge"].eq("left_only"),
            [
                "territory_id",
                "month_id",
            ],
        ]
    )

    territory_names = (
        dataframe[
            [
                "territory_id",
                "territory_name",
            ]
        ]
        .dropna(subset=["territory_id"])
        .drop_duplicates(subset=["territory_id"])
    )

    missing_combinations = missing_combinations.merge(
        territory_names,
        on="territory_id",
        how="left",
        validate="many_to_one",
    )

    missing_combinations["missing_reason"] = (
        "no_primary_demand_metric_available"
    )

    missing_combinations = (
        missing_combinations[
            [
                "territory_id",
                "territory_name",
                "month_id",
                "missing_reason",
            ]
        ]
        .sort_values(
            [
                "territory_id",
                "month_id",
            ]
        )
        .reset_index(drop=True)
    )

    return missing_combinations

def calculate_sha256(file_path: Path) -> str:
    """Calcula el hash SHA-256 de un fichero."""

    sha256 = hashlib.sha256()

    with file_path.open("rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            sha256.update(block)

    return sha256.hexdigest()

def validate_snapshot_traceability(
    dataframe: pd.DataFrame,
    results: list[dict[str, str]],
) -> None:
    """
    Comprueba que los snapshots raw utilizados por la capa gold
    están registrados en download_log.csv y que los ficheros raw
    conservan el SHA-256 esperado.
    """

    if not DOWNLOAD_LOG_PATH.exists():
        add_result(
            results,
            check_name="download_log_exists",
            passed=False,
            details=(
                "No se encontró "
                "data/metadata/download_log.csv."
            ),
        )
        return

    download_log = pd.read_csv(
        DOWNLOAD_LOG_PATH,
        dtype="string",
    )

    required_columns = {
        "source_id",
        "raw_file_path",
        "file_hash",
    }

    missing_columns = sorted(
        required_columns - set(download_log.columns)
    )

    if missing_columns:
        add_result(
            results,
            check_name="download_log_schema",
            passed=False,
            details=(
                "Faltan columnas en download_log.csv: "
                + ", ".join(missing_columns)
            ),
        )
        return

    add_result(
        results,
        check_name="download_log_schema",
        passed=True,
        details=(
            "download_log.csv contiene las columnas "
            "necesarias para validar la trazabilidad."
        ),
    )

    snapshot_checks = [
        (
            "demand_snapshot_id",
            "ine_eotr_demand_province",
        ),
        (
            "supply_snapshot_id",
            "ine_eotr_supply_province",
        ),
    ]

    for snapshot_column, source_id in snapshot_checks:
        gold_hashes = set(
            dataframe[snapshot_column]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        registered_hashes = set(
            download_log.loc[
                download_log["source_id"].eq(source_id),
                "file_hash",
            ]
            .dropna()
            .astype(str)
            .tolist()
        )

        missing_hashes = sorted(
            gold_hashes - registered_hashes
        )

        add_result(
            results,
            check_name=(
                f"snapshot_registered::{snapshot_column}"
            ),
            passed=(
                bool(gold_hashes)
                and not missing_hashes
            ),
            details=(
                "Todos los hashes de la capa gold están "
                "registrados en download_log.csv."
                if gold_hashes and not missing_hashes
                else (
                    "Hashes no registrados: "
                    + (
                        ", ".join(missing_hashes)
                        if missing_hashes
                        else "ningún hash presente en la gold"
                    )
                )
            ),
        )

        matching_rows = (
            download_log.loc[
                download_log["source_id"].eq(source_id)
                & download_log["file_hash"].isin(gold_hashes),
                [
                    "raw_file_path",
                    "file_hash",
                ],
            ]
            .drop_duplicates()
        )

        integrity_errors: list[str] = []

        for _, log_row in matching_rows.iterrows():
            raw_file_path = (
                PROJECT_ROOT
                / Path(str(log_row["raw_file_path"]))
            )

            expected_hash = str(
                log_row["file_hash"]
            )

            if not raw_file_path.exists():
                integrity_errors.append(
                    "Fichero inexistente: "
                    + str(log_row["raw_file_path"])
                )
                continue

            actual_hash = calculate_sha256(
                raw_file_path
            )

            if actual_hash != expected_hash:
                integrity_errors.append(
                    "Hash distinto: "
                    + str(log_row["raw_file_path"])
                )

        complete_traceability = (
            bool(gold_hashes)
            and len(matching_rows) == len(gold_hashes)
            and not integrity_errors
        )

        add_result(
            results,
            check_name=(
                f"raw_file_integrity::{snapshot_column}"
            ),
            passed=complete_traceability,
            details=(
                "Los ficheros raw existen y su SHA-256 "
                "coincide con download_log.csv."
                if complete_traceability
                else (
                    "; ".join(integrity_errors)
                    if integrity_errors
                    else (
                        "No se encontró un fichero raw "
                        "registrado para todos los snapshots."
                    )
                )
            ),
        )

def validate_schema_contract(
    dataframe: pd.DataFrame,
    schema: dict[str, Any],
    results: list[dict[str, str]],
) -> None:
    """Valida columnas, tipos y restricciones del contrato gold."""

    schema_columns = schema["columns"]
    documented_columns = list(schema_columns.keys())
    actual_columns = dataframe.columns.tolist()

    # Número de columnas esperado.
    expected_column_count = int(
        schema["dataset"]["column_count"]
    )

    add_result(
        results,
        check_name="schema_column_count",
        passed=len(actual_columns) == expected_column_count,
        details=(
            f"Esperadas: {expected_column_count}; "
            f"encontradas: {len(actual_columns)}"
        ),
    )

    # Columnas ausentes y columnas no documentadas.
    missing_columns = sorted(
        set(documented_columns) - set(actual_columns)
    )

    unexpected_columns = sorted(
        set(actual_columns) - set(documented_columns)
    )

    add_result(
        results,
        check_name="schema_columns_match",
        passed=not missing_columns and not unexpected_columns,
        details=(
            "Las columnas coinciden con el contrato."
            if not missing_columns and not unexpected_columns
            else (
                f"Faltan: {missing_columns}; "
                f"sobran: {unexpected_columns}"
            )
        ),
    )

    # Si no coinciden las columnas, no es seguro continuar.
    if missing_columns or unexpected_columns:
        return

    # Orden de columnas.
    order_matches = actual_columns == documented_columns

    add_result(
        results,
        check_name="schema_column_order",
        passed=order_matches,
        details=(
            "El orden de las columnas coincide con el contrato."
            if order_matches
            else "El orden de las columnas no coincide con el contrato."
        ),
    )

    # Validaciones individuales por columna.
    for column_name, column_rules in schema_columns.items():
        expected_dtype = str(column_rules["dtype"])
        actual_dtype = normalize_dtype_name(
            dataframe[column_name].dtype
        )

        dtype_matches = actual_dtype == expected_dtype

        add_result(
            results,
            check_name=f"schema_dtype::{column_name}",
            passed=dtype_matches,
            details=(
                f"Esperado: {expected_dtype}; "
                f"encontrado: {actual_dtype}"
            ),
        )

        nullable = bool(
            column_rules.get("nullable", True)
        )

        if not nullable:
            null_count = int(
                dataframe[column_name].isna().sum()
            )

            add_result(
                results,
                check_name=f"schema_not_null::{column_name}",
                passed=null_count == 0,
                details=f"Valores nulos: {null_count:,}",
            )

        allowed_values = column_rules.get(
            "allowed_values"
        )

        if allowed_values is not None:
            observed_values = set(
                dataframe[column_name]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            invalid_values = sorted(
                observed_values - set(
                    str(value)
                    for value in allowed_values
                )
            )

            add_result(
                results,
                check_name=f"schema_allowed_values::{column_name}",
                passed=not invalid_values,
                details=(
                    "Todos los valores están permitidos."
                    if not invalid_values
                    else (
                        "Valores no permitidos: "
                        + ", ".join(invalid_values)
                    )
                ),
            )

        minimum = column_rules.get("minimum")
        maximum = column_rules.get("maximum")

        if minimum is not None or maximum is not None:
            numeric_values = pd.to_numeric(
                dataframe[column_name],
                errors="coerce",
            ).dropna()

            below_minimum = 0
            above_maximum = 0

            if minimum is not None:
                below_minimum = int(
                    numeric_values.lt(minimum).sum()
                )

            if maximum is not None:
                above_maximum = int(
                    numeric_values.gt(maximum).sum()
                )

            add_result(
                results,
                check_name=f"schema_range::{column_name}",
                passed=(
                    below_minimum == 0
                    and above_maximum == 0
                ),
                details=(
                    f"Por debajo del mínimo: {below_minimum:,}; "
                    f"por encima del máximo: {above_maximum:,}"
                ),
            )

# ---------------------------------------------------------------------
# Validaciones principales
# ---------------------------------------------------------------------

def validate_dataset(
    dataframe: pd.DataFrame,
    rules: dict[str, Any],
    schema: dict[str, Any],
) -> tuple[
    list[dict[str, str]],
    list[dict[str, object]],
]:
    """
    Ejecuta todas las reglas.

    Devuelve:
    - resultados de controles;
    - resumen de valores nulos.
    """

    results: list[dict[str, str]] = []
    missingness: list[dict[str, object]] = []

    validate_schema_contract(
        dataframe=dataframe,
        schema=schema,
        results=results,
    )

    dataset_rules = rules["dataset"]
    validation_rules = rules["validation"]

    # -------------------------------------------------------------
    # Dataset no vacío
    # -------------------------------------------------------------

    add_result(
        results,
        check_name="dataset_not_empty",
        passed=not dataframe.empty,
        details=f"Filas encontradas: {len(dataframe):,}",
    )

    # -------------------------------------------------------------
    # Columnas obligatorias
    # -------------------------------------------------------------

    required_columns = set(
        dataset_rules["required_columns"]
    )

    missing_columns = sorted(
        required_columns.difference(
            dataframe.columns
        )
    )

    add_result(
        results,
        check_name="required_columns",
        passed=not missing_columns,
        details=(
            "Todas las columnas obligatorias están presentes."
            if not missing_columns
            else "Faltan: " + ", ".join(missing_columns)
        ),
    )

    # Si faltan columnas estructurales, no se pueden ejecutar
    # con seguridad el resto de reglas.
    if missing_columns:
        return results, missingness

    # -------------------------------------------------------------
    # Clave primaria
    # -------------------------------------------------------------

    key_columns = dataset_rules["key_columns"]

    null_key_rows = int(
        dataframe[key_columns]
        .isna()
        .any(axis=1)
        .sum()
    )

    add_result(
        results,
        check_name="key_not_null",
        passed=null_key_rows == 0,
        details=(
            f"Filas con claves nulas: {null_key_rows:,}"
        ),
    )

    duplicate_rows = int(
        dataframe.duplicated(
            subset=key_columns
        ).sum()
    )

    add_result(
        results,
        check_name="key_uniqueness",
        passed=duplicate_rows == 0,
        details=(
            f"Claves duplicadas: {duplicate_rows:,}"
        ),
    )

    # -------------------------------------------------------------
    # Cobertura territorial
    # -------------------------------------------------------------

    expected_territory_count = int(
        dataset_rules["expected_territory_count"]
    )

    actual_territory_count = int(
        dataframe["territory_id"].nunique()
    )

    add_result(
        results,
        check_name="territory_count",
        passed=(
            actual_territory_count
            == expected_territory_count
        ),
        details=(
            f"Esperados: {expected_territory_count}; "
            f"encontrados: {actual_territory_count}"
        ),
    )

    territory_levels = sorted(
        dataframe["territory_level"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    add_result(
        results,
        check_name="territory_level",
        passed=territory_levels == ["province"],
        details=(
            "Niveles encontrados: "
            + ", ".join(territory_levels)
        ),
    )

    # -------------------------------------------------------------
    # Cobertura temporal
    # -------------------------------------------------------------

    dates = pd.to_datetime(
        dataframe["date_month"],
        errors="coerce",
    )

    invalid_dates = int(dates.isna().sum())

    add_result(
        results,
        check_name="valid_month_dates",
        passed=invalid_dates == 0,
        details=(
            f"Fechas mensuales no válidas: {invalid_dates:,}"
        ),
    )

    if invalid_dates == 0:
        actual_minimum = dates.min()
        expected_minimum = pd.Timestamp(
            dataset_rules["minimum_start_month"]
        )

        add_result(
            results,
            check_name="minimum_start_month",
            passed=actual_minimum == expected_minimum,
            details=(
                f"Esperado: {expected_minimum.date()}; "
                f"encontrado: {actual_minimum.date()}"
            ),
        )

        valid_month_ids = (
            dataframe["month_id"]
            .astype("string")
            .eq(
                dates.dt.strftime("%Y-%m")
            )
        )

        invalid_month_ids = int(
            (~valid_month_ids.fillna(False)).sum()
        )

        add_result(
            results,
            check_name="month_id_consistency",
            passed=invalid_month_ids == 0,
            details=(
                "Filas donde month_id no coincide con "
                f"date_month: {invalid_month_ids:,}"
            ),
        )

        unique_dates = (
            dates.drop_duplicates()
            .sort_values()
        )

        expected_dates = pd.date_range(
            start=unique_dates.min(),
            end=unique_dates.max(),
            freq="MS",
        )

        missing_global_months = {
            date.strftime("%Y-%m")
            for date in (
                set(expected_dates)
                - set(unique_dates)
            )
        }

        allowed_missing_months = set(
            dataset_rules.get(
                "allowed_missing_global_months",
                [],
            )
        )

        unexpected_missing_months = sorted(
            missing_global_months
            - allowed_missing_months
        )

        documented_missing_months = sorted(
            missing_global_months
            & allowed_missing_months
        )

        if unexpected_missing_months:
            continuity_details = (
                "Meses ausentes no documentados: "
                + ", ".join(
                    unexpected_missing_months
                )
            )
        elif documented_missing_months:
            continuity_details = (
                "Ausencias globales documentadas "
                "en la fuente: "
                + ", ".join(
                    documented_missing_months
                )
            )
        else:
            continuity_details = (
                "No existen huecos en la serie "
                "mensual global."
            )

        add_result(
            results,
            check_name="global_month_continuity",
            passed=not unexpected_missing_months,
            details=continuity_details,
        )

    # -------------------------------------------------------------
    # Valores no negativos
    # -------------------------------------------------------------

    for column in validation_rules[
        "non_negative_columns"
    ]:
        negative_count = count_negative_values(
            dataframe[column]
        )

        add_result(
            results,
            check_name=f"non_negative::{column}",
            passed=negative_count == 0,
            details=(
                f"Valores negativos: {negative_count:,}"
            ),
        )

    # -------------------------------------------------------------
    # Porcentajes 0-100
    # -------------------------------------------------------------

    for column in validation_rules[
        "percentage_columns_0_100"
    ]:
        invalid_count = count_values_outside_range(
            dataframe[column],
            0,
            100,
        )

        add_result(
            results,
            check_name=f"range_0_100::{column}",
            passed=invalid_count == 0,
            details=(
                "Valores fuera del rango 0-100: "
                f"{invalid_count:,}"
            ),
        )

    # -------------------------------------------------------------
    # Proporciones 0-1
    # -------------------------------------------------------------

    for column in validation_rules[
        "share_columns_0_1"
    ]:
        invalid_count = count_values_outside_range(
            dataframe[column],
            0,
            1,
        )

        add_result(
            results,
            check_name=f"range_0_1::{column}",
            passed=invalid_count == 0,
            details=(
                "Valores fuera del rango 0-1: "
                f"{invalid_count:,}"
            ),
        )

    # -------------------------------------------------------------
    # Pares de proporciones que deben sumar 1
    # -------------------------------------------------------------

    tolerance = float(
        validation_rules["numeric_tolerance"]
    )

    for pair_rule in validation_rules["share_pairs"]:
        first_column = pair_rule["first"]
        second_column = pair_rule["second"]

        complete_mask = (
            dataframe[first_column].notna()
            & dataframe[second_column].notna()
        )

        share_sum = (
            dataframe.loc[
                complete_mask,
                first_column,
            ].astype(float)
            +
            dataframe.loc[
                complete_mask,
                second_column,
            ].astype(float)
        )

        invalid_count = int(
            (
                (share_sum - 1).abs()
                > tolerance
            ).sum()
        )

        add_result(
            results,
            check_name=(
                f"share_sum::{pair_rule['name']}"
            ),
            passed=invalid_count == 0,
            details=(
                "Filas completas que no suman 1: "
                f"{invalid_count:,}"
            ),
        )

    # -------------------------------------------------------------
    # Totales y componentes
    # -------------------------------------------------------------

    for total_rule in validation_rules["total_rules"]:
        total_column = total_rule["total"]
        components = total_rule["components"]

        complete_mask = (
            dataframe[components]
            .notna()
            .all(axis=1)
        )

        expected_total = (
            dataframe.loc[
                complete_mask,
                components,
            ]
            .astype(float)
            .sum(axis=1)
        )

        actual_total = (
            dataframe.loc[
                complete_mask,
                total_column,
            ]
            .astype(float)
        )

        invalid_count = int(
            (
                (actual_total - expected_total).abs()
                > tolerance
            ).sum()
        )

        add_result(
            results,
            check_name=(
                f"total_consistency::{total_rule['name']}"
            ),
            passed=invalid_count == 0,
            details=(
                "Filas completas con total incoherente: "
                f"{invalid_count:,}"
            ),
        )

    # -------------------------------------------------------------
    # Estancia media
    # -------------------------------------------------------------

    average_stay_mask = (
        dataframe["travellers_total"].notna()
        & dataframe["overnight_stays_total"].notna()
        & dataframe["average_stay"].notna()
        & dataframe["travellers_total"].gt(0)
    )

    expected_average_stay = (
        dataframe.loc[
            average_stay_mask,
            "overnight_stays_total",
        ].astype(float)
        /
        dataframe.loc[
            average_stay_mask,
            "travellers_total",
        ].astype(float)
    )

    actual_average_stay = dataframe.loc[
        average_stay_mask,
        "average_stay",
    ].astype(float)

    invalid_average_stay = int(
        (
            (
                actual_average_stay
                - expected_average_stay
            ).abs()
            > tolerance
        ).sum()
    )

    add_result(
        results,
        check_name="average_stay_consistency",
        passed=invalid_average_stay == 0,
        details=(
            "Filas con estancia media incoherente: "
            f"{invalid_average_stay:,}"
        ),
    )

    # -------------------------------------------------------------
    # Estado provisional
    # -------------------------------------------------------------

    provisional_from = pd.Timestamp(
        dataset_rules["provisional_from"]
    )

    expected_provisional = dates >= provisional_from

    actual_provisional = (
        dataframe["is_provisional"]
        .astype("boolean")
    )

    invalid_provisional = int(
        (
            actual_provisional.fillna(False)
            != expected_provisional
        ).sum()
    )

    add_result(
        results,
        check_name="provisional_period",
        passed=invalid_provisional == 0,
        details=(
            "Filas con clasificación provisional "
            f"incoherente: {invalid_provisional:,}"
        ),
    )

    expected_status = pd.Series(
        "final_or_not_marked_provisional",
        index=dataframe.index,
        dtype="string",
    )

    expected_status.loc[
        expected_provisional
    ] = "provisional"

    invalid_status = int(
        (
            dataframe["data_status"].astype("string")
            != expected_status
        ).fillna(True).sum()
    )

    add_result(
        results,
        check_name="data_status_consistency",
        passed=invalid_status == 0,
        details=(
            "Filas con data_status incoherente: "
            f"{invalid_status:,}"
        ),
    )

    # -------------------------------------------------------------
    # Trazabilidad
    # -------------------------------------------------------------

    validate_snapshot_traceability(
        dataframe=dataframe,
        results=results,
    )

    for column in validation_rules[
        "unique_per_pipeline_columns"
    ]:
        unique_count = int(
            dataframe[column]
            .dropna()
            .nunique()
        )

        add_result(
            results,
            check_name=f"single_value::{column}",
            passed=unique_count == 1,
            details=(
                f"Valores distintos encontrados: {unique_count}"
            ),
        )

    # -------------------------------------------------------------
    # Integridad con dimensiones
    # -------------------------------------------------------------

    territory_path = resolve_project_path(
        dataset_rules["territory_dimension_path"]
    )

    if territory_path.exists():
        territory_dimension = pd.read_parquet(
            territory_path
        )

        missing_territories = sorted(
            set(dataframe["territory_id"])
            - set(territory_dimension["territory_id"])
        )

        add_result(
            results,
            check_name="territory_referential_integrity",
            passed=not missing_territories,
            details=(
                "Todos los territorios existen en dim_territory."
                if not missing_territories
                else (
                    "Territorios sin dimensión: "
                    + ", ".join(missing_territories)
                )
            ),
        )

    else:
        add_result(
            results,
            check_name="territory_dimension_available",
            passed=False,
            details="No se encontró dim_territory.parquet.",
        )

    calendar_path = resolve_project_path(
        dataset_rules["calendar_dimension_path"]
    )

    if calendar_path.exists():
        calendar_dimension = pd.read_parquet(
            calendar_path
        )

        missing_months = sorted(
            set(dataframe["month_id"])
            - set(calendar_dimension["month_id"])
        )

        add_result(
            results,
            check_name="calendar_referential_integrity",
            passed=not missing_months,
            details=(
                "Todos los meses existen "
                "en dim_calendar_month."
                if not missing_months
                else (
                    "Meses sin dimensión: "
                    + ", ".join(missing_months)
                )
            ),
        )

    else:
        add_result(
            results,
            check_name="calendar_dimension_available",
            passed=False,
            details=(
                "No se encontró dim_calendar_month.parquet."
            ),
        )

    # -------------------------------------------------------------
    # Columnas contextuales pendientes
    # -------------------------------------------------------------

    for column in validation_rules[
        "context_columns_expected_null"
    ]:
        non_null_count = int(
            dataframe[column].notna().sum()
        )

        add_result(
            results,
            check_name=f"context_pending::{column}",
            passed=non_null_count == 0,
            severity="warning",
            details=(
                "Valores no nulos encontrados: "
                f"{non_null_count:,}. "
                "La fuente contextual todavía no está integrada."
            ),
        )

    # -------------------------------------------------------------
    # Informe de valores nulos
    # -------------------------------------------------------------

    total_rows = len(dataframe)

    for column in validation_rules[
        "missingness_report_columns"
    ]:
        null_count = int(
            dataframe[column].isna().sum()
        )

        null_percentage = (
            null_count / total_rows * 100
            if total_rows
            else 0.0
        )

        missingness.append(
            {
                "column": column,
                "null_count": null_count,
                "null_percentage": round(
                    null_percentage,
                    2,
                ),
            }
        )

    return results, missingness


# ---------------------------------------------------------------------
# Informe Markdown
# ---------------------------------------------------------------------

def build_report(
    dataframe: pd.DataFrame,
    results: list[dict[str, str]],
    missingness: list[dict[str, object]],
    dataset_path: Path,
) -> str:
    """Construye el contenido del informe de calidad."""

    generated_at = datetime.now(
        timezone.utc
    ).isoformat()

    pass_count = sum(
        result["status"] == "PASS"
        for result in results
    )

    warning_count = sum(
        result["status"] == "WARN"
        for result in results
    )

    fail_count = sum(
        result["status"] == "FAIL"
        for result in results
    )

    overall_status = (
        "FAIL"
        if fail_count
        else "PASS"
    )

    result_rows = [
        [
            result["check"],
            result["status"],
            result["details"],
        ]
        for result in results
    ]

    missingness_rows = [
        [
            item["column"],
            f"{item['null_count']:,}",
            f"{item['null_percentage']:.2f} %",
        ]
        for item in missingness
    ]

    period_start = (
        pd.to_datetime(
            dataframe["date_month"]
        ).min().strftime("%Y-%m")
        if not dataframe.empty
        else "N/A"
    )

    period_end = (
        pd.to_datetime(
            dataframe["date_month"]
        ).max().strftime("%Y-%m")
        if not dataframe.empty
        else "N/A"
    )

    report_parts = [
        "# Informe de calidad de datos",
        "",
        "## 1. Resumen de ejecución",
        "",
        f"- **Dataset:** `{dataset_path.relative_to(PROJECT_ROOT).as_posix()}`",
        f"- **Fecha de validación UTC:** `{generated_at}`",
        f"- **Estado general:** **{overall_status}**",
        f"- **Filas:** `{len(dataframe):,}`",
        f"- **Columnas:** `{len(dataframe.columns)}`",
        f"- **Territorios:** `{dataframe['territory_id'].nunique()}`",
        f"- **Periodo:** `{period_start}` a `{period_end}`",
        f"- **Controles superados:** `{pass_count}`",
        f"- **Advertencias:** `{warning_count}`",
        f"- **Controles fallidos:** `{fail_count}`",
        "",
        "## 2. Resultado de los controles",
        "",
        markdown_table(
            ["Control", "Estado", "Detalle"],
            result_rows,
        ),
        "",
        "## 3. Valores nulos en variables principales",
        "",
        markdown_table(
            [
                "Variable",
                "Nulos",
                "Porcentaje",
            ],
            missingness_rows,
        ),
        "",
        "## 4. Interpretación",
        "",
    ]

    if fail_count:
        report_parts.extend(
            [
                "La validación ha detectado errores críticos. "
                "La tabla gold no debe utilizarse para modelado "
                "ni publicación hasta corregir los controles "
                "marcados como `FAIL`.",
                "",
            ]
        )
    else:
        report_parts.extend(
            [
                "La tabla gold cumple las reglas estructurales, "
                "territoriales, temporales, numéricas y de "
                "trazabilidad definidas para esta versión.",
                "",
                "Las ausencias esperadas de las variables contextuales se "
                "registran como controles superados porque las fuentes de precios, "
                "gasto y contexto empresarial todavía no se han integrado. Sus valores nulos "
                "no representan un error del pipeline actual.",
                "",
            ]
        )

    report_parts.extend(
        [
            "## 5. Alcance del control",
            "",
            "Este informe valida la consistencia técnica del "
            "dataset, pero no sustituye la interpretación "
            "estadística ni la revisión metodológica de las "
            "fuentes originales.",
            "",
        ]
    )

    return "\n".join(report_parts)


# ---------------------------------------------------------------------
# Ejecución
# ---------------------------------------------------------------------

def main() -> int:
    """Ejecuta la validación y genera el informe."""

    print("=" * 70)
    print("VALIDACIÓN DE CALIDAD DE LA CAPA GOLD")
    print("=" * 70)
    print()

    try:
        rules = load_rules()
        schema = load_schema()

        dataset_path = resolve_project_path(
            rules["dataset"]["path"]
        )

        report_path = resolve_project_path(
            rules["dataset"]["report_path"]
        )

        if not dataset_path.exists():
            raise FileNotFoundError(
                "No se encontró el dataset gold: "
                f"{dataset_path.relative_to(PROJECT_ROOT)}"
            )

        dataframe = pd.read_parquet(
            dataset_path
        )

        results, missingness = validate_dataset(
            dataframe,
            rules,
            schema,
        )

        missing_territory_months = (
            find_missing_territory_months(
                dataframe=dataframe,
                allowed_missing_global_months=(
                    rules["dataset"].get(
                        "allowed_missing_global_months",
                        [],
                    )
                ),
            )
        )

        MISSING_TERRITORY_MONTHS_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        missing_territory_months.to_csv(
            MISSING_TERRITORY_MONTHS_PATH,
            index=False,
            encoding="utf-8",
        )

        missing_territory_month_count = len(
            missing_territory_months
        )

        add_result(
            results,
            check_name="territory_month_coverage",
            passed=missing_territory_month_count == 0,
            severity="warning",
            details=(
                "No existen combinaciones provincia-mes ausentes."
                if missing_territory_month_count == 0
                else (
                    "Combinaciones provincia-mes ausentes: "
                    f"{missing_territory_month_count:,}. "
                    "Detalle disponible en "
                    "`data/metadata/"
                    "missing_territory_months.csv`."
                )
            ),
        )

        report_content = build_report(
            dataframe=dataframe,
            results=results,
            missingness=missingness,
            dataset_path=dataset_path,
        )

        report_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        report_path.write_text(
            report_content,
            encoding="utf-8",
        )

        pass_count = sum(
            result["status"] == "PASS"
            for result in results
        )

        warning_count = sum(
            result["status"] == "WARN"
            for result in results
        )

        fail_count = sum(
            result["status"] == "FAIL"
            for result in results
        )

        print(f"[OK] Controles superados: {pass_count}")
        print(f"[WARN] Advertencias: {warning_count}")
        print(f"[ERROR] Controles fallidos: {fail_count}")
        print(
            "[OK] Informe generado: "
            f"{report_path.relative_to(PROJECT_ROOT).as_posix()}"
        )

        if fail_count:
            print()
            print(
                "[ERROR] La validación ha detectado "
                "errores críticos."
            )
            return 1

        print()
        print(
            "[OK] La capa gold supera "
            "las validaciones críticas."
        )

        return 0

    except Exception as error:
        print(
            f"[ERROR] {error}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())