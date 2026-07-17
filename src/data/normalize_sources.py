"""
Normaliza las fuentes provinciales de turismo rural del INE.

Entradas raw:
- Tabla 2073: viajeros y pernoctaciones por provincias.
- Tabla 2070: establecimientos, plazas, ocupación y personal.

Salidas processed:
- processed_ocupacion_rural_demand_province_monthly.parquet
- processed_ocupacion_rural_supply_province_monthly.parquet
- processed_ocupacion_rural_monthly.parquet
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------
# Rutas del proyecto
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ine_ocupacion_rural"
)

PROCESSED_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "processed"
)


# ---------------------------------------------------------------------
# Estado provisional indicado por el INE
# ---------------------------------------------------------------------

# En la revisión de las tablas realizada el 14 de julio de 2026,
# el INE señala como provisionales los datos desde junio de 2025.
PROVISIONAL_FROM = pd.Timestamp("2025-06-01")


# ---------------------------------------------------------------------
# Ficheros de salida
# ---------------------------------------------------------------------

DEMAND_OUTPUT = (
    PROCESSED_DIRECTORY
    / "processed_ocupacion_rural_demand_province_monthly.parquet"
)

SUPPLY_OUTPUT = (
    PROCESSED_DIRECTORY
    / "processed_ocupacion_rural_supply_province_monthly.parquet"
)

COMBINED_OUTPUT = (
    PROCESSED_DIRECTORY
    / "processed_ocupacion_rural_monthly.parquet"
)


# ---------------------------------------------------------------------
# Nombres originales de columnas
# ---------------------------------------------------------------------

DEMAND_COLUMNS = {
    "Provincias con mayor número de pernoctaciones": (
        "source_territory"
    ),
    "Viajeros y pernoctaciones": "source_metric",
    "Residencia": "source_residence",
    "Periodo": "source_period",
    "Total": "source_value_raw",
}

SUPPLY_COLUMNS = {
    "Provincias con mayor número de pernoctaciones": (
        "source_territory"
    ),
    "Establecimientos y personal empleado (plazas)": (
        "source_metric"
    ),
    "Periodo": "source_period",
    "Total": "source_value_raw",
}


# ---------------------------------------------------------------------
# Traducción de categorías a nombres técnicos
# ---------------------------------------------------------------------

DEMAND_METRIC_MAP = {
    "Viajero": "travellers",
    "Pernoctaciones": "overnight_stays",
}

RESIDENCE_MAP = {
    "Residentes en España": "residents_in_spain",
    "Residentes en el Extranjero": "residents_abroad",
}

SUPPLY_METRIC_MAP = {
    "Número de establecimientos abiertos estimados": (
        "estimated_open_establishments"
    ),
    "Número de plazas estimadas": (
        "estimated_bed_places"
    ),
    "Grado de ocupación por plazas": (
        "occupancy_rate_by_bed_places"
    ),
    "Grado de ocupación por plazas en fin de semana": (
        "weekend_occupancy_rate_by_bed_places"
    ),
    "Grado de ocupación por habitaciones": (
        "occupancy_rate_by_rooms"
    ),
    "Personal empleado": "employed_personnel",
}

SUPPLY_UNIT_MAP = {
    "estimated_open_establishments": "establishments",
    "estimated_bed_places": "bed_places",
    "occupancy_rate_by_bed_places": "percent",
    "weekend_occupancy_rate_by_bed_places": "percent",
    "occupancy_rate_by_rooms": "percent",
    "employed_personnel": "persons",
}


# ---------------------------------------------------------------------
# Funciones generales
# ---------------------------------------------------------------------

def calculate_sha256(file_path: Path) -> str:
    """
    Calcula la huella SHA-256 de un fichero.

    Permite identificar exactamente qué snapshot raw se utilizó.
    """

    sha256 = hashlib.sha256()

    with file_path.open("rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            sha256.update(block)

    return sha256.hexdigest()


def find_latest_raw(pattern: str) -> Path:
    """
    Encuentra el snapshot raw más reciente según el timestamp
    UTC incluido al principio del nombre del fichero.

    No se utiliza la fecha de modificación del sistema de archivos,
    porque puede cambiar al copiar o clonar el repositorio.
    """

    candidates = sorted(
        RAW_DIRECTORY.glob(pattern),
        key=lambda path: path.name,
        reverse=True,
    )

    if not candidates:
        raise FileNotFoundError(
            "No se encontró ningún fichero raw "
            f"con el patrón: {pattern}"
        )

    return candidates[0]


def read_raw_csv(file_path: Path) -> pd.DataFrame:
    """
    Lee un CSV del INE sin convertir todavía sus valores.

    keep_default_na=False permite conservar literalmente
    símbolos como '.' y '..' para tratarlos de forma explícita.
    """

    return pd.read_csv(
        file_path,
        sep=";",
        encoding="utf-8-sig",
        dtype="string",
        keep_default_na=False,
    )


def require_columns(
    dataframe: pd.DataFrame,
    expected_columns: set[str],
) -> None:
    """
    Comprueba que la fuente mantiene las columnas esperadas.

    Si el INE cambia el esquema, el proceso se detiene en vez de
    generar silenciosamente un resultado incorrecto.
    """

    missing_columns = expected_columns.difference(
        dataframe.columns
    )

    if missing_columns:
        raise ValueError(
            "Faltan columnas esperadas en el CSV: "
            + ", ".join(sorted(missing_columns))
        )


def parse_spanish_number(
    series: pd.Series,
) -> pd.Series:
    """
    Convierte números publicados con formato español.

    Ejemplos:
    - '9.220'  -> 9220
    - '45,73'  -> 45.73
    - '.'      -> nulo
    - '..'     -> nulo
    - vacío    -> nulo
    """

    cleaned = series.astype("string").str.strip()

    cleaned = cleaned.replace(
        {
            "": pd.NA,
            ".": pd.NA,
            "..": pd.NA,
        }
    )

    cleaned = cleaned.str.replace(
        ".",
        "",
        regex=False,
    )

    cleaned = cleaned.str.replace(
        ",",
        ".",
        regex=False,
    )

    return pd.to_numeric(
        cleaned,
        errors="coerce",
    ).astype("Float64")


# ---------------------------------------------------------------------
# Territorios
# ---------------------------------------------------------------------

def add_territory_columns(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Separa el código provincial y el nombre.

    Ejemplo:
    '38 Santa Cruz de Tenerife'
        province_code = '38'
        province_name = 'Santa Cruz de Tenerife'
    """

    extracted = dataframe[
        "source_territory"
    ].str.extract(
        r"^(?P<province_code>\d{2})\s+"
        r"(?P<province_name>.+)$"
    )

    invalid_mask = extracted.isna().any(axis=1)

    if invalid_mask.any():
        invalid_values = dataframe.loc[
            invalid_mask,
            "source_territory",
        ].drop_duplicates()

        raise ValueError(
            "No se pudieron interpretar algunos territorios: "
            + ", ".join(invalid_values.tolist())
        )

    result = dataframe.copy()

    result["province_code"] = (
        extracted["province_code"]
        .astype("string")
    )

    result["province_name"] = (
        extracted["province_name"]
        .str.strip()
        .astype("string")
    )

    result["territory_id"] = (
        "ES-PROV-" + result["province_code"]
    )

    result["territory_level"] = "province"
    result["country_code"] = "ES"

    return result


# ---------------------------------------------------------------------
# Periodos mensuales
# ---------------------------------------------------------------------

def add_time_columns(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convierte periodos como 2026M05 en fechas mensuales.

    El mes se representa mediante su primer día:
    2026M05 -> 2026-05-01
    """

    valid_period = dataframe[
        "source_period"
    ].str.fullmatch(
        r"\d{4}M(0[1-9]|1[0-2])"
    )

    if not valid_period.all():
        invalid_values = dataframe.loc[
            ~valid_period,
            "source_period",
        ].drop_duplicates()

        raise ValueError(
            "Se encontraron periodos no válidos: "
            + ", ".join(invalid_values.tolist())
        )

    result = dataframe.copy()

    result["month"] = pd.to_datetime(
        (
            result["source_period"].str.replace(
                "M",
                "-",
                regex=False,
            )
            + "-01"
        ),
        format="%Y-%m-%d",
        errors="raise",
    )

    result["month_id"] = (
        result["month"]
        .dt.strftime("%Y-%m")
        .astype("string")
    )

    result["year"] = (
        result["month"]
        .dt.year
        .astype("Int16")
    )

    result["month_number"] = (
        result["month"]
        .dt.month
        .astype("Int8")
    )

    result["is_provisional"] = (
        result["month"] >= PROVISIONAL_FROM
    ).astype("boolean")

    result["data_status"] = (
        result["is_provisional"]
        .map(
            {
                True: "provisional",
                False: "final_or_not_marked_provisional",
            }
        )
        .astype("string")
    )

    return result


# ---------------------------------------------------------------------
# Trazabilidad
# ---------------------------------------------------------------------

def add_common_metadata(
    dataframe: pd.DataFrame,
    *,
    source_file: Path,
    source_id: str,
    table_id: int,
) -> pd.DataFrame:
    """
    Añade metadatos que permiten reconstruir el origen
    de cada registro.
    """

    result = dataframe.copy()

    result["source_id"] = source_id
    result["source_table_id"] = table_id
    result["source_file_name"] = source_file.name
    result["source_snapshot_id"] = (
        calculate_sha256(source_file)
    )

    result["processed_at_utc"] = (
        datetime.now(timezone.utc).isoformat()
    )

    result["value_missing"] = (
        result["value"]
        .isna()
        .astype("boolean")
    )

    result["quality_flag"] = (
        result["value_missing"]
        .map(
            {
                True: "missing_in_source",
                False: "published_value",
            }
        )
        .astype("string")
    )

    return result


# ---------------------------------------------------------------------
# Validaciones
# ---------------------------------------------------------------------

def validate_processed(
    dataframe: pd.DataFrame,
    *,
    key_columns: list[str],
    expected_territories: int = 50,
) -> None:
    """
    Aplica controles mínimos antes de guardar el Parquet.
    """

    duplicate_count = int(
        dataframe.duplicated(
            subset=key_columns
        ).sum()
    )

    if duplicate_count:
        raise ValueError(
            "Se detectaron "
            f"{duplicate_count} filas duplicadas "
            "para la clave esperada."
        )

    territory_count = (
        dataframe["province_code"].nunique()
    )

    if territory_count != expected_territories:
        raise ValueError(
            "Número de provincias inesperado: "
            f"{territory_count}"
        )

    negative_count = int(
        (
            dataframe["value"]
            .dropna()
            < 0
        ).sum()
    )

    if negative_count:
        raise ValueError(
            "Se detectaron "
            f"{negative_count} valores negativos."
        )

    percentage_rows = (
        dataframe["unit"].eq("percent")
        & dataframe["value"].notna()
    )

    invalid_percentages = int(
        (
            ~dataframe.loc[
                percentage_rows,
                "value",
            ].between(0, 100)
        ).sum()
    )

    if invalid_percentages:
        raise ValueError(
            "Se detectaron "
            f"{invalid_percentages} porcentajes "
            "fuera del rango 0-100."
        )


# ---------------------------------------------------------------------
# Normalización de la tabla 2073
# ---------------------------------------------------------------------

def normalize_demand(
    source_file: Path,
) -> pd.DataFrame:
    """
    Normaliza viajeros y pernoctaciones provinciales.
    """

    dataframe = read_raw_csv(source_file)

    require_columns(
        dataframe,
        set(DEMAND_COLUMNS),
    )

    dataframe = dataframe.rename(
        columns=DEMAND_COLUMNS
    )[
        list(DEMAND_COLUMNS.values())
    ]

    dataframe = add_territory_columns(dataframe)
    dataframe = add_time_columns(dataframe)

    dataframe["metric"] = (
        dataframe["source_metric"]
        .map(DEMAND_METRIC_MAP)
        .astype("string")
    )

    dataframe["residence"] = (
        dataframe["source_residence"]
        .map(RESIDENCE_MAP)
        .astype("string")
    )

    if (
        dataframe["metric"].isna().any()
        or dataframe["residence"].isna().any()
    ):
        raise ValueError(
            "Existen métricas o categorías "
            "de residencia sin mapear "
            "en la tabla 2073."
        )

    dataframe["value"] = parse_spanish_number(
        dataframe["source_value_raw"]
    )

    dataframe["unit"] = (
        "persons_or_overnight_stays"
    )

    dataframe = add_common_metadata(
        dataframe,
        source_file=source_file,
        source_id="ine_eotr_demand_province",
        table_id=2073,
    )

    validate_processed(
        dataframe,
        key_columns=[
            "province_code",
            "metric",
            "residence",
            "month_id",
        ],
    )

    columns = [
        "territory_id",
        "territory_level",
        "country_code",
        "province_code",
        "province_name",
        "metric",
        "source_metric",
        "residence",
        "source_residence",
        "month_id",
        "month",
        "year",
        "month_number",
        "value",
        "unit",
        "source_value_raw",
        "value_missing",
        "quality_flag",
        "data_status",
        "is_provisional",
        "source_id",
        "source_table_id",
        "source_file_name",
        "source_snapshot_id",
        "processed_at_utc",
    ]

    return (
        dataframe[columns]
        .sort_values(
            [
                "province_code",
                "month",
                "metric",
                "residence",
            ]
        )
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------
# Normalización de la tabla 2070
# ---------------------------------------------------------------------

def normalize_supply(
    source_file: Path,
) -> pd.DataFrame:
    """
    Normaliza oferta, ocupación y empleo provinciales.
    """

    dataframe = read_raw_csv(source_file)

    require_columns(
        dataframe,
        set(SUPPLY_COLUMNS),
    )

    dataframe = dataframe.rename(
        columns=SUPPLY_COLUMNS
    )[
        list(SUPPLY_COLUMNS.values())
    ]

    dataframe = add_territory_columns(dataframe)
    dataframe = add_time_columns(dataframe)

    dataframe["metric"] = (
        dataframe["source_metric"]
        .map(SUPPLY_METRIC_MAP)
        .astype("string")
    )

    if dataframe["metric"].isna().any():
        raise ValueError(
            "Existen métricas sin mapear "
            "en la tabla 2070."
        )

    dataframe["value"] = parse_spanish_number(
        dataframe["source_value_raw"]
    )

    dataframe["unit"] = (
        dataframe["metric"]
        .map(SUPPLY_UNIT_MAP)
        .astype("string")
    )

    # La tabla de oferta no tiene dimensión de residencia.
    # Se crean las columnas vacías para que ambas tablas puedan
    # combinarse manteniendo el mismo esquema.
    dataframe["residence"] = pd.Series(
        pd.NA,
        index=dataframe.index,
        dtype="string",
    )

    dataframe["source_residence"] = pd.Series(
        pd.NA,
        index=dataframe.index,
        dtype="string",
    )

    dataframe = add_common_metadata(
        dataframe,
        source_file=source_file,
        source_id="ine_eotr_supply_province",
        table_id=2070,
    )

    validate_processed(
        dataframe,
        key_columns=[
            "province_code",
            "metric",
            "month_id",
        ],
    )

    columns = [
        "territory_id",
        "territory_level",
        "country_code",
        "province_code",
        "province_name",
        "metric",
        "source_metric",
        "residence",
        "source_residence",
        "month_id",
        "month",
        "year",
        "month_number",
        "value",
        "unit",
        "source_value_raw",
        "value_missing",
        "quality_flag",
        "data_status",
        "is_provisional",
        "source_id",
        "source_table_id",
        "source_file_name",
        "source_snapshot_id",
        "processed_at_utc",
    ]

    return (
        dataframe[columns]
        .sort_values(
            [
                "province_code",
                "month",
                "metric",
            ]
        )
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------
# Escritura y resumen
# ---------------------------------------------------------------------

def write_parquet(
    dataframe: pd.DataFrame,
    output_path: Path,
) -> None:
    """Guarda un DataFrame como fichero Parquet."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_parquet(
        output_path,
        index=False,
        engine="pyarrow",
    )


def print_summary(
    label: str,
    dataframe: pd.DataFrame,
    output_path: Path,
) -> None:
    """Muestra un resumen del resultado generado."""

    print(f"[OK] {label}")

    print(
        f"     Filas: "
        f"{len(dataframe):,}"
    )

    print(
        f"     Provincias: "
        f"{dataframe['province_code'].nunique()}"
    )

    print(
        "     Periodo: "
        f"{dataframe['month'].min().date()} "
        "-> "
        f"{dataframe['month'].max().date()}"
    )

    print(
        f"     Valores nulos: "
        f"{int(dataframe['value'].isna().sum()):,}"
    )

    print(
        "     Salida: "
        f"{output_path.relative_to(PROJECT_ROOT).as_posix()}"
    )

    print()


# ---------------------------------------------------------------------
# Argumentos de terminal
# ---------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    """Lee qué fuente debe normalizarse."""

    parser = argparse.ArgumentParser(
        description=(
            "Normaliza las tablas 2073 y 2070 del INE."
        )
    )

    parser.add_argument(
        "--source",
        choices=[
            "2073",
            "2070",
            "all",
        ],
        default="all",
        help=(
            "Fuente que se desea normalizar. "
            "El valor por defecto es all."
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Ejecución principal
# ---------------------------------------------------------------------

def main() -> int:
    """Ejecuta la normalización."""

    arguments = parse_arguments()

    PROCESSED_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("NORMALIZACIÓN DE DATOS PROCESSED")
    print("=" * 70)
    print()

    processed_frames: list[pd.DataFrame] = []

    try:
        if arguments.source in {
            "2073",
            "all",
        }:
            demand_file = find_latest_raw(
                "*2073*.csv"
            )

            demand = normalize_demand(
                demand_file
            )

            write_parquet(
                demand,
                DEMAND_OUTPUT,
            )

            print_summary(
                "Tabla 2073 normalizada",
                demand,
                DEMAND_OUTPUT,
            )

            processed_frames.append(demand)

        if arguments.source in {
            "2070",
            "all",
        }:
            supply_file = find_latest_raw(
                "*2070*.csv"
            )

            supply = normalize_supply(
                supply_file
            )

            write_parquet(
                supply,
                SUPPLY_OUTPUT,
            )

            print_summary(
                "Tabla 2070 normalizada",
                supply,
                SUPPLY_OUTPUT,
            )

            processed_frames.append(supply)

        if arguments.source == "all":
            combined = pd.concat(
                processed_frames,
                ignore_index=True,
            )

            write_parquet(
                combined,
                COMBINED_OUTPUT,
            )

            print_summary(
                "Dataset processed combinado",
                combined,
                COMBINED_OUTPUT,
            )

        gitkeep_path = (
            PROCESSED_DIRECTORY
            / ".gitkeep"
        )

        if gitkeep_path.exists():
            gitkeep_path.unlink()

        print(
            "[OK] Normalización completada "
            "sin errores."
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