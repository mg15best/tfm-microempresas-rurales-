"""
Construye las dimensiones de territorio y calendario del proyecto.

Entrada:
- data/processed/processed_ocupacion_rural_monthly.parquet

Salidas:
- data/processed/dim_territory.parquet
- data/processed/dim_calendar_month.parquet
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------
# Rutas del proyecto
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIRECTORY = PROJECT_ROOT / "data" / "processed"

INPUT_PATH = (
    PROCESSED_DIRECTORY
    / "processed_ocupacion_rural_monthly.parquet"
)

TERRITORY_OUTPUT = (
    PROCESSED_DIRECTORY
    / "dim_territory.parquet"
)

CALENDAR_OUTPUT = (
    PROCESSED_DIRECTORY
    / "dim_calendar_month.parquet"
)


# ---------------------------------------------------------------------
# Catálogo de comunidades autónomas
# ---------------------------------------------------------------------

AUTONOMOUS_COMMUNITIES = {
    "01": "Andalucía",
    "02": "Aragón",
    "03": "Principado de Asturias",
    "04": "Illes Balears",
    "05": "Canarias",
    "06": "Cantabria",
    "07": "Castilla y León",
    "08": "Castilla-La Mancha",
    "09": "Cataluña",
    "10": "Comunitat Valenciana",
    "11": "Extremadura",
    "12": "Galicia",
    "13": "Comunidad de Madrid",
    "14": "Región de Murcia",
    "15": "Comunidad Foral de Navarra",
    "16": "País Vasco",
    "17": "La Rioja",
}


# Relación entre código de provincia y comunidad autónoma.
PROVINCE_TO_AUTONOMOUS_COMMUNITY = {
    "01": "16",  # Araba/Álava
    "02": "08",  # Albacete
    "03": "10",  # Alicante/Alacant
    "04": "01",  # Almería
    "05": "07",  # Ávila
    "06": "11",  # Badajoz
    "07": "04",  # Illes Balears
    "08": "09",  # Barcelona
    "09": "07",  # Burgos
    "10": "11",  # Cáceres
    "11": "01",  # Cádiz
    "12": "10",  # Castellón/Castelló
    "13": "08",  # Ciudad Real
    "14": "01",  # Córdoba
    "15": "12",  # A Coruña
    "16": "08",  # Cuenca
    "17": "09",  # Girona
    "18": "01",  # Granada
    "19": "08",  # Guadalajara
    "20": "16",  # Gipuzkoa
    "21": "01",  # Huelva
    "22": "02",  # Huesca
    "23": "01",  # Jaén
    "24": "07",  # León
    "25": "09",  # Lleida
    "26": "17",  # La Rioja
    "27": "12",  # Lugo
    "28": "13",  # Madrid
    "29": "01",  # Málaga
    "30": "14",  # Murcia
    "31": "15",  # Navarra
    "32": "12",  # Ourense
    "33": "03",  # Asturias
    "34": "07",  # Palencia
    "35": "05",  # Las Palmas
    "36": "12",  # Pontevedra
    "37": "07",  # Salamanca
    "38": "05",  # Santa Cruz de Tenerife
    "39": "06",  # Cantabria
    "40": "07",  # Segovia
    "41": "01",  # Sevilla
    "42": "07",  # Soria
    "43": "09",  # Tarragona
    "44": "02",  # Teruel
    "45": "08",  # Toledo
    "46": "10",  # Valencia/València
    "47": "07",  # Valladolid
    "48": "16",  # Bizkaia
    "49": "07",  # Zamora
    "50": "02",  # Zaragoza
}


# ---------------------------------------------------------------------
# Nombres mensuales y temporadas
# ---------------------------------------------------------------------

MONTH_NAMES_ES = {
    1: "enero",
    2: "febrero",
    3: "marzo",
    4: "abril",
    5: "mayo",
    6: "junio",
    7: "julio",
    8: "agosto",
    9: "septiembre",
    10: "octubre",
    11: "noviembre",
    12: "diciembre",
}


def assign_season(month_number: int) -> str:
    """
    Asigna una estación meteorológica aproximada.

    La clasificación no representa todavía temporada turística
    alta o baja. Esa clasificación se calculará posteriormente
    a partir de los datos reales de demanda.
    """

    if month_number in {12, 1, 2}:
        return "winter"

    if month_number in {3, 4, 5}:
        return "spring"

    if month_number in {6, 7, 8}:
        return "summer"

    return "autumn"


# ---------------------------------------------------------------------
# Carga del dataset processed
# ---------------------------------------------------------------------

def load_processed_dataset() -> pd.DataFrame:
    """Carga y valida el dataset processed principal."""

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            "No se encontró el dataset processed principal: "
            f"{INPUT_PATH.relative_to(PROJECT_ROOT)}"
        )

    dataframe = pd.read_parquet(INPUT_PATH)

    required_columns = {
        "territory_id",
        "territory_level",
        "country_code",
        "province_code",
        "province_name",
        "month_id",
        "month",
        "value",
        "source_id",
        "source_file_name",
    }

    missing_columns = required_columns.difference(
        dataframe.columns
    )

    if missing_columns:
        raise ValueError(
            "Faltan columnas obligatorias en el dataset processed: "
            + ", ".join(sorted(missing_columns))
        )

    if dataframe.empty:
        raise ValueError(
            "El dataset processed está vacío."
        )

    return dataframe


# ---------------------------------------------------------------------
# Dimensión territorial
# ---------------------------------------------------------------------

def classify_coverage(coverage_ratio: float) -> str:
    """
    Clasifica la cobertura mensual de un territorio.

    high:         al menos 95 %
    medium:       al menos 80 %
    low:          al menos 50 %
    insufficient: menos del 50 %
    """

    if coverage_ratio >= 0.95:
        return "high"

    if coverage_ratio >= 0.80:
        return "medium"

    if coverage_ratio >= 0.50:
        return "low"

    return "insufficient"


def build_territory_dimension(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construye una fila por provincia normalizada.

    La disponibilidad y cobertura se calculan usando todos los
    registros de ocupación rural presentes en processed.
    """

    territory_base = (
        dataframe[
            [
                "territory_id",
                "territory_level",
                "province_code",
                "province_name",
            ]
        ]
        .drop_duplicates()
        .copy()
    )

    duplicate_territories = int(
        territory_base["territory_id"].duplicated().sum()
    )

    if duplicate_territories:
        raise ValueError(
            "Un territory_id aparece asociado a más de "
            "una combinación territorial."
        )

    global_first_month = dataframe["month"].min()
    global_last_month = dataframe["month"].max()

    expected_months = len(
        pd.date_range(
            start=global_first_month,
            end=global_last_month,
            freq="MS",
        )
    )

    availability_rows: list[dict[str, object]] = []

    for territory_id, group in dataframe.groupby(
        "territory_id",
        sort=False,
    ):
        published_rows = group[
            group["value"].notna()
        ]

        available_months = (
            published_rows["month_id"].nunique()
        )

        coverage_ratio = (
            available_months / expected_months
            if expected_months
            else 0.0
        )

        first_available_month = (
            published_rows["month"].min()
            if not published_rows.empty
            else pd.NaT
        )

        last_available_month = (
            published_rows["month"].max()
            if not published_rows.empty
            else pd.NaT
        )

        availability_rows.append(
            {
                "territory_id": territory_id,
                "is_rural_tourism_available": (
                    not published_rows.empty
                ),
                "first_available_month": (
                    first_available_month.strftime("%Y-%m")
                    if pd.notna(first_available_month)
                    else pd.NA
                ),
                "last_available_month": (
                    last_available_month.strftime("%Y-%m")
                    if pd.notna(last_available_month)
                    else pd.NA
                ),
                "coverage_quality": classify_coverage(
                    coverage_ratio
                ),
            }
        )

    availability = pd.DataFrame(
        availability_rows
    )

    dimension = territory_base.merge(
        availability,
        on="territory_id",
        how="left",
        validate="one_to_one",
    )

    dimension["autonomous_community_code"] = (
        dimension["province_code"].map(
            PROVINCE_TO_AUTONOMOUS_COMMUNITY
        )
    )

    missing_mapping = dimension[
        "autonomous_community_code"
    ].isna()

    if missing_mapping.any():
        missing_provinces = dimension.loc[
            missing_mapping,
            "province_code",
        ].tolist()

        raise ValueError(
            "Faltan comunidades autónomas para las provincias: "
            + ", ".join(missing_provinces)
        )

    dimension["autonomous_community_id"] = (
        "ES-CCAA-"
        + dimension[
            "autonomous_community_code"
        ].astype("string")
    )

    # Campo auxiliar útil para futuras uniones y revisiones.
    dimension["autonomous_community_name"] = (
        dimension["autonomous_community_code"]
        .map(AUTONOMOUS_COMMUNITIES)
        .astype("string")
    )

    dimension["territory_name"] = (
        dimension["province_name"]
        .astype("string")
    )

    dimension["source_territory_code"] = (
        dimension["province_code"]
        .astype("string")
    )

    dimension["source_territory_name"] = (
        dimension["province_name"]
        .astype("string")
    )

    # En esta primera versión cada territorio es una provincia,
    # por lo que province_id coincide con territory_id.
    dimension["province_id"] = (
        dimension["territory_id"]
        .astype("string")
    )

    dimension["generated_at_utc"] = (
        datetime.now(timezone.utc).isoformat()
    )

    columns = [
        "territory_id",
        "territory_name",
        "territory_level",
        "source_territory_code",
        "source_territory_name",
        "autonomous_community_id",
        "autonomous_community_name",
        "province_id",
        "is_rural_tourism_available",
        "first_available_month",
        "last_available_month",
        "coverage_quality",
        "generated_at_utc",
    ]

    dimension = (
        dimension[columns]
        .sort_values("source_territory_code")
        .reset_index(drop=True)
    )

    validate_territory_dimension(dimension)

    return dimension


def validate_territory_dimension(
    dimension: pd.DataFrame,
) -> None:
    """Valida la dimensión territorial."""

    if len(dimension) != 50:
        raise ValueError(
            "Se esperaban 50 provincias, pero se han "
            f"generado {len(dimension)}."
        )

    if dimension["territory_id"].duplicated().any():
        raise ValueError(
            "La dimensión territorial tiene claves duplicadas."
        )

    mandatory_columns = [
        "territory_id",
        "territory_name",
        "territory_level",
        "source_territory_code",
        "autonomous_community_id",
        "province_id",
        "coverage_quality",
    ]

    null_counts = (
        dimension[mandatory_columns]
        .isna()
        .sum()
    )

    if int(null_counts.sum()) > 0:
        raise ValueError(
            "La dimensión territorial contiene nulos "
            "en campos obligatorios:\n"
            f"{null_counts[null_counts > 0]}"
        )

    invalid_levels = set(
        dimension["territory_level"].unique()
    ).difference({"province"})

    if invalid_levels:
        raise ValueError(
            "Se encontraron niveles territoriales inesperados: "
            + ", ".join(sorted(invalid_levels))
        )


# ---------------------------------------------------------------------
# Dimensión temporal
# ---------------------------------------------------------------------

def build_calendar_dimension(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Construye una fila por cada mes consecutivo."""

    first_month = dataframe["month"].min()
    last_month = dataframe["month"].max()

    monthly_dates = pd.date_range(
        start=first_month,
        end=last_month,
        freq="MS",
    )

    dimension = pd.DataFrame(
        {
            "date_month": monthly_dates,
        }
    )

    dimension["month_id"] = (
        dimension["date_month"]
        .dt.strftime("%Y-%m")
        .astype("string")
    )

    dimension["year"] = (
        dimension["date_month"]
        .dt.year
        .astype("Int16")
    )

    dimension["month"] = (
        dimension["date_month"]
        .dt.month
        .astype("Int8")
    )

    dimension["month_name"] = (
        dimension["month"]
        .map(MONTH_NAMES_ES)
        .astype("string")
    )

    dimension["quarter"] = (
        dimension["date_month"]
        .dt.quarter
        .astype("Int8")
    )

    dimension["season"] = (
        dimension["month"]
        .map(assign_season)
        .astype("string")
    )

    # En este proyecto se considera periodo estival ampliado
    # de junio a septiembre.
    dimension["is_summer"] = (
        dimension["month"]
        .isin([6, 7, 8, 9])
        .astype("boolean")
    )

    # A escala mensual, Navidad se representa mediante
    # diciembre y enero.
    dimension["is_christmas_period"] = (
        dimension["month"]
        .isin([12, 1])
        .astype("boolean")
    )

    # Este campo queda nulo hasta integrar un calendario
    # de Semana Santa por año.
    dimension["is_easter_period"] = pd.Series(
        pd.NA,
        index=dimension.index,
        dtype="boolean",
    )

    dimension["covid_period"] = (
        dimension["date_month"].between(
            pd.Timestamp("2020-03-01"),
            pd.Timestamp("2021-12-01"),
            inclusive="both",
        )
        .astype("boolean")
    )

    snapshot_timestamps = pd.to_datetime(
        dataframe["source_file_name"]
        .astype("string")
        .str.extract(
            r"^(\d{8}T\d{6}Z)_",
            expand=False,
        ),
        format="%Y%m%dT%H%M%SZ",
        utc=True,
        errors="coerce",
    )

    if snapshot_timestamps.isna().any():
        raise ValueError(
            "No se pudo obtener la fecha de adquisición "
            "desde source_file_name."
        )

    snapshot_month = (
        snapshot_timestamps.max()
        .tz_localize(None)
        .to_period("M")
        .to_timestamp()
    )

    dimension["complete_month_available"] = (
        dimension["date_month"] < snapshot_month
    ).astype("boolean")

    dimension["generated_at_utc"] = (
        datetime.now(timezone.utc).isoformat()
    )

    columns = [
        "month_id",
        "date_month",
        "year",
        "month",
        "month_name",
        "quarter",
        "season",
        "is_summer",
        "is_christmas_period",
        "is_easter_period",
        "covid_period",
        "complete_month_available",
        "generated_at_utc",
    ]

    dimension = dimension[columns]

    validate_calendar_dimension(
        dimension,
        first_month=first_month,
        last_month=last_month,
    )

    return dimension


def validate_calendar_dimension(
    dimension: pd.DataFrame,
    *,
    first_month: pd.Timestamp,
    last_month: pd.Timestamp,
) -> None:
    """Valida continuidad, claves y tipos temporales."""

    expected_count = len(
        pd.date_range(
            start=first_month,
            end=last_month,
            freq="MS",
        )
    )

    if len(dimension) != expected_count:
        raise ValueError(
            "La dimensión calendario no contiene todos "
            "los meses esperados."
        )

    if dimension["month_id"].duplicated().any():
        raise ValueError(
            "La dimensión calendario tiene meses duplicados."
        )

    if not dimension["date_month"].is_monotonic_increasing:
        raise ValueError(
            "La dimensión calendario no está ordenada."
        )

    if not dimension["date_month"].dt.is_month_start.all():
        raise ValueError(
            "Todas las fechas deben ser el primer día del mes."
        )

    expected_month_ids = pd.date_range(
        start=first_month,
        end=last_month,
        freq="MS",
    ).strftime("%Y-%m")

    actual_month_ids = dimension[
        "month_id"
    ].astype(str)

    if list(actual_month_ids) != list(expected_month_ids):
        raise ValueError(
            "Se detectaron huecos en la serie mensual."
        )


# ---------------------------------------------------------------------
# Escritura
# ---------------------------------------------------------------------

def write_dimension(
    dataframe: pd.DataFrame,
    output_path: Path,
) -> None:
    """Guarda una dimensión como Parquet."""

    dataframe.to_parquet(
        output_path,
        index=False,
        engine="pyarrow",
    )


def print_summary(
    name: str,
    dataframe: pd.DataFrame,
    output_path: Path,
) -> None:
    """Muestra el resumen de una dimensión."""

    print(f"[OK] {name}")
    print(f"     Filas: {len(dataframe):,}")
    print(f"     Columnas: {len(dataframe.columns)}")
    print(
        "     Salida: "
        f"{output_path.relative_to(PROJECT_ROOT).as_posix()}"
    )
    print()


# ---------------------------------------------------------------------
# Ejecución principal
# ---------------------------------------------------------------------

def main() -> int:
    """Genera ambas dimensiones."""

    print("=" * 70)
    print("CONSTRUCCIÓN DE DIMENSIONES")
    print("=" * 70)
    print()

    try:
        processed = load_processed_dataset()

        territory_dimension = (
            build_territory_dimension(processed)
        )

        calendar_dimension = (
            build_calendar_dimension(processed)
        )

        write_dimension(
            territory_dimension,
            TERRITORY_OUTPUT,
        )

        write_dimension(
            calendar_dimension,
            CALENDAR_OUTPUT,
        )

        print_summary(
            "Dimensión territorial",
            territory_dimension,
            TERRITORY_OUTPUT,
        )

        print_summary(
            "Dimensión calendario",
            calendar_dimension,
            CALENDAR_OUTPUT,
        )

        print(
            "[OK] Dimensiones construidas y validadas."
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