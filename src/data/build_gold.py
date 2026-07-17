"""
Construye la tabla gold principal de demanda turística rural.

Entradas:
- processed_ocupacion_rural_demand_province_monthly.parquet
- processed_ocupacion_rural_supply_province_monthly.parquet
- dim_territory.parquet
- dim_calendar_month.parquet

Salidas:
- data/gold/gold_tourism_demand_monthly.parquet
- data/gold/exports_csv/gold_tourism_demand_monthly.csv

Granularidad:
- Una fila por territorio y mes.
"""

from __future__ import annotations

import hashlib
import sys
import uuid
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------
# Configuración general
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIRECTORY = PROJECT_ROOT / "data" / "processed"
GOLD_DIRECTORY = PROJECT_ROOT / "data" / "gold"
CSV_EXPORT_DIRECTORY = GOLD_DIRECTORY / "exports_csv"

DEMAND_INPUT = (
    PROCESSED_DIRECTORY
    / "processed_ocupacion_rural_demand_province_monthly.parquet"
)

SUPPLY_INPUT = (
    PROCESSED_DIRECTORY
    / "processed_ocupacion_rural_supply_province_monthly.parquet"
)

TERRITORY_INPUT = (
    PROCESSED_DIRECTORY
    / "dim_territory.parquet"
)

CALENDAR_INPUT = (
    PROCESSED_DIRECTORY
    / "dim_calendar_month.parquet"
)

GOLD_OUTPUT = (
    GOLD_DIRECTORY
    / "gold_tourism_demand_monthly.parquet"
)

CSV_OUTPUT = (
    CSV_EXPORT_DIRECTORY
    / "gold_tourism_demand_monthly.csv"
)

DATA_VERSION = "gold_tourism_demand_monthly_v1.0.0"


# ---------------------------------------------------------------------
# Correspondencia de métricas processed -> gold
# ---------------------------------------------------------------------

DEMAND_PIVOT_COLUMNS = {
    (
        "travellers",
        "residents_in_spain",
    ): "travellers_domestic",
    (
        "travellers",
        "residents_abroad",
    ): "travellers_foreign",
    (
        "overnight_stays",
        "residents_in_spain",
    ): "overnight_stays_domestic",
    (
        "overnight_stays",
        "residents_abroad",
    ): "overnight_stays_foreign",
}

SUPPLY_PIVOT_COLUMNS = {
    "estimated_open_establishments": (
        "establishments_estimated"
    ),
    "estimated_bed_places": (
        "places_estimated"
    ),
    "occupancy_rate_by_bed_places": (
        "occupancy_rate_pct"
    ),
    "weekend_occupancy_rate_by_bed_places": (
        "weekend_occupancy_rate_pct"
    ),
    "occupancy_rate_by_rooms": (
        "room_occupancy_rate_pct"
    ),
    "employed_personnel": (
        "staff_employed"
    ),
}


# ---------------------------------------------------------------------
# Utilidades de carga
# ---------------------------------------------------------------------

def require_file(file_path: Path) -> None:
    """Comprueba que un fichero necesario existe."""

    if not file_path.exists():
        raise FileNotFoundError(
            "No se encontró el fichero necesario: "
            f"{file_path.relative_to(PROJECT_ROOT)}"
        )


def load_inputs() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Carga los datasets processed y sus dimensiones."""

    input_paths = [
        DEMAND_INPUT,
        SUPPLY_INPUT,
        TERRITORY_INPUT,
        CALENDAR_INPUT,
    ]

    for file_path in input_paths:
        require_file(file_path)

    demand = pd.read_parquet(DEMAND_INPUT)
    supply = pd.read_parquet(SUPPLY_INPUT)
    territory = pd.read_parquet(TERRITORY_INPUT)
    calendar = pd.read_parquet(CALENDAR_INPUT)

    if demand.empty:
        raise ValueError(
            "El dataset processed de demanda está vacío."
        )

    if supply.empty:
        raise ValueError(
            "El dataset processed de oferta está vacío."
        )

    if territory.empty:
        raise ValueError(
            "La dimensión territorial está vacía."
        )

    if calendar.empty:
        raise ValueError(
            "La dimensión calendario está vacía."
        )

    return demand, supply, territory, calendar


def require_columns(
    dataframe: pd.DataFrame,
    expected_columns: set[str],
    dataset_name: str,
) -> None:
    """Comprueba que están presentes las columnas requeridas."""

    missing_columns = expected_columns.difference(
        dataframe.columns
    )

    if missing_columns:
        raise ValueError(
            f"Faltan columnas en {dataset_name}: "
            + ", ".join(sorted(missing_columns))
        )


# ---------------------------------------------------------------------
# Trazabilidad
# ---------------------------------------------------------------------

def get_single_snapshot_id(
    dataframe: pd.DataFrame,
    dataset_name: str,
) -> str:
    """
    Obtiene el identificador del snapshot raw utilizado.

    Cada dataset processed debe provenir de un único snapshot
    en esta primera versión del pipeline.
    """

    snapshot_ids = (
        dataframe["source_snapshot_id"]
        .dropna()
        .astype(str)
        .unique()
    )

    if len(snapshot_ids) != 1:
        raise ValueError(
            f"{dataset_name} contiene "
            f"{len(snapshot_ids)} snapshots diferentes. "
            "Se esperaba exactamente uno."
        )

    return str(snapshot_ids[0])


def combine_snapshot_ids(
    demand_snapshot_id: str,
    supply_snapshot_id: str,
) -> str:
    """
    Genera una huella conjunta de los dos snapshots utilizados.
    """

    snapshot_text = (
        f"demand:{demand_snapshot_id}|"
        f"supply:{supply_snapshot_id}"
    )

    return hashlib.sha256(
        snapshot_text.encode("utf-8")
    ).hexdigest()


# ---------------------------------------------------------------------
# Conversión de tipos
# ---------------------------------------------------------------------

def cast_nullable_integer(
    series: pd.Series,
    column_name: str,
) -> pd.Series:
    """
    Convierte una serie numérica en entero nullable.

    Se detiene si encuentra valores con decimales reales,
    porque viajeros y pernoctaciones deben ser recuentos.
    """

    numeric = pd.to_numeric(
        series,
        errors="coerce",
    ).astype("Float64")

    non_null = numeric.dropna()

    fractional_difference = (
        non_null - non_null.round()
    ).abs()

    if (
        not fractional_difference.empty
        and (fractional_difference > 0.000001).any()
    ):
        raise ValueError(
            f"La columna {column_name} contiene "
            "valores decimales inesperados."
        )

    return numeric.round().astype("Int64")


def safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    """
    Divide dos series solo cuando el denominador es mayor que cero.
    """

    numerator_numeric = pd.to_numeric(
        numerator,
        errors="coerce",
    ).astype("Float64")

    denominator_numeric = pd.to_numeric(
        denominator,
        errors="coerce",
    ).astype("Float64")

    result = (
        numerator_numeric
        / denominator_numeric
    )

    valid_denominator = (
        denominator_numeric.notna()
        & denominator_numeric.gt(0)
    )

    return (
        result
        .where(valid_denominator)
        .astype("Float64")
    )


def percentage_change(
    current: pd.Series,
    previous: pd.Series,
) -> pd.Series:
    """
    Calcula la variación porcentual respecto a un periodo anterior.
    """

    difference = (
        pd.to_numeric(
            current,
            errors="coerce",
        ).astype("Float64")
        -
        pd.to_numeric(
            previous,
            errors="coerce",
        ).astype("Float64")
    )

    return (
        safe_divide(
            difference,
            previous,
        )
        * 100
    ).astype("Float64")


# ---------------------------------------------------------------------
# Demanda en formato ancho
# ---------------------------------------------------------------------

def build_demand_wide(
    demand: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convierte demanda desde formato largo a una fila por
    territorio y mes.
    """

    required_columns = {
        "territory_id",
        "month_id",
        "metric",
        "residence",
        "value",
        "is_provisional",
        "source_snapshot_id",
    }

    require_columns(
        demand,
        required_columns,
        "processed de demanda",
    )

    duplicate_count = int(
        demand.duplicated(
            subset=[
                "territory_id",
                "month_id",
                "metric",
                "residence",
            ]
        ).sum()
    )

    if duplicate_count:
        raise ValueError(
            "La demanda processed contiene "
            f"{duplicate_count} claves duplicadas."
        )

    pivoted = demand.pivot_table(
        index=[
            "territory_id",
            "month_id",
        ],
        columns=[
            "metric",
            "residence",
        ],
        values="value",
        aggfunc="first",
        dropna=False,
    )

    # Las columnas resultantes son tuplas:
    # ('travellers', 'residents_in_spain'), etc.
    flattened_columns: list[str] = []

    for column in pivoted.columns:
        technical_name = DEMAND_PIVOT_COLUMNS.get(
            tuple(column)
        )

        if technical_name is None:
            technical_name = (
                f"{column[0]}__{column[1]}"
            )

        flattened_columns.append(
            technical_name
        )

    pivoted.columns = flattened_columns

    pivoted = pivoted.reset_index()

    expected_value_columns = [
        "travellers_domestic",
        "travellers_foreign",
        "overnight_stays_domestic",
        "overnight_stays_foreign",
    ]

    for column in expected_value_columns:
        if column not in pivoted.columns:
            pivoted[column] = pd.Series(
                pd.NA,
                index=pivoted.index,
                dtype="Float64",
            )

    pivoted["travellers_total"] = (
        pivoted[
            [
                "travellers_domestic",
                "travellers_foreign",
            ]
        ]
        .sum(
            axis=1,
            min_count=2,
        )
    )

    pivoted["overnight_stays_total"] = (
        pivoted[
            [
                "overnight_stays_domestic",
                "overnight_stays_foreign",
            ]
        ]
        .sum(
            axis=1,
            min_count=2,
        )
    )

    count_columns = [
        "travellers_total",
        "travellers_domestic",
        "travellers_foreign",
        "overnight_stays_total",
        "overnight_stays_domestic",
        "overnight_stays_foreign",
    ]

    for column in count_columns:
        pivoted[column] = cast_nullable_integer(
            pivoted[column],
            column,
        )

    provisional_status = (
        demand.groupby(
            [
                "territory_id",
                "month_id",
            ],
            as_index=False,
        )["is_provisional"]
        .max()
    )

    pivoted = pivoted.merge(
        provisional_status,
        on=[
            "territory_id",
            "month_id",
        ],
        how="left",
        validate="one_to_one",
    )

    return pivoted[
        [
            "territory_id",
            "month_id",
            "travellers_total",
            "travellers_domestic",
            "travellers_foreign",
            "overnight_stays_total",
            "overnight_stays_domestic",
            "overnight_stays_foreign",
            "is_provisional",
        ]
    ]


# ---------------------------------------------------------------------
# Oferta en formato ancho
# ---------------------------------------------------------------------

def build_supply_wide(
    supply: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convierte oferta, ocupación y empleo desde formato largo
    a una fila por territorio y mes.
    """

    required_columns = {
        "territory_id",
        "month_id",
        "metric",
        "value",
        "source_snapshot_id",
    }

    require_columns(
        supply,
        required_columns,
        "processed de oferta",
    )

    duplicate_count = int(
        supply.duplicated(
            subset=[
                "territory_id",
                "month_id",
                "metric",
            ]
        ).sum()
    )

    if duplicate_count:
        raise ValueError(
            "La oferta processed contiene "
            f"{duplicate_count} claves duplicadas."
        )

    pivoted = supply.pivot_table(
        index=[
            "territory_id",
            "month_id",
        ],
        columns="metric",
        values="value",
        aggfunc="first",
        dropna=False,
    )

    pivoted = (
        pivoted
        .rename(
            columns=SUPPLY_PIVOT_COLUMNS
        )
        .reset_index()
    )

    expected_columns = list(
        SUPPLY_PIVOT_COLUMNS.values()
    )

    for column in expected_columns:
        if column not in pivoted.columns:
            pivoted[column] = pd.Series(
                pd.NA,
                index=pivoted.index,
                dtype="Float64",
            )

        pivoted[column] = pd.to_numeric(
            pivoted[column],
            errors="coerce",
        ).astype("Float64")

    return pivoted[
        [
            "territory_id",
            "month_id",
            "establishments_estimated",
            "places_estimated",
            "occupancy_rate_pct",
            "weekend_occupancy_rate_pct",
            "room_occupancy_rate_pct",
            "staff_employed",
        ]
    ]


# ---------------------------------------------------------------------
# Base territorio x mes
# ---------------------------------------------------------------------

def build_base_grid(
    territory: pd.DataFrame,
    calendar: pd.DataFrame,
) -> pd.DataFrame:
    """
    Crea todas las combinaciones válidas de provincia y mes.

    Esto garantiza una serie temporal continua antes de filtrar
    registros sin ninguna métrica principal de demanda.
    """

    territory_columns = [
        "territory_id",
        "territory_name",
        "territory_level",
        "source_territory_code",
        "source_territory_name",
        "autonomous_community_id",
        "autonomous_community_name",
        "province_id",
        "coverage_quality",
    ]

    calendar_columns = [
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
    ]

    require_columns(
        territory,
        set(territory_columns),
        "dimensión territorial",
    )

    require_columns(
        calendar,
        set(calendar_columns),
        "dimensión calendario",
    )

    if territory["territory_id"].duplicated().any():
        raise ValueError(
            "dim_territory contiene territory_id duplicados."
        )

    if calendar["month_id"].duplicated().any():
        raise ValueError(
            "dim_calendar_month contiene month_id duplicados."
        )

    return territory[territory_columns].merge(
        calendar[calendar_columns],
        how="cross",
    )


# ---------------------------------------------------------------------
# Variables derivadas
# ---------------------------------------------------------------------

def add_profile_ratios(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Añade estancia media, proporciones y ratios operativos."""

    result = dataframe.copy()

    result["average_stay"] = safe_divide(
        result["overnight_stays_total"],
        result["travellers_total"],
    )

    result["domestic_travellers_share"] = safe_divide(
        result["travellers_domestic"],
        result["travellers_total"],
    )

    result["foreign_travellers_share"] = safe_divide(
        result["travellers_foreign"],
        result["travellers_total"],
    )

    result[
        "domestic_overnight_stays_share"
    ] = safe_divide(
        result["overnight_stays_domestic"],
        result["overnight_stays_total"],
    )

    result[
        "foreign_overnight_stays_share"
    ] = safe_divide(
        result["overnight_stays_foreign"],
        result["overnight_stays_total"],
    )

    result["overnight_stays_per_place"] = safe_divide(
        result["overnight_stays_total"],
        result["places_estimated"],
    )

    result[
        "travellers_per_establishment"
    ] = safe_divide(
        result["travellers_total"],
        result["establishments_estimated"],
    )

    occupancy_difference = (
        result["weekend_occupancy_rate_pct"]
        - result["occupancy_rate_pct"]
    )

    result["weekend_dependence_index"] = (
        safe_divide(
            occupancy_difference,
            result["occupancy_rate_pct"],
        )
        * 100
    )

    return result


def add_temporal_changes(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calcula variación mensual e interanual de pernoctaciones.

    La base todavía contiene todos los meses consecutivos,
    por lo que shift(12) representa exactamente el mismo mes
    del año anterior.
    """

    result = dataframe.sort_values(
        [
            "territory_id",
            "date_month",
        ]
    ).copy()

    grouped_overnight_stays = result.groupby(
        "territory_id",
        sort=False,
    )["overnight_stays_total"]

    previous_month = grouped_overnight_stays.shift(1)
    previous_year = grouped_overnight_stays.shift(12)

    result[
        "overnight_stays_mom_change_pct"
    ] = percentage_change(
        result["overnight_stays_total"],
        previous_month,
    )

    result[
        "overnight_stays_yoy_change_pct"
    ] = percentage_change(
        result["overnight_stays_total"],
        previous_year,
    )

    return result


def add_seasonality_index(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construye un índice de estacionalidad territorial 0-100.

    Lógica:
    1. Se excluyen los meses COVID y los datos provisionales.
    2. Se calcula la media histórica de pernoctaciones para
       cada provincia y mes del año.
    3. El mes con mayor media histórica en cada provincia
       recibe 100.
    4. Los demás meses se expresan de forma relativa a ese máximo.

    El índice describe el patrón histórico del territorio.
    No representa una predicción.
    """

    result = dataframe.copy()

    reference_mask = (
        result["overnight_stays_total"].notna()
        & ~result["covid_period"].fillna(False)
        & ~result["is_provisional"].fillna(False)
    )

    reference = result.loc[
        reference_mask,
        [
            "territory_id",
            "month",
            "overnight_stays_total",
        ],
    ].copy()

    seasonal_profile = (
        reference.groupby(
            [
                "territory_id",
                "month",
            ],
            as_index=False,
        )["overnight_stays_total"]
        .mean()
        .rename(
            columns={
                "overnight_stays_total": (
                    "historical_monthly_mean"
                )
            }
        )
    )

    seasonal_profile[
        "territory_monthly_max"
    ] = seasonal_profile.groupby(
        "territory_id"
    )["historical_monthly_mean"].transform(
        "max"
    )

    seasonal_profile["seasonality_index"] = (
        safe_divide(
            seasonal_profile[
                "historical_monthly_mean"
            ],
            seasonal_profile[
                "territory_monthly_max"
            ],
        )
        * 100
    )

    result = result.merge(
        seasonal_profile[
            [
                "territory_id",
                "month",
                "seasonality_index",
            ]
        ],
        on=[
            "territory_id",
            "month",
        ],
        how="left",
        validate="many_to_one",
    )

    return result


def weighted_available_score(
    dataframe: pd.DataFrame,
    components: list[tuple[str, float]],
) -> pd.Series:
    """
    Calcula una media ponderada usando solo componentes disponibles.

    Si un componente es nulo, su peso no se utiliza en esa fila.
    """

    numerator = pd.Series(
        0.0,
        index=dataframe.index,
        dtype="float64",
    )

    denominator = pd.Series(
        0.0,
        index=dataframe.index,
        dtype="float64",
    )

    for column, weight in components:
        available = dataframe[column].notna()

        numerator = (
            numerator
            + dataframe[column]
            .fillna(0)
            .astype(float)
            * weight
        )

        denominator = (
            denominator
            + available.astype(float)
            * weight
        )

    result = numerator / denominator

    return (
        result
        .where(denominator > 0)
        .astype("Float64")
    )


def add_tourism_pressure_index(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calcula un índice interpretable de presión turística 0-100.

    Pesos definidos en el diseño:
    - 40 % ocupación general.
    - 30 % pernoctaciones por plaza.
    - 20 % tendencia interanual.
    - 10 % ocupación de fin de semana.

    Las variables que no son porcentajes se transforman en
    percentiles mensuales entre provincias.
    """

    result = dataframe.copy()

    result["_occupancy_score"] = (
        result["occupancy_rate_pct"]
        .clip(
            lower=0,
            upper=100,
        )
        .astype("Float64")
    )

    result[
        "_overnight_stays_per_place_score"
    ] = (
        result.groupby(
            "month_id"
        )["overnight_stays_per_place"]
        .rank(
            method="average",
            pct=True,
        )
        * 100
    ).astype("Float64")

    result["_demand_trend_score"] = (
        result.groupby(
            "month_id"
        )["overnight_stays_yoy_change_pct"]
        .rank(
            method="average",
            pct=True,
        )
        * 100
    ).astype("Float64")

    result["_weekend_pressure_score"] = (
        result["weekend_occupancy_rate_pct"]
        .clip(
            lower=0,
            upper=100,
        )
        .astype("Float64")
    )

    result["tourism_pressure_index"] = (
        weighted_available_score(
            result,
            [
                (
                    "_occupancy_score",
                    0.40,
                ),
                (
                    "_overnight_stays_per_place_score",
                    0.30,
                ),
                (
                    "_demand_trend_score",
                    0.20,
                ),
                (
                    "_weekend_pressure_score",
                    0.10,
                ),
            ],
        )
        .round(2)
    )

    result = result.drop(
        columns=[
            "_occupancy_score",
            "_overnight_stays_per_place_score",
            "_demand_trend_score",
            "_weekend_pressure_score",
        ]
    )

    return result


# ---------------------------------------------------------------------
# Campos contextuales todavía no integrados
# ---------------------------------------------------------------------

def add_context_placeholders(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Añade campos previstos en el contrato gold que todavía
    no disponen de una fuente integrada.

    Permanecen como nulos reales. No se inventan datos.
    """

    result = dataframe.copy()

    float_columns = [
        "price_index",
        "price_yoy_change_pct",
        "resident_avg_spend_context",
        "foreign_avg_spend_context",
    ]

    string_columns = [
        "price_source_frequency",
        "price_territory_level",
        "spend_context_frequency",
        "spend_context_territory_level",
        "business_context_frequency",
        "business_context_territory_level",
    ]

    for column in float_columns:
        result[column] = pd.Series(
            pd.NA,
            index=result.index,
            dtype="Float64",
        )

    for column in string_columns:
        result[column] = pd.Series(
            pd.NA,
            index=result.index,
            dtype="string",
        )

    result["demand_source_frequency"] = (
        "monthly"
    )

    return result


# ---------------------------------------------------------------------
# Construcción principal
# ---------------------------------------------------------------------

def build_gold_dataset(
    demand: pd.DataFrame,
    supply: pd.DataFrame,
    territory: pd.DataFrame,
    calendar: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    """Construye el dataset gold completo."""

    demand_snapshot_id = get_single_snapshot_id(
        demand,
        "processed de demanda",
    )

    supply_snapshot_id = get_single_snapshot_id(
        supply,
        "processed de oferta",
    )

    combined_snapshot_id = combine_snapshot_ids(
        demand_snapshot_id,
        supply_snapshot_id,
    )

    pipeline_run_id = str(uuid.uuid4())
    created_at = pd.Timestamp.now(tz="UTC")

    demand_wide = build_demand_wide(demand)
    supply_wide = build_supply_wide(supply)

    gold = build_base_grid(
        territory,
        calendar,
    )

    gold = gold.merge(
        demand_wide,
        on=[
            "territory_id",
            "month_id",
        ],
        how="left",
        validate="one_to_one",
    )

    gold = gold.merge(
        supply_wide,
        on=[
            "territory_id",
            "month_id",
        ],
        how="left",
        validate="one_to_one",
    )

    gold = add_profile_ratios(gold)
    gold = add_temporal_changes(gold)
    gold = add_seasonality_index(gold)
    gold = add_tourism_pressure_index(gold)
    gold = add_context_placeholders(gold)

    gold["data_status"] = pd.Series(
        "unknown",
        index=gold.index,
        dtype="string",
    )

    gold.loc[
        gold["is_provisional"].eq(True),
        "data_status",
    ] = "provisional"

    gold.loc[
        gold["is_provisional"].eq(False),
        "data_status",
    ] = "final_or_not_marked_provisional"

    gold["demand_snapshot_id"] = (
        demand_snapshot_id
    )

    gold["supply_snapshot_id"] = (
        supply_snapshot_id
    )

    gold["source_snapshot_id"] = (
        combined_snapshot_id
    )

    gold["pipeline_run_id"] = (
        pipeline_run_id
    )

    gold["data_version"] = (
        DATA_VERSION
    )

    gold["created_at"] = (
        created_at
    )

    # Un registro gold descriptivo debe disponer de al menos
    # una de las métricas principales de demanda.
    valid_demand_mask = (
        gold[
            [
                "travellers_total",
                "overnight_stays_total",
            ]
        ]
        .notna()
        .any(axis=1)
    )

    removed_rows = int(
        (~valid_demand_mask).sum()
    )

    gold = (
        gold.loc[valid_demand_mask]
        .sort_values(
            [
                "territory_id",
                "date_month",
            ]
        )
        .reset_index(drop=True)
    )

    columns = [
        # Claves y territorio
        "territory_id",
        "source_territory_code",
        "source_territory_name",
        "territory_name",
        "territory_level",
        "autonomous_community_id",
        "autonomous_community_name",
        "province_id",
        "coverage_quality",

        # Tiempo
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

        # Demanda
        "travellers_total",
        "travellers_domestic",
        "travellers_foreign",
        "overnight_stays_total",
        "overnight_stays_domestic",
        "overnight_stays_foreign",
        "average_stay",

        # Oferta, ocupación y empleo
        "establishments_estimated",
        "places_estimated",
        "occupancy_rate_pct",
        "weekend_occupancy_rate_pct",
        "room_occupancy_rate_pct",
        "staff_employed",

        # Variables derivadas
        "domestic_travellers_share",
        "foreign_travellers_share",
        "domestic_overnight_stays_share",
        "foreign_overnight_stays_share",
        "overnight_stays_per_place",
        "travellers_per_establishment",
        "weekend_dependence_index",
        "overnight_stays_mom_change_pct",
        "overnight_stays_yoy_change_pct",
        "seasonality_index",
        "tourism_pressure_index",

        # Contexto futuro
        "price_index",
        "price_yoy_change_pct",
        "resident_avg_spend_context",
        "foreign_avg_spend_context",
        "demand_source_frequency",
        "price_source_frequency",
        "price_territory_level",
        "spend_context_frequency",
        "spend_context_territory_level",
        "business_context_frequency",
        "business_context_territory_level",

        # Calidad y trazabilidad
        "data_status",
        "is_provisional",
        "demand_snapshot_id",
        "supply_snapshot_id",
        "source_snapshot_id",
        "pipeline_run_id",
        "data_version",
        "created_at",
    ]

    return gold[columns], removed_rows


# ---------------------------------------------------------------------
# Validaciones de la capa gold
# ---------------------------------------------------------------------

def validate_gold(
    gold: pd.DataFrame,
) -> None:
    """Aplica controles antes de guardar el resultado."""

    if gold.empty:
        raise ValueError(
            "El dataset gold está vacío."
        )

    duplicate_count = int(
        gold.duplicated(
            subset=[
                "territory_id",
                "month_id",
            ]
        ).sum()
    )

    if duplicate_count:
        raise ValueError(
            "Se encontraron "
            f"{duplicate_count} claves gold duplicadas."
        )

    if gold["territory_id"].isna().any():
        raise ValueError(
            "Existen registros sin territory_id."
        )

    if gold["month_id"].isna().any():
        raise ValueError(
            "Existen registros sin month_id."
        )

    has_main_demand = (
        gold[
            [
                "travellers_total",
                "overnight_stays_total",
            ]
        ]
        .notna()
        .any(axis=1)
    )

    if not has_main_demand.all():
        raise ValueError(
            "Hay filas sin ninguna métrica principal de demanda."
        )

    non_negative_columns = [
        "travellers_total",
        "travellers_domestic",
        "travellers_foreign",
        "overnight_stays_total",
        "overnight_stays_domestic",
        "overnight_stays_foreign",
        "average_stay",
        "establishments_estimated",
        "places_estimated",
        "staff_employed",
        "overnight_stays_per_place",
        "travellers_per_establishment",
    ]

    for column in non_negative_columns:
        negative_count = int(
            gold[column]
            .dropna()
            .lt(0)
            .sum()
        )

        if negative_count:
            raise ValueError(
                f"{column} contiene "
                f"{negative_count} valores negativos."
            )

    percentage_columns = [
        "occupancy_rate_pct",
        "weekend_occupancy_rate_pct",
        "room_occupancy_rate_pct",
        "seasonality_index",
        "tourism_pressure_index",
    ]

    for column in percentage_columns:
        values = gold[column].dropna()

        invalid_count = int(
            (~values.between(0, 100)).sum()
        )

        if invalid_count:
            raise ValueError(
                f"{column} contiene "
                f"{invalid_count} valores fuera de 0-100."
            )

    share_columns = [
        "domestic_travellers_share",
        "foreign_travellers_share",
        "domestic_overnight_stays_share",
        "foreign_overnight_stays_share",
    ]

    for column in share_columns:
        values = gold[column].dropna()

        invalid_count = int(
            (~values.between(0, 1)).sum()
        )

        if invalid_count:
            raise ValueError(
                f"{column} contiene "
                f"{invalid_count} valores fuera de 0-1."
            )

    travellers_complete = (
        gold["travellers_domestic"].notna()
        & gold["travellers_foreign"].notna()
    )

    expected_travellers = (
        gold["travellers_domestic"]
        + gold["travellers_foreign"]
    )

    invalid_traveller_totals = int(
        (
            gold.loc[
                travellers_complete,
                "travellers_total",
            ]
            != expected_travellers.loc[
                travellers_complete
            ]
        ).sum()
    )

    if invalid_traveller_totals:
        raise ValueError(
            "Se detectaron incoherencias en travellers_total."
        )

    overnight_complete = (
        gold["overnight_stays_domestic"].notna()
        & gold["overnight_stays_foreign"].notna()
    )

    expected_overnight_stays = (
        gold["overnight_stays_domestic"]
        + gold["overnight_stays_foreign"]
    )

    invalid_overnight_totals = int(
        (
            gold.loc[
                overnight_complete,
                "overnight_stays_total",
            ]
            != expected_overnight_stays.loc[
                overnight_complete
            ]
        ).sum()
    )

    if invalid_overnight_totals:
        raise ValueError(
            "Se detectaron incoherencias "
            "en overnight_stays_total."
        )


# ---------------------------------------------------------------------
# Escritura
# ---------------------------------------------------------------------

def write_outputs(
    gold: pd.DataFrame,
) -> None:
    """Guarda la versión Parquet y la exportación CSV."""

    GOLD_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    CSV_EXPORT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    gold.to_parquet(
        GOLD_OUTPUT,
        index=False,
        engine="pyarrow",
    )

    csv_export = gold.copy()

    csv_export["date_month"] = (
        pd.to_datetime(
            csv_export["date_month"],
            errors="raise",
        )
        .dt.strftime("%Y-%m-%d")
)

    csv_export["created_at"] = (
        pd.to_datetime(
            csv_export["created_at"],
            errors="raise",
            utc=True,
        )
        .dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    )

    csv_export.to_csv(
        CSV_OUTPUT,
        index=False,
        encoding="utf-8",
    )

    gitkeep_paths = [
        GOLD_DIRECTORY / ".gitkeep",
        CSV_EXPORT_DIRECTORY / ".gitkeep",
    ]

    for gitkeep_path in gitkeep_paths:
        if gitkeep_path.exists():
            gitkeep_path.unlink()


def print_summary(
    gold: pd.DataFrame,
    removed_rows: int,
) -> None:
    """Muestra un resumen de la tabla gold."""

    print("[OK] Dataset gold generado")
    print(f"     Filas: {len(gold):,}")
    print(f"     Columnas: {len(gold.columns)}")
    print(
        "     Territorios: "
        f"{gold['territory_id'].nunique()}"
    )
    print(
        "     Periodo: "
        f"{gold['date_month'].min().date()} "
        "-> "
        f"{gold['date_month'].max().date()}"
    )
    print(
        "     Claves duplicadas: "
        f"{gold.duplicated(['territory_id', 'month_id']).sum()}"
    )
    print(
        "     Filas sin demanda eliminadas: "
        f"{removed_rows:,}"
    )
    print(
        "     Pernoctaciones disponibles: "
        f"{gold['overnight_stays_total'].notna().sum():,}"
    )
    print(
        "     Filas provisionales: "
        f"{gold['is_provisional'].fillna(False).sum():,}"
    )
    print(
        "     Parquet: "
        f"{GOLD_OUTPUT.relative_to(PROJECT_ROOT).as_posix()}"
    )
    print(
        "     CSV: "
        f"{CSV_OUTPUT.relative_to(PROJECT_ROOT).as_posix()}"
    )


# ---------------------------------------------------------------------
# Ejecución principal
# ---------------------------------------------------------------------

def main() -> int:
    """Ejecuta la construcción de la tabla gold."""

    print("=" * 70)
    print("CONSTRUCCIÓN DE LA CAPA GOLD")
    print("=" * 70)
    print()

    try:
        (
            demand,
            supply,
            territory,
            calendar,
        ) = load_inputs()

        gold, removed_rows = build_gold_dataset(
            demand=demand,
            supply=supply,
            territory=territory,
            calendar=calendar,
        )

        validate_gold(gold)
        write_outputs(gold)

        print_summary(
            gold,
            removed_rows,
        )

        print()
        print(
            "[OK] Construcción gold completada "
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