"""
Construye el dataset gold específico de modelado temporal.

Entrada:
- data/gold/gold_tourism_demand_monthly.parquet
- data/metadata/modeling_config.yml

Salida:
- data/gold/gold_modeling_dataset_monthly.parquet

Granularidad:
- Una fila por territorio, mes objetivo y horizonte de predicción.

Principio central:
- Todas las variables históricas se obtienen por mes calendario real.
  Un hueco temporal produce un valor nulo; nunca se sustituye por la
  fila anterior disponible ni por cero.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_PATH = PROJECT_ROOT / "data" / "metadata" / "modeling_config.yml"
DEFAULT_SOURCE_PATH = (
    PROJECT_ROOT / "data" / "gold" / "gold_tourism_demand_monthly.parquet"
)
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "gold" / "gold_modeling_dataset_monthly.parquet"
)

MODELING_DATA_VERSION = "gold_modeling_dataset_monthly_v1.0.0"


REQUIRED_SOURCE_COLUMNS = {
    "territory_id",
    "territory_name",
    "territory_level",
    "month_id",
    "date_month",
    "year",
    "month",
    "quarter",
    "is_summer",
    "is_christmas_period",
    "covid_period",
    "overnight_stays_total",
    "occupancy_rate_pct",
    "weekend_occupancy_rate_pct",
    "average_stay",
    "domestic_overnight_stays_share",
    "foreign_overnight_stays_share",
    "places_estimated",
    "establishments_estimated",
    "staff_employed",
    "is_provisional",
    "source_snapshot_id",
}


OUTPUT_COLUMNS = [
    "territory_id",
    "territory_name",
    "territory_level",
    "target_month_id",
    "target_date_month",
    "forecast_horizon",
    "target_overnight_stays_total",
    "target_occupancy_rate_pct",
    "year",
    "month",
    "quarter",
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
    "evaluation_split",
    "data_quality_flag",
    "is_provisional",
    "source_snapshot_id",
    "pipeline_run_id",
    "data_version",
    "created_at",
]


def require_file(file_path: Path) -> None:
    """Comprueba que existe un fichero requerido."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"No se encontró el fichero requerido: "
            f"{file_path.relative_to(PROJECT_ROOT)}"
        )


def require_columns(
    dataframe: pd.DataFrame,
    expected_columns: set[str],
    dataset_name: str,
) -> None:
    """Comprueba que un dataframe contiene las columnas necesarias."""
    missing = expected_columns.difference(dataframe.columns)

    if missing:
        raise ValueError(
            f"Faltan columnas en {dataset_name}: "
            + ", ".join(sorted(missing))
        )


def load_config() -> dict:
    """Carga la configuración reproducible de modelado."""
    require_file(CONFIG_PATH)

    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("modeling_config.yml no contiene un objeto YAML válido.")

    return config


def resolve_project_path(path_text: str, default_path: Path) -> Path:
    """Convierte una ruta del YAML en una ruta absoluta del proyecto."""
    if not path_text:
        return default_path

    path = Path(path_text)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def load_source(config: dict) -> pd.DataFrame:
    """Carga la tabla gold descriptiva y comprueba su estructura mínima."""
    source_path = resolve_project_path(
        config.get("source_dataset", {}).get("path", ""),
        DEFAULT_SOURCE_PATH,
    )

    require_file(source_path)

    source = pd.read_parquet(source_path)

    if source.empty:
        raise ValueError("La tabla gold descriptiva está vacía.")

    require_columns(
        source,
        REQUIRED_SOURCE_COLUMNS,
        "gold_tourism_demand_monthly",
    )

    source = source.copy()
    source["date_month"] = pd.to_datetime(source["date_month"])

    duplicate_count = int(
        source.duplicated(
            subset=["territory_id", "date_month"]
        ).sum()
    )

    if duplicate_count:
        raise ValueError(
            "La tabla gold descriptiva contiene "
            f"{duplicate_count} claves territorio-mes duplicadas."
        )

    return source.sort_values(
        ["territory_id", "date_month"]
    ).reset_index(drop=True)


def add_calendar_lag(
    dataframe: pd.DataFrame,
    source: pd.DataFrame,
    source_column: str,
    offset_months: int,
    output_column: str,
) -> pd.DataFrame:
    """
    Añade un lag utilizando una unión por fecha calendario.

    Para una fila objetivo de agosto de 2024 y offset=1 se busca
    exactamente julio de 2024. Si julio no existe, el lag queda nulo.
    """
    lookup = source[
        ["territory_id", "date_month", source_column]
    ].copy()

    lookup["target_date_month"] = (
        lookup["date_month"]
        + pd.DateOffset(months=offset_months)
    )

    lookup = (
        lookup[
            ["territory_id", "target_date_month", source_column]
        ]
        .rename(columns={source_column: output_column})
    )

    if lookup.duplicated(
        subset=["territory_id", "target_date_month"]
    ).any():
        raise ValueError(
            f"La tabla de lookup para {output_column} contiene duplicados."
        )

    return dataframe.merge(
        lookup,
        on=["territory_id", "target_date_month"],
        how="left",
        validate="one_to_one",
    )


def add_calendar_rolling_mean(
    dataframe: pd.DataFrame,
    source: pd.DataFrame,
    source_column: str,
    window_months: int,
    output_column: str,
) -> pd.DataFrame:
    """
    Calcula una media móvil usando exclusivamente meses anteriores.

    La media solo se calcula cuando están presentes todos los meses
    calendario de la ventana. Un hueco produce un valor nulo.
    """
    result = dataframe.copy()
    temporary_columns: list[str] = []

    for offset in range(1, window_months + 1):
        temporary_column = (
            f"__{source_column}_calendar_lag_{offset}"
        )
        temporary_columns.append(temporary_column)

        result = add_calendar_lag(
            result,
            source,
            source_column,
            offset,
            temporary_column,
        )

    complete_window = result[
        temporary_columns
    ].notna().all(axis=1)

    result[output_column] = (
        result[temporary_columns]
        .mean(axis=1, skipna=False)
        .where(complete_window)
        .astype("Float64")
    )

    return result.drop(columns=temporary_columns)


def safe_percentage_change(
    current: pd.Series,
    previous: pd.Series,
) -> pd.Series:
    """Calcula una variación porcentual evitando división por cero."""
    current_numeric = pd.to_numeric(
        current, errors="coerce"
    ).astype("Float64")

    previous_numeric = pd.to_numeric(
        previous, errors="coerce"
    ).astype("Float64")

    valid = (
        current_numeric.notna()
        & previous_numeric.notna()
        & previous_numeric.ne(0)
    )

    result = (
        (current_numeric - previous_numeric)
        / previous_numeric
        * 100
    )

    return result.where(valid).astype("Float64")


def assign_evaluation_split(
    dataframe: pd.DataFrame,
    config: dict,
) -> pd.Series:
    """Asigna train, validaciones, test o seguimiento provisional."""
    dates = dataframe["target_date_month"]

    split = pd.Series(
        "train",
        index=dataframe.index,
        dtype="string",
    )

    validation_config = config["validation"]

    for fold in validation_config["folds"]:
        start = pd.Timestamp(f"{fold['validation_start']}-01")
        end = (
            pd.Timestamp(f"{fold['validation_end']}-01")
            + pd.offsets.MonthEnd(0)
        )

        split.loc[dates.between(start, end)] = fold["name"]

    final_test = validation_config["final_test"]
    test_start = pd.Timestamp(f"{final_test['start']}-01")
    test_end = (
        pd.Timestamp(f"{final_test['end']}-01")
        + pd.offsets.MonthEnd(0)
    )
    split.loc[dates.between(test_start, test_end)] = "test"

    provisional = validation_config["provisional_monitoring"]
    provisional_start = pd.Timestamp(f"{provisional['start']}-01")
    provisional_end = (
        pd.Timestamp(f"{provisional['end']}-01")
        + pd.offsets.MonthEnd(0)
    )
    split.loc[
        dates.between(provisional_start, provisional_end)
    ] = "provisional_monitoring"

    return split


def assign_data_quality_flag(
    dataframe: pd.DataFrame,
    source: pd.DataFrame,
) -> pd.Series:
    """Clasifica la aptitud temporal básica de cada fila."""
    flags = pd.Series(
        "ok",
        index=dataframe.index,
        dtype="string",
    )

    target_missing = dataframe[
        "target_overnight_stays_total"
    ].isna()
    flags.loc[target_missing] = "missing_target"

    first_observation = (
        source.groupby("territory_id")["date_month"]
        .min()
        .rename("first_observation")
    )

    first_observation_for_row = dataframe[
        "territory_id"
    ].map(first_observation)

    has_twelve_months_possible = (
        dataframe["target_date_month"]
        >= (
            first_observation_for_row
            + pd.DateOffset(months=12)
        )
    )

    core_history_columns = [
        "lag_1_overnight_stays",
        "lag_3_overnight_stays",
        "lag_12_overnight_stays",
        "rolling_mean_3m_overnight_stays",
        "rolling_mean_12m_overnight_stays",
    ]

    core_history_missing = dataframe[
        core_history_columns
    ].isna().any(axis=1)

    insufficient_history = (
        ~target_missing
        & ~has_twelve_months_possible
    )
    flags.loc[insufficient_history] = "insufficient_history"

    missing_lag = (
        ~target_missing
        & has_twelve_months_possible
        & core_history_missing
    )
    flags.loc[missing_lag] = "missing_lag"

    provisional = (
        ~target_missing
        & dataframe["is_provisional"].fillna(False)
    )
    flags.loc[provisional] = "provisional_target"

    return flags


def cast_output_types(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Aplica tipos estables antes de escribir el Parquet."""
    result = dataframe.copy()

    string_columns = [
        "territory_id",
        "territory_name",
        "territory_level",
        "target_month_id",
        "evaluation_split",
        "data_quality_flag",
        "source_snapshot_id",
        "pipeline_run_id",
        "data_version",
    ]

    for column in string_columns:
        result[column] = result[column].astype("string")

    result["forecast_horizon"] = (
        pd.to_numeric(
            result["forecast_horizon"],
            errors="raise",
        ).astype("int8")
    )

    result["year"] = pd.to_numeric(
        result["year"], errors="raise"
    ).astype("int16")

    result["month"] = pd.to_numeric(
        result["month"], errors="raise"
    ).astype("int8")

    result["quarter"] = pd.to_numeric(
        result["quarter"], errors="raise"
    ).astype("int8")

    boolean_columns = [
        "is_summer",
        "is_christmas_period",
        "covid_period",
        "is_provisional",
    ]

    for column in boolean_columns:
        result[column] = result[column].astype("boolean")

    float_columns = [
        "target_overnight_stays_total",
        "target_occupancy_rate_pct",
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

    for column in float_columns:
        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        ).astype("Float64")

    result["target_date_month"] = pd.to_datetime(
        result["target_date_month"]
    )
    result["created_at"] = pd.to_datetime(
        result["created_at"],
        utc=True,
    )

    return result


def validate_modeling_dataset(dataframe: pd.DataFrame) -> None:
    """Aplica controles mínimos antes de guardar el dataset."""
    if dataframe.empty:
        raise ValueError("El dataset de modelado está vacío.")

    require_columns(
        dataframe,
        set(OUTPUT_COLUMNS),
        "gold_modeling_dataset_monthly",
    )

    duplicate_count = int(
        dataframe.duplicated(
            subset=[
                "territory_id",
                "target_month_id",
                "forecast_horizon",
            ]
        ).sum()
    )

    if duplicate_count:
        raise ValueError(
            "El dataset de modelado contiene "
            f"{duplicate_count} claves duplicadas."
        )

    if dataframe[
        [
            "territory_id",
            "target_month_id",
            "target_date_month",
            "forecast_horizon",
        ]
    ].isna().any().any():
        raise ValueError("Existen claves de modelado nulas.")

    if not dataframe["forecast_horizon"].eq(1).all():
        raise ValueError(
            "Se encontraron horizontes diferentes de 1."
        )

    invalid_territory_level = (
        dataframe["territory_level"]
        .dropna()
        .ne("province")
        .any()
    )
    if invalid_territory_level:
        raise ValueError(
            "Se encontraron niveles territoriales distintos de province."
        )

    expected_month_id = (
        dataframe["target_date_month"]
        .dt.strftime("%Y-%m")
    )

    if not dataframe["target_month_id"].eq(
        expected_month_id
    ).all():
        raise ValueError(
            "target_month_id no coincide con target_date_month."
        )

    target_values = dataframe[
        "target_overnight_stays_total"
    ].dropna()

    if target_values.lt(0).any():
        raise ValueError(
            "La variable objetivo contiene valores negativos."
        )

    allowed_splits = {
        "train",
        "validation_1",
        "validation_2",
        "validation_3",
        "test",
        "provisional_monitoring",
    }

    unexpected_splits = set(
        dataframe["evaluation_split"].dropna().unique()
    ).difference(allowed_splits)

    if unexpected_splits:
        raise ValueError(
            "Se encontraron evaluation_split no permitidos: "
            + ", ".join(sorted(unexpected_splits))
        )


def build_modeling_dataset(
    source: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """Construye todas las filas y features del dataset de modelado."""
    horizon = int(
        config["problem"]["forecast_horizon_months"]
    )

    if horizon != 1:
        raise ValueError(
            "Esta primera versión solo implementa forecast_horizon = 1."
        )

    modeling = source[
        [
            "territory_id",
            "territory_name",
            "territory_level",
            "month_id",
            "date_month",
            "year",
            "month",
            "quarter",
            "is_summer",
            "is_christmas_period",
            "covid_period",
            "overnight_stays_total",
            "occupancy_rate_pct",
            "is_provisional",
            "source_snapshot_id",
        ]
    ].copy()

    modeling = modeling.rename(
        columns={
            "month_id": "target_month_id",
            "date_month": "target_date_month",
            "overnight_stays_total": (
                "target_overnight_stays_total"
            ),
            "occupancy_rate_pct": (
                "target_occupancy_rate_pct"
            ),
        }
    )

    modeling["forecast_horizon"] = horizon

    lag_specs = [
        ("overnight_stays_total", 1, "lag_1_overnight_stays"),
        ("overnight_stays_total", 3, "lag_3_overnight_stays"),
        ("overnight_stays_total", 12, "lag_12_overnight_stays"),
        ("occupancy_rate_pct", 1, "lag_1_occupancy_rate_pct"),
        ("occupancy_rate_pct", 12, "lag_12_occupancy_rate_pct"),
        (
            "weekend_occupancy_rate_pct",
            1,
            "lag_1_weekend_occupancy_rate_pct",
        ),
        ("average_stay", 1, "lag_1_average_stay"),
        ("average_stay", 12, "lag_12_average_stay"),
        (
            "domestic_overnight_stays_share",
            1,
            "lag_1_domestic_overnight_stays_share",
        ),
        (
            "foreign_overnight_stays_share",
            1,
            "lag_1_foreign_overnight_stays_share",
        ),
        ("places_estimated", 1, "lag_1_places_estimated"),
        (
            "establishments_estimated",
            1,
            "lag_1_establishments_estimated",
        ),
        ("staff_employed", 1, "lag_1_staff_employed"),
    ]

    for source_column, offset, output_column in lag_specs:
        modeling = add_calendar_lag(
            modeling,
            source,
            source_column,
            offset,
            output_column,
        )

    modeling = add_calendar_rolling_mean(
        modeling,
        source,
        "overnight_stays_total",
        3,
        "rolling_mean_3m_overnight_stays",
    )

    modeling = add_calendar_rolling_mean(
        modeling,
        source,
        "overnight_stays_total",
        12,
        "rolling_mean_12m_overnight_stays",
    )

    modeling = add_calendar_lag(
        modeling,
        source,
        "overnight_stays_total",
        13,
        "__lag_13_overnight_stays",
    )

    modeling["yoy_change_overnight_stays"] = (
        safe_percentage_change(
            modeling["lag_1_overnight_stays"],
            modeling["__lag_13_overnight_stays"],
        )
    )

    modeling = modeling.drop(
        columns=["__lag_13_overnight_stays"]
    )

    modeling["evaluation_split"] = assign_evaluation_split(
        modeling,
        config,
    )

    modeling["data_quality_flag"] = assign_data_quality_flag(
        modeling,
        source,
    )

    modeling["pipeline_run_id"] = str(uuid.uuid4())
    modeling["data_version"] = MODELING_DATA_VERSION
    modeling["created_at"] = pd.Timestamp.now(tz="UTC")

    modeling = cast_output_types(modeling)

    modeling = (
        modeling[OUTPUT_COLUMNS]
        .sort_values(
            ["territory_id", "target_date_month"]
        )
        .reset_index(drop=True)
    )

    validate_modeling_dataset(modeling)

    return modeling


def write_output(
    dataframe: pd.DataFrame,
    config: dict,
) -> Path:
    """Guarda el dataset de modelado en Parquet."""
    output_path = resolve_project_path(
        config.get("modeling_dataset", {}).get(
            "path", ""
        ),
        DEFAULT_OUTPUT_PATH,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_parquet(
        output_path,
        index=False,
    )

    return output_path


def main() -> int:
    """Ejecuta la construcción completa."""
    config = load_config()
    source = load_source(config)
    modeling = build_modeling_dataset(source, config)
    output_path = write_output(modeling, config)

    split_counts = (
        modeling["evaluation_split"]
        .value_counts(dropna=False)
        .sort_index()
    )

    quality_counts = (
        modeling["data_quality_flag"]
        .value_counts(dropna=False)
        .sort_index()
    )

    print("Dataset de modelado construido correctamente.")
    print(
        f"Salida: {output_path.relative_to(PROJECT_ROOT)}"
    )
    print(f"Filas: {len(modeling):,}")
    print(f"Columnas: {len(modeling.columns)}")
    print(
        "Periodo: "
        f"{modeling['target_month_id'].min()} "
        f"a {modeling['target_month_id'].max()}"
    )
    print(
        "Territorios: "
        f"{modeling['territory_id'].nunique()}"
    )
    print("\nSplits:")
    print(split_counts.to_string())
    print("\nCalidad:")
    print(quality_counts.to_string())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())