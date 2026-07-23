# Informe de calidad de datos

## 1. Resumen de ejecución

- **Dataset:** `data/gold/gold_tourism_demand_monthly.parquet`
- **Fecha de validación UTC:** `2026-07-23T12:25:34.596015+00:00`
- **Estado general:** **PASS**
- **Filas:** `12,691`
- **Columnas:** `64`
- **Territorios:** `50`
- **Periodo:** `2005-01` a `2026-05`
- **Controles superados:** `165`
- **Advertencias:** `1`
- **Controles fallidos:** `0`

## 2. Resultado de los controles

| Control | Estado | Detalle |
| --- | --- | --- |
| schema_column_count | PASS | Esperadas: 64; encontradas: 64 |
| schema_columns_match | PASS | Las columnas coinciden con el contrato. |
| schema_column_order | PASS | El orden de las columnas coincide con el contrato. |
| schema_dtype::territory_id | PASS | Esperado: string; encontrado: string |
| schema_not_null::territory_id | PASS | Valores nulos: 0 |
| schema_dtype::source_territory_code | PASS | Esperado: string; encontrado: string |
| schema_not_null::source_territory_code | PASS | Valores nulos: 0 |
| schema_dtype::source_territory_name | PASS | Esperado: string; encontrado: string |
| schema_not_null::source_territory_name | PASS | Valores nulos: 0 |
| schema_dtype::territory_name | PASS | Esperado: string; encontrado: string |
| schema_not_null::territory_name | PASS | Valores nulos: 0 |
| schema_dtype::territory_level | PASS | Esperado: string; encontrado: string |
| schema_not_null::territory_level | PASS | Valores nulos: 0 |
| schema_allowed_values::territory_level | PASS | Todos los valores están permitidos. |
| schema_dtype::autonomous_community_id | PASS | Esperado: string; encontrado: string |
| schema_not_null::autonomous_community_id | PASS | Valores nulos: 0 |
| schema_dtype::autonomous_community_name | PASS | Esperado: string; encontrado: string |
| schema_not_null::autonomous_community_name | PASS | Valores nulos: 0 |
| schema_dtype::province_id | PASS | Esperado: string; encontrado: string |
| schema_not_null::province_id | PASS | Valores nulos: 0 |
| schema_dtype::coverage_quality | PASS | Esperado: string; encontrado: string |
| schema_not_null::coverage_quality | PASS | Valores nulos: 0 |
| schema_allowed_values::coverage_quality | PASS | Todos los valores están permitidos. |
| schema_dtype::month_id | PASS | Esperado: string; encontrado: string |
| schema_not_null::month_id | PASS | Valores nulos: 0 |
| schema_dtype::date_month | PASS | Esperado: datetime64[ns]; encontrado: datetime64[ns] |
| schema_not_null::date_month | PASS | Valores nulos: 0 |
| schema_dtype::year | PASS | Esperado: Int16; encontrado: Int16 |
| schema_not_null::year | PASS | Valores nulos: 0 |
| schema_dtype::month | PASS | Esperado: Int8; encontrado: Int8 |
| schema_not_null::month | PASS | Valores nulos: 0 |
| schema_range::month | PASS | Por debajo del mínimo: 0; por encima del máximo: 0 |
| schema_dtype::month_name | PASS | Esperado: string; encontrado: string |
| schema_not_null::month_name | PASS | Valores nulos: 0 |
| schema_dtype::quarter | PASS | Esperado: Int8; encontrado: Int8 |
| schema_not_null::quarter | PASS | Valores nulos: 0 |
| schema_range::quarter | PASS | Por debajo del mínimo: 0; por encima del máximo: 0 |
| schema_dtype::season | PASS | Esperado: string; encontrado: string |
| schema_not_null::season | PASS | Valores nulos: 0 |
| schema_allowed_values::season | PASS | Todos los valores están permitidos. |
| schema_dtype::is_summer | PASS | Esperado: boolean; encontrado: boolean |
| schema_not_null::is_summer | PASS | Valores nulos: 0 |
| schema_dtype::is_christmas_period | PASS | Esperado: boolean; encontrado: boolean |
| schema_not_null::is_christmas_period | PASS | Valores nulos: 0 |
| schema_dtype::is_easter_period | PASS | Esperado: boolean; encontrado: boolean |
| schema_dtype::covid_period | PASS | Esperado: boolean; encontrado: boolean |
| schema_not_null::covid_period | PASS | Valores nulos: 0 |
| schema_dtype::complete_month_available | PASS | Esperado: boolean; encontrado: boolean |
| schema_not_null::complete_month_available | PASS | Valores nulos: 0 |
| schema_dtype::travellers_total | PASS | Esperado: Int64; encontrado: Int64 |
| schema_dtype::travellers_domestic | PASS | Esperado: Int64; encontrado: Int64 |
| schema_dtype::travellers_foreign | PASS | Esperado: Int64; encontrado: Int64 |
| schema_dtype::overnight_stays_total | PASS | Esperado: Int64; encontrado: Int64 |
| schema_dtype::overnight_stays_domestic | PASS | Esperado: Int64; encontrado: Int64 |
| schema_dtype::overnight_stays_foreign | PASS | Esperado: Int64; encontrado: Int64 |
| schema_dtype::average_stay | PASS | Esperado: Float64; encontrado: Float64 |
| schema_dtype::establishments_estimated | PASS | Esperado: Float64; encontrado: Float64 |
| schema_dtype::places_estimated | PASS | Esperado: Float64; encontrado: Float64 |
| schema_dtype::occupancy_rate_pct | PASS | Esperado: Float64; encontrado: Float64 |
| schema_range::occupancy_rate_pct | PASS | Por debajo del mínimo: 0; por encima del máximo: 0 |
| schema_dtype::weekend_occupancy_rate_pct | PASS | Esperado: Float64; encontrado: Float64 |
| schema_range::weekend_occupancy_rate_pct | PASS | Por debajo del mínimo: 0; por encima del máximo: 0 |
| schema_dtype::room_occupancy_rate_pct | PASS | Esperado: Float64; encontrado: Float64 |
| schema_range::room_occupancy_rate_pct | PASS | Por debajo del mínimo: 0; por encima del máximo: 0 |
| schema_dtype::staff_employed | PASS | Esperado: Float64; encontrado: Float64 |
| schema_dtype::domestic_travellers_share | PASS | Esperado: Float64; encontrado: Float64 |
| schema_range::domestic_travellers_share | PASS | Por debajo del mínimo: 0; por encima del máximo: 0 |
| schema_dtype::foreign_travellers_share | PASS | Esperado: Float64; encontrado: Float64 |
| schema_range::foreign_travellers_share | PASS | Por debajo del mínimo: 0; por encima del máximo: 0 |
| schema_dtype::domestic_overnight_stays_share | PASS | Esperado: Float64; encontrado: Float64 |
| schema_range::domestic_overnight_stays_share | PASS | Por debajo del mínimo: 0; por encima del máximo: 0 |
| schema_dtype::foreign_overnight_stays_share | PASS | Esperado: Float64; encontrado: Float64 |
| schema_range::foreign_overnight_stays_share | PASS | Por debajo del mínimo: 0; por encima del máximo: 0 |
| schema_dtype::overnight_stays_per_place | PASS | Esperado: Float64; encontrado: Float64 |
| schema_dtype::travellers_per_establishment | PASS | Esperado: Float64; encontrado: Float64 |
| schema_dtype::weekend_dependence_index | PASS | Esperado: Float64; encontrado: Float64 |
| schema_dtype::overnight_stays_mom_change_pct | PASS | Esperado: Float64; encontrado: Float64 |
| schema_dtype::overnight_stays_yoy_change_pct | PASS | Esperado: Float64; encontrado: Float64 |
| schema_dtype::seasonality_index | PASS | Esperado: Float64; encontrado: Float64 |
| schema_range::seasonality_index | PASS | Por debajo del mínimo: 0; por encima del máximo: 0 |
| schema_dtype::tourism_pressure_index | PASS | Esperado: Float64; encontrado: Float64 |
| schema_range::tourism_pressure_index | PASS | Por debajo del mínimo: 0; por encima del máximo: 0 |
| schema_dtype::price_index | PASS | Esperado: Float64; encontrado: Float64 |
| schema_dtype::price_yoy_change_pct | PASS | Esperado: Float64; encontrado: Float64 |
| schema_dtype::resident_avg_spend_context | PASS | Esperado: Float64; encontrado: Float64 |
| schema_dtype::foreign_avg_spend_context | PASS | Esperado: Float64; encontrado: Float64 |
| schema_dtype::demand_source_frequency | PASS | Esperado: string; encontrado: string |
| schema_not_null::demand_source_frequency | PASS | Valores nulos: 0 |
| schema_allowed_values::demand_source_frequency | PASS | Todos los valores están permitidos. |
| schema_dtype::price_source_frequency | PASS | Esperado: string; encontrado: string |
| schema_dtype::price_territory_level | PASS | Esperado: string; encontrado: string |
| schema_dtype::spend_context_frequency | PASS | Esperado: string; encontrado: string |
| schema_dtype::spend_context_territory_level | PASS | Esperado: string; encontrado: string |
| schema_dtype::business_context_frequency | PASS | Esperado: string; encontrado: string |
| schema_dtype::business_context_territory_level | PASS | Esperado: string; encontrado: string |
| schema_dtype::data_status | PASS | Esperado: string; encontrado: string |
| schema_not_null::data_status | PASS | Valores nulos: 0 |
| schema_allowed_values::data_status | PASS | Todos los valores están permitidos. |
| schema_dtype::is_provisional | PASS | Esperado: boolean; encontrado: boolean |
| schema_not_null::is_provisional | PASS | Valores nulos: 0 |
| schema_dtype::demand_snapshot_id | PASS | Esperado: string; encontrado: string |
| schema_not_null::demand_snapshot_id | PASS | Valores nulos: 0 |
| schema_dtype::supply_snapshot_id | PASS | Esperado: string; encontrado: string |
| schema_not_null::supply_snapshot_id | PASS | Valores nulos: 0 |
| schema_dtype::source_snapshot_id | PASS | Esperado: string; encontrado: string |
| schema_not_null::source_snapshot_id | PASS | Valores nulos: 0 |
| schema_dtype::pipeline_run_id | PASS | Esperado: string; encontrado: string |
| schema_not_null::pipeline_run_id | PASS | Valores nulos: 0 |
| schema_dtype::data_version | PASS | Esperado: string; encontrado: string |
| schema_not_null::data_version | PASS | Valores nulos: 0 |
| schema_dtype::created_at | PASS | Esperado: datetime64[ns, UTC]; encontrado: datetime64[ns, UTC] |
| schema_not_null::created_at | PASS | Valores nulos: 0 |
| dataset_not_empty | PASS | Filas encontradas: 12,691 |
| required_columns | PASS | Todas las columnas obligatorias están presentes. |
| key_not_null | PASS | Filas con claves nulas: 0 |
| key_uniqueness | PASS | Claves duplicadas: 0 |
| territory_count | PASS | Esperados: 50; encontrados: 50 |
| territory_level | PASS | Niveles encontrados: province |
| valid_month_dates | PASS | Fechas mensuales no válidas: 0 |
| minimum_start_month | PASS | Esperado: 2005-01-01; encontrado: 2005-01-01 |
| month_id_consistency | PASS | Filas donde month_id no coincide con date_month: 0 |
| global_month_continuity | PASS | Ausencias globales documentadas en la fuente: 2020-04, 2020-05, 2020-11 |
| non_negative::travellers_total | PASS | Valores negativos: 0 |
| non_negative::travellers_domestic | PASS | Valores negativos: 0 |
| non_negative::travellers_foreign | PASS | Valores negativos: 0 |
| non_negative::overnight_stays_total | PASS | Valores negativos: 0 |
| non_negative::overnight_stays_domestic | PASS | Valores negativos: 0 |
| non_negative::overnight_stays_foreign | PASS | Valores negativos: 0 |
| non_negative::average_stay | PASS | Valores negativos: 0 |
| non_negative::establishments_estimated | PASS | Valores negativos: 0 |
| non_negative::places_estimated | PASS | Valores negativos: 0 |
| non_negative::staff_employed | PASS | Valores negativos: 0 |
| non_negative::overnight_stays_per_place | PASS | Valores negativos: 0 |
| non_negative::travellers_per_establishment | PASS | Valores negativos: 0 |
| range_0_100::occupancy_rate_pct | PASS | Valores fuera del rango 0-100: 0 |
| range_0_100::weekend_occupancy_rate_pct | PASS | Valores fuera del rango 0-100: 0 |
| range_0_100::room_occupancy_rate_pct | PASS | Valores fuera del rango 0-100: 0 |
| range_0_100::seasonality_index | PASS | Valores fuera del rango 0-100: 0 |
| range_0_100::tourism_pressure_index | PASS | Valores fuera del rango 0-100: 0 |
| range_0_1::domestic_travellers_share | PASS | Valores fuera del rango 0-1: 0 |
| range_0_1::foreign_travellers_share | PASS | Valores fuera del rango 0-1: 0 |
| range_0_1::domestic_overnight_stays_share | PASS | Valores fuera del rango 0-1: 0 |
| range_0_1::foreign_overnight_stays_share | PASS | Valores fuera del rango 0-1: 0 |
| share_sum::traveller_shares | PASS | Filas completas que no suman 1: 0 |
| share_sum::overnight_stay_shares | PASS | Filas completas que no suman 1: 0 |
| total_consistency::travellers_total | PASS | Filas completas con total incoherente: 0 |
| total_consistency::overnight_stays_total | PASS | Filas completas con total incoherente: 0 |
| average_stay_consistency | PASS | Filas con estancia media incoherente: 0 |
| provisional_period | PASS | Filas con clasificación provisional incoherente: 0 |
| data_status_consistency | PASS | Filas con data_status incoherente: 0 |
| single_value::source_snapshot_id | PASS | Valores distintos encontrados: 1 |
| single_value::pipeline_run_id | PASS | Valores distintos encontrados: 1 |
| single_value::data_version | PASS | Valores distintos encontrados: 1 |
| territory_referential_integrity | PASS | Todos los territorios existen en dim_territory. |
| calendar_referential_integrity | PASS | Todos los meses existen en dim_calendar_month. |
| context_pending::price_index | PASS | Valores no nulos encontrados: 0. La fuente contextual todavía no está integrada. |
| context_pending::price_yoy_change_pct | PASS | Valores no nulos encontrados: 0. La fuente contextual todavía no está integrada. |
| context_pending::resident_avg_spend_context | PASS | Valores no nulos encontrados: 0. La fuente contextual todavía no está integrada. |
| context_pending::foreign_avg_spend_context | PASS | Valores no nulos encontrados: 0. La fuente contextual todavía no está integrada. |
| context_pending::price_source_frequency | PASS | Valores no nulos encontrados: 0. La fuente contextual todavía no está integrada. |
| context_pending::price_territory_level | PASS | Valores no nulos encontrados: 0. La fuente contextual todavía no está integrada. |
| context_pending::spend_context_frequency | PASS | Valores no nulos encontrados: 0. La fuente contextual todavía no está integrada. |
| context_pending::spend_context_territory_level | PASS | Valores no nulos encontrados: 0. La fuente contextual todavía no está integrada. |
| context_pending::business_context_frequency | PASS | Valores no nulos encontrados: 0. La fuente contextual todavía no está integrada. |
| context_pending::business_context_territory_level | PASS | Valores no nulos encontrados: 0. La fuente contextual todavía no está integrada. |
| territory_month_coverage | WARN | Combinaciones provincia-mes ausentes: 9. Detalle disponible en `data/metadata/missing_territory_months.csv`. |

## 3. Valores nulos en variables principales

| Variable | Nulos | Porcentaje |
| --- | --- | --- |
| travellers_total | 0 | 0.00 % |
| overnight_stays_total | 0 | 0.00 % |
| average_stay | 16 | 0.13 % |
| establishments_estimated | 0 | 0.00 % |
| places_estimated | 0 | 0.00 % |
| occupancy_rate_pct | 0 | 0.00 % |
| weekend_occupancy_rate_pct | 0 | 0.00 % |
| room_occupancy_rate_pct | 1,191 | 9.38 % |
| staff_employed | 0 | 0.00 % |
| overnight_stays_yoy_change_pct | 768 | 6.05 % |
| tourism_pressure_index | 0 | 0.00 % |

## 4. Interpretación

La tabla gold cumple las reglas estructurales, territoriales, temporales, numéricas y de trazabilidad definidas para esta versión.

Las ausencias esperadas de las variables contextuales se registran como controles superados porque las fuentes de precios, gasto y contexto empresarial todavía no se han integrado. Sus valores nulos no representan un error del pipeline actual.

## 5. Alcance del control

Este informe valida la consistencia técnica del dataset, pero no sustituye la interpretación estadística ni la revisión metodológica de las fuentes originales.
