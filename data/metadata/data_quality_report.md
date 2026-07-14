# Informe de calidad de datos

## 1. Resumen de ejecución

- **Dataset:** `data/gold/gold_tourism_demand_monthly.parquet`
- **Fecha de validación UTC:** `2026-07-14T10:54:31.490918+00:00`
- **Estado general:** **PASS**
- **Filas:** `12,693`
- **Columnas:** `64`
- **Territorios:** `50`
- **Periodo:** `2005-01` a `2026-05`
- **Controles superados:** `53`
- **Advertencias:** `0`
- **Controles fallidos:** `0`

## 2. Resultado de los controles

| Control | Estado | Detalle |
| --- | --- | --- |
| dataset_not_empty | PASS | Filas encontradas: 12,693 |
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
| room_occupancy_rate_pct | 1,193 | 9.40 % |
| staff_employed | 0 | 0.00 % |
| overnight_stays_yoy_change_pct | 766 | 6.03 % |
| tourism_pressure_index | 0 | 0.00 % |

## 4. Interpretación

La tabla gold cumple las reglas estructurales, territoriales, temporales, numéricas y de trazabilidad definidas para esta versión.

Las advertencias sobre variables contextuales son esperadas porque las fuentes de precios, gasto y contexto empresarial todavía no se han integrado. Sus valores nulos no representan un error del pipeline actual.

## 5. Alcance del control

Este informe valida la consistencia técnica del dataset, pero no sustituye la interpretación estadística ni la revisión metodológica de las fuentes originales.
