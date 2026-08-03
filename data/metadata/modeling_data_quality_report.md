# Informe de calidad del dataset de modelado

- **Dataset:** `data\gold\gold_modeling_dataset_monthly.parquet`
- **Generado en UTC:** `2026-08-03T09:18:59.955056+00:00`
- **Filas:** 12,691
- **Columnas:** 37
- **Territorios:** 50
- **Periodo:** 2005-01 → 2026-05
- **Resultado:** 53 PASS / 0 WARN / 0 FAIL

## Distribución temporal

| Split | Filas |
| --- | --- |
| provisional_monitoring | 600 |
| test | 600 |
| train | 9691 |
| validation_1 | 600 |
| validation_2 | 600 |
| validation_3 | 600 |

## Indicadores de calidad

| Indicador | Filas |
| --- | --- |
| insufficient_history | 595 |
| missing_lag | 890 |
| ok | 10606 |
| provisional_target | 600 |

## Validaciones

| Control | Estado | Severidad | Detalle |
| --- | --- | --- | --- |
| required_columns | PASS | error | Todas las columnas obligatorias están presentes. |
| modeling_key_not_null | PASS | error | Filas con clave nula: 0. |
| unique_modeling_key | PASS | error | Claves duplicadas: 0. |
| forecast_horizon | PASS | error | Filas con horizonte distinto de 1: 0. |
| territory_count | PASS | error | Territorios observados: 50; esperados: 50. |
| territory_level | PASS | error | Filas con nivel distinto de province: 0. |
| non_negative::target_overnight_stays_total | PASS | error | Valores negativos: 0. |
| non_negative::lag_1_overnight_stays | PASS | error | Valores negativos: 0. |
| non_negative::lag_3_overnight_stays | PASS | error | Valores negativos: 0. |
| non_negative::lag_12_overnight_stays | PASS | error | Valores negativos: 0. |
| non_negative::rolling_mean_3m_overnight_stays | PASS | error | Valores negativos: 0. |
| non_negative::rolling_mean_12m_overnight_stays | PASS | error | Valores negativos: 0. |
| non_negative::lag_1_average_stay | PASS | error | Valores negativos: 0. |
| non_negative::lag_12_average_stay | PASS | error | Valores negativos: 0. |
| non_negative::lag_1_places_estimated | PASS | error | Valores negativos: 0. |
| non_negative::lag_1_establishments_estimated | PASS | error | Valores negativos: 0. |
| non_negative::lag_1_staff_employed | PASS | error | Valores negativos: 0. |
| range_0_100::target_occupancy_rate_pct | PASS | error | Valores fuera de 0-100: 0. |
| range_0_100::lag_1_occupancy_rate_pct | PASS | error | Valores fuera de 0-100: 0. |
| range_0_100::lag_12_occupancy_rate_pct | PASS | error | Valores fuera de 0-100: 0. |
| range_0_100::lag_1_weekend_occupancy_rate_pct | PASS | error | Valores fuera de 0-100: 0. |
| range_0_1::lag_1_domestic_overnight_stays_share | PASS | error | Valores fuera de 0-1: 0. |
| range_0_1::lag_1_foreign_overnight_stays_share | PASS | error | Valores fuera de 0-1: 0. |
| share_pair::lag_1_overnight_stay_shares | PASS | error | Filas completas que no suman 1.0: 0. |
| allowed_evaluation_splits | PASS | error | Todos los splits están permitidos. |
| allowed_data_quality_flags | PASS | error | Todos los indicadores de calidad están permitidos. |
| target_month_matches_date | PASS | error | Filas incoherentes: 0. |
| target_date_first_day | PASS | error | Fechas que no son día 1: 0. |
| split_dates::validation_1 | PASS | error | Filas asignadas incorrectamente: 0. |
| split_dates::validation_2 | PASS | error | Filas asignadas incorrectamente: 0. |
| split_dates::validation_3 | PASS | error | Filas asignadas incorrectamente: 0. |
| split_dates::test | PASS | error | Filas asignadas incorrectamente: 0. |
| split_dates::provisional_monitoring | PASS | error | Filas asignadas incorrectamente: 0. |
| provisional_monitoring_rows | PASS | error | Filas de seguimiento no marcadas como provisionales: 0. |
| provisional_excluded_from_selection | PASS | error | Filas provisionales en validación o test: 0. |
| calendar_lag::lag_1_overnight_stays | PASS | error | Diferencias respecto al mes exacto: 0. |
| calendar_lag::lag_3_overnight_stays | PASS | error | Diferencias respecto al mes exacto: 0. |
| calendar_lag::lag_12_overnight_stays | PASS | error | Diferencias respecto al mes exacto: 0. |
| calendar_lag::lag_1_occupancy_rate_pct | PASS | error | Diferencias respecto al mes exacto: 0. |
| calendar_lag::lag_12_occupancy_rate_pct | PASS | error | Diferencias respecto al mes exacto: 0. |
| calendar_lag::lag_1_average_stay | PASS | error | Diferencias respecto al mes exacto: 0. |
| calendar_lag::lag_12_average_stay | PASS | error | Diferencias respecto al mes exacto: 0. |
| calendar_rolling::rolling_mean_3m_overnight_stays | PASS | error | Diferencias respecto a la ventana calendárica completa: 0. |
| calendar_rolling::rolling_mean_12m_overnight_stays | PASS | error | Diferencias respecto a la ventana calendárica completa: 0. |
| historical_yoy_change | PASS | error | Diferencias respecto a t-1 frente a t-13: 0. |
| traceability_not_null::source_snapshot_id | PASS | error | Valores nulos: 0. |
| traceability_not_null::pipeline_run_id | PASS | error | Valores nulos: 0. |
| traceability_not_null::data_version | PASS | error | Valores nulos: 0. |
| traceability_not_null::created_at | PASS | error | Valores nulos: 0. |
| single_value::source_snapshot_id | PASS | error | Valores únicos observados: 1. |
| single_value::pipeline_run_id | PASS | error | Valores únicos observados: 1. |
| single_value::data_version | PASS | error | Valores únicos observados: 1. |
| single_value::created_at | PASS | error | Valores únicos observados: 1. |
