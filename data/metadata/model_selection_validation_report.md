# Selección reproducible de modelos mediante validación temporal

- **Generado en UTC:** `2026-08-11T11:54:50.719338+00:00`
- **Dataset:** `data/gold/gold_modeling_dataset_monthly.parquet`
- **Estrategia:** validación temporal expansiva
- **Splits de selección:** validation_1, validation_2 y validation_3
- **Test final utilizado en selección:** no
- **Estado de la ventana de test:** `already_opened`
- **Filas comparables agregadas:** 1750
- **Predicciones negativas:** recortadas a cero
- **Forecast origin:** `end_of_month_t_before_target_t_plus_1`
- **Desfase EOTR mínimo seguro:** 3 meses
- **Predictores operacionales:** `year`, `is_summer`, `is_christmas_period`, `lag_3_overnight_stays`, `lag_12_overnight_stays`, `lag_12_occupancy_rate_pct`, `lag_12_average_stay`, `territory_id`, `month`, `quarter`

## Alcance reproducido

La selección ejecutable compara el baseline estacional, la rejilla
documentada de Ridge y la configuración congelada `hgb_raw_02`.

El repositorio histórico no conserva la rejilla completa de
configuraciones HGB mencionada en la memoria. Por ello este script no
inventa configuraciones adicionales ni afirma reproducir una búsqueda
HGB que no quedó registrada.

## Baseline estacional lag-12

| Métrica | Resultado |
|---|---:|
| MAE | 5.018,64 |
| RMSE | 10.411,37 |
| WAPE | 24,19 % |
| Sesgo medio | -3.268,00 |

## Búsqueda de Ridge

| Configuración | MAE | RMSE | WAPE |
|---|---:|---:|---:|
| ridge_alpha_0_01 | 5.175,99 | 10.290,46 | 24,95 % |
| ridge_alpha_0_1 | 5.176,00 | 10.290,60 | 24,95 % |
| ridge_alpha_1 | 5.176,16 | 10.291,96 | 24,95 % |
| ridge_alpha_10 | 5.178,84 | 10.305,34 | 24,96 % |
| ridge_alpha_100 | 5.230,48 | 10.406,40 | 25,21 % |
| ridge_alpha_1000 | 5.650,11 | 11.157,53 | 27,24 % |

Mejor Ridge: `ridge_alpha_0_01`.

- MAE: 5.175,99
- Mejora frente al baseline: -3,14 %

## HistGradientBoosting congelado

Configuración evaluada: `hgb_raw_02`.

- MAE: 7.349,42
- RMSE: 12.967,02
- WAPE: 35,43 %
- Sesgo medio: -1.024,74
- Mejora frente al baseline: -46,44 %

## Decisión de validación

- Mejor candidato de machine learning: `ridge_alpha_0_01`
- Mejora agregada del mejor candidato: -3,14 %
- Umbral mínimo configurado: 5,00 %
- Solución seleccionada tras validación: `seasonal_naive_lag_12`

El conjunto de test permanece excluido de este proceso.
