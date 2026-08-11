# Selección reproducible de modelos mediante validación temporal

- **Generado en UTC:** `2026-08-11T12:46:18.184332+00:00`
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

## Disponibilidad point-in-time de etiquetas de entrenamiento

Ridge y HGB aplican el mismo purge mensual. El límite estructural del fold
no presupone que una etiqueta EOTR ya esté publicada: el límite de
disponibilidad es `validation_start - minimum_safe_eotr_lag_months` y el
límite efectivo es el más restrictivo de ambos. El baseline lag-12 no se
entrena y no depende de este purge.

| Fold | Inicio validación | Fin estructural train | Fin por disponibilidad | Fin efectivo train | Máxima etiqueta usada | Filas antes | Filas después | Filas purgadas |
|---|---|---|---|---|---|---:|---:|---:|
| validation_1 | 2021-06 | 2021-05 | 2021-03 | 2021-03 | 2021-03 | 8987 | 8987 | 0 |
| validation_2 | 2022-06 | 2022-05 | 2022-03 | 2022-03 | 2022-03 | 9537 | 9437 | 100 |
| validation_3 | 2023-06 | 2023-05 | 2023-03 | 2023-03 | 2023-03 | 10137 | 10037 | 100 |

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
| ridge_alpha_1 | 5.164,60 | 10.298,30 | 24,90 % |
| ridge_alpha_0_1 | 5.164,69 | 10.296,95 | 24,90 % |
| ridge_alpha_0_01 | 5.164,70 | 10.296,82 | 24,90 % |
| ridge_alpha_10 | 5.165,14 | 10.311,37 | 24,90 % |
| ridge_alpha_100 | 5.204,43 | 10.409,75 | 25,09 % |
| ridge_alpha_1000 | 5.605,25 | 11.165,26 | 27,02 % |

Mejor Ridge: `ridge_alpha_1`.

- MAE: 5.164,60
- Mejora frente al baseline: -2,91 %

## HistGradientBoosting congelado

Configuración evaluada: `hgb_raw_02`.

- MAE: 6.852,14
- RMSE: 12.755,40
- WAPE: 33,03 %
- Sesgo medio: -1.773,75
- Mejora frente al baseline: -36,53 %

## Decisión de validación

- Mejor candidato de machine learning: `ridge_alpha_1`
- Mejora agregada del mejor candidato: -2,91 %
- Umbral mínimo configurado: 5,00 %
- Solución seleccionada tras validación: `seasonal_naive_lag_12`

El umbral pooled de mejora igual o superior al 5,00 %
es la gate automática. La estabilidad entre folds se interpreta como
diagnóstico: ante evidencia inestable o insuficiente se prefiere el baseline,
sin consultar el test. El conjunto de test permanece excluido de este proceso.
