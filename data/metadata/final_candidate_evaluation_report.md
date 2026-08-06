# Evaluación final del candidato HistGradientBoosting

- **Dataset:** `data\gold\gold_modeling_dataset_monthly.parquet`
- **Candidato:** `hgb_raw_02`
- **Baseline:** `seasonal_naive_lag_12`
- **Generado en UTC:** `2026-08-06T17:29:45.846839+00:00`
- **Filas de test:** 600
- **Periodo de test:** 2024-06 → 2025-05
- **Territorios de test:** 50

## Configuración congelada

| Parámetro | Valor |
|---|---:|
| target_transform | raw |
| learning_rate | 0,05 |
| max_iter | 300 |
| max_leaf_nodes | 31 |
| min_samples_leaf | 20 |
| l2_regularization | 1,0 |
| early_stopping | False |
| random_state | 42 |

La configuración fue seleccionada exclusivamente con las tres ventanas de
validación. El test final no se utilizó para modificar hiperparámetros,
variables ni transformaciones.

## Comparación temporal

| evaluation_split | baseline_MAE | model_MAE | mae_improvement_pct | baseline_RMSE | model_RMSE | baseline_WAPE_pct | model_WAPE_pct | baseline_mean_bias | model_mean_bias |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| validation_1 | 9,116.56 | 8,081.94 | 11.35 | 16,454.67 | 16,089.15 | 44.60 | 39.54 | -8,782.89 | -7,810.56 |
| validation_2 | 3,071.71 | 4,133.83 | -34.58 | 6,094.11 | 6,738.02 | 14.91 | 20.07 | -909.25 | 3,260.49 |
| validation_3 | 3,209.14 | 3,404.34 | -6.08 | 5,552.10 | 6,234.74 | 15.16 | 16.08 | -571.43 | 33.21 |
| test | 3,045.00 | 2,760.59 | 9.34 | 5,236.82 | 4,550.60 | 14.35 | 13.01 | -49.18 | 378.82 |
| validation_pooled | 5,018.64 | 5,124.56 | -2.11 | 10,411.37 | 10,499.99 | 24.19 | 24.70 | -3,268.00 | -1,325.48 |

## Resultado principal de test

El candidato obtiene un MAE de **2,760.59**,
frente a **3,045.00** del baseline. La mejora
relativa en MAE es del **9.34 %**.

El RMSE mejora un **13.10 %**.
El WAPE pasa de **14.35 %** a
**13.01 %**.

El sesgo medio del candidato es **378.82**,
por lo que presenta una ligera tendencia agregada a sobreestimar. El baseline
presenta un sesgo de **-49.18**.

## Consistencia territorial y mensual

El candidato mejora el MAE en **41 de 50
territorios (82.00 %)**.

También mejora en **7 de 12 meses calendario**.

| month | rows | baseline_MAE | model_MAE | mae_improvement_pct | model_improves |
| --- | --- | --- | --- | --- | --- |
| 1 | 50 | 1,702.94 | 1,760.11 | -3.36 | False |
| 2 | 50 | 2,177.02 | 1,928.90 | 11.40 | True |
| 3 | 50 | 6,033.50 | 4,630.82 | 23.25 | True |
| 4 | 50 | 5,525.56 | 2,972.44 | 46.21 | True |
| 5 | 50 | 3,748.56 | 3,567.18 | 4.84 | True |
| 6 | 50 | 1,811.70 | 1,676.40 | 7.47 | True |
| 7 | 50 | 2,864.44 | 2,704.99 | 5.57 | True |
| 8 | 50 | 3,002.40 | 3,580.36 | -19.25 | False |
| 9 | 50 | 2,062.54 | 2,402.02 | -16.46 | False |
| 10 | 50 | 2,524.72 | 2,759.70 | -9.31 | False |
| 11 | 50 | 2,749.82 | 2,325.74 | 15.42 | True |
| 12 | 50 | 2,336.82 | 2,818.42 | -20.61 | False |

## Territorios con mayor mejora

| territory_id | territory_name | baseline_MAE | model_MAE | mae_improvement_pct |
| --- | --- | --- | --- | --- |
| ES-PROV-37 | Salamanca | 2,471.08 | 1,500.27 | 39.29 |
| ES-PROV-05 | Ávila | 4,391.00 | 2,716.52 | 38.13 |
| ES-PROV-06 | Badajoz | 1,977.67 | 1,315.79 | 33.47 |
| ES-PROV-14 | Córdoba | 2,099.83 | 1,428.82 | 31.96 |
| ES-PROV-02 | Albacete | 3,361.92 | 2,289.70 | 31.89 |

## Territorios con mayor deterioro

| territory_id | territory_name | baseline_MAE | model_MAE | mae_improvement_pct |
| --- | --- | --- | --- | --- |
| ES-PROV-07 | Balears, Illes | 10,499.25 | 13,961.60 | -32.98 |
| ES-PROV-15 | Coruña, A | 1,354.00 | 1,791.90 | -32.34 |
| ES-PROV-24 | León | 3,083.08 | 3,871.85 | -25.58 |
| ES-PROV-35 | Palmas, Las | 1,146.00 | 1,382.10 | -20.60 |
| ES-PROV-29 | Málaga | 9,310.75 | 11,130.33 | -19.54 |

## Criterios de promoción

| Criterio | Resultado |
|---|---|
| Mejora MAE agregada en validación ≥ 5 % | FAIL |
| Mejora MAE en test ≥ 5 % | PASS |
| Mejora en la mayoría de territorios | PASS |
| Promoción completa | FAIL |

## Decisión

El candidato mejora claramente el test final, pero no satisface el criterio de mejora agregada en validación. Se documenta como mejor modelo de machine learning en test, mientras el baseline se conserva como referencia y fallback operativo por su mayor estabilidad temporal.

No se realizarán ajustes posteriores basados en el test final. Cualquier
mejora futura deberá definirse como un nuevo experimento y volver a validarse
sin reutilizar este test para seleccionar hiperparámetros.
