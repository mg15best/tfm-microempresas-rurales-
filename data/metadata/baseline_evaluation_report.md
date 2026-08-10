# Evaluación del baseline estacional lag-12

- **Dataset:** `data\gold\gold_modeling_dataset_monthly.parquet`
- **Modelo:** `seasonal_naive_lag_12`
- **Predicción:** pernoctaciones del mismo mes del año anterior
- **Generado en UTC:** `2026-08-10T10:52:28.018952+00:00`
- **Filas evaluadas:** 2,350
- **MAE global:** 4,514.73
- **RMSE global:** 9,366.05
- **WAPE global:** 21.64 %
- **Sesgo medio global:** -2,446.17

## Cobertura de evaluación

| evaluation_split | total_rows | evaluable_rows | excluded_rows | baseline_missing | provisional_rows |
| --- | --- | --- | --- | --- | --- |
| validation_1 | 600 | 550 | 50 | 50 | 0 |
| validation_2 | 600 | 600 | 0 | 0 | 0 |
| validation_3 | 600 | 600 | 0 | 0 | 0 |
| test | 600 | 600 | 0 | 0 | 0 |

Las filas sin `lag_12_overnight_stays` se excluyen porque el baseline no puede
generar una predicción válida. No se imputan como cero ni se sustituyen por
otra observación disponible.

## Métricas por partición temporal

| split | rows | MAE | RMSE | WAPE_pct | mean_bias |
| --- | --- | --- | --- | --- | --- |
| validation_1 | 550 | 9,116.56 | 16,454.67 | 44.60 | -8,782.89 |
| validation_2 | 600 | 3,071.71 | 6,094.11 | 14.91 | -909.25 |
| validation_3 | 600 | 3,209.14 | 5,552.10 | 15.16 | -571.43 |
| test | 600 | 3,045.00 | 5,236.82 | 14.35 | -49.18 |
| overall | 2350 | 4,514.73 | 9,366.05 | 21.64 | -2,446.17 |

El split con mayor MAE es **validation_1**
(9,116.56), mientras que el menor MAE se observa
en **test** (3,045.00).

El sesgo se define como `predicción - valor real`. Un valor negativo indica
infraestimación de la demanda.

## Diagnóstico temporal básico

El mes calendario con mayor MAE agregado es el **4**,
con un MAE de **7,733.32**.

## Criterio de comparación posterior

Los modelos candidatos deberán compararse con este baseline sobre las mismas
filas evaluables y las mismas particiones temporales.
