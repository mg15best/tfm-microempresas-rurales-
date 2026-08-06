# Selección reproducible de modelos mediante validación temporal

- **Generado en UTC:** `2026-08-06T16:55:06.579638+00:00`
- **Dataset:** `data/gold/gold_modeling_dataset_monthly.parquet`
- **Estrategia:** validación temporal expansiva
- **Splits de selección:** validation_1, validation_2 y validation_3
- **Test final utilizado en selección:** no
- **Filas comparables agregadas:** 1750
- **Predicciones negativas:** recortadas a cero

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
| ridge_alpha_100 | 5.450,77 | 10.609,57 | 26,27 % |
| ridge_alpha_10 | 5.515,10 | 10.909,13 | 26,58 % |
| ridge_alpha_1 | 5.545,07 | 11.005,18 | 26,73 % |
| ridge_alpha_0_1 | 5.549,41 | 11.017,85 | 26,75 % |
| ridge_alpha_0_01 | 5.549,86 | 11.019,16 | 26,75 % |
| ridge_alpha_1000 | 5.566,55 | 10.647,82 | 26,83 % |

Mejor Ridge: `ridge_alpha_100`.

- MAE: 5.450,77
- Mejora frente al baseline: -8,61 %

## HistGradientBoosting congelado

Configuración evaluada: `hgb_raw_02`.

- MAE: 5.124,56
- RMSE: 10.499,99
- WAPE: 24,70 %
- Sesgo medio: -1.325,48
- Mejora frente al baseline: -2,11 %

## Decisión de validación

- Mejor candidato de machine learning: `hgb_raw_02`
- Mejora agregada del mejor candidato: -2,11 %
- Solución seleccionada tras validación: `seasonal_naive_lag_12`

El conjunto de test permanece excluido de este proceso.
