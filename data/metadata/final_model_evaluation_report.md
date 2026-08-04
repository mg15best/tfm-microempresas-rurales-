# Evaluaci?n final del modelo predictivo

## Alcance

Este informe documenta la evaluaci?n final del candidato
`HistGradientBoostingRegressor` frente al baseline estacional lag-12.

La configuraci?n del modelo fue seleccionada exclusivamente mediante las
tres ventanas de validaci?n temporal. El conjunto de test no se utiliz?
para modificar variables, transformaciones ni hiperpar?metros.

## Configuraci?n congelada

- Modelo: `HistGradientBoostingRegressor`
- Identificador: `hgb_raw_02`
- Transformaci?n del target: ninguna
- `learning_rate`: 0,05
- `max_iter`: 300
- `max_leaf_nodes`: 31
- `min_samples_leaf`: 20
- `l2_regularization`: 1,0
- `early_stopping`: False
- `random_state`: 42

## Cobertura

| Elemento | Resultado |
|---|---:|
| Filas de entrenamiento | 10.737 |
| Periodo de entrenamiento | 2006-01 a 2024-05 |
| Filas de test | 600 |
| Periodo de test | 2024-06 a 2025-05 |
| Provincias | 50 |
| Predicciones negativas | 0 |

## Resultados globales del test

| Modelo | MAE | RMSE | WAPE | Sesgo medio |
|---|---:|---:|---:|---:|
| Baseline lag-12 | 3.045,00 | 5.236,82 | 14,35 % | -49,18 |
| HistGradientBoosting | 2.760,59 | 4.550,60 | 13,01 % | 378,82 |

La mejora de MAE de HistGradientBoosting frente al baseline es del
**9,34 %**.

El candidato reduce tambi?n el RMSE aproximadamente un **13,10 %** y el
WAPE en **1,34 puntos porcentuales**.

El sesgo medio positivo de `378,82` indica una ligera tendencia agregada
a sobreestimar la demanda. El baseline presenta un sesgo pr?cticamente
neutro.

## Consistencia territorial

HistGradientBoosting mejora el MAE en **41 de las 50 provincias**, lo que
representa un **82 %** del total.

### Provincias con mayor mejora

| Provincia | MAE baseline | MAE modelo | Mejora |
|---|---:|---:|---:|
| Salamanca | 2.471,08 | 1.500,27 | 39,29 % |
| ?vila | 4.391,00 | 2.716,52 | 38,13 % |
| Badajoz | 1.977,67 | 1.315,79 | 33,47 % |
| C?rdoba | 2.099,83 | 1.428,82 | 31,96 % |
| Albacete | 3.361,92 | 2.289,70 | 31,89 % |

### Provincias con mayor deterioro

| Provincia | MAE baseline | MAE modelo | Mejora |
|---|---:|---:|---:|
| Balears, Illes | 10.499,25 | 13.961,60 | -32,98 % |
| Coru?a, A | 1.354,00 | 1.791,90 | -32,34 % |
| Le?n | 3.083,08 | 3.871,85 | -25,58 % |
| Palmas, Las | 1.146,00 | 1.382,10 | -20,60 % |
| M?laga | 9.310,75 | 11.130,33 | -19,54 % |

## Resultados por mes

| Mes | MAE baseline | MAE modelo | Mejora |
|---:|---:|---:|---:|
| 1 | 1.702,94 | 1.760,11 | -3,36 % |
| 2 | 2.177,02 | 1.928,90 | 11,40 % |
| 3 | 6.033,50 | 4.630,82 | 23,25 % |
| 4 | 5.525,56 | 2.972,44 | 46,21 % |
| 5 | 3.748,56 | 3.567,18 | 4,84 % |
| 6 | 1.811,70 | 1.676,40 | 7,47 % |
| 7 | 2.864,44 | 2.704,99 | 5,57 % |
| 8 | 3.002,40 | 3.580,36 | -19,25 % |
| 9 | 2.062,54 | 2.402,02 | -16,46 % |
| 10 | 2.524,72 | 2.759,70 | -9,31 % |
| 11 | 2.749,82 | 2.325,74 | 15,42 % |
| 12 | 2.336,82 | 2.818,42 | -20,61 % |

El candidato mejora en 7 de los 12 meses. Las mayores mejoras se observan
en marzo y abril, mientras que los principales deterioros aparecen en
agosto, septiembre y diciembre.

## Evaluaci?n de los criterios

| Criterio | Resultado |
|---|---|
| Mejora agregada en validaci?n igual o superior al 5 % | FAIL |
| Mejora en el test final igual o superior al 5 % | PASS |
| Mejora en la mayor?a de las provincias | PASS |

## Decisi?n final

HistGradientBoosting es el modelo con mejor rendimiento en el conjunto de
test final y el mejor candidato de machine learning desarrollado.

Sin embargo, no supera al baseline de forma agregada en las ventanas de
validaci?n temporal. Esto revela una menor estabilidad entre periodos y
aconseja evitar su promoci?n como ?nico modelo operativo.

La decisi?n adoptada es:

1. Mantener el baseline estacional lag-12 como referencia robusta y
   mecanismo de fallback.
2. Conservar `hgb_raw_02` como mejor candidato predictivo y modelo ganador
   en el test final.
3. Mostrar ambos resultados en la Entrega 4 de forma transparente.
4. No realizar nuevos ajustes basados en el test final.
5. Considerar en trabajos futuros modelos especializados por provincia,
   r?gimen estacional o comportamiento tur?stico.

Esta decisi?n evita seleccionar el modelo ?nicamente por un resultado
favorable en test y mantiene la coherencia con los criterios definidos antes
de abrir dicho conjunto.
