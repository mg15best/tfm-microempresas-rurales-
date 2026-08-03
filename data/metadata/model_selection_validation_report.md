# Selección de modelos mediante validación temporal

## Alcance

Este informe documenta la selección preliminar de modelos realizada
exclusivamente sobre las tres ventanas de validación temporal.

El conjunto de test final no se ha utilizado para ajustar hiperparámetros
ni para seleccionar el modelo.

- Filas comparables agregadas: 1.750
- Validaciones: validation_1, validation_2 y validation_3
- Comparación realizada sobre las mismas filas que el baseline
- Predicciones negativas recortadas a cero
- Entrenamiento expansivo en cada ventana temporal
- Datos provisionales excluidos

## Baseline estacional lag-12

El baseline utiliza como predicción las pernoctaciones observadas en el
mismo mes del año anterior.

| Métrica agregada | Resultado |
|---|---:|
| MAE | 5.018,64 |
| Filas | 1.750 |

## Ridge

Se probaron los siguientes valores de regularización:

- 0,01
- 0,1
- 1
- 10
- 100
- 1.000

La mejor configuración fue `alpha=100`.

| Métrica | Resultado |
|---|---:|
| MAE | 5.450,77 |
| MAE baseline | 5.018,64 |
| Mejora frente al baseline | -8,61 % |
| RMSE | 10.609,57 |
| WAPE | 26,27 % |
| Sesgo medio | -2.844,36 |

Ridge no supera al baseline y queda descartado como modelo candidato
principal.

## HistGradientBoostingRegressor

La mejor configuración de validación fue `hgb_raw_02`.

| Parámetro | Valor |
|---|---:|
| Transformación del target | raw |
| learning_rate | 0,05 |
| max_leaf_nodes | 31 |
| min_samples_leaf | 20 |
| l2_regularization | 1,0 |
| max_iter | 300 |
| random_state | 42 |

### Resultado agregado

| Métrica | Resultado |
|---|---:|
| MAE | 5.124,56 |
| MAE baseline | 5.018,64 |
| Mejora frente al baseline | -2,11 % |
| RMSE | 10.499,99 |
| WAPE | 24,70 % |
| Sesgo medio | -1.325,48 |

### Resultado por ventana

| Ventana | MAE modelo | MAE baseline | Mejora |
|---|---:|---:|---:|
| validation_1 | 8.081,94 | 9.116,56 | 11,35 % |
| validation_2 | 4.133,83 | 3.071,71 | -34,58 % |
| validation_3 | 3.404,34 | 3.209,14 | -6,08 % |

HistGradientBoosting mejora claramente durante la recuperación
pospandemia de `validation_1`, pero empeora en las dos validaciones
posteriores.

## Decisión preliminar

Ninguno de los modelos candidatos supera al baseline de forma agregada
en las tres ventanas de validación.

Por tanto:

1. El baseline estacional lag-12 permanece como modelo de referencia y
   campeón provisional.
2. Ridge queda descartado.
3. `hgb_raw_02` se conserva como mejor candidato no lineal.
4. El conjunto de test permanece intacto hasta la evaluación final.
5. La decisión definitiva deberá considerar también la estabilidad
   territorial, el comportamiento por mes y la evaluación final en test.
