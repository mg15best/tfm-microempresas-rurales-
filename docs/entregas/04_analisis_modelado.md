# Entrega 4 - Diseño del análisis y estrategia de modelado

## Trazabilidad del repositorio

Este documento corresponde al archivo `docs/entregas/04_analisis_modelado.md`.

La entrega parte del trabajo realizado en las tres fases anteriores:

- `01_ideas_producto.md`: definición de las ideas iniciales y selección de la línea de proyecto.
- `02_datos_necesarios.md`: análisis de las fuentes necesarias, viabilidad y alcance del MVP.
- `03_modelo_datos.md`: diseño de las capas raw, processed y gold, definición de la unidad analítica y construcción de la tabla descriptiva mensual.

El proyecto mantiene como núcleo la **analítica predictiva de la demanda turística rural en España**, utilizando datos oficiales del Instituto Nacional de Estadística. La variable principal será el número mensual de pernoctaciones en alojamientos de turismo rural por provincia.

La tabla descriptiva implementada en la entrega anterior es:

```text
data/gold/gold_tourism_demand_monthly.parquet
```

Su unidad analítica es una fila por provincia y mes. La versión disponible contiene 12.691 registros, 64 columnas, 50 provincias y un periodo comprendido entre enero de 2005 y mayo de 2026.

La entrega se inició definiendo de forma completa y trazable la estrategia de análisis y modelado y, en su versión final, incorpora también la implementación reproducible de esa estrategia. El trabajo desarrollado permite:

- concretar el problema predictivo;
- construir y validar el dataset específico de modelado;
- implementar el baseline estacional;
- entrenar y comparar los modelos candidatos;
- aplicar backtesting con ventanas temporales expansivas;
- evaluar una única vez el test final no provisional;
- generar predicciones, métricas e informes reproducibles;
- documentar los resultados, limitaciones y decisión final.

Las fuentes de precios, gasto turístico y tejido empresarial se mantienen como ampliaciones contextuales opcionales. No forman parte del núcleo obligatorio de esta estrategia porque su incorporación no es necesaria para demostrar la viabilidad del modelo principal de demanda.

### Evolución respecto a la Entrega 3

La presente estrategia concreta varias decisiones que en `03_modelo_datos.md` se habían dejado abiertas para fases posteriores. Estas decisiones no sustituyen la Entrega 3, sino que representan la evolución incremental del proyecto al pasar del diseño de la capa de datos al diseño del análisis y del modelado.

En particular:

- se mantiene `target_overnight_stays_total` como variable objetivo principal;
- se concreta un horizonte inicial de predicción de un mes (`forecast_horizon = 1`);
- se confirma `lag_12_overnight_stays` como baseline estacional principal;
- se plantea inicialmente la comparación entre una regresión Ridge y un modelo de boosting de árboles;
- se concreta una validación estrictamente temporal mediante backtesting con ventana expansiva;
- se adopta MAE como métrica principal, complementada con RMSE, WAPE y análisis de errores por provincia;
- las variables de precios, gasto y contexto empresarial pasan a considerarse ampliaciones opcionales y no serán necesarias para el primer modelo;
- `seasonality_index` y `tourism_pressure_index` se mantienen como indicadores descriptivos, pero no se utilizarán directamente como predictores por su riesgo de incorporar información no disponible en el momento real de la predicción;
- el dataset de modelado previsto en la Entrega 3 se mantiene como siguiente capa gold, pero su construcción deberá garantizar que todos los lags y variables derivadas utilizan exclusivamente información anterior al mes objetivo.

De esta forma se conserva la trazabilidad entre entregas sin modificar retrospectivamente las decisiones documentadas y entregadas en fases anteriores.

---

# 1. Problema que se busca resolver

## 1.1. Situación actual y problema

Los alojamientos rurales y otros agentes vinculados al turismo de los territorios suelen planificar su actividad con información limitada. Pueden conocer su experiencia pasada o consultar estadísticas históricas, pero no disponen necesariamente de una previsión mensual territorial validada que les ayude a anticipar cambios en la demanda.

Esta falta de anticipación dificulta decisiones como:

- ajustar personal y turnos;
- planificar mantenimiento o cierres temporales;
- preparar compras y aprovisionamiento;
- programar actividades y experiencias;
- decidir cuándo lanzar campañas;
- anticipar meses de alta o baja presión turística;
- coordinarse con otros negocios o entidades del territorio.

El problema no consiste en predecir las ventas o el beneficio de un establecimiento concreto. Los datos disponibles describen actividad turística agregada por provincia y no contienen reservas, precios internos, costes, capacidad comercial ni facturación de empresas individuales.

Por tanto, el problema que se abordará es:

> **predecir las pernoctaciones mensuales en alojamientos de turismo rural de cada provincia española utilizando únicamente información que estaría disponible antes del mes que se quiere estimar.**

## 1.2. Usuarios del resultado

Los usuarios principales del resultado serán:

1. **Alojamientos de turismo rural**, que podrán interpretar la previsión como una señal territorial para apoyar la planificación operativa.
2. **Asociaciones empresariales y entidades de desarrollo local**, que podrán identificar periodos de crecimiento, descenso o incertidumbre.
3. **Pequeños negocios vinculados al flujo turístico**, como restauración, comercio local o empresas de actividades, que podrán utilizar la previsión como contexto general, sin interpretarla como una predicción directa de ventas.
4. **Responsables del análisis y del futuro dashboard**, que necesitarán comparar el comportamiento real, el baseline y el modelo seleccionado.

## 1.3. Decisiones que se quieren apoyar

La salida del proyecto deberá ayudar a responder preguntas operativas como:

- ¿Se espera que las pernoctaciones del próximo mes aumenten o disminuyan respecto al mismo mes del año anterior?
- ¿La previsión se sitúa por encima o por debajo del comportamiento habitual de la provincia?
- ¿El nivel de incertidumbre permite utilizar la previsión con confianza?
- ¿En qué provincias o periodos el modelo funciona de forma suficientemente estable?
- ¿Cuándo es más prudente utilizar una referencia estacional sencilla en lugar de un modelo más complejo?

El modelo no sustituirá la decisión empresarial. Proporcionará una señal cuantitativa y explicable que podrá combinarse con información local, reservas reales, eventos o conocimiento operativo.

## 1.4. Resultado esperado y criterio de utilidad

El resultado principal será una previsión mensual de pernoctaciones por provincia.

La estrategia inicial trabajará con:

```text
forecast_horizon = 1
```

Esto significa que, utilizando información disponible hasta el mes `t`, se predecirán las pernoctaciones del mes `t+1`.

El proyecto se considerará útil si consigue:

- producir previsiones reproducibles para las provincias con histórico suficiente;
- evitar el uso de información futura;
- superar de forma estable un baseline estacional razonable;
- mantener un error interpretable en número de pernoctaciones;
- mostrar incertidumbre y limitaciones;
- permitir analizar el rendimiento por provincia, mes y temporada;
- integrarse posteriormente en un dashboard o aplicación ligera.

No se considerará útil un modelo que obtenga una buena métrica global únicamente porque predice bien las provincias de mayor volumen, pero empeora de forma sistemática en la mayoría de territorios.

---

# 2. Análisis de datos planteado y utilidad esperada

## 2.1. Preguntas que debe responder el análisis

El análisis se orientará a responder las siguientes preguntas:

### Evolución temporal

- ¿Cómo evolucionan las pernoctaciones mensuales en cada provincia?
- ¿Existe una tendencia de crecimiento, estabilidad o descenso?
- ¿La tendencia es homogénea entre provincias?
- ¿Qué impacto tienen los años 2020 y 2021 sobre la continuidad de las series?

### Estacionalidad

- ¿Qué meses concentran mayor y menor demanda?
- ¿El patrón anual se repite con suficiente estabilidad?
- ¿Hasta qué punto las pernoctaciones del mismo mes del año anterior explican el valor actual?
- ¿Qué provincias presentan una estacionalidad más intensa?

### Dependencia temporal

- ¿Qué relación existe entre la demanda actual y los valores de uno, tres y doce meses anteriores?
- ¿Las medias móviles de tres y doce meses aportan información adicional?
- ¿La variación interanual histórica ayuda a anticipar el siguiente mes?

### Diferencias territoriales

- ¿Qué provincias presentan mayor volumen medio?
- ¿Qué territorios tienen mayor volatilidad?
- ¿Existen provincias en las que el baseline estacional sea especialmente difícil de superar?
- ¿El rendimiento del modelo cambia según el tamaño o el patrón estacional del territorio?

### Variables auxiliares

- ¿La ocupación histórica aporta información adicional cuando se utiliza con desfase?
- ¿La estancia media del mes anterior mejora la previsión?
- ¿La composición nacional o extranjera de la demanda histórica contiene señal predictiva?
- ¿Las variables de capacidad, como plazas y establecimientos, deben utilizarse directamente, con desfase o únicamente como contexto?

### Calidad y disponibilidad

- ¿Qué filas pueden generar correctamente todos los lags?
- ¿Cómo afectan los meses globalmente ausentes y las combinaciones provincia-mes sin demanda publicada?
- ¿Qué variables presentan más valores nulos?
- ¿Qué registros deben marcarse como histórico insuficiente?

## 2.2. Análisis previo al modelado

Antes de entrenar modelos se realizarán los siguientes análisis.

### 2.2.1. Cobertura y continuidad temporal

Se comprobará:

- número de observaciones por provincia;
- primer y último mes disponible;
- continuidad mensual;
- meses globalmente ausentes;
- combinaciones provincia-mes ausentes;
- disponibilidad de la variable objetivo;
- cobertura necesaria para construir `lag_1`, `lag_3` y `lag_12`;
- pérdida de filas provocada por las medias móviles.

Los meses `2020-04`, `2020-05` y `2020-11` no contienen observaciones provinciales publicadas. Además, existen nueve combinaciones provincia-mes sin viajeros ni pernoctaciones. Estos huecos no se sustituirán automáticamente por cero.

### 2.2.2. Distribución de la variable objetivo

Se estudiará la distribución de `overnight_stays_total`:

- globalmente;
- por provincia;
- por año;
- por mes;
- mediante escala original y transformación logarítmica.

El objetivo será comprobar la asimetría y las diferencias de escala. Provincias con gran volumen pueden dominar las métricas agregadas, por lo que será necesario complementar la evaluación global con resultados territoriales.

### 2.2.3. Tendencia y estacionalidad

Se analizarán:

- medias y medianas por mes del año;
- perfiles estacionales por provincia;
- evolución interanual;
- descomposición descriptiva cuando sea útil;
- relación entre `overnight_stays_total` y su valor doce meses antes.

Este bloque permitirá comprobar si `lag_12_overnight_stays` constituye un baseline fuerte.

### 2.2.4. Autocorrelación y lags

Se estudiará la relación de la variable objetivo con:

```text
lag_1_overnight_stays
lag_3_overnight_stays
lag_12_overnight_stays
rolling_mean_3m_overnight_stays
rolling_mean_12m_overnight_stays
```

También se analizará la autocorrelación en distintos retardos, globalmente y en una selección de provincias representativas.

### 2.2.5. Variables auxiliares históricas

Se evaluará la relación entre la demanda futura y las siguientes variables utilizadas siempre con desfase:

- ocupación por plazas;
- ocupación por habitaciones;
- ocupación de fin de semana;
- estancia media;
- peso de demanda nacional;
- peso de demanda extranjera;
- plazas estimadas;
- establecimientos abiertos estimados;
- personal empleado.

El análisis distinguirá correlación descriptiva de utilidad predictiva. Una relación contemporánea elevada no implica que la variable pueda utilizarse para predecir el futuro.

### 2.2.6. Periodo COVID-19

El periodo comprendido entre marzo de 2020 y diciembre de 2021 se considerará una ruptura estructural.

Se compararán, como mínimo:

- distribución de la demanda antes, durante y después del periodo;
- errores del baseline durante la ruptura;
- comportamiento de los modelos con y sin esos registros en entrenamiento;
- utilidad de incluir `covid_period` como indicador de control histórico.

No se eliminará el periodo COVID-19 sin análisis previo. La decisión se justificará mediante una comparación de sensibilidad.

### 2.2.7. Datos provisionales

Los registros desde junio de 2025 hasta mayo de 2026 están marcados como provisionales.

Se utilizarán con cautela:

- no formarán parte del test final principal utilizado para seleccionar el modelo;
- podrán emplearse como evaluación adicional o periodo de seguimiento;
- cualquier resultado basado en ellos se identificará como provisional.

## 2.3. Análisis durante el modelado

Durante el entrenamiento y comparación se estudiará:

- evolución de MAE y RMSE entre folds temporales;
- diferencia de rendimiento entre baseline y candidatos;
- estabilidad de los hiperparámetros;
- sensibilidad a la transformación `log1p`;
- impacto de añadir o eliminar bloques de variables;
- rendimiento con y sin periodo COVID-19;
- importancia de variables o coeficientes;
- diferencias de error entre provincias;
- señales de sobreajuste.

Se realizarán pruebas de ablación sencillas:

1. solo calendario y lags de demanda;
2. calendario, lags y medias móviles;
3. incorporación de variables auxiliares históricas;
4. comparación con y sin variables territoriales;
5. comparación con y sin periodo COVID-19.

La finalidad será comprobar si cada bloque de información mejora realmente el rendimiento.

## 2.4. Análisis posterior al modelado

Una vez obtenidas las predicciones de validación y test, se analizarán los residuos:

- error absoluto;
- error con signo;
- errores extremos;
- sesgo a sobreestimar o infraestimar;
- rendimiento por provincia;
- rendimiento por mes del año;
- rendimiento por temporada;
- rendimiento por nivel de demanda;
- rendimiento en provincias de bajo, medio y alto volumen;
- rendimiento antes, durante y después de COVID-19.

También se identificarán:

- provincias en las que el modelo supera claramente el baseline;
- provincias en las que ambos enfoques son equivalentes;
- provincias en las que el baseline sigue siendo superior;
- meses o patrones con mayor incertidumbre.

## 2.5. Hipótesis y patrones que se comprobarán

Las hipótesis principales serán:

### H1. Estacionalidad anual

Las pernoctaciones del mismo mes del año anterior contienen una señal predictiva fuerte.

### H2. Persistencia reciente

Los valores de uno y tres meses anteriores aportan información sobre la tendencia más reciente.

### H3. Medias móviles

Las medias móviles de tres y doce meses ayudan a reducir volatilidad y mejoran la estabilidad de la predicción.

### H4. Heterogeneidad territorial

La provincia modifica el patrón de demanda, por lo que debe incorporarse al modelo global o considerarse explícitamente en la evaluación.

### H5. Variables auxiliares

La ocupación, la estancia media y la composición de la demanda aportan señal adicional cuando se utilizan con desfase temporal.

### H6. Mayor complejidad no implica necesariamente mejor solución

Un modelo flexible puede mejorar el error medio, pero puede resultar menos estable o menos útil que una regresión sencilla o el baseline estacional.

### H7. COVID-19 reduce la estabilidad

Los meses afectados por la pandemia aumentan el error y pueden alterar las relaciones aprendidas por los modelos.

## 2.6. Visualizaciones e indicadores para el MVP

Las visualizaciones previstas para el futuro MVP serán:

1. Serie histórica de pernoctaciones por provincia.
2. Comparación entre valor real, baseline y predicción.
3. Banda de incertidumbre de la previsión.
4. Variación prevista respecto al mismo mes del año anterior.
5. Error histórico del modelo por provincia.
6. MAE y mejora relativa frente al baseline.
7. Mapa o ranking territorial de rendimiento.
8. Distribución de errores por mes y temporada.
9. Indicador del estado del dato: definitivo o provisional.
10. Explicación resumida de los factores principales.

Estas visualizaciones deberán ayudar a interpretar la previsión y no limitarse a mostrar métricas técnicas.

---

# 3. Tipo de modelos que se van a plantear

## 3.1. Tipo de tarea de modelado

El proyecto abordará una tarea de:

> **forecasting mensual formulado como regresión supervisada sobre un panel temporal de provincias.**

El dataset contendrá observaciones de múltiples territorios a lo largo del tiempo. Cada fila representará un territorio, un mes objetivo y un horizonte de predicción.

El enfoque inicial será un **modelo global**, entrenado con todas las provincias. Esta estrategia permite:

- aprovechar patrones compartidos;
- aumentar el número de observaciones;
- evitar mantener un modelo independiente por provincia;
- comparar el rendimiento territorial dentro de un mismo sistema.

La evaluación seguirá siendo temporal. El hecho de formular el problema como regresión supervisada no permite aplicar divisiones aleatorias.

## 3.2. Baseline o modelo de referencia

El baseline principal será el modelo estacional naive:

```text
predicción para el mes t = pernoctaciones observadas en t - 12
```

Campo equivalente:

```text
lag_12_overnight_stays
```

Se selecciona porque:

- el turismo rural presenta una estacionalidad anual intensa;
- es sencillo y reproducible;
- utiliza únicamente información disponible antes del mes objetivo;
- es fácil de explicar;
- representa una referencia difícil de superar de forma trivial.

Como baseline complementario podrá calcularse una media histórica del mismo mes o una media móvil, pero la referencia principal será `lag_12`.

## 3.3. Modelo candidato interpretable

El primer modelo candidato será una **regresión Ridge**.

### Motivos

- permite combinar variables numéricas y categóricas;
- reduce la inestabilidad causada por variables correlacionadas;
- es rápida y reproducible;
- facilita interpretar el signo y magnitud relativa de los coeficientes;
- proporciona una primera alternativa supervisada frente al baseline.

### Limitaciones

- supone relaciones principalmente lineales;
- puede no capturar interacciones complejas;
- puede tener dificultades con cambios bruscos;
- requiere preprocesamiento, escalado y codificación.

## 3.4. Modelo candidato flexible

El segundo modelo candidato será un **HistGradientBoostingRegressor** o un modelo de boosting equivalente disponible en el entorno final.

### Motivos

- permite capturar relaciones no lineales;
- puede modelar interacciones entre calendario, territorio y lags;
- suele funcionar bien con datos tabulares;
- permite comprobar si una mayor flexibilidad mejora realmente la predicción;
- puede manejar patrones diferentes según el nivel de demanda.

### Limitaciones

- menor interpretabilidad directa;
- mayor riesgo de sobreajuste;
- mayor sensibilidad a hiperparámetros;
- necesita una validación temporal rigurosa;
- puede ofrecer una mejora global pequeña concentrada en pocas provincias.

La implementación final utilizará una alternativa compatible con las dependencias y el entorno reproducible del repositorio.

## 3.5. Comparación inicial de alternativas

| Alternativa | Tipo | Motivo | Limitación principal |
|---|---|---|---|
| Baseline estacional | Regla naive `t-12` | Referencia simple, fuerte y explicable para series estacionales | No incorpora tendencia reciente ni otras variables |
| Regresión Ridge | Modelo lineal regularizado | Primera solución interpretable y reproducible | Puede no capturar relaciones no lineales |
| Boosting de árboles | Modelo flexible no lineal | Permite comprobar si la complejidad mejora el resultado | Mayor riesgo de sobreajuste y menor interpretabilidad |

No se propondrá un número elevado de modelos. La finalidad es comparar alternativas con niveles de complejidad claramente diferentes.

Como alternativa futura podrá evaluarse un modelo estadístico por serie, como SARIMA o ETS, pero no será el núcleo inicial porque obligaría a ajustar y mantener múltiples modelos provinciales.

---

# 4. Datos de entrada del análisis y los modelos

## 4.1. Dataset descriptivo de origen

El dataset de origen será:

```text
data/gold/gold_tourism_demand_monthly.parquet
```

Representa una fila por:

```text
territory_id + month_id
```

Incluye:

- claves territoriales y temporales;
- viajeros;
- pernoctaciones;
- estancia media;
- establecimientos y plazas;
- ocupación;
- personal empleado;
- proporciones de demanda nacional y extranjera;
- indicadores derivados;
- metadatos de calidad y trazabilidad.

Este dataset se utilizará para análisis descriptivo y como fuente para construir el dataset específico de modelado.

## 4.2. Dataset específico de modelado

Se construirá:

```text
data/gold/gold_modeling_dataset_monthly.parquet
```

No será una copia de la tabla descriptiva. Contendrá únicamente:

- variable objetivo;
- variables conocidas antes del mes objetivo;
- variables históricas correctamente desplazadas;
- campos de control temporal;
- claves y metadatos de trazabilidad.

## 4.3. Unidad de análisis, clave y horizonte

La unidad de análisis será:

> una provincia, un mes objetivo y un horizonte de predicción.

La clave prevista será:

```text
territory_id + target_month_id + forecast_horizon
```

Campos principales:

```text
territory_id
target_month_id
target_date_month
forecast_horizon
```

El horizonte inicial será:

```text
forecast_horizon = 1
```

## 4.4. Variable objetivo

La variable objetivo principal será:

```text
target_overnight_stays_total
```

Representa las pernoctaciones mensuales reales en alojamientos de turismo rural de la provincia y mes objetivo.

Se selecciona porque:

- mide la presencia turística total mejor que el número de viajeros;
- incorpora tanto afluencia como duración de la estancia;
- está disponible en la fuente principal;
- tiene cobertura suficiente;
- es relevante para la planificación territorial.

La ocupación podrá utilizarse como indicador secundario de evaluación o como posible objetivo de una ampliación futura, pero no será el objetivo principal de esta estrategia.

## 4.5. Variables de entrada previstas

### 4.5.1. Variables de calendario conocidas

```text
year
month
quarter
is_summer
is_christmas_period
covid_period
```

`is_easter_period` solo se utilizará cuando se construya mediante un calendario anual reproducible.

### 4.5.2. Variables históricas de demanda

```text
lag_1_overnight_stays
lag_3_overnight_stays
lag_12_overnight_stays
rolling_mean_3m_overnight_stays
rolling_mean_12m_overnight_stays
yoy_change_overnight_stays
```

### 4.5.3. Variables auxiliares con desfase

```text
lag_1_occupancy_rate_pct
lag_12_occupancy_rate_pct
lag_1_weekend_occupancy_rate_pct
lag_1_average_stay
lag_12_average_stay
lag_1_domestic_overnight_stays_share
lag_1_foreign_overnight_stays_share
lag_1_places_estimated
lag_1_establishments_estimated
lag_1_staff_employed
```

Estas variables solo se incorporarán si su disponibilidad histórica y utilidad predictiva quedan demostradas.

### 4.5.4. Variable territorial

```text
territory_id
```

Se tratará como variable categórica mediante una codificación incluida dentro del pipeline de entrenamiento.

### 4.5.5. Metadatos de control

```text
data_status
is_provisional
source_snapshot_id
pipeline_run_id
data_version
```

Estos campos se conservarán para trazabilidad, pero no se utilizarán como predictores, salvo indicadores explícitos de estado cuya utilidad esté metodológicamente justificada.

## 4.6. Transformaciones y creación de features

Las variables temporales se construirán por provincia, ordenando previamente por fecha.

Reglas principales:

1. Los lags se calcularán con `shift`.
2. Las medias móviles se calcularán después de aplicar el desplazamiento.
3. Ninguna media móvil incluirá el valor del mes objetivo.
4. Las variaciones históricas se calcularán únicamente con valores anteriores.
5. Las transformaciones se realizarán dentro de una función reproducible.
6. Se validará que las fechas estén ordenadas y que las claves sean únicas.

Ejemplo conceptual:

```text
lag_1 = overnight_stays_total.shift(1)
rolling_mean_3m = overnight_stays_total.shift(1).rolling(3).mean()
```

También se evaluará una transformación:

```text
log1p(target_overnight_stays_total)
```

La transformación logarítmica puede reducir la influencia de provincias de gran volumen. Si se utiliza, las predicciones se devolverán a la escala original antes de calcular las métricas finales.

## 4.7. Variables excluidas

No se utilizarán inicialmente las siguientes variables:

### Variables contemporáneas del mes objetivo

- `travellers_total` del mes objetivo;
- `occupancy_rate_pct` del mes objetivo;
- `average_stay` del mes objetivo;
- `staff_employed` del mes objetivo;
- cualquier métrica observada durante el mismo mes que se intenta predecir.

Motivo: no estarían disponibles en el momento real de la predicción.

### Indicadores con riesgo de fuga

- `seasonality_index` actual;
- `tourism_pressure_index` actual.

El `seasonality_index` se calcula utilizando el histórico completo disponible, por lo que incorpora información posterior a observaciones antiguas. El `tourism_pressure_index` contiene métricas contemporáneas de demanda y ocupación.

Si se desea utilizar una señal equivalente, deberá recalcularse dentro de cada ventana de entrenamiento o aplicarse un desfase adecuado.

### Variables contextuales no integradas

- `price_index`;
- `price_yoy_change_pct`;
- `resident_avg_spend_context`;
- `foreign_avg_spend_context`;
- densidad empresarial;
- variables de oportunidad económica.

Motivo: sus fuentes todavía no están integradas en el núcleo reproducible.

### Campos técnicos como predictores

- identificadores de snapshots;
- identificadores de ejecución;
- fecha de creación;
- versión del dataset.

Se conservarán para trazabilidad, no para explicar la demanda.

### Variables redundantes

No se utilizarán simultáneamente todas las proporciones complementarias cuando una pueda derivarse exactamente de otra. Por ejemplo, si:

```text
domestic_share + foreign_share = 1
```

se podrá conservar una sola variable para evitar redundancia perfecta.

## 4.8. Disponibilidad real de la información

Para predecir el mes `t+1`, solo podrá utilizarse:

- calendario conocido del mes objetivo;
- datos publicados hasta el mes `t`;
- lags y medias móviles construidos hasta `t`;
- variables externas cuya fecha de publicación permita demostrar que estaban disponibles.

No se asumirá que una estadística oficial se publica inmediatamente al terminar el mes. En una implementación operativa futura deberá registrarse también la fecha real de publicación.

Para esta prueba de concepto se trabajará con la regla conservadora de no usar información del mes objetivo.

---

# 5. Datos de salida y forma de consumo

## 5.1. Salida principal del modelo

La salida principal será:

```text
predicted_overnight_stays_total
```

Representará el número previsto de pernoctaciones para una provincia, un mes objetivo y un horizonte determinado.

## 5.2. Granularidad y campos de salida

La futura tabla de predicciones podrá denominarse:

```text
data/gold/gold_tourism_demand_forecast_monthly.parquet
```

La granularidad será:

```text
territory_id + target_month_id + forecast_horizon
```

Campos previstos:

| Campo | Tipo | Descripción |
|---|---|---|
| `territory_id` | string | Provincia predicha |
| `territory_name` | string | Nombre legible |
| `target_month_id` | string | Mes objetivo |
| `target_date_month` | date | Fecha del mes objetivo |
| `forecast_horizon` | int8 | Horizonte en meses |
| `predicted_overnight_stays_total` | float | Predicción principal |
| `prediction_interval_lower` | float nullable | Límite inferior |
| `prediction_interval_upper` | float nullable | Límite superior |
| `baseline_prediction` | float nullable | Predicción del baseline |
| `actual_overnight_stays_total` | float nullable | Valor real cuando esté disponible |
| `absolute_error` | float nullable | Error absoluto en backtesting |
| `model_name` | string | Modelo utilizado |
| `model_version` | string | Versión del modelo |
| `training_cutoff` | date | Último mes utilizado para entrenar |
| `data_status` | string | Estado definitivo o provisional |
| `source_snapshot_id` | string | Trazabilidad de datos |
| `pipeline_run_id` | string | Ejecución del pipeline |
| `generated_at` | datetime | Fecha de generación |
| `explanation` | string | Interpretación resumida |

## 5.3. Formatos previstos

Las salidas se almacenarán principalmente en:

- Parquet, como formato canónico;
- CSV, como exportación legible;
- tablas de métricas;
- figuras para informes;
- dashboard o aplicación ligera en una fase posterior.

## 5.4. Integración en el MVP

El usuario podrá seleccionar una provincia y consultar:

- demanda histórica;
- previsión del siguiente mes;
- comparación con el mismo mes del año anterior;
- intervalo de incertidumbre;
- rendimiento histórico del modelo;
- estado provisional o definitivo del dato;
- explicación resumida.

La predicción se interpretará como una señal territorial. No se traducirá directamente en ventas, reservas o beneficio empresarial.

## 5.5. Explicación e incertidumbre

La salida deberá mostrar:

- modelo utilizado;
- baseline de referencia;
- diferencia prevista respecto al año anterior;
- principales variables asociadas a la predicción;
- intervalo de incertidumbre;
- calidad histórica del modelo en esa provincia;
- advertencia cuando el rendimiento territorial sea insuficiente.

Los intervalos podrán estimarse mediante:

- cuantiles de residuos de backtesting;
- residuos segmentados por nivel de demanda o temporada;
- método conformal, si se implementa posteriormente.

La técnica elegida deberá utilizar únicamente errores obtenidos fuera de muestra.

---

# 6. Estrategia para diseñar y seleccionar el modelo

## 6.1. Preparación del dataset de modelado

El proceso será:

1. Cargar la tabla gold descriptiva.
2. Verificar claves, tipos y orden temporal.
3. Ordenar por `territory_id` y `date_month`.
4. Crear el mes objetivo y el horizonte.
5. Generar lags por provincia.
6. Generar medias móviles usando solo meses anteriores.
7. Crear variables de calendario.
8. Identificar filas con histórico insuficiente.
9. Excluir filas sin objetivo durante el entrenamiento.
10. Conservar metadatos de trazabilidad.
11. Exportar el dataset de modelado.
12. Validar la nueva tabla mediante reglas específicas.

El dataset deberá incluir un campo:

```text
data_quality_flag
```

Con valores como:

```text
ok
missing_target
insufficient_history
missing_lag
provisional_target
outlier_review
```

## 6.2. Construcción del baseline

Para cada fila evaluable:

```text
baseline_prediction = lag_12_overnight_stays
```

Las filas sin `lag_12` no se utilizarán para comparar modelos con este baseline.

Se calcularán las mismas métricas para el baseline y para los modelos candidatos.

## 6.3. Entrenamiento de modelos candidatos

### Ridge

Se integrará en un pipeline con:

- imputación;
- escalado de variables numéricas;
- codificación de `territory_id`;
- regularización;
- búsqueda limitada del parámetro `alpha`.

### Boosting

Se integrará en un pipeline con:

- imputación;
- codificación territorial compatible;
- ajuste limitado de profundidad, tasa de aprendizaje y número de iteraciones;
- control de sobreajuste.

La búsqueda de hiperparámetros será pequeña y coherente con el volumen de datos. No se realizará una exploración masiva.

## 6.4. Preprocesamiento

### Nulos

- No se imputará la variable objetivo.
- Los lags ausentes por falta de histórico se marcarán.
- La imputación de features se ajustará exclusivamente con train.
- Se priorizarán medianas u otros métodos simples y reproducibles.
- Podrán añadirse indicadores de ausencia cuando aporten valor.

### Escalado

Se aplicará a la regresión Ridge. No será necesario para los modelos basados en árboles.

### Codificación

`territory_id` se tratará como categórica.

La codificación deberá:

- ajustarse solo con el conjunto de entrenamiento;
- gestionar categorías desconocidas;
- formar parte del pipeline.

### Selección de variables

La selección se basará en:

- disponibilidad temporal;
- calidad;
- redundancia;
- mejora fuera de muestra;
- interpretabilidad;
- riesgo de leakage.

## 6.5. Criterios de comparación

Los modelos se compararán mediante:

1. MAE global.
2. RMSE global.
3. WAPE global.
4. MAE por provincia.
5. Mejora frente al baseline.
6. Porcentaje de provincias en las que mejora.
7. Estabilidad entre folds.
8. Sesgo de predicción.
9. Error por mes y temporada.
10. Coste computacional.
11. Interpretabilidad.
12. Facilidad de integración.
13. Reproducibilidad.

## 6.6. Regla de selección final

Un modelo candidato será seleccionado si cumple todas estas condiciones:

1. Mejora el MAE del baseline en el conjunto de validación agregado.
2. Mantiene la mejora en el test final.
3. Obtiene una mejora práctica aproximada de al menos un 5 % en MAE, salvo que una mejora menor se justifique por una estabilidad o utilidad claramente superior.
4. Supera el baseline en una mayoría clara de provincias.
5. No empeora de forma grave en territorios o temporadas relevantes.
6. Presenta resultados estables entre folds.
7. No utiliza variables con fuga de información.
8. Puede reproducirse con el pipeline y las dependencias del repositorio.
9. Su complejidad se justifica por la mejora obtenida.
10. Puede explicarse e integrarse en el MVP.

Si dos modelos tienen un rendimiento similar, se seleccionará el más simple, estable e interpretable.

---

# 7. Estrategia de validación y evaluación

## 7.1. Separación temporal de los datos

No se utilizará una división aleatoria.

La separación será temporal y común a todas las provincias. Para cada fecha de evaluación:

- el entrenamiento solo contendrá meses anteriores;
- ninguna provincia aportará información futura;
- los transformadores se ajustarán solo con train;
- el test final no se utilizará para seleccionar hiperparámetros.

La propuesta inicial será:

```text
Entrenamiento inicial: 2005-01 a 2021-05
Validación 1:          2021-06 a 2022-05
Validación 2:          2022-06 a 2023-05
Validación 3:          2023-06 a 2024-05
Test final:            2024-06 a 2025-05
Seguimiento provisional: 2025-06 a 2026-05
```

Estas fechas podrán ajustarse ligeramente durante la implementación si el análisis de publicación, cobertura o lags lo requiere, manteniendo siempre:

- bloques completos de doce meses;
- un test final no provisional;
- un periodo provisional separado.

## 7.2. Backtesting

Se empleará backtesting con ventana expansiva.

Ejemplo:

```text
Fold 1:
train hasta 2021-05
validación 2021-06 a 2022-05

Fold 2:
train hasta 2022-05
validación 2022-06 a 2023-05

Fold 3:
train hasta 2023-05
validación 2023-06 a 2024-05
```

En cada fold:

1. se construirán o filtrarán las features disponibles;
2. se ajustará el preprocesamiento;
3. se entrenará el modelo;
4. se generarán predicciones;
5. se calcularán métricas;
6. se guardarán resultados por fila;
7. se comparará con el baseline.

## 7.3. Prevención de fuga de información

Se aplicarán las siguientes reglas:

- no usar variables contemporáneas del objetivo;
- usar `shift` antes de cualquier media móvil;
- recalcular features dentro de la ventana temporal correcta;
- ajustar imputadores, escaladores y codificadores solo con train;
- no usar el test para seleccionar variables;
- no utilizar `seasonality_index` actual como predictor;
- no utilizar `tourism_pressure_index` actual como predictor;
- no utilizar datos provisionales para elegir el modelo principal;
- mantener fechas de corte explícitas;
- incorporar pruebas automatizadas para comprobar los desplazamientos.

También se comprobará que:

```text
feature_date < target_date
```

para todas las variables que dependan de observaciones históricas.

## 7.4. Métricas de evaluación

### Métrica principal: MAE

```text
MAE = media de |valor real - predicción|
```

Se selecciona porque:

- se interpreta en número de pernoctaciones;
- no eleva al cuadrado los errores;
- facilita explicar el rendimiento.

### Métricas complementarias

#### RMSE

Penaliza con mayor intensidad los errores extremos.

#### WAPE

```text
WAPE = suma de errores absolutos / suma de valores reales
```

Permite contextualizar el error respecto al volumen total.

#### Sesgo medio

Indica si el modelo tiende a sobreestimar o infraestimar.

#### Mejora relativa frente al baseline

```text
mejora = (MAE_baseline - MAE_modelo) / MAE_baseline
```

#### Métricas segmentadas

- MAE por provincia;
- MAE por mes;
- MAE por temporada;
- MAE por nivel de demanda;
- porcentaje de provincias mejoradas.

MAPE no será la métrica principal porque puede ser inestable cuando los valores reales son bajos.

## 7.5. Comparación con el baseline

Todos los modelos utilizarán:

- las mismas filas evaluables;
- los mismos folds;
- las mismas fechas;
- la misma variable objetivo;
- las mismas métricas.

La mejora no se evaluará únicamente en el agregado nacional. Se comprobará también:

- cuántas provincias mejoran;
- en cuáles empeoran;
- si la mejora es estable;
- si depende de meses concretos;
- si se concentra en provincias de gran volumen.

## 7.6. Análisis de errores

Se revisarán:

- diez mayores errores absolutos;
- provincias con mayor MAE;
- meses con mayor error;
- predicciones negativas, que deberán truncarse o evitarse;
- errores durante COVID-19;
- errores en meses de demanda extrema;
- residuos con patrón temporal;
- sesgo sistemático;
- territorios con histórico irregular.

Las predicciones no podrán ser negativas. Si el modelo genera valores inferiores a cero, se aplicará una regla documentada, como:

```text
predicción_final = max(0, predicción)
```

## 7.7. Criterio mínimo de aceptación

El modelo se considerará aceptable si:

- mejora el baseline estacional en MAE;
- mantiene la mejora en el test final;
- alcanza una mejora práctica aproximada del 5 % o superior;
- mejora en una mayoría clara de provincias;
- no presenta leakage;
- mantiene estabilidad entre folds;
- produce salidas coherentes y no negativas;
- puede integrarse y explicarse.

Si ningún modelo cumple estas condiciones:

1. se seleccionará el baseline estacional;
2. se documentará que la complejidad adicional no aporta mejora suficiente;
3. el MVP mostrará previsiones estacionales y análisis descriptivo;
4. se identificarán las fuentes o variables necesarias para una futura mejora.

---

# 8. Riesgos y alternativas

## 8.1. Representatividad de la variable objetivo

`overnight_stays_total` representa pernoctaciones en alojamientos de turismo rural a nivel provincial.

No representa:

- reservas de una empresa;
- facturación;
- margen;
- demanda municipal;
- visitantes que no utilizan alojamiento rural;
- consumo directo en restauración o comercio.

La interpretación deberá mantenerse en el nivel territorial real de la fuente.

## 8.2. Riesgo de fuga de información

Los principales riesgos son:

- utilizar ocupación del mes objetivo;
- calcular medias móviles sin desplazar;
- usar indicadores derivados con todo el histórico;
- ajustar transformaciones con validación o test;
- utilizar datos revisados o publicados después del momento simulado;
- usar el test para elegir variables.

La prevención del leakage será uno de los criterios principales de aceptación.

## 8.3. Calidad, volumen y cobertura temporal

El histórico y el volumen son suficientes para plantear un modelo global mensual, pero existen limitaciones:

- tres meses globalmente ausentes;
- nueve combinaciones provincia-mes ausentes;
- valores nulos en algunas variables;
- primeros meses sin lags;
- datos provisionales desde junio de 2025;
- diferencias de cobertura entre métricas.

No se imputarán ausencias de demanda como cero sin evidencia.

## 8.4. Cambios estructurales y periodo COVID-19

COVID-19 altera la distribución y puede reducir la capacidad de generalización.

Alternativas:

- incluir indicador COVID;
- entrenar con todos los datos;
- comparar con entrenamiento sin el periodo;
- reducir su peso si se justifica;
- analizar errores de forma separada.

La decisión final se basará en validación temporal.

## 8.5. Diferencias entre territorios

Las provincias presentan:

- escalas muy diferentes;
- estacionalidad distinta;
- volatilidad desigual;
- diferente composición nacional y extranjera;
- comportamientos estructurales específicos.

Un modelo global puede funcionar bien en promedio y mal en algunos territorios.

Alternativas:

- evaluar por provincia;
- transformar el objetivo;
- ponderar observaciones;
- entrenar modelos por grupos de territorios;
- usar baseline en provincias con bajo rendimiento;
- limitar el despliegue a territorios validados.

## 8.6. Incertidumbres principales

Las mayores incertidumbres son:

1. cuánto puede mejorarse el baseline estacional;
2. si las variables auxiliares con desfase aportan señal real;
3. cómo tratar COVID-19;
4. si un modelo global representa correctamente provincias muy distintas;
5. cómo estimar intervalos de predicción fiables;
6. cómo afectarán futuras revisiones de datos provisionales;
7. qué fecha real de disponibilidad tiene cada estadística oficial.

## 8.7. Alternativa si ningún modelo supera el baseline

Si ningún modelo supera el baseline con rigor:

- se seleccionará la regla `t-12`;
- se mantendrá el análisis descriptivo y territorial;
- se mostrarán intervalos derivados de errores históricos;
- se documentarán los segmentos en los que el baseline es fiable;
- se evitará presentar un modelo más complejo como mejor solución;
- se planteará como ampliación la incorporación de calendario, clima o precios cuando exista compatibilidad;
- se revisará un posible enfoque estadístico por provincia.

Esta alternativa sigue siendo útil porque proporciona una referencia estacional reproducible y transparente.

---

# 9. Resultado esperado de la estrategia

Al finalizar esta fase, el repositorio deberá permitir comprender:

- qué problema predictivo se quiere resolver;
- quién utilizará el resultado;
- qué decisión se quiere apoyar;
- qué análisis se realizará;
- qué variable se predecirá;
- qué datos podrán utilizarse;
- qué variables se excluirán;
- qué baseline y modelos se compararán;
- cómo se evitará la fuga de información;
- cómo se dividirán los datos;
- qué métricas se utilizarán;
- qué criterio determinará la selección;
- cómo se integrará la salida en el MVP;
- qué riesgos y alternativas existen.

Durante esta fase se han desarrollado, de forma reproducible, los siguientes componentes principales:

```text
src/features/build_modeling_dataset.py
src/features/validate_modeling_dataset.py
src/models/evaluate_baseline.py
src/models/evaluate_final_candidate.py
data/gold/gold_modeling_dataset_monthly.parquet
data/model_outputs/final_candidate_predictions.parquet
```

También se han creado el contrato del dataset de modelado, las reglas de validación, la configuración temporal de folds, pruebas automatizadas de lags y medias móviles, métricas desagregadas e informes de selección y evaluación final.

La Entrega 4 queda así alineada con el alcance actual del TFM: se ha construido un núcleo predictivo riguroso, comparable con una referencia sencilla y preparado para evolucionar hacia un futuro sistema de apoyo a la planificación turística rural.

---

# 10. Implementación y resultados obtenidos

## 10.1. Componentes implementados

La estrategia descrita en este documento se ha llevado a una implementación reproducible dentro del repositorio.

### Configuración y contratos

```text
data/metadata/modeling_config.yml
data/metadata/schema_modeling.yml
data/metadata/modeling_validation_rules.yml
```

### Ingeniería y validación de datos

```text
src/features/build_modeling_dataset.py
src/features/validate_modeling_dataset.py
data/gold/gold_modeling_dataset_monthly.parquet
data/metadata/modeling_data_quality_report.md
tests/test_build_modeling_dataset.py
```

### Evaluación de modelos

```text
src/models/evaluate_baseline.py
src/models/evaluate_final_candidate.py
data/model_outputs/final_candidate_predictions.parquet
```

### Informes y métricas

```text
data/metadata/baseline_evaluation_report.md
data/metadata/baseline_metrics_summary.csv
data/metadata/baseline_metrics_by_territory.csv
data/metadata/baseline_metrics_by_month.csv
data/metadata/baseline_metrics_by_season.csv
data/metadata/model_selection_validation_report.md
data/metadata/final_candidate_metrics_by_split.csv
data/metadata/final_candidate_test_by_territory.csv
data/metadata/final_candidate_test_by_month.csv
```

## 10.2. Dataset de modelado construido

El dataset final de modelado es:

```text
data/gold/gold_modeling_dataset_monthly.parquet
```

Sus principales características son:

| Indicador | Resultado |
|---|---:|
| Filas | 12.691 |
| Columnas | 37 |
| Provincias | 50 |
| Periodo | 2005-01 a 2026-05 |
| Claves duplicadas | 0 |
| Train inicial | 9.691 |
| Validation 1 | 600 |
| Validation 2 | 600 |
| Validation 3 | 600 |
| Test final | 600 |
| Seguimiento provisional | 600 |

La clasificación de calidad de las filas es:

| Estado | Filas |
|---|---:|
| `ok` | 10.606 |
| `insufficient_history` | 595 |
| `missing_lag` | 890 |
| `provisional_target` | 600 |

La validación reproducible del dataset terminó con:

```text
62 PASS / 0 WARN / 0 FAIL
```

La suite automatizada contiene 12 pruebas que cubren la construcción temporal del dataset y utilidades compartidas de modelado. Entre otros controles, verifica que los lags buscan el mes calendario exacto, que los huecos no se sustituyen por cero, que las medias móviles excluyen el mes objetivo y requieren ventanas completas, y que la configuración de inputs, folds, filas comparables y métricas de modelado mantiene el comportamiento esperado.

Las dependencias de desarrollo se declaran en `requirements-dev.txt` y la suite se ejecuta también automáticamente mediante GitHub Actions en cada `push` y `pull_request` sobre `main`.

## 10.3. Cobertura común de evaluación

La comparación con el baseline utiliza únicamente filas:

- con target conocido;
- con `lag_12_overnight_stays` disponible;
- no provisionales;
- pertenecientes a los folds temporales definidos.

La cobertura común es:

| Split | Filas totales | Filas evaluables |
|---|---:|---:|
| Validation 1 | 600 | 550 |
| Validation 2 | 600 | 600 |
| Validation 3 | 600 | 600 |
| Test | 600 | 600 |

Las 50 filas no evaluables de `validation_1` corresponden al mes cuyo lag anual apunta a noviembre de 2020, uno de los meses globalmente ausentes.

## 10.4. Resultado del baseline estacional

El baseline utiliza:

```text
predicción = lag_12_overnight_stays
```

Resultados por partición:

| Split | Filas | MAE | RMSE | WAPE | Sesgo medio |
|---|---:|---:|---:|---:|---:|
| Validation 1 | 550 | 9.116,56 | 16.454,67 | 44,60 % | -8.782,89 |
| Validation 2 | 600 | 3.071,71 | 6.094,11 | 14,91 % | -909,25 |
| Validation 3 | 600 | 3.209,14 | 5.552,10 | 15,16 % | -571,43 |
| Test | 600 | 3.045,00 | 5.236,82 | 14,35 % | -49,18 |

El comportamiento excepcionalmente desfavorable de `validation_1` refleja la comparación entre la recuperación posterior y meses afectados por la ruptura de COVID-19. En las ventanas posteriores, el baseline muestra un rendimiento mucho más estable.

## 10.5. Resultado de Ridge

Ridge se implementó mediante un pipeline con:

- imputación por mediana ajustada exclusivamente con cada entrenamiento;
- indicadores de ausencia;
- escalado de variables numéricas;
- codificación one-hot de provincia, mes y trimestre;
- predicciones recortadas a cero;
- búsqueda temporal limitada del parámetro `alpha`.

La mejor configuración fue:

```text
alpha = 100
```

Resultado agregado sobre las tres validaciones:

| Modelo | Filas | MAE | MAE baseline | Mejora |
|---|---:|---:|---:|---:|
| Ridge | 1.750 | 5.450,77 | 5.018,64 | -8,61 % |

Ridge solo mejoró el baseline en `validation_2`, con una mejora del 3,71 %, y no alcanzó el umbral práctico definido. Por ello quedó descartado como candidato principal y no se utilizó el test para reajustarlo.

## 10.6. Selección de HistGradientBoosting

Se probaron configuraciones limitadas de `HistGradientBoostingRegressor` con target original y transformación `log1p`, utilizando únicamente las tres validaciones.

La configuración seleccionada fue:

```text
config_id = hgb_raw_02
target_transform = raw
learning_rate = 0.05
max_iter = 300
max_leaf_nodes = 31
min_samples_leaf = 20
l2_regularization = 1.0
early_stopping = False
random_state = 42
```

Resultado agregado de validación:

| Modelo | Filas | MAE | MAE baseline | Mejora |
|---|---:|---:|---:|---:|
| `hgb_raw_02` | 1.750 | 5.124,56 | 5.018,64 | -2,11 % |

Resultado por fold:

| Split | MAE HGB | MAE baseline | Mejora |
|---|---:|---:|---:|
| Validation 1 | 8.081,94 | 9.116,56 | 11,35 % |
| Validation 2 | 4.133,83 | 3.071,71 | -34,58 % |
| Validation 3 | 3.404,34 | 3.209,14 | -6,08 % |

El candidato mejora durante la recuperación posterior a COVID-19, pero empeora en las dos validaciones más recientes. Por tanto, no cumple el criterio de mejora agregada y estabilidad temporal.

## 10.7. Evaluación única del test final

Tras congelar la configuración seleccionada, el modelo se entrenó con 10.737 filas evaluables hasta mayo de 2024 y se evaluó una sola vez en el periodo junio de 2024 a mayo de 2025.

| Modelo | Filas | MAE | RMSE | WAPE | Sesgo medio |
|---|---:|---:|---:|---:|---:|
| Baseline lag-12 | 600 | 3.045,00 | 5.236,82 | 14,35 % | -49,18 |
| `hgb_raw_02` | 600 | 2.760,59 | 4.550,60 | 13,01 % | 378,82 |

En el test final, HistGradientBoosting:

- mejora el MAE un 9,34 %;
- reduce el RMSE aproximadamente un 13,10 %;
- reduce el WAPE de 14,35 % a 13,01 %;
- mejora en 41 de 50 provincias, un 82 %;
- mejora en 7 de los 12 meses;
- no produce predicciones negativas.

Las mayores mejoras mensuales se observan en marzo y abril. Los principales deterioros aparecen en agosto, septiembre y diciembre.

## 10.8. Decisión final

Los criterios definidos antes de abrir el test producen el siguiente resultado:

| Criterio | Resultado |
|---|---|
| Mejora agregada en validación igual o superior al 5 % | FAIL |
| Mejora en test igual o superior al 5 % | PASS |
| Mejora en la mayoría de provincias | PASS |
| Promoción completa como único modelo operativo | FAIL |

La decisión final es:

1. Mantener el baseline estacional lag-12 como referencia robusta y mecanismo de fallback.
2. Conservar `hgb_raw_02` como mejor candidato de machine learning y modelo ganador en el test final.
3. Mostrar ambos resultados de forma transparente.
4. No realizar nuevos ajustes basados en el test final.
5. Definir cualquier mejora posterior como un nuevo experimento temporal independiente.

Esta decisión evita seleccionar el modelo únicamente por un resultado favorable en test y preserva la metodología definida antes de conocer dicho resultado.

Por coherencia con esta decisión, no se serializa ni publica `hgb_raw_02`
como artefacto de producción. El candidato permanece reproducible mediante
el pipeline y la configuración versionados en el repositorio, pero no se
presenta como modelo desplegable mientras no satisfaga los criterios de
promoción definidos. Si una futura iteración supera dichos criterios, la
serialización y el pipeline de inferencia deberán incorporarse como parte
de esa nueva fase, sin reutilizar el test actual para ajustar el modelo.

## 10.9. Limitaciones observadas y siguientes mejoras

Los resultados muestran que la demanda turística rural presenta una estacionalidad anual muy fuerte y que un modelo más complejo no garantiza una mejora estable.

Las ampliaciones futuras deberán considerar:

- modelos especializados por grupos de provincias;
- variables de calendario móvil, especialmente Semana Santa;
- clima y eventos cuando puedan integrarse sin fuga temporal;
- enfoques estadísticos por provincia;
- intervalos de predicción basados en errores fuera de muestra;
- reglas territoriales de fallback;
- seguimiento separado de datos provisionales;
- actualización periódica sin reutilizar el test ya abierto para reajustar hiperparámetros.
