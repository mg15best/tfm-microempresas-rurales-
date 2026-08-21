# TFM - Analítica predictiva de demanda turística rural

Este repositorio contiene el desarrollo incremental del Trabajo Final de Máster del Máster en Big Data e Inteligencia Artificial.

El proyecto desarrolla una prueba de concepto de **analítica predictiva de demanda turística rural**, orientada a alojamientos rurales y a pequeñas empresas vinculadas al flujo de visitantes: restauración, comercio local, actividades, guías, transporte y entidades de apoyo territorial.

El alcance se centra en datos turísticos territoriales agregados. El proyecto no pretende predecir la facturación ni el beneficio de una empresa concreta.

## Objetivo

El objetivo es transformar estadísticas oficiales españolas sobre turismo rural en un sistema de apoyo a la toma de decisiones que permita:

* analizar viajeros, pernoctaciones, estancia media y ocupación;
* identificar patrones de estacionalidad;
* comparar provincias;
* distinguir demanda residente y extranjera;
* anticipar la demanda turística mensual;
* generar indicadores y orientaciones operativas explicables.

La variable objetivo principal del modelado es:

```text
pernoctaciones mensuales por provincia
```

El horizonte implementado es de un mes y la evaluación se realiza mediante backtesting temporal con ventanas expansivas.

## Estado actual del proyecto

Actualmente están implementados:

* descarga reproducible de fuentes oficiales del INE;
* registro de descargas y hashes SHA-256;
* capas raw y processed;
* dimensiones de territorio y calendario;
* tabla gold descriptiva;
* contrato formal de las 64 columnas de la tabla gold;
* validación automatizada de calidad;
* análisis exploratorio en notebook;
* dataset específico de modelado con lags calendario y ventanas históricas;
* contrato y validación reproducible del dataset de modelado;
* control point-in-time que separa mes de referencia y fecha de publicación;
* purge de etiquetas de entrenamiento todavía no publicadas en cada forecast origin;
* infraestructura de evaluación point-in-time V2 y backtesting rolling;
* candidatos `lag_12`, tendencia estacional y ETS evaluados sin fuga temporal;
* artefacto canónico de validación ETS congelado y verificable por hash;
* ETS como `provisional_validation_champion`, sin confirmación en una nueva
  ventana final intacta e independiente;
* inferencia operacional ETS con `lag_12` solo como fallback de disponibilidad,
  no como router competitivo entre modelos;
* intervalo empírico operacional del 80 % calibrado prequentialmente;
* contexto de decisión basado en posición histórica Q25/Q75;
* frontal funcional con Streamlit y Plotly para las 50 provincias;
* estados controlados de error, caché y descarga CSV;
* test histórico anterior protegido frente a reutilización.

Estado técnico actual tras B5:

| Indicador | Resultado |
|---|---:|
| Modelo seleccionado | ETS (`holt_winters_additive_damped_v1`) |
| Estado de selección | `provisional_validation_champion` |
| Evidencia | `canonical_rolling_validation` |
| MAE pooled canónico | 4.084,574535196216 |
| RMSE pooled canónico | 7.770,827125343509 |
| WAPE pooled canónico | 19,68924738072576 % |
| Sesgo medio pooled canónico | -1.862,1237643616084 |
| Fallback | `seasonal_naive_lag_12`, solo por disponibilidad |
| Incertidumbre | Intervalo empírico operacional del 80 % (`operational_prequential_scaled_absolute_residual_interval_v1`) |
| Frontal | Streamlit + Plotly |

La Entrega 4 conserva la comparación histórica entre baseline, Ridge y HGB y
su test ya abierto en `docs/entregas/04_analisis_modelado.md`. Esa evidencia
permanece como trazabilidad, pero no representa la selección operacional B5 ni
se reutiliza para reajustar el sistema actual.

## Fuentes utilizadas

La primera versión funcional utiliza la Encuesta de Ocupación en Alojamientos de Turismo Rural del Instituto Nacional de Estadística:

| Tabla INE | Contenido                                               | Nivel     | Frecuencia |
| --------: | ------------------------------------------------------- | --------- | ---------- |
|    `2073` | Viajeros y pernoctaciones por residencia                | Provincia | Mensual    |
|    `2070` | Establecimientos, plazas, ocupación y personal empleado | Provincia | Mensual    |

Las fuentes de precios, gasto turístico, empresas y climatología se encuentran documentadas como posibles ampliaciones, pero no se integran todavía en el núcleo del MVP.

## Resultados principales

La tabla gold descriptiva se encuentra en:

```text
data/gold/gold_tourism_demand_monthly.parquet
```

El dataset específico de modelado se encuentra en:

```text
data/gold/gold_modeling_dataset_monthly.parquet
```

La evidencia canónica vigente para la selección ETS contiene las ventanas de
validación rolling point-in-time y se encuentra en:

```text
data/model_outputs/ets_v2_rolling_validation_predictions.parquet
data/metadata/ets_v2_rolling_validation_predictions.metadata.yml
```

El modelo seleccionado es ETS con estado `provisional_validation_champion`.
`lag_12` se utiliza únicamente cuando falta un input requerido por ETS en el
corte disponible. No existe un router que elija modelos por territorio.

Los resultados de test de la especificación anterior se conservan únicamente
como archivo histórico en
`data/model_outputs/historical_pre_point_in_time_final_candidate_predictions.parquet`.

También existe una exportación CSV de la tabla gold descriptiva:

```text
data/gold/exports_csv/gold_tourism_demand_monthly.csv
```

Características de la versión actual:

| Característica          |                            Resultado |
| ----------------------- | -----------------------------------: |
| Filas                   |                               12.741 |
| Columnas                |                                   64 |
| Provincias              |                                   50 |
| Periodo                 |                    2005-01 a 2026-06 |
| Claves duplicadas       |                                    0 |
| Registros provisionales |                                  600 |
| Tamaño del parquet      |                      1.809.283 bytes |
| Versión                 | `gold_tourism_demand_monthly_v1.0.0` |

La clave principal es:

```text
territory_id + month_id
```

Los datos desde julio de 2025 hasta junio de 2026 están marcados como
provisionales. El último mes disponible en la tabla gold es junio de 2026.

Los meses `2020-04`, `2020-05` y `2020-11` no contienen observaciones provinciales publicadas. No se rellenan con cero.

También existen nueve combinaciones provincia-mes sin viajeros ni pernoctaciones disponibles: Albacete (`2006-05`); Badajoz (`2005-04`, `2005-05`, `2005-07`, `2005-10`, `2005-11`, `2006-01` y `2006-06`); y Ciudad Real (`2006-05`). Estas combinaciones se excluyen de la capa gold sin imputarlas con cero y se registran en `data/metadata/missing_territory_months.csv`.

## Hallazgos iniciales

El análisis exploratorio identifica que:

* Illes Balears fue la provincia con más pernoctaciones rurales en 2024, con 1.688.709;
* agosto es el mes con mayor demanda media;
* Santa Cruz de Tenerife alcanzó su máximo histórico de la serie en agosto de 2005, con 36.085 pernoctaciones;
* el último mes disponible es junio de 2026.

Estos resultados son agregados territoriales y no representan la demanda o rentabilidad de una empresa individual.

## Estructura del repositorio

```text
app.py
data/
├── raw/
├── processed/
├── gold/
├── metadata/
└── model_outputs/

docs/
├── assets/
│   └── 05_mockup_frontal.png
└── entregas/
    ├── 01_ideas_producto.md
    ├── 02_datos_necesarios.md
    ├── 03_modelo_datos.md
    ├── 04_analisis_modelado.md
    └── 05_diseno_frontal.md

notebooks/
└── 01_data_exploration.ipynb

reports/
└── figures/

src/
├── application/
│   ├── forecast_service.py
│   └── decision_support.py
├── data/
│   ├── download_sources.py
│   ├── normalize_sources.py
│   ├── build_dimensions.py
│   ├── build_gold.py
│   └── validate_gold.py
├── features/
│   ├── build_modeling_dataset.py
│   └── validate_modeling_dataset.py
├── models/
│   ├── modeling_common.py
│   ├── evaluate_baseline.py
│   ├── select_models.py
│   ├── ets_v2.py
│   ├── inference.py
│   └── prediction_intervals_v2.py
└── visualization/
    ├── dashboard_data.py
    └── streamlit_app.py

tests/
├── test_build_gold.py
├── test_ets_v2.py
├── test_inference.py
├── test_prediction_intervals_v2.py
├── test_forecast_service.py
├── test_decision_support.py
├── test_dashboard_data.py
└── test_streamlit_app.py

requirements.txt
README.md
```

## Requisitos

* Python 3.14.2, versión utilizada y comprobada en el desarrollo actual
* Git
* Visual Studio Code, recomendado
* Extensiones Python y Jupyter de VS Code, para ejecutar el notebook

Dependencias principales:

```text
pandas
pyarrow
PyYAML
matplotlib
ipykernel
nbconvert
scikit-learn
statsmodels
plotly
streamlit
```

Las versiones exactas comprobadas se encuentran fijadas en `requirements.txt`. El uso de otras versiones puede funcionar, pero no forma parte del entorno validado en esta entrega.

## Instalación en Windows

Clonar el repositorio y entrar en su carpeta:

```powershell
git clone https://github.com/mg15best/tfm-microempresas-rurales-.git
cd tfm-microempresas-rurales-
```

Crear un entorno virtual:

```powershell
python -m venv .venv
```

Instalar las dependencias dentro del entorno:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Comprobar la instalación:

```powershell
.\.venv\Scripts\python.exe -c "import pandas, pyarrow, yaml, matplotlib, sklearn, statsmodels, plotly, streamlit; print('Dependencias instaladas correctamente')"
```

La carpeta `.venv/` no debe subirse al repositorio.

## Ejecución del pipeline

Todos los comandos deben ejecutarse desde la raíz del repositorio.

### 1. Descargar las fuentes principales

```powershell
.\.venv\Scripts\python.exe src/data/download_sources.py --source all
```

También se puede descargar una tabla concreta:

```powershell
.\.venv\Scripts\python.exe src/data/download_sources.py --source 2073
.\.venv\Scripts\python.exe src/data/download_sources.py --source 2070
```

Cada ejecución crea un nuevo snapshot raw y añade una fila a:

```text
data/metadata/download_log.csv
```

### 2. Normalizar las fuentes

```powershell
.\.venv\Scripts\python.exe src/data/normalize_sources.py --source all
```

### 3. Construir las dimensiones

```powershell
.\.venv\Scripts\python.exe src/data/build_dimensions.py
```

### 4. Construir la capa gold

```powershell
.\.venv\Scripts\python.exe src/data/build_gold.py
```

### 5. Ejecutar las validaciones

```powershell
.\.venv\Scripts\python.exe src/data/validate_gold.py
```

Una ejecución correcta devuelve código de salida `0` y actualiza:

```text
data/metadata/data_quality_report.md
```

En PowerShell puede comprobarse mediante:

```powershell
$LASTEXITCODE
```

### 6. Construir el dataset de modelado

```powershell
.\.venv\Scripts\python.exe src/features/build_modeling_dataset.py
```

### 7. Validar el dataset de modelado

```powershell
.\.venv\Scripts\python.exe src/features/validate_modeling_dataset.py
```

Una ejecución correcta devuelve código `0` y actualiza:

```text
data/metadata/modeling_data_quality_report.md
```

### 8. Consultar la evidencia canónica y la selección B5

La selección operacional usa el bundle ETS congelado en:

```text
data/model_outputs/ets_v2_rolling_validation_predictions.parquet
data/metadata/ets_v2_rolling_validation_predictions.metadata.yml
```

ETS gana la regla de validación en dos de tres folds y queda como campeón
provisional. La aplicación verifica y consume ese bundle; no lo regenera al
arrancar. La evaluación se basa exclusivamente en validación rolling
point-in-time y no existe una nueva ventana final intacta e independiente.
El test histórico de Entrega 4 continúa protegido y no interviene en B5.

### 9. Ejecutar el frontal Streamlit

```powershell
streamlit run app.py
```

La aplicación permite seleccionar una provincia y consultar el pronóstico del
mes siguiente, su intervalo empírico operacional del 80 %, la posición
histórica, la orientación de planificación, la evidencia territorial y la
procedencia. También puede iniciarse con el ejecutable del entorno:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

### 10. Ejecutar las pruebas automatizadas

Instalar las dependencias de desarrollo:

```powershell
python -m pip install -r requirements-dev.txt
```

Ejecutar la suite:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

La suite actual contiene 287 pruebas automatizadas, más 38 subtests, y se
ejecuta también mediante GitHub Actions en cada `push` y `pull_request` sobre
`main`.

## Análisis exploratorio

El análisis se encuentra en:

```text
notebooks/01_data_exploration.ipynb
```

Para ejecutarlo en VS Code:

1. abrir el notebook;
2. seleccionar el entorno `.venv` como kernel;
3. ejecutar todas las celdas mediante **Run All**;
4. guardar el notebook para conservar tablas y gráficos.

También puede comprobarse la ejecución completa del notebook desde terminal:

```powershell
.\.venv\Scripts\python.exe -m jupyter nbconvert `
  --to notebook `
  --execute notebooks\01_data_exploration.ipynb `
  --output 01_data_exploration.executed.ipynb `
  --output-dir notebooks `
  --ExecutePreprocessor.timeout=600
```

El notebook analiza:

* estructura y valores nulos;
* evolución mensual y anual;
* estacionalidad;
* impacto del periodo COVID-19;
* ranking provincial;
* demanda residente y extranjera;
* ocupación y presión turística;
* caso de Santa Cruz de Tenerife.

## Calidad y trazabilidad

Las reglas se encuentran en:

```text
data/metadata/validation_rules.yml
```

El contrato formal de la tabla gold se encuentra en:

```text
data/metadata/schema_gold.yml
```

El validador comprueba:

* columnas obligatorias;
* claves no nulas y únicas;
* cobertura territorial;
* coherencia temporal;
* valores no negativos;
* porcentajes entre 0 y 100;
* proporciones entre 0 y 1;
* coherencia de totales;
* cálculo de estancia media;
* clasificación provisional;
* integridad con las dimensiones;
* snapshots, versión e identificador de ejecución.

Los marcadores no numéricos de la fuente se convierten en nulos reales. No se imputan con cero salvo que la fuente indique explícitamente que el valor observado es cero.

## Documentación académica

Las entregas incrementales se encuentran en:

```text
docs/entregas/
```

* `01_ideas_producto.md`: definición y selección inicial de la idea.
* `02_datos_necesarios.md`: identificación y evaluación de datos necesarios.
* `03_modelo_datos.md`: arquitectura raw, processed y gold, granularidades, claves, calidad y transformaciones.
* `04_analisis_modelado.md`: problema predictivo, dataset de modelado, baseline, modelos candidatos, backtesting temporal, resultados de validación y test, decisión final, limitaciones y alternativas.
* `05_diseno_frontal.md`: usuario, solución de producto, mockup funcional, flujo, UX, comunicación de resultados, explicabilidad y alcance del MVP.

## Limitaciones actuales

* El núcleo actual trabaja únicamente a nivel provincial.
* La fuente mide alojamientos de turismo rural, no todo el turismo de un territorio.
* Los datos provisionales pueden ser revisados por el INE.
* Las variables de precios, gasto, clima y eventos todavía no están integradas.
* El índice de presión turística es una señal descriptiva y no una medida directa de rentabilidad.
* ETS es un campeón provisional de validación: supera la regla de selección en
  dos de tres folds, con variabilidad temporal que debe permanecer visible.
* No existe una nueva ventana final intacta e independiente para confirmar la
  selección y el test histórico ya abierto no debe reutilizarse.
* La evidencia canónica usa el último vintage revisado y métricas pooled; no
  garantiza comportamiento o cobertura individual por provincia y temporada.
* El intervalo del 80 % es empírico y operacional, no una garantía condicional.
* `lag_12` es fallback solo por disponibilidad y no una política de selección
  competitiva por provincia.

## Siguientes pasos

1. Confirmar la selección únicamente cuando exista una nueva ventana final
   intacta e independiente.
2. Evaluar horizontes multiperiodo y cobertura por provincia y temporada.
3. Incorporar precios, gasto, clima, eventos o datos internos del negocio con
   disponibilidad point-in-time demostrada.
4. Añadir escenarios what-if sin confundirlos con predicciones observadas.
5. Estudiar autenticación, API y despliegue multiusuario si el MVP evoluciona a
   producto operativo.
6. Completar una auditoría formal de accesibilidad y pruebas con usuarios.
