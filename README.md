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
* generar posteriormente indicadores y recomendaciones operativas explicables.

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
* dataset específico de modelado con lags calendario y ventanas históricas sin fuga;
* contrato y validación reproducible del dataset de modelado;
* baseline estacional `lag_12`;
* regresión Ridge como candidato interpretable;
* `HistGradientBoostingRegressor` como candidato flexible;
* backtesting temporal con ventana expansiva;
* evaluación final no provisional;
* métricas globales, territoriales y mensuales;
* predicciones reproducibles del candidato final.

Resultado resumido de la Entrega 4:

| Indicador | Resultado |
|---|---:|
| Dataset de modelado | 12.691 filas y 37 columnas |
| Validación del dataset | 62 PASS / 0 WARN / 0 FAIL |
| MAE baseline en test | 3.045,00 |
| MAE `hgb_raw_02` en test | 2.760,59 |
| Mejora de MAE en test | 9,34 % |
| Provincias mejoradas | 41 de 50 |
| Mejora agregada en validación | -2,11 % |

`hgb_raw_02` es el mejor candidato de machine learning y gana en el test final, pero no cumple la regla completa de promoción porque no mejora el baseline de forma agregada y estable en validación. El baseline estacional se mantiene como referencia robusta y mecanismo de fallback.

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

Las predicciones reproducibles de validación y test del candidato final se encuentran en:

```text
data/model_outputs/final_candidate_predictions.parquet
```

También existe una exportación CSV de la tabla gold descriptiva:

```text
data/gold/exports_csv/gold_tourism_demand_monthly.csv
```

Características de la versión actual:

| Característica          |                            Resultado |
| ----------------------- | -----------------------------------: |
| Filas                   |                               12.691 |
| Columnas                |                                   64 |
| Provincias              |                                   50 |
| Periodo                 |                    2005-01 a 2026-05 |
| Claves duplicadas       |                                    0 |
| Registros provisionales |                                  600 |
| Versión                 | `gold_tourism_demand_monthly_v1.0.0` |

La clave principal es:

```text
territory_id + month_id
```

Los datos desde junio de 2025 hasta mayo de 2026 están marcados como provisionales.

Los meses `2020-04`, `2020-05` y `2020-11` no contienen observaciones provinciales publicadas. No se rellenan con cero.

También existen nueve combinaciones provincia-mes sin viajeros ni pernoctaciones disponibles: Albacete (`2006-05`); Badajoz (`2005-04`, `2005-05`, `2005-07`, `2005-10`, `2005-11`, `2006-01` y `2006-06`); y Ciudad Real (`2006-05`). Estas combinaciones se excluyen de la capa gold sin imputarlas con cero y se registran en `data/metadata/missing_territory_months.csv`.

## Hallazgos iniciales

El análisis exploratorio identifica que:

* Illes Balears fue la provincia con más pernoctaciones rurales en 2024, con 1.688.709;
* agosto es el mes con mayor demanda media;
* Santa Cruz de Tenerife alcanzó su máximo histórico de la serie en agosto de 2005, con 36.085 pernoctaciones;
* el último mes disponible es mayo de 2026 y tiene carácter provisional.

Estos resultados son agregados territoriales y no representan la demanda o rentabilidad de una empresa individual.

## Estructura del repositorio

```text
app/
data/
├── raw/
├── processed/
├── gold/
└── metadata/

docs/
└── entregas/
    ├── 01_ideas_producto.md
    ├── 02_datos_necesarios.md
    ├── 03_modelo_datos.md
    └── 04_analisis_modelado.md

notebooks/
└── 01_data_exploration.ipynb

reports/
└── figures/

src/
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
│   ├── evaluate_baseline.py
│   └── evaluate_final_candidate.py
└── visualization/

tests/
├── test_build_gold.py
└── test_build_modeling_dataset.py

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
.\.venv\Scripts\python.exe -c "import pandas, pyarrow, yaml, matplotlib, ipykernel, nbconvert, sklearn; print('Dependencias instaladas correctamente')"
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

### 8. Evaluar el baseline estacional

```powershell
.\.venv\Scripts\python.exe src/models/evaluate_baseline.py
```

### 9. Reproducir la evaluación del candidato final

```powershell
.\.venv\Scripts\python.exe src/models/evaluate_final_candidate.py
```

Esta ejecución reproduce los folds de validación y el test final con la configuración congelada `hgb_raw_02`, y genera:

```text
data/model_outputs/final_candidate_predictions.parquet
data/metadata/final_candidate_evaluation_report.md
data/metadata/final_candidate_metrics_by_split.csv
data/metadata/final_candidate_test_by_territory.csv
data/metadata/final_candidate_test_by_month.csv
```

### 10. Ejecutar las pruebas automatizadas

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
```

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

## Limitaciones actuales

* El núcleo actual trabaja únicamente a nivel provincial.
* La fuente mide alojamientos de turismo rural, no todo el turismo de un territorio.
* Los datos provisionales pueden ser revisados por el INE.
* Las variables de precios, gasto, clima y eventos todavía no están integradas.
* El índice de presión turística es una señal descriptiva y no una medida directa de rentabilidad.
* El mejor candidato de machine learning no mejora el baseline de forma estable en las tres validaciones.
* El test final ya se ha abierto y no debe reutilizarse para reajustar hiperparámetros.
* Todavía no se han implementado intervalos de predicción ni una política operativa de selección por provincia.

## Siguientes pasos

1. Diseñar el dashboard o aplicación ligera para consultar histórico, baseline, predicción y error territorial.
2. Incorporar intervalos de predicción calculados con errores fuera de muestra.
3. Definir reglas de fallback por provincia, mes o temporada.
4. Analizar el periodo provisional de junio de 2025 a mayo de 2026 como seguimiento separado.
5. Evaluar nuevas variables o familias de modelos únicamente mediante un nuevo experimento temporal independiente.
6. Preparar la presentación académica y la demostración reproducible de la Entrega 4.
