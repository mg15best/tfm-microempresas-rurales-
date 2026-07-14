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

La variable objetivo principal prevista para el modelado es:

```text
pernoctaciones mensuales por provincia
```

## Estado actual del proyecto

Actualmente están implementados:

* descarga reproducible de fuentes oficiales del INE;
* registro de descargas y hashes SHA-256;
* capas raw y processed;
* dimensiones de territorio y calendario;
* tabla gold descriptiva;
* validación automatizada de calidad;
* informe de calidad;
* análisis exploratorio en notebook.

Pendiente para las siguientes fases:

* contrato formal del esquema gold;
* construcción de features temporales;
* dataset específico de modelado;
* baselines y modelos predictivos;
* backtesting temporal;
* dashboard;
* recomendaciones por tipo de microempresa.

## Fuentes utilizadas

La primera versión funcional utiliza la Encuesta de Ocupación en Alojamientos de Turismo Rural del Instituto Nacional de Estadística:

| Tabla INE | Contenido                                               | Nivel     | Frecuencia |
| --------: | ------------------------------------------------------- | --------- | ---------- |
|    `2073` | Viajeros y pernoctaciones por residencia                | Provincia | Mensual    |
|    `2070` | Establecimientos, plazas, ocupación y personal empleado | Provincia | Mensual    |

Las fuentes de precios, gasto turístico, empresas y climatología se encuentran documentadas como posibles ampliaciones, pero no se integran todavía en el núcleo del MVP.

## Resultado principal

La tabla gold se encuentra en:

```text
data/gold/gold_tourism_demand_monthly.parquet
```

También existe una exportación CSV:

```text
data/gold/exports_csv/gold_tourism_demand_monthly.csv
```

Características de la versión actual:

| Característica          |                            Resultado |
| ----------------------- | -----------------------------------: |
| Filas                   |                               12.693 |
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
    └── 03_modelo_datos.md

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
├── models/
└── visualization/

requirements.txt
README.md
```

## Requisitos

* Python 3
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
```

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
.\.venv\Scripts\python.exe -c "import pandas, pyarrow, yaml, matplotlib, ipykernel; print('Dependencias instaladas correctamente')"
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

## Limitaciones actuales

* El núcleo actual trabaja únicamente a nivel provincial.
* La fuente mide alojamientos de turismo rural, no todo el turismo de un territorio.
* Los datos provisionales pueden ser revisados por el INE.
* Las variables de precios y gasto todavía no están integradas.
* El índice de presión turística es una señal relativa y no una medida directa de rentabilidad.
* Todavía no se ha construido ni evaluado el modelo predictivo.

## Siguientes pasos

1. Crear el contrato formal `schema_gold.yml`.
2. Construir features temporales y lags sin fuga de información.
3. Generar `gold_modeling_dataset_monthly.parquet`.
4. Definir train, validación y test mediante separación temporal.
5. Comparar baselines y modelos predictivos.
6. Desarrollar el dashboard y las recomendaciones explicables.
