# TFM - Analítica predictiva de demanda turística rural

Este repositorio contiene el desarrollo incremental del Trabajo Final de Máster del Máster en Big Data e Inteligencia Artificial.

El proyecto propone una prueba de concepto de **analítica predictiva de demanda turística rural** orientada a alojamientos rurales y microempresas locales vinculadas al flujo de visitantes: restauración, comercio local, actividades, guías, transporte y entidades de apoyo territorial.

Tras el feedback recibido en la segunda entrega, el proyecto se ha acotado a un sector concreto: **alojamientos de turismo rural y ecosistema económico local asociado**. Esta decisión permite trabajar con fuentes oficiales españolas y evita entrenar modelos con datos no representativos del usuario final.

## Objetivo del proyecto

El objetivo es transformar estadísticas oficiales españolas sobre turismo rural en un sistema de apoyo a la toma de decisiones que permita:

- analizar la evolución de viajeros, pernoctaciones, estancia media y ocupación;
- anticipar demanda turística mensual por territorio;
- identificar patrones de estacionalidad;
- comparar territorios;
- estimar oportunidades relativas para distintos tipos de microempresas locales;
- generar recomendaciones operativas explicables.

El proyecto no pretende predecir la facturación exacta ni el beneficio neto de una empresa concreta. Su alcance se centra en estimar **demanda turística territorial**, **intensidad turística esperada** y **oportunidades operativas relativas**.

## Fuentes de datos previstas

El proyecto utilizará únicamente fuentes españolas oficiales o plataformas públicas estatales basadas en datos oficiales:

- INE / Dataestur - Encuesta de Ocupación en Alojamientos de Turismo Rural.
- INE - Índice de Precios de Alojamientos de Turismo Rural.
- INE / Dataestur - Turismo de residentes en España.
- INE / Dataestur - EGATUR, gasto turístico de visitantes internacionales.
- INE / Dataestur - Empresas activas asociadas a la actividad turística.
- Fuentes oficiales opcionales para festivos, calendario o climatología.

La fuente principal del proyecto será la Encuesta de Ocupación en Alojamientos de Turismo Rural. Las demás fuentes se utilizarán como contexto o enriquecimiento, siempre que su granularidad territorial y temporal sea compatible.

## Estructura del repositorio

```text
docs/
└── entregas/
    ├── 01_ideas_producto.md
    ├── 02_datos_necesarios.md
    └── 03_modelo_datos.md

data/
├── raw/
├── processed/
├── gold/
└── metadata/

notebooks/
src/
reports/
app/
