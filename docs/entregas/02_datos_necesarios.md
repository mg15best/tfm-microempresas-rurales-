# Entrega 2 - Selección de idea de proyecto y análisis de datos necesarios

## Trazabilidad del repositorio

Este documento corresponde al archivo `docs/entregas/02_datos_necesarios.md`.

La entrega anterior se mantiene en `docs/entregas/01_ideas_producto.md`, garantizando la trazabilidad del proyecto desde la fase inicial de ideación hasta la selección y posterior acotación de la idea definitiva.

---

# 1. Idea seleccionada

La idea seleccionada para continuar el proyecto es un **sistema de analítica predictiva de demanda turística rural y estimación de oportunidades para microempresas locales**. La solución estará dirigida principalmente a alojamientos de turismo rural, pero también a pequeños negocios de los territorios vinculados directa o indirectamente con sus visitantes: restaurantes, cafeterías, comercios de producto local, artesanía, empresas de actividades, guías, transporte local y asociaciones empresariales o entidades públicas de apoyo al tejido económico rural.

## Problema que resuelve

Las microempresas rurales suelen tomar decisiones sobre apertura, personal, compras, existencias, campañas, actividades y colaboraciones con información limitada. En muchos casos conocen su experiencia pasada, pero no disponen de herramientas para anticipar con suficiente rigor cuándo aumentará o disminuirá la demanda turística del territorio, cuánto tiempo permanecerán los visitantes, qué peso tendrá la demanda nacional o extranjera o qué meses presentan mayor riesgo de baja actividad. Esta incertidumbre afecta tanto a los propios alojamientos rurales como a otros negocios que dependen del flujo de viajeros alojados en la zona. El problema no consiste únicamente en la falta de digitalización interna, sino también en la dificultad para transformar estadísticas oficiales dispersas en información comprensible y útil para la planificación de pequeños negocios.

## Solución planteada

La solución se abordará desde un enfoque de Data Science, analítica territorial y visualización de datos. El proyecto integrará exclusivamente **fuentes españolas oficiales**, principalmente la Encuesta de Ocupación en Alojamientos de Turismo Rural y el Índice de Precios de Alojamientos de Turismo Rural del Instituto Nacional de Estadística. Estas fuentes se complementarán, cuando la granularidad sea compatible, con información oficial sobre gasto turístico de residentes y visitantes internacionales y con datos agregados del tejido empresarial turístico. A partir de estas fuentes se construirá un conjunto de datos reproducible para analizar estacionalidad, comparar territorios, predecir viajeros, pernoctaciones u ocupación y generar indicadores de oportunidad para distintas actividades económicas locales.

El alojamiento rural se utilizará como **indicador principal de presencia turística** en el territorio. Las pernoctaciones, los viajeros, la estancia media y el grado de ocupación permiten aproximar cuándo existe una mayor concentración de visitantes potenciales. Esta demanda prevista se traducirá en recomendaciones operativas diferenciadas según el tipo de negocio. Por ejemplo, un restaurante podría anticipar periodos de mayor afluencia, una empresa de actividades podría programar experiencias en meses con mayor estancia media y un comercio local podría preparar campañas o existencias antes de periodos de alta ocupación.

## MVP del proyecto final

El producto mínimo viable consistirá en un **demostrador analítico interactivo** con cinco componentes:

1. Un pipeline reproducible de descarga, limpieza, normalización e integración de datos oficiales.
2. Un módulo descriptivo de demanda turística rural por territorio y periodo.
3. Un modelo predictivo para estimar una variable principal, previsiblemente pernoctaciones o grado de ocupación, con validación temporal.
4. Un módulo de segmentación o comparación de territorios según estacionalidad, ocupación, estancia media y procedencia de los viajeros.
5. Un sistema de recomendaciones explicables para alojamientos rurales y negocios locales relacionados con la actividad turística.

El resultado se mostrará mediante notebooks o scripts, documentación técnica y una aplicación ligera o dashboard interactivo. La interfaz permitirá seleccionar territorio y periodo, consultar evolución histórica, visualizar previsiones, comparar zonas y obtener recomendaciones operativas.

## Alcance del MVP

El MVP se centrará en **predecir demanda turística territorial y estimar oportunidades económicas relativas**, no en calcular las ventas o beneficios exactos de una empresa concreta.

Queda dentro del alcance:

- Analizar viajeros, pernoctaciones, estancia media, establecimientos, plazas, ocupación y personal ocupado.
- Analizar la evolución del índice de precios de alojamientos rurales.
- Comparar comunidades autónomas, provincias, zonas turísticas o puntos turísticos cuando existan datos suficientes.
- Predecir demanda a nivel territorial y mensual.
- Identificar temporadas altas, bajas y cambios anómalos.
- Agrupar territorios con comportamientos similares.
- Crear indicadores de oportunidad para alojamiento, restauración, actividades, comercio local y otros servicios.
- Generar recomendaciones explicables basadas en predicciones, indicadores y reglas transparentes.

Queda fuera del alcance:

- Predecir reservas, ventas o beneficios netos de un establecimiento individual.
- Construir un CRM, sistema de reservas o plataforma SaaS completa.
- Realizar recomendaciones personalizadas a clientes.
- Utilizar datos no españoles o fuentes privadas no oficiales como base del modelo.
- Afirmar que el gasto turístico total se distribuye íntegramente entre negocios del municipio.
- Presentar el prototipo como una solución comercial validada para cualquier territorio rural sin una validación local posterior.

## Naturaleza del resultado

El proyecto se presentará como una **prueba de concepto basada en datos oficiales**, válida para demostrar el potencial de la analítica territorial. Las predicciones indicarán demanda esperada del territorio, mientras que las estimaciones para negocios relacionados se formularán como:

- oportunidad económica potencial;
- presión de demanda comercial;
- intensidad turística esperada;
- escenario de actividad;
- recomendación operativa.

No se utilizará el término “beneficio previsto” salvo que en una fase posterior se disponga de datos internos de ingresos, costes, capacidad y márgenes de una empresa real.

---

# 2. Datos necesarios

## 2.1. Preguntas de negocio que deben responder los datos

Los datos deberán permitir responder, como mínimo, a las siguientes preguntas:

### Demanda turística rural

- ¿Cómo evolucionan los viajeros y las pernoctaciones por mes y territorio?
- ¿Qué meses presentan mayor o menor grado de ocupación?
- ¿Cuál es la estancia media y cómo varía a lo largo del año?
- ¿Qué territorios dependen más del turismo residente en España y cuáles de viajeros extranjeros?
- ¿Qué diferencias existen entre la ocupación general y la ocupación en fin de semana?
- ¿Qué territorios presentan una demanda estable y cuáles una estacionalidad elevada?
- ¿Se puede predecir la demanda de los próximos meses con un error razonable?

### Oferta y capacidad

- ¿Cuántos establecimientos y plazas están disponibles en cada territorio?
- ¿Cómo se relaciona la capacidad ofertada con las pernoctaciones y la ocupación?
- ¿Qué territorios presentan mayor presión de demanda respecto a su capacidad?
- ¿Cómo evoluciona el personal ocupado en relación con la actividad turística?
- ¿Existe relación entre evolución de precios y evolución de demanda?

### Ecosistema empresarial

- ¿Qué actividades económicas relacionadas con el turismo tienen mayor presencia en cada territorio?
- ¿Existe una concentración suficiente de alojamiento, restauración, transporte o actividades recreativas para absorber la demanda?
- ¿Qué territorios muestran un volumen de pernoctaciones elevado respecto al número de empresas turísticas?
- ¿Dónde podrían existir oportunidades de colaboración entre alojamientos y negocios complementarios?

### Oportunidad económica y recomendaciones

- ¿Qué meses son más adecuados para reforzar personal, existencias o programación de actividades?
- ¿Cuándo conviene iniciar campañas para reducir la estacionalidad?
- ¿Qué tipos de negocio pueden beneficiarse más de una subida prevista de pernoctaciones?
- ¿Cuándo se concentra la demanda en fines de semana?
- ¿Qué territorios presentan patrones similares y pueden compararse entre sí?
- ¿Qué recomendaciones operativas pueden formularse de manera transparente y justificable?

## 2.2. Variables o campos necesarios

### Fuente principal: ocupación en alojamientos de turismo rural

| Bloque | Variables o campos | Uso previsto |
|---|---|---|
| Tiempo | año, mes, periodo, fecha normalizada, trimestre, temporada | Series temporales, estacionalidad y validación temporal |
| Territorio | España, comunidad autónoma, provincia, zona turística, punto turístico, código territorial | Comparación geográfica y selección de caso de estudio |
| Procedencia | residentes en España, residentes en el extranjero, comunidad autónoma de procedencia o país de residencia cuando esté disponible | Perfil agregado de la demanda |
| Demanda | viajeros, pernoctaciones, estancia media | Medición y predicción de actividad turística |
| Oferta | establecimientos abiertos estimados, plazas estimadas | Capacidad disponible |
| Ocupación | grado de ocupación por plazas, por plazas en fin de semana y por habitaciones | Intensidad de uso y presión de demanda |
| Empleo | personal ocupado | Aproximación a necesidades operativas del sector |
| Modalidad | modalidad de alojamiento o alquiler, cuando esté disponible | Comparación entre categorías |
| Estado del dato | provisional/definitivo, disponible/no disponible, observación estadística | Control de calidad y trazabilidad |

### Índice de precios de alojamientos de turismo rural

| Bloque | Variables o campos | Uso previsto |
|---|---|---|
| Tiempo | año y mes | Evolución y comparación temporal |
| Territorio | nacional y comunidad autónoma cuando esté disponible | Comparación territorial |
| Precio | índice general, tasa de variación, modalidad de alquiler, tipo de tarifa | Variable explicativa y análisis de presión de precios |

### Turismo de residentes en España

| Bloque | Variables o campos | Uso previsto |
|---|---|---|
| Viaje | destino, duración, motivo, alojamiento principal, transporte | Caracterización agregada de la demanda |
| Gasto | gasto total, gasto medio y categorías de gasto | Construcción de escenarios de oportunidad |
| Categorías de gasto | paquete turístico, alojamiento, transporte, bares y restaurantes, actividades recreativas/culturales/deportivas, bienes duraderos y resto | Estimación relativa por tipo de negocio |
| Perfil | variables sociodemográficas agregadas disponibles | Contexto de demanda, no personalización individual |
| Tiempo y territorio | trimestre/año y destino según nivel publicado | Integración con la demanda rural cuando sea compatible |

### Gasto turístico de visitantes internacionales

| Bloque | Variables o campos | Uso previsto |
|---|---|---|
| Viaje | destino principal, país de residencia, vía de acceso, motivo, alojamiento, organización | Contextualización del visitante internacional |
| Gasto | gasto total, gasto medio, gasto medio diario y partidas de gasto disponibles | Escenarios de oportunidad para demanda extranjera |
| Tiempo | año y mes | Integración temporal |
| Territorio | España y destino principal según publicación | Uso solo cuando el nivel territorial sea compatible |

### Empresas activas asociadas al turismo

| Bloque | Variables o campos | Uso previsto |
|---|---|---|
| Tiempo | año | Evolución del tejido empresarial |
| Territorio | España, comunidad autónoma, provincia u otra desagregación disponible | Comparación territorial |
| Actividad | transporte, hostelería, agencias de viajes y otras actividades turísticas | Identificación del ecosistema local |
| Subactividad | alojamiento, servicios de comida y bebida, transporte, industria cultural, actividades deportivas, recreativas y de entretenimiento | Recomendaciones por tipo de negocio |
| Unidad | empresas activas y unidades locales | Densidad y capacidad empresarial |

### Variables derivadas

El proyecto generará variables adicionales:

- `year`
- `month`
- `quarter`
- `season`
- `territory_id`
- `territory_level`
- `travellers_total`
- `overnight_stays_total`
- `average_stay`
- `occupancy_rate`
- `weekend_occupancy_rate`
- `establishments_estimated`
- `places_estimated`
- `staff_employed`
- `domestic_share`
- `foreign_share`
- `overnight_stays_per_place`
- `travellers_per_establishment`
- `overnight_stays_per_tourism_business`
- `year_on_year_change`
- `month_on_month_change`
- `rolling_mean_3m`
- `rolling_mean_12m`
- `lag_1`
- `lag_3`
- `lag_12`
- `seasonality_index`
- `tourism_pressure_index`
- `business_opportunity_index`
- `covid_period`
- `forecast_value`
- `forecast_interval_lower`
- `forecast_interval_upper`

## 2.3. Variable objetivo del modelo

La variable objetivo principal se decidirá después del análisis exploratorio y de calidad. Las candidatas son:

1. **Pernoctaciones mensuales**, por representar mejor el volumen total de presencia turística que el número de viajeros.
2. **Grado de ocupación por plazas**, por relacionar demanda y capacidad.
3. **Viajeros mensuales**, como medida complementaria de afluencia.

La primera opción recomendada es predecir **pernoctaciones mensuales por territorio**, porque:

- refleja el volumen de presencia turística;
- se relaciona con consumo potencial de restauración, actividades y comercio;
- permite construir series temporales suficientemente extensas;
- puede agregarse y compararse con capacidad, estancia media y ocupación;
- evita depender de datos individuales.

El grado de ocupación se mantendrá como segunda variable objetivo o como indicador complementario.

## 2.4. Granularidad adecuada

La unidad analítica principal será:

> **territorio × mes**

El territorio se definirá mediante el nivel con mayor equilibrio entre detalle y estabilidad:

1. Provincia.
2. Zona turística.
3. Comunidad autónoma.
4. Punto turístico, únicamente cuando la serie tenga cobertura suficiente.

La granularidad prevista por fuente será:

| Fuente | Granularidad temporal | Granularidad territorial | Granularidad temática |
|---|---|---|---|
| Ocupación de turismo rural | mensual | nacional, comunidad autónoma, provincia, zona y punto turístico | oferta, demanda, procedencia y ocupación |
| Índice de precios rural | mensual | nacional y comunidad autónoma según tabla | índice, tarifa y modalidad |
| Turismo de residentes | trimestral o anual según tabla | destino publicado | viajes, motivo, alojamiento y gasto |
| EGATUR | mensual | destino publicado | visitante internacional, viaje y gasto |
| Empresas activas en turismo | anual | territorio publicado | actividad y subactividad turística |

No se afirmará que los resultados son municipales cuando la fuente solo permita trabajar por provincia, zona o punto turístico. Cuando una zona turística agrupe varios municipios, el dashboard mostrará expresamente su composición territorial.

## 2.5. Profundidad histórica necesaria

Para detectar estacionalidad y validar predicciones se necesita una serie temporal extensa.

Se plantea:

- **Mínimo aceptable:** 5 años completos de datos mensuales.
- **Objetivo del MVP:** entre 10 y 15 años de datos mensuales.
- **Escenario deseable:** utilizar toda la serie comparable disponible.
- **Comparación interanual:** al menos 24 meses.
- **Variables empresariales:** histórico anual disponible compatible.
- **Gasto turístico:** histórico suficiente para calcular medias y escenarios por periodo.

La Encuesta de Ocupación en Alojamientos de Turismo Rural dispone de resultados históricos anuales publicados desde 2001. La extensión exacta de las series mensuales y la comparabilidad de cada tabla se confirmarán durante la descarga.

Los años 2020 y 2021 requerirán tratamiento específico por el cierre de establecimientos y las alteraciones provocadas por la pandemia. Se evaluarán tres estrategias:

1. Mantenerlos e incluir una variable indicadora de periodo COVID.
2. Excluirlos del entrenamiento principal y utilizarlos como análisis de choque.
3. Comparar modelos con y sin esos periodos.

## 2.6. Volumen aproximado de datos

El proyecto no necesita millones de registros, ya que su dificultad principal está en integrar series oficiales heterogéneas y validarlas correctamente.

Volumen esperado después de normalizar las tablas a formato largo:

| Escenario | Volumen aproximado |
|---|---:|
| MVP mínimo, comunidad autónoma × mes | 3.000 - 10.000 observaciones |
| MVP provincial con varias variables | 10.000 - 50.000 observaciones |
| Provincia + zona turística + procedencia | 30.000 - 150.000 observaciones |
| Integración ampliada con precios, gasto y tejido empresarial | 50.000 - 300.000 observaciones |

El volumen final dependerá del número de territorios, periodos, procedencias y métricas integradas. La suficiencia del proyecto se valorará por la profundidad temporal, calidad, reproducibilidad y coherencia de las variables, no por alcanzar artificialmente un volumen masivo.

## 2.7. Datos imprescindibles y deseables

### Datos imprescindibles

- Año y mes.
- Identificador y nivel territorial.
- Viajeros.
- Pernoctaciones.
- Estancia media.
- Establecimientos abiertos estimados.
- Plazas estimadas.
- Grado de ocupación por plazas.
- Grado de ocupación en fin de semana.
- Procedencia agregada nacional/extranjera.
- Estado o disponibilidad del dato.

### Datos deseables

- Grado de ocupación por habitaciones.
- Personal ocupado.
- Índice de precios.
- Modalidad de alquiler.
- Tipo de tarifa.
- Procedencia detallada.
- Categorías de gasto turístico.
- Número de empresas y unidades locales por actividad turística.
- Festivos nacionales y autonómicos.
- Variables climatológicas oficiales.
- Información de eventos o temporadas locales.
- Datos internos agregados de una empresa piloto para validación externa, sin incorporarlos obligatoriamente al repositorio público.

---

# 3. Fuentes de datos previstas

## 3.1. Principio de selección

El proyecto utilizará únicamente:

- organismos públicos españoles;
- estadísticas oficiales españolas;
- plataformas públicas estatales que redistribuyen datos oficiales;
- datos agregados, documentados y accesibles para reutilización académica.

No se utilizarán como fuente principal:

- datasets extranjeros;
- plataformas privadas;
- datos extraídos de portales comerciales;
- datos sintéticos para validar el modelo;
- información sin documentación metodológica suficiente.

## 3.2. Fuente principal

### Instituto Nacional de Estadística — Encuesta de Ocupación en Alojamientos de Turismo Rural

**Uso previsto**

- Construcción del dataset principal.
- Análisis de oferta y demanda.
- Predicción de viajeros, pernoctaciones u ocupación.
- Comparación territorial.
- Estudio de estacionalidad.
- Creación de recomendaciones operativas.

**Información disponible**

- Viajeros.
- Pernoctaciones.
- Estancia media.
- Procedencia de viajeros.
- Establecimientos abiertos estimados.
- Plazas estimadas.
- Grados de ocupación.
- Personal ocupado.
- Niveles nacional, autonómico, provincial, zona turística y punto turístico.
- Periodicidad mensual.

**Formato esperado**

Descarga tabular desde INEbase y extracción automatizada cuando sea posible. Los datos se almacenarán en el proyecto como CSV o Parquet normalizado, conservando una copia inmutable de los ficheros originales.

**Enlaces**

- [INE — Alojamientos de turismo rural: encuesta de ocupación e índice de precios](https://www.ine.es/dyngs/INEbase/es/operacion.htm?c=Estadistica_C&cid=1254736176963&idp=1254735576863&menu=ultiDatos)
- [INE — Resultados mensuales y anuales](https://www.ine.es/dyngs/INEbase/operacion.htm?c=Estadistica_C&cid=1254736176963&idp=1254735576863&menu=resultados)
- [Dataestur — Ocupación en alojamientos de turismo rural](https://www.dataestur.es/alojamientos/encuesta-ocupacion-turismo-rural/)

**Estabilidad**

Alta. Es una operación oficial del INE con publicación periódica, metodología documentada y actualización mensual.

## 3.3. Fuente principal complementaria

### Instituto Nacional de Estadística — Índice de Precios de Alojamientos de Turismo Rural

**Uso previsto**

- Analizar evolución de precios.
- Incorporar una variable explicativa al modelo.
- Comparar demanda y precios.
- Detectar periodos en los que el precio aumenta o disminuye sin una variación equivalente de demanda.

**Información disponible**

- Índice general.
- Tasas de variación.
- Desglose por comunidades autónomas.
- Modalidad de alquiler.
- Tipo de tarifa.

**Formato esperado**

Descarga tabular de INEbase y normalización a CSV o Parquet.

**Enlace**

- [INE — Índice de precios de alojamientos de turismo rural](https://www.ine.es/dyngs/INEbase/es/operacion.htm?c=Estadistica_C&cid=1254736176963&idp=1254735576863&menu=ultiDatos)

**Estabilidad**

Alta. Forma parte de la misma operación estadística oficial.

## 3.4. Fuentes para estimar oportunidad económica

### Instituto Nacional de Estadística / Dataestur — Turismo de residentes en España

**Uso previsto**

- Contextualizar los viajes de residentes.
- Analizar duración, motivo, alojamiento y transporte.
- Obtener categorías agregadas de gasto.
- Construir escenarios relativos para restauración, actividades, transporte, comercio y alojamiento.

**Categorías útiles**

- Gasto en alojamiento.
- Gasto en transporte.
- Gasto en bares o restaurantes.
- Gasto en actividades recreativas, culturales y deportivas.
- Gasto en bienes duraderos.
- Resto de gastos.

**Limitación**

La fuente representa viajes de residentes en España y no exclusivamente turismo rural. Por tanto, se utilizará como ponderación contextual o escenario, no como prueba de gasto efectivo en un municipio concreto.

**Enlace**

- [Dataestur — Turismo de los residentes](https://www.dataestur.es/viajes-ocio/turismo-residente-etr/)

**Periodicidad y formato**

Publicación trimestral en Dataestur. La plataforma permite consulta mediante API y descarga de ficheros XLSX.

### Instituto Nacional de Estadística / Dataestur — EGATUR

**Uso previsto**

- Contextualizar el gasto de visitantes internacionales.
- Incorporar país de residencia, motivo, alojamiento y destino principal.
- Construir escenarios diferenciados entre demanda nacional e internacional.

**Limitación**

La fuente no se utilizará para atribuir gasto exacto a negocios rurales. Solo se integrará cuando el nivel temporal y territorial sea compatible con la encuesta de ocupación rural.

**Enlace**

- [Dataestur — Gasto turístico de visitantes internacionales](https://www.dataestur.es/economia/gasto-turistico-visitantes-egatur/)

**Periodicidad y formato**

Mensual. Consulta mediante Dataestur y descarga por API/XLSX.

## 3.5. Fuente para caracterizar el ecosistema empresarial

### Instituto Nacional de Estadística / Dataestur — Empresas activas en turismo

**Uso previsto**

- Medir el tejido empresarial relacionado con el turismo.
- Identificar empresas y unidades locales de alojamiento, comida y bebida, transporte y actividades.
- Crear ratios de demanda respecto a oferta empresarial.
- Comparar territorios.
- Generar recomendaciones por tipo de actividad.

**Actividades consideradas**

- Transporte.
- Hostelería.
- Agencias de viajes.
- Otras actividades asociadas al turismo.
- Alojamiento.
- Servicios de comida y bebida.
- Industria cultural.
- Actividades deportivas, recreativas y de entretenimiento.

**Limitación**

Los datos son agregados y anuales. No muestran ingresos, clientes ni capacidad de cada empresa.

**Enlace**

- [Dataestur — Empresas activas en turismo](https://www.dataestur.es/economia/empresas-activas-asociadas-a-la-actividad-turistica/)

**Periodicidad y formato**

Anual. Consulta mediante Dataestur y descarga por API/XLSX.

## 3.6. API y automatización

Dataestur ofrece una API oficial que permite:

- seleccionar conjuntos de datos;
- definir parámetros de consulta;
- generar una URL de petición;
- automatizar descargas;
- descargar directamente resultados en formato XLSX.

**Enlace**

- [Dataestur — API de datos turísticos](https://www.dataestur.es/apidata/)

El proyecto conservará:

- fecha de descarga;
- fuente;
- tabla o endpoint;
- parámetros usados;
- versión del fichero;
- hash del archivo original;
- script de transformación.

## 3.7. Fuentes opcionales de enriquecimiento

Solo se añadirán si el núcleo del proyecto ya funciona.

### AEMET OpenData

Posibles variables:

- temperatura media;
- precipitación;
- días de lluvia;
- fenómenos adversos.

Uso: analizar si el clima mejora la predicción en territorios donde las actividades rurales dependen especialmente de las condiciones meteorológicas.

### BOE y calendarios oficiales

Posibles variables:

- festivos nacionales;
- festivos autonómicos;
- puentes;
- Semana Santa;
- Navidad.

Uso: mejorar la explicación de picos y valles mensuales.

Estas fuentes serán complementarias y no críticas para la viabilidad del MVP.

## 3.8. Arquitectura de datos prevista

Los datos se organizarán en tres niveles:

```text
data/
├── raw/
│   ├── ine_ocupacion_rural/
│   ├── ine_precios_rurales/
│   ├── dataestur_etr/
│   ├── dataestur_egatur/
│   └── dataestur_empresas_turisticas/
├── interim/
│   ├── ocupacion_normalizada/
│   ├── gasto_normalizado/
│   └── empresas_normalizadas/
└── processed/
    ├── tourism_demand_monthly.parquet
    ├── business_context_annual.parquet
    └── modeling_dataset.parquet
```

La tabla principal de modelado tendrá una fila por territorio y mes. Las fuentes trimestrales o anuales se integrarán mediante reglas documentadas y sin crear una precisión temporal artificial.

## 3.9. Riesgos detectados y mitigaciones

| Riesgo | Impacto | Mitigación |
|---|---|---|
| Datos agregados, no empresariales | No permiten predecir ventas de un negocio concreto | Presentar el resultado como demanda territorial y oportunidad relativa |
| Falta de datos para todos los municipios | Limita la utilidad estrictamente municipal | Trabajar por provincia, zona o punto turístico y documentar su composición |
| Gasto ETR/EGATUR no exclusivo del turismo rural | Puede sobreestimar la aplicabilidad | Usarlo solo como contexto o ponderación, nunca como gasto real local |
| Diferencias entre territorios | Un único modelo nacional puede generalizar mal | Incluir variables territoriales, segmentar zonas y comparar modelos |
| Periodo COVID | Introduce ruptura estructural | Incluir variable COVID y evaluar modelos con y sin 2020-2021 |
| Datos provisionales y revisiones | Los valores pueden cambiar | Versionar descargas y registrar fecha/estado del dato |
| Valores ausentes o protegidos por secreto estadístico | Reduce continuidad de algunas series | Aplicar reglas de imputación prudentes o excluir series insuficientes |
| Cambios metodológicos | Pueden romper comparabilidad | Consultar metadatos y restringir el periodo cuando sea necesario |
| Índice de precios no equivale a precio real | Puede interpretarse incorrectamente | Tratarlo como indicador de evolución, no como tarifa monetaria |
| Predicción temporal con fuga de información | Métricas artificialmente optimistas | Separación temporal, backtesting y validación walk-forward |
| Exceso de alcance | Riesgo de no completar el proyecto | Priorizar predicción de una variable y dashboard mínimo |
| Interpretar oportunidad como beneficio | Puede inducir conclusiones falsas | Usar escenarios e índices, no beneficio neto |
| Integración de distintas periodicidades | Puede generar falsa granularidad | Mantener la frecuencia original y documentar cualquier agregación |

---

# 4. Consideraciones de privacidad y protección de datos

Las fuentes principales contienen estadísticas agregadas y no incluyen nombres, teléfonos, correos electrónicos, reservas individuales ni otros identificadores personales.

Por tanto:

- no se trabajará con información personal identificable;
- no será necesario crear perfiles individuales;
- no se incluirán datos de clientes;
- no se analizarán comportamientos de personas concretas;
- el repositorio podrá ser público sin publicar datos personales;
- se respetarán las condiciones de reutilización y atribución de cada fuente.

La privacidad presenta un riesgo bajo porque el propio organismo oficial aplica secreto estadístico y publica datos agregados. Aun así, se mantendrán las siguientes medidas:

- conservar únicamente datos publicados oficialmente;
- no intentar reidentificar establecimientos o personas;
- no combinar tablas con la finalidad de inferir información protegida;
- documentar valores suprimidos o no disponibles;
- evitar conclusiones sobre negocios individuales;
- presentar resultados a nivel territorial agregado.

Si posteriormente se incorporaran datos internos de una empresa piloto:

- serían opcionales;
- se necesitaría autorización expresa;
- se anonimizarían o agregarían;
- no se subirían al repositorio público;
- se usarían únicamente para comprobar si la tendencia territorial se relaciona con la actividad real del negocio.

---

# 5. Viabilidad inicial del proyecto

## 5.1. Viabilidad de obtención de datos

La obtención de datos es viable porque el proyecto se apoya en fuentes oficiales españolas de acceso público y actualización periódica. La encuesta de ocupación rural constituye una base directamente relacionada con el sector objetivo. Las fuentes complementarias permiten incorporar precios, gasto y estructura empresarial sin depender de empresas privadas.

La API de Dataestur y las descargas de INEbase permiten construir un pipeline reproducible.

## 5.2. Calidad, granularidad e histórico

La información principal presenta:

- periodicidad mensual;
- múltiples variables de oferta y demanda;
- desglose territorial;
- histórico suficiente para estacionalidad;
- documentación metodológica;
- estabilidad institucional;
- ausencia de datos personales.

La granularidad no llega necesariamente a cada municipio, pero las zonas y puntos turísticos permiten aproximarse a territorios rurales concretos. El MVP será defendible siempre que se comunique correctamente el nivel geográfico real.

## 5.3. Viabilidad técnica

El proyecto puede desarrollarse de forma realista durante el curso mediante esta priorización:

1. Descargar y versionar las tablas principales.
2. Construir un diccionario de datos.
3. Normalizar tiempo, territorio y métricas.
4. Realizar análisis exploratorio y controles de calidad.
5. Seleccionar una variable objetivo.
6. Crear un baseline estacional.
7. Entrenar modelos predictivos.
8. Validar con separación temporal.
9. Crear indicadores derivados.
10. Diseñar reglas de recomendación explicables.
11. Construir dashboard o aplicación.
12. Documentar limitaciones y reproducibilidad.

## 5.4. Modelos previstos

Se compararán modelos simples y avanzados.

### Baselines

- Valor del mes anterior.
- Valor del mismo mes del año anterior.
- Media móvil.
- Media histórica del mes.

### Modelos candidatos

- Regresión lineal regularizada.
- Random Forest.
- Gradient Boosting.
- XGBoost o HistGradientBoosting, si el alcance lo permite.
- Modelos específicos de series temporales, si la estructura de los datos lo justifica.

### Segmentación

- K-Means.
- Clustering jerárquico.
- PCA para visualización, si resulta necesario.

### Evaluación

- MAE.
- RMSE.
- MAPE o SMAPE cuando sea adecuado.
- R² como métrica complementaria.
- Backtesting temporal.
- Comparación frente al baseline.

El modelo solo se considerará útil si mejora de manera consistente al baseline y mantiene resultados razonables en periodos no vistos.

## 5.5. Recomendaciones previstas

Las recomendaciones serán deterministas y explicables. Ejemplos:

- Si la ocupación prevista supera claramente la media histórica, recomendar refuerzo de capacidad, personal o existencias.
- Si la ocupación de fin de semana es muy superior a la mensual, concentrar acciones de viernes a domingo.
- Si la demanda prevista es baja y el territorio depende del turismo nacional, recomendar campañas de escapadas de proximidad.
- Si aumenta la estancia media, recomendar paquetes conjuntos entre alojamiento, restauración y actividades.
- Si existe alta demanda y baja densidad de empresas de actividades, señalar una oportunidad relativa para experiencias.
- Si aumenta el índice de precios sin caída equivalente de demanda, recomendar estrategias de valor añadido en lugar de descuentos indiscriminados.
- Si el modelo presenta alta incertidumbre, mostrar la recomendación como escenario y no como acción prioritaria.

## 5.6. Principal riesgo

El mayor riesgo es confundir una predicción territorial con una predicción empresarial. La disponibilidad de turistas en una zona no garantiza ventas para todos los negocios. La conversión depende de ubicación, calidad, capacidad, precios, competencia, visibilidad y gestión interna.

La mitigación será:

- separar claramente demanda prevista y oportunidad empresarial;
- utilizar intervalos de predicción;
- crear escenarios conservador, central y favorable;
- no estimar beneficio neto;
- documentar supuestos;
- proponer validación futura con negocios reales.

## 5.7. Alternativa si una fuente complementaria no funciona

El proyecto seguirá siendo viable con la fuente principal del INE.

Plan alternativo:

1. Mantener ocupación, viajeros, pernoctaciones, estancia media y capacidad.
2. Crear predicción y segmentación territorial.
3. Sustituir estimaciones monetarias por un índice no monetario de oportunidad.
4. Formular recomendaciones según intensidad turística, estacionalidad y tejido empresarial disponible.
5. Incorporar gasto o empresas activas únicamente cuando su integración sea metodológicamente válida.

## Valoración final de viabilidad

La idea es viable y defendible porque:

- se acota a un sector concreto;
- utiliza datos oficiales españoles directamente relacionados con el entorno de aplicación;
- evita depender de datos privados;
- dispone de suficiente profundidad temporal;
- permite análisis descriptivo, predicción, segmentación y visualización;
- tiene utilidad para alojamientos y otros negocios rurales;
- reconoce explícitamente sus límites.

El proyecto final no afirmará que predice la facturación exacta de una microempresa. Su aportación será demostrar que los datos oficiales de turismo rural pueden convertirse en un sistema de anticipación de demanda y apoyo a decisiones para el ecosistema económico de territorios rurales.

---

# 6. Fuentes oficiales de referencia

- [Instituto Nacional de Estadística — Alojamientos de turismo rural](https://www.ine.es/dyngs/INEbase/es/operacion.htm?c=Estadistica_C&cid=1254736176963&idp=1254735576863&menu=ultiDatos)
- [Instituto Nacional de Estadística — Resultados de ocupación rural](https://www.ine.es/dyngs/INEbase/operacion.htm?c=Estadistica_C&cid=1254736176963&idp=1254735576863&menu=resultados)
- [Dataestur — Ocupación en alojamientos de turismo rural](https://www.dataestur.es/alojamientos/encuesta-ocupacion-turismo-rural/)
- [Dataestur — Turismo de los residentes](https://www.dataestur.es/viajes-ocio/turismo-residente-etr/)
- [Dataestur — Gasto turístico de visitantes internacionales](https://www.dataestur.es/economia/gasto-turistico-visitantes-egatur/)
- [Dataestur — Empresas activas en turismo](https://www.dataestur.es/economia/empresas-activas-asociadas-a-la-actividad-turistica/)
- [Dataestur — API de datos turísticos](https://www.dataestur.es/apidata/)
