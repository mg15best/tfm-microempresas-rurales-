# Entrega 2 - Selección de idea de proyecto y análisis de datos necesarios

## Trazabilidad del repositorio

Este documento corresponde al archivo `docs/entregas/02_datos_necesarios.md`.

La entrega anterior se mantiene en `docs/entregas/01_ideas_producto.md`, garantizando la trazabilidad del proyecto desde la fase inicial de ideación hasta la selección de la idea definitiva.

---

# 1. Idea seleccionada

La idea seleccionada para continuar el proyecto es una **suite digital “marca blanca” para microempresas pequeñas y rurales**, orientada a ayudar a negocios con baja digitalización a gestionar mejor sus datos de ventas, clientes, productos y actividad comercial. La visión general de producto sería una herramienta que pudiera ser ofrecida por ayuntamientos, asociaciones empresariales, cámaras de comercio, consultoras locales, grupos de desarrollo rural o entidades de apoyo a pymes bajo su propia marca. Sin embargo, para mantener el proyecto dentro de un alcance realista de Data Science / IA, el MVP no consistirá en construir una suite empresarial completa, sino en desarrollar un **módulo analítico y de recomendaciones operativas** que demuestre cómo una microempresa puede convertir datos simples en decisiones útiles.

## Problema que resuelve

Muchas microempresas, especialmente en entornos rurales o municipios pequeños, siguen gestionando su actividad diaria con herramientas dispersas como WhatsApp, hojas de cálculo, libretas, agendas manuales, correos aislados o conversaciones informales con clientes. Esta forma de trabajo puede ser suficiente en fases iniciales, pero limita la capacidad de seguimiento comercial, dificulta recordar tareas pendientes, impide medir con claridad qué productos o servicios funcionan mejor y reduce la posibilidad de anticiparse a la estacionalidad. El problema no es únicamente tecnológico, sino también operativo y competitivo: negocios pequeños con pocos recursos compiten frente a cadenas, plataformas digitales o empresas mejor organizadas, pero no cuentan con sistemas que conviertan sus datos cotidianos en decisiones útiles. Resolver este problema aportaría valor porque permitiría mejorar la eficiencia operativa, aumentar la fidelización de clientes, profesionalizar la comunicación comercial y fortalecer la economía local mediante herramientas de decisión accesibles.

## Solución planteada

La solución se abordará desde un enfoque de Data Science, IA aplicada y visualización de datos. El núcleo del proyecto será un sistema capaz de importar datos sencillos desde CSV o Excel —ventas, clientes anonimizados, productos o servicios, tareas, interacciones y campañas—, limpiarlos, estructurarlos y transformarlos en indicadores comprensibles para personas sin perfil técnico. Sobre esos datos se aplicarán técnicas de análisis descriptivo, segmentación de clientes, detección de clientes inactivos, análisis de productos más vendidos, identificación de patrones temporales, cálculo de métricas comerciales y generación de recomendaciones accionables. La solución no se plantea como una plataforma SaaS completa, sino como un prototipo funcional que demuestre el valor de ordenar y explotar datos básicos en microempresas.

## MVP del proyecto final

El producto mínimo viable consistirá en un **prototipo funcional de dashboard analítico y asistente de recomendaciones operativas para microempresas**, con una interfaz sencilla y enfoque de marca blanca. El MVP permitirá cargar o simular datos de un pequeño negocio, visualizar indicadores clave de ventas, clientes, productos, tareas y campañas, y generar recomendaciones como “clientes a los que conviene contactar”, “productos con mayor margen o recurrencia”, “días o meses con mayor actividad”, “posibles periodos de baja demanda” o “acciones comerciales prioritarias”. El resultado final deberá poder mostrarse funcionando mediante un repositorio con código, datos de ejemplo, notebooks o scripts de análisis, documentación técnica y un dashboard interactivo o aplicación web ligera.

## Alcance del MVP

El MVP se limitará a un módulo analítico y de recomendaciones. Incluirá carga de datos de ejemplo, limpieza, análisis descriptivo, segmentación básica de clientes, detección de inactividad, visualización de indicadores y generación de recomendaciones explicables.

Queda fuera del alcance del MVP construir una plataforma SaaS completa, integrar WhatsApp en producción, gestionar pagos reales, automatizar campañas comerciales reales o almacenar datos personales identificables. La suite completa se mantiene como visión futura del producto, pero no como objetivo de desarrollo de esta entrega ni del MVP final.

---

# 2. Datos necesarios

Para desarrollar esta idea de forma rigurosa, el proyecto necesitará datos que representen la actividad diaria de una microempresa. El foco principal no estará en grandes volúmenes de datos desde el inicio, sino en construir un modelo de datos coherente, reutilizable y ampliable, capaz de adaptarse a negocios pequeños con información incompleta o poco estructurada.

## 2.1. Preguntas de negocio que deben responder los datos

Los datos deberán permitir responder, como mínimo, a las siguientes preguntas:

- ¿Cuáles son los productos o servicios más vendidos y cuáles generan mayor valor económico?
- ¿Qué clientes compran con más frecuencia, cuáles llevan tiempo inactivos y cuáles podrían estar en riesgo de pérdida?
- ¿Existen patrones de estacionalidad por día de la semana, mes, campaña, festivo o temporada turística?
- ¿Qué canales generan más ventas o interacciones: presencial, teléfono, WhatsApp, web, redes sociales u otros?
- ¿Qué tareas comerciales o recordatorios están pendientes, vencidos o asociados a clientes relevantes?
- ¿Qué campañas o acciones de comunicación generan respuesta, recompra o incremento de ventas?
- ¿Qué recomendaciones simples puede ofrecer el sistema para mejorar seguimiento, ventas y fidelización?

Estas preguntas conectan directamente el problema de negocio con el uso de Data Science: no se trata solo de almacenar datos, sino de convertir información operativa en decisiones prácticas.

## 2.2. Variables o campos necesarios

El proyecto necesitará un modelo de datos compuesto por varias entidades. Para el MVP no será obligatorio disponer de todas ellas en un entorno real, pero sí conviene definirlas desde el inicio para que el proyecto pueda crecer en entregas posteriores.

| Bloque de datos | Variables o campos necesarios | Uso dentro del proyecto |
|---|---|---|
| Negocio | `business_id`, nombre comercial anonimizado, sector, CNAE o categoría, municipio, provincia, tipo de entorno, tamaño aproximado, fecha de alta, canal principal de venta | Permite diferenciar negocios, adaptar el análisis por sector y preparar una futura arquitectura multiempresa o marca blanca. |
| Clientes | `customer_id_hash`, fecha de primera compra, fecha de última compra, municipio o zona agregada, segmento, canal preferido, número de compras, valor acumulado, consentimiento comercial si aplica | Permite segmentación, análisis RFM, detección de clientes inactivos y recomendaciones de seguimiento sin almacenar datos personales directos. |
| Productos o servicios | `product_id`, nombre normalizado, categoría, precio, coste estimado, margen estimado, estado activo/inactivo, proveedor opcional, estacionalidad esperada | Permite analizar productos top, categorías rentables, evolución de demanda y recomendaciones de surtido. |
| Ventas o pedidos | `order_id`, `business_id`, `customer_id_hash`, fecha y hora, canal, importe bruto, descuentos, impuestos, importe neto, método de pago agrupado, estado del pedido, devolución o cancelación | Es el bloque imprescindible para medir actividad, ingresos, frecuencia, ticket medio, recurrencia y evolución temporal. |
| Líneas de venta | `order_line_id`, `order_id`, `product_id`, cantidad, precio unitario, descuento por línea, importe total de línea, margen estimado | Permite análisis por producto, cesta de compra, productos relacionados y recomendaciones comerciales. |
| Tareas comerciales | `task_id`, `business_id`, `customer_id_hash` opcional, tipo de tarea, fecha de creación, fecha límite, estado, prioridad, resultado | Permite analizar tareas vencidas, carga operativa, seguimiento comercial y generación de alertas. |
| Interacciones | `interaction_id`, `customer_id_hash`, canal, fecha, tipo de interacción, motivo, resultado, campaña asociada | Permite medir relación con clientes, respuesta a acciones comerciales y oportunidades de fidelización. |
| Campañas | `campaign_id`, fecha de inicio, fecha de fin, canal, objetivo, segmento impactado, coste estimado, ventas asociadas, respuestas | Permite evaluar campañas y generar recomendaciones futuras. |
| Calendario | fecha, día de la semana, mes, trimestre, año, festivo nacional/autonómico/local, temporada, evento local | Permite análisis temporal, estacionalidad y comparación de periodos. |
| Datos externos | población municipal, densidad empresarial, indicadores de digitalización, clima, eventos, puntos de interés, turismo o actividad económica local | Permiten contextualizar el negocio y enriquecer el análisis, aunque no sustituyen los datos operativos. |

## 2.3. Granularidad adecuada

La granularidad principal del proyecto debe ser a nivel de **línea de venta o transacción**, no únicamente a nivel de resumen mensual. Esto es importante porque muchas preguntas relevantes —productos más vendidos, ticket medio, recurrencia, cesta de compra, margen por categoría o comportamiento de clientes— solo pueden analizarse correctamente si existe detalle por operación.

La granularidad recomendada será la siguiente:

| Tipo de información | Granularidad adecuada |
|---|---|
| Ventas | Línea de venta, pedido o transacción individual |
| Clientes | Cliente anonimizado o pseudonimizado |
| Productos o servicios | Producto o servicio individual, agrupable por categoría |
| Tiempo | Fecha y hora para ventas; agregación diaria, semanal y mensual para análisis |
| Tareas | Tarea individual con estado y fecha límite |
| Interacciones | Evento individual de contacto o comunicación |
| Campañas | Campaña individual, con clientes impactados y resultados |
| Contexto territorial | Municipio, provincia o comarca |
| Datos externos | Día, mes, municipio, sector o indicador agregado |

Para el dashboard, las visualizaciones podrán trabajar con agregaciones diarias, semanales o mensuales. Sin embargo, la base de datos debe conservar el mayor nivel de detalle posible para permitir análisis posteriores.

## 2.4. Profundidad histórica necesaria

La profundidad histórica dependerá del tipo de análisis:

- Para un dashboard descriptivo básico, podrían ser suficientes entre 3 y 6 meses de datos.
- Para analizar recurrencia de clientes, conviene disponer de al menos 6 a 12 meses.
- Para detectar estacionalidad anual, se necesita como mínimo 1 año completo.
- Para comparar temporadas y mejorar predicciones, sería deseable disponer de 24 a 36 meses.
- Para negocios rurales con alta estacionalidad —turismo rural, hostelería, comercio vinculado a campañas agrícolas o festividades locales— será especialmente importante cubrir periodos de alta y baja demanda.

Como criterio inicial, el MVP debería trabajar con al menos **12 meses de datos transaccionales**, aunque podría complementarse con datos sintéticos o datasets públicos para demostrar funcionalidades si no se dispone todavía de datos reales.

## 2.5. Volumen aproximado de datos

El volumen razonable para el MVP no necesita ser masivo, porque el problema principal de las microempresas suele ser la falta de estructura y calidad de los datos, no el exceso de volumen. Aun así, el proyecto debe estar diseñado para escalar a varias empresas en el futuro.

Un volumen adecuado sería:

| Escenario | Volumen aproximado |
|---|---|
| MVP mínimo con datos simulados | 3.000 - 10.000 líneas de venta |
| MVP sólido con dataset público transaccional | 50.000 - 500.000 líneas de venta |
| Piloto con una microempresa real | 1.000 - 20.000 líneas de venta anuales, según sector |
| Piloto multiempresa | 10.000 - 200.000 líneas de venta |
| Prueba de escalabilidad académica | Dataset sintético ampliado hasta 1 - 5 millones de registros |

Para un proyecto de Máster en IA y Big Data, el valor no dependerá únicamente del tamaño del dataset, sino de la calidad del pipeline: integración de fuentes, limpieza, modelado, trazabilidad, análisis reproducible, visualización, evaluación de modelos y tratamiento responsable de datos.

## 2.6. Datos imprescindibles y datos deseables

### Datos imprescindibles para el MVP

- Fecha de venta o pedido.
- Identificador de venta o pedido.
- Identificador de producto o servicio.
- Categoría del producto o servicio.
- Cantidad vendida.
- Precio unitario o importe total.
- Identificador de cliente anonimizado o, si no existe, indicador de venta anónima.
- Canal de venta.
- Estado de la venta: completada, cancelada o devuelta.
- Negocio o tienda a la que pertenece el dato.

Sin estos datos, el proyecto podría realizar visualizaciones básicas de ventas, pero perdería capacidad analítica para segmentar clientes, analizar recurrencia o generar recomendaciones personalizadas.

### Datos deseables pero no obligatorios

- Coste estimado y margen por producto.
- Interacciones con clientes.
- Tareas o recordatorios asociados.
- Campañas de marketing.
- Canal de comunicación preferido.
- Ubicación agregada del cliente.
- Inventario o stock.
- Festivos, eventos locales o temporadas.
- Datos meteorológicos para sectores sensibles al clima.
- Información agregada del municipio y sector.
- Datos de competencia local o puntos de interés cercanos.

Estos datos ampliarían el valor de la solución, pero no son imprescindibles para demostrar el MVP.

---

# 3. Fuentes de datos previstas

La estrategia de datos se plantea en cuatro capas: datos públicos transaccionales, datos sintéticos de demostración, datos reales anonimizados si se consiguieran y datos públicos contextuales. Esta combinación permite mantener la viabilidad académica del proyecto sin depender desde el inicio de conseguir datos privados de empresas reales.

## 3.1. Tipos de datos utilizados

- **Datos públicos transaccionales:** se usarán para construir y validar el pipeline analítico.
- **Datos sintéticos:** se generarán para adaptar el caso a microempresas rurales.
- **Datos reales anonimizados:** solo se incorporarán si existe una fuente piloto segura, con permisos suficientes y sin datos personales identificables.
- **Datos públicos contextuales:** INE, Eurostat, datos.gob.es, BOE, AEMET u OpenStreetMap se usarán únicamente para enriquecer el contexto, no como sustituto de los datos operativos internos.

## 3.2. Estrategia general de obtención de datos

La fuente ideal serían datos reales anonimizados de una o varias microempresas, pero esto puede ser difícil por privacidad, disponibilidad o falta de digitalización previa. Por tanto, el proyecto se desarrollará con una estrategia progresiva:

1. **Fase inicial:** uso de datasets públicos transaccionales y datos sintéticos para construir el pipeline, el dashboard y los modelos.
2. **Fase intermedia:** adaptación de plantillas CSV/Excel para simular cómo una microempresa cargaría sus ventas, clientes, productos y tareas.
3. **Fase avanzada:** incorporación opcional de datos reales anonimizados de una microempresa piloto, siempre que sea legal, viable y seguro.
4. **Fase de enriquecimiento:** uso de datos públicos de INE, Eurostat, OpenStreetMap, AEMET o calendarios oficiales para contexto territorial, sectorial y temporal.

## 3.3. Fuentes concretas previstas

| Fuente | Uso previsto | Formato esperado | Histórico disponible | Estabilidad y riesgos |
|---|---|---|---|---|
| UCI Machine Learning Repository - Online Retail | Dataset transaccional de referencia para construir análisis de ventas, productos, clientes, recurrencia y segmentación. | Excel/CSV tras conversión. | Transacciones entre diciembre de 2010 y diciembre de 2011. UCI indica 541.909 instancias. | Fuente estable y muy usada en proyectos académicos. Riesgo: no representa microempresas rurales españolas y está limitada a un comercio online concreto. |
| UCI Machine Learning Repository - Online Retail II | Alternativa o ampliación con más histórico para pruebas de segmentación, recurrencia y estacionalidad. | Excel/CSV. | Transacciones entre diciembre de 2009 y diciembre de 2011. | Útil para robustecer el análisis temporal, aunque mantiene el mismo riesgo de desajuste sectorial y geográfico. |
| Plantillas propias CSV/Excel del proyecto | Representar datos realistas de microempresas: ventas, clientes anonimizados, productos, tareas, campañas e interacciones. | CSV o Excel. | Histórico simulado de 12 a 36 meses. | Máxima adaptación al caso de uso. Riesgo: al ser sintético, deberá documentarse claramente cómo se genera para no confundirlo con datos reales. |
| Datos reales anonimizados de una microempresa piloto, si se consiguen | Validar el MVP con datos cercanos al problema real. | CSV, Excel o exportación de TPV/ERP/hojas de cálculo. | Dependerá del negocio; objetivo mínimo de 6 a 12 meses. | Alto valor aplicado, pero riesgo de privacidad, baja calidad, formatos inconsistentes y permisos. No será dependencia obligatoria del MVP. |
| INE - API JSON e INEbase | Obtener datos estadísticos oficiales de población, municipios y estructura empresarial. | JSON, CSV, XLSX o PC-Axis según tabla. | Series estadísticas explotables mediante API JSON. | Fuente pública, oficial y estable. Riesgo bajo, aunque algunas tablas requieren selección previa de dimensiones. |
| datos.gob.es - Empresas por municipio y actividad principal, DIRCE | Contextualizar la densidad empresarial por municipio y actividad económica. | CSV, XLSX, JSON o HTML. | Datos anuales según disponibilidad de la ficha concreta. | Fuente relevante para justificar enfoque rural y sectorial. Riesgo bajo, aunque son datos agregados y no sustituyen datos operativos internos. |
| datos.gob.es - Cifras oficiales del padrón por municipio | Caracterizar municipios pequeños, población potencial, ruralidad y contexto territorial. | CSV, XLSX, JSON o PC-Axis. | Serie anual por municipio. | Fuente oficial y estable. Su uso será contextual, no predictivo a nivel cliente. |
| Eurostat - Digital economy and society database | Justificar el contexto de digitalización empresarial y comparar indicadores de uso TIC, comercio electrónico o intensidad digital por tamaño de empresa. | API, descarga tabular y bases detalladas. | Eurostat publica datasets recientes e históricos sobre economía y sociedad digital. | Fuente oficial europea y estable. Riesgo bajo. Su granularidad es agregada, no sirve para entrenar modelos de clientes. |
| OpenStreetMap / Overpass API | Obtener puntos de interés, comercios, restaurantes, alojamientos o servicios cercanos para análisis territorial o competencia local. | API, JSON, GeoJSON u OSM. | Datos vivos, actualizados por la comunidad. | Fuente útil para contexto geográfico. Riesgos: cobertura irregular, etiquetas incompletas, cambios de calidad por zona y obligación de respetar la licencia ODbL. |
| AEMET OpenData | Enriquecer análisis de estacionalidad en sectores sensibles al clima, como hostelería, turismo rural o comercio de temporada. | API y descarga en formatos reutilizables. | Datos meteorológicos históricos y actuales según catálogo. | Fuente oficial. Riesgos: posible necesidad de clave API, límites de uso y complejidad de seleccionar estaciones representativas. |
| BOE / calendario de fiestas laborales | Incorporar festivos nacionales y autonómicos como variables de calendario para explicar picos o caídas de demanda. | HTML/PDF; posible transformación manual a CSV. | Publicación anual. | Fuente oficial, pero requiere transformación y los festivos locales pueden necesitar fuentes municipales adicionales. |
| datos.gob.es - Catálogo nacional de datos abiertos | Localizar datasets complementarios por territorio, turismo, comercio, transporte o economía local. | Variable según dataset: CSV, JSON, XLSX o API. | Depende de cada publicador. | Útil como repositorio de búsqueda. Riesgo: heterogeneidad, datasets incompletos o sin actualización frecuente. |

## 3.4. Enlaces de referencia previstos

- UCI Online Retail: https://archive.ics.uci.edu/dataset/352/online+retail
- UCI Online Retail II: https://archive.ics.uci.edu/dataset/502/online+retail+ii
- INEbase: https://www.ine.es
- API del INE: https://servicios.ine.es/wstempus/js/es/DATOS_TABLA/
- datos.gob.es: https://datos.gob.es
- Eurostat Digital Economy and Society: https://ec.europa.eu/eurostat/web/digital-economy-and-society
- OpenStreetMap: https://www.openstreetmap.org
- Overpass API: https://overpass-api.de
- AEMET OpenData: https://opendata.aemet.es
- BOE: https://www.boe.es

## 3.5. Riesgos detectados en las fuentes

| Riesgo | Impacto | Mitigación propuesta |
|---|---|---|
| Falta de datos reales de microempresas | Puede limitar la validación aplicada del MVP. | Usar UCI Online Retail, plantillas propias y datos sintéticos documentados. |
| Datos reales incompletos o desordenados | Puede afectar limpieza, modelado y visualización. | Diseñar validaciones automáticas, diccionario de datos y reglas de calidad. |
| Ausencia de identificador de cliente | Limita segmentación y fidelización. | Permitir ventas anónimas y aplicar análisis agregado cuando no exista cliente. |
| Desajuste entre dataset público y negocio rural español | Puede afectar realismo del caso. | Usar dataset público solo como base técnica y adaptar datos sintéticos al contexto rural. |
| Datos personales en clientes o interacciones | Riesgo legal y ético. | Evitar nombres, teléfonos, direcciones y mensajes; usar identificadores hash o datos sintéticos. |
| OpenStreetMap incompleto en zonas rurales | Puede sesgar análisis territorial. | Usarlo solo como complemento, no como fuente crítica del MVP. |
| Festivos locales difíciles de automatizar | Puede complicar la estacionalidad local. | Empezar con festivos nacionales/autonómicos y permitir carga manual de eventos locales. |
| AEMET requiere tratamiento técnico adicional | Puede aumentar complejidad. | Considerar clima como variable opcional, no imprescindible para el MVP. |
| Cambios en APIs o formatos | Puede romper procesos de extracción. | Guardar muestras versionadas y documentar scripts de descarga. |

---

# 4. Consideraciones de privacidad y protección de datos

El proyecto puede implicar datos personales si se utilizan datos reales de clientes, especialmente nombres, teléfonos, correos electrónicos, direcciones, mensajes de WhatsApp, notas internas, preferencias de compra o historial de comunicaciones. Por este motivo, el diseño del MVP debe seguir un enfoque de minimización de datos: solo se recogerán los campos estrictamente necesarios para el análisis y se evitará almacenar información personal identificable cuando no sea imprescindible.

Para el desarrollo académico, se propone trabajar preferentemente con datos sintéticos, datasets públicos o datos anonimizados. En caso de disponer de datos reales, el identificador de cliente deberá sustituirse por un `customer_id_hash` o un identificador interno pseudonimizado. Es importante distinguir anonimización y pseudonimización: la pseudonimización reduce la vinculación directa con una persona, pero no elimina totalmente la posibilidad de reidentificación, mientras que la anonimización busca hacer los datos no vinculables a ninguna persona concreta.

No se deben almacenar en el repositorio público nombres, teléfonos, emails, direcciones completas, conversaciones reales, observaciones sensibles ni datos de pago. Si se incorporan datos de una empresa real, deberán agregarse, filtrarse o transformarse antes de subirlos al repositorio. El repositorio podrá contener esquemas, scripts, notebooks y datos de muestra, pero no datos personales reales.

Desde el punto de vista ético, también deben evitarse funcionalidades que puedan perjudicar injustamente a clientes o negocios. Por ejemplo, una recomendación comercial no debería etiquetar a una persona de forma opaca o discriminatoria, ni generar acciones invasivas de marketing. Las recomendaciones del sistema deberán ser explicables, orientadas a la mejora operativa y siempre supervisadas por el usuario del negocio. El sistema no tomará decisiones automáticas con efectos relevantes sobre personas, sino que propondrá acciones como recordatorios, priorización de clientes inactivos o análisis de productos.

También se deben aplicar medidas específicas de seguridad y privacidad:

- Separar datos de distintos negocios mediante `business_id` y control lógico de acceso.
- Evitar subir datos reales al repositorio público.
- Usar datos sintéticos para demostraciones.
- Documentar cualquier transformación aplicada a los datos.
- Eliminar campos de texto libre que puedan contener información sensible.
- Agregar localización a nivel de municipio o zona, evitando direcciones concretas.
- Usar métricas agregadas en visualizaciones públicas.
- Mantener trazabilidad de fuentes y versiones de datasets.
- Incluir un archivo README o sección de privacidad explicando qué datos se usan y cuáles se han excluido.

En conclusión, el proyecto puede usarse de forma segura en un contexto académico siempre que se trabaje con datos públicos, sintéticos o correctamente anonimizados, y siempre que los datos reales de clientes se eviten o se traten con garantías suficientes.

---

# 5. Viabilidad inicial del proyecto

El proyecto parece viable desde el punto de vista de los datos, siempre que se acote correctamente el MVP. La idea completa de una suite digital para microempresas puede ser amplia, pero el enfoque propuesto para el curso es realista porque reduce el alcance inicial a un módulo analítico con importación de datos, dashboard, segmentación básica y recomendaciones operativas. Esto permite demostrar valor sin construir una plataforma empresarial completa.

La obtención de datos necesarios también parece viable. Existen datasets públicos transaccionales adecuados para prototipar análisis de ventas y clientes, como Online Retail de UCI, y fuentes oficiales para enriquecer el contexto territorial y sectorial, como INE, Eurostat, datos.gob.es, BOE, AEMET u OpenStreetMap. Estas fuentes permiten avanzar aunque no se consiga inmediatamente una microempresa piloto. La principal limitación es que los datasets públicos no representan exactamente a microempresas rurales españolas, por lo que será necesario adaptar el caso con datos sintéticos realistas y documentar claramente esa decisión.

La información disponible tiene suficiente calidad para una primera versión del proyecto si se usa con una finalidad adecuada. Los datos transaccionales públicos permiten entrenar y validar técnicas de segmentación, análisis de recurrencia, cálculo de indicadores y visualización. Los datos agregados de INE y Eurostat permiten justificar el contexto, pero no sirven para sustituir datos operativos internos. Por tanto, el proyecto deberá separar claramente los datos de negocio, que alimentan el dashboard y los modelos, de los datos externos, que solo enriquecen o contextualizan.

El proyecto puede desarrollarse de forma realista durante el curso si se mantiene la siguiente priorización:

1. Diseñar el modelo de datos común para microempresas.
2. Crear o adaptar datasets de ejemplo.
3. Construir un pipeline de limpieza y transformación.
4. Desarrollar indicadores descriptivos.
5. Implementar segmentación de clientes y detección de inactividad.
6. Crear un dashboard o aplicación ligera.
7. Añadir recomendaciones explicables basadas en reglas y modelos simples.
8. Documentar privacidad, limitaciones, riesgos y futuras mejoras.

La parte más arriesgada en este momento es conseguir datos reales de una microempresa con suficiente calidad, histórico y permisos de uso. También existe riesgo de ampliar demasiado el alcance funcional: CRM, tareas, campañas, IA, dashboard, predicción y marca blanca podrían convertirse en un proyecto excesivo si se intentan desarrollar todos al mismo nivel. Para evitarlo, el MVP se centrará en la parte de datos y analítica, dejando la suite completa como visión futura.

Si la fuente principal de datos reales no funciona, la alternativa será utilizar una combinación de:

- Dataset Online Retail de UCI para transacciones.
- Datos sintéticos generados con estructura de microempresa rural.
- Datos abiertos de INE para contexto municipal y sectorial.
- Datos de Eurostat u ONTSI para justificar digitalización empresarial.
- OpenStreetMap para contexto territorial opcional.
- Calendario de festivos y eventos simulados para estacionalidad.

Esta estrategia reduce la dependencia de fuentes privadas y mantiene la continuidad del proyecto para siguientes entregas. Además, permite que el repositorio evolucione con una estructura profesional: datos brutos, datos procesados, scripts de limpieza, notebooks de análisis, modelos, documentación, dashboard y entregas parciales.

## Valoración final de viabilidad

La idea es viable, relevante y adecuada para un proyecto de IA y Big Data si se desarrolla con un alcance controlado. El valor académico reside en convertir un problema real de baja digitalización en una solución basada en datos, con un pipeline reproducible, fuentes documentadas, análisis visual, segmentación, recomendaciones y tratamiento responsable de la privacidad. El proyecto no dependerá de construir una suite empresarial completa, sino de demostrar que una microempresa puede obtener decisiones útiles a partir de datos simples, bien estructurados y presentados de forma comprensible.

El MVP final será suficientemente realista para el curso y suficientemente extensible para futuras entregas, ya que podrá evolucionar hacia modelos predictivos más avanzados, versiones sectorizadas, integración con herramientas externas, automatización de campañas, despliegue en la nube o arquitectura multiempresa de marca blanca.
