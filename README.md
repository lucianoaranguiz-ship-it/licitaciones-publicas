# Predicción de precio unitario de adjudicación en Mercado Público

Aplicación web que expone un modelo de regresión entrenado sobre licitaciones
públicas de equipamiento TI, para estimar el precio unitario al que se
adjudicaría una licitación según cómo se diseñe.

**[Ver aplicación en vivo](https://licitaciones-publicas.streamlit.app/)**

## El problema

Un Servicio de Salud de la Región de Antofagasta debe adquirir 800 notebooks
corporativos a través de Mercado Público, con un presupuesto máximo referencial
de $560.000 por unidad.

En el Estado, el precio finalmente adjudicado no depende solo del producto:
depende de cómo se diseña la licitación —tipo por tramo UTM, plazos de cierre,
modalidad de pago, estimación del monto— y del contexto en que ocurre —región,
sector del organismo, tamaño del proveedor que se adjudica y, sobre todo, nivel
de competencia. Configuraciones que parecen convenientes terminan adjudicándose
más caras por falta de oferentes o por recargos logísticos en zonas extremas.

Tres áreas internas proponen configuraciones distintas para la misma compra. La
pregunta operativa es cuál de ellas se adjudicaría dentro del presupuesto.

Esta aplicación permite responder esa pregunta de forma interactiva: se ajustan
los parámetros de diseño de la licitación y el modelo devuelve el precio
unitario esperado.

## Los datos

Extracto de licitaciones adjudicadas de equipos computacionales. Las variables
corresponden a campos reales publicados por ChileCompra a través de la API de
Datos Abiertos de Mercado Público (datos-abiertos.chilecompra.cl); los registros
son sintéticos.

Variables de diseño de la licitación:

| Variable | Descripción |
|---|---|
| `tipo_licitacion` | Tipo por tramo UTM (L1, LE, LP, LQ, LR) |
| `tipo_estimacion` | Presupuesto disponible, precio referencial o no estimable |
| `modalidad_pago` | 30 días, contra entrega, etc. |
| `dias_cierre` | Días que la licitación estuvo abierta |
| `dias_adjudicacion` | Días entre cierre y adjudicación |
| `duracion_contrato_dias` | Duración del contrato |
| `extension_plazo` | 1 si el cierre se amplió por recibir 2 o menos ofertas (art. 25) |

Variables de contexto:

| Variable | Descripción |
|---|---|
| `region` / `macrozona` | Región de la unidad compradora, agrupada en macrozonas |
| `sector_comprador` | Municipalidad, Servicio de Salud, Ministerio, Universidad Estatal, Gobierno Regional |
| `tamano_proveedor` | Micro, pequeña, mediana o grande |
| `monto_estimado` | Monto estimado en CLP |
| `cantidad_adjudicada`, `n_items` | Tamaño de la compra |
| `n_oferentes` | Competencia efectiva |
| `n_reclamos`, `toma_razon` | Fricción administrativa |

Variable objetivo: `precio_unitario_adj`, el precio unitario adjudicado en CLP.

### Sobre la estructura causal

Hay un punto que conviene tener presente al leer la importancia de variables.
`dias_cierre` correlaciona con el precio, pero el mecanismo no es directo: más
días abierta significa más difusión, más difusión significa más oferentes, y más
competencia presiona el precio a la baja. El motor real es `n_oferentes`.

En la misma línea, `extension_plazo` no causa precios altos: es un marcador de
baja competencia, porque se activa justamente cuando llegaron dos ofertas o
menos.

Un modelo predictivo captura estas asociaciones sin distinguir causa de
marcador. Sirve para anticipar el precio, no para concluir que alargar el plazo
lo baja por sí solo.

## Preparación de los datos

- `monto_estimado` viene vacío cuando el monto no es estimable; se imputa con la
  mediana del conjunto de entrenamiento.
- `sector_comprador` llega con texto inconsistente —mayúsculas, espacios
  sobrantes— y se homologa.
- `region` se agrupa en macrozonas (Norte Grande, Norte Chico, Centro, Centro
  Sur, Sur, Austral). Con dieciséis regiones, varias quedan con muy pocos casos
  y los árboles terminan haciendo cortes sobre ruido; las macrozonas capturan el
  gradiente logístico norte-sur con categorías que tienen masa suficiente.
- División 70/30 estratificada por la variable objetivo.

El orden importa: la imputación y la codificación se ajustan **después** de
separar entrenamiento y prueba, y solo con los datos de entrenamiento. Si se
calcula la mediana sobre el total, información del conjunto de prueba se filtra
al modelo y las métricas quedan infladas. Todo eso vive dentro del `Pipeline`,
que es lo que garantiza que el orden se respete también en producción.

## Modelos evaluados

Se compararon dos familias:

**Métodos de ensamblaje** — Bagging (25 árboles), Random Forest (300 árboles,
grilla sobre `mtry`) y XGBoost (grilla sobre `nrounds`, `max_depth` y `eta`),
todos con validación cruzada de 5 pliegues.

**Redes neuronales** — arquitecturas de una y dos capas con activación logística
y tanh, más una búsqueda en grilla sobre el número de nodos por capa, y una
variante con softplus.

Métricas: MAE, RMSE y correlación entre predicho y observado. Se reportan las
tres porque miden cosas distintas. El MAE es el error promedio en pesos,
directamente interpretable frente al presupuesto. El RMSE penaliza los errores
grandes de forma cuadrática, y en esta decisión eso importa: subestimar
gravemente el precio de una configuración es peor que equivocarse un poco en
todas. La correlación indica si el modelo ordena bien las configuraciones aunque
tenga sesgo de nivel.

Las redes neuronales pagan un costo de preparación que los ensambles no
necesitan: exigen una matriz completamente numérica, así que las categóricas hay
que dummificarlas, y exigen normalización para que el descenso de gradiente
converja. Eso multiplica las columnas y obliga a desnormalizar antes de calcular
métricas en pesos.

El modelo desplegado acá es el de mejor desempeño en el conjunto de prueba.

## Qué hace la aplicación

**Estimación individual.** Se configuran los parámetros de la licitación en el
panel lateral y se obtiene el precio unitario esperado, contrastado contra el
presupuesto máximo de $560.000.

**Qué pesa en la estimación.** Importancia por permutación, medida sobre la
variable original y no sobre las columnas dummy, así que una categórica con
varios niveles aparece como una sola barra.

**Desempeño del modelo.** MAE, RMSE y correlación sobre el conjunto de prueba,
con el gráfico de predicho contra observado.

**Evaluación por lote.** Carga de un CSV con varias configuraciones —por
ejemplo, las tres propuestas de las unidades internas— para compararlas de una
vez.

## Estructura del repositorio

```
train_export.py      entrena, evalúa y exporta artefactos
app.py               la aplicación web
requirements.txt     versiones fijas
.streamlit/
  config.toml        tema
artifacts/           salida de train_export.py
  modelo.skops       pipeline serializado
  model_meta.json    esquema, métricas e importancias
  muestra.csv        muestra del conjunto de prueba
```

## Reproducir

```bash
pip install -r requirements.txt
python train_export.py --data Licitaciones_TI_ChileCompra.csv
streamlit run app.py
```

El script deja en `artifacts/` el pipeline serializado y un JSON con métricas,
importancias y el esquema de cada variable. La aplicación no entrena ni
recalcula nada: solo carga y predice.

Anote la versión de scikit-learn que imprime al terminar y fíjela en
`requirements.txt`. Un pipeline serializado con una versión y cargado con otra
falla, o peor, carga con advertencias y predice distinto.

## Decisiones de diseño

**Se serializa el `Pipeline` completo, no el estimador.** Imputación,
codificación y escalado viajan dentro del mismo objeto. Guardar solo el modelo
obliga a replicar el preprocesamiento a mano en producción, y ahí aparecen los
desalineamientos de columnas que no lanzan excepción y simplemente predicen mal.

**Todo lo costoso se precomputa.** La importancia por permutación con diez
repeticiones tarda minutos; calcularla en cada visita agotaría el contenedor. Se
calcula una vez al entrenar y viaja en el JSON.

**skops en lugar de pickle.** Un `.pkl` ejecuta código arbitrario al abrirse. En
un repositorio público eso es un vector de ataque real. El formato `.skops`
reconstruye solo tipos declarados.

**El formulario se genera desde el esquema.** Los controles de la interfaz no
están escritos a mano: se construyen leyendo `model_meta.json`. Cambiar de
dataset o de modelo no requiere tocar `app.py`.

## Limitaciones

- Los registros son sintéticos. La estructura de variables es real, los valores
  no, así que las magnitudes no deben leerse como estimaciones del mercado.
- El modelo captura asociaciones, no efectos causales. No responde qué pasaría
  si se cambiara el diseño de la licitación, sino a qué precio se han adjudicado
  históricamente licitaciones con ese perfil.
- Desplegado en el plan gratuito de Streamlit Community Cloud: la aplicación se
  suspende tras 12 horas sin visitas, por lo que el primer acceso puede tardar
  unos segundos en iniciar.
- No hay monitoreo de drift. Si la composición del mercado de equipamiento TI
  cambia, el modelo se degrada en silencio.

## Contexto

Material desarrollado para el curso Machine Learning para la Gestión Pública y
Privada, Magíster en Ciencias de Datos, Universidad de Santiago de Chile.
