# Model Card

Este documento resume el estado de los modelos implementados en el repositorio.

## Nombre del sistema

Modelos de regresion para estimacion de `total_amount_usd` en ordenes de e-commerce.

## Tipo de tarea

Regresion supervisada.

## Variable objetivo

```txt
total_amount_usd
```

## Datos de entrada

El dataset se descarga desde Kaggle mediante `src/download_data.py` y se guarda como `data/orders.csv`.

El notebook muestra un dataset de 28 columnas. Entre las columnas observadas en el repositorio se encuentran:

```txt
order_id
customer_id
order_date
year
month
quarter
day_of_week
product_name
category
unit_price_usd
payment_method
device_used
delivery_days
delivery_date
order_status
returned
customer_rating
session_duration_minutes
pages_viewed_before_purchase
is_repeat_customer
```

La lista anterior no debe interpretarse como diccionario completo del dataset, ya que el repositorio no incluye un archivo de esquema formal.

## Preprocesamiento

Incluye imputacion KNN, codificacion ordinal temporal, eliminacion de constantes, eliminacion de alta correlacion, seleccion por informacion mutua, One-Hot Encoding, Target Encoding y escalado `MinMaxScaler` para algunos modelos.

## Modelos evaluados

```txt
Lasso
DecisionTreeRegressor
RandomForestRegressor
MLPRegressor
XGBRegressor
SVR
ElasticNet
```

## Metricas

```txt
MSE
R2
```

## Resultados

No hay resultados numericos persistidos en el repositorio actual. Los valores se imprimen en consola al ejecutar `src/models.py`.

## Uso previsto

Uso academico y experimental para comparar modelos de regresion sobre datos de comportamiento y ventas de e-commerce.

## Usos no recomendados

No se recomienda usar el proyecto actual como sistema productivo de prediccion porque no incluye validacion robusta de entradas, persistencia de modelos, API de inferencia, monitoreo, tests automatizados ni analisis documentado de sesgos o estabilidad.

## Limitaciones

- Dataset no versionado.
- Sin metricas persistidas.
- Sin validacion cruzada identificada.
- Sin busqueda sistematica de hiperparametros identificada.
- Sin evaluacion de deriva de datos.
- Sin documentacion de licenciamiento del dataset dentro del repositorio.

