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

El pipeline deja disponible el archivo local `data/orders.csv` mediante `src/download_data.py`.

El notebook exploratorio carga `../data/orders.csv` y muestra ejemplos de consulta sobre `category`, pero el repositorio no incluye un diccionario formal del dataset ni un esquema completo documentado.

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

Uso experimental para comparar modelos de regresion sobre datos de comportamiento y ventas de e-commerce.

## Usos no recomendados

No se recomienda interpretar este repositorio como una aplicacion de produccion; el alcance actual corresponde a un pipeline local de exploracion, preprocesamiento y comparacion de modelos.

## Limitaciones

- Dataset no versionado.
- Sin metricas persistidas.
- Sin validacion cruzada identificada.
- Sin busqueda sistematica de hiperparametros identificada.
- Sin documentacion de licenciamiento del dataset dentro del repositorio.

