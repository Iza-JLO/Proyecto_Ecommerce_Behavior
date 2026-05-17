# Descripcion del proyecto

Este repositorio contiene un proyecto de Machine Learning aplicado al analisis de comportamiento de usuarios en e-commerce. El flujo implementado descarga datos de ordenes desde Kaggle, realiza una etapa de preprocesamiento y seleccion de caracteristicas, y entrena varios modelos de regresion para predecir el valor total de una orden.

La variable objetivo usada en el codigo es `total_amount_usd`. Por lo tanto, el problema modelado actualmente es una tarea de regresion orientada a estimar el monto total de compra, no una clasificacion directa de intencion de compra.

## Problema que resuelve

El proyecto busca identificar relaciones entre atributos de navegacion, compra, producto, cliente y transaccion para estimar el importe total de una orden de e-commerce. Este enfoque puede ayudar a comparar modelos predictivos y estudiar que variables tienen relacion con el valor economico de una compra.

## Fuente de datos

La descarga de datos se implementa en `src/download_data.py` usando `kagglehub`.

Dataset configurado:

```txt
meruvakodandasuraj/e-commerce-customer-behavior-and-sales-20202026
```

Archivo descargado:

```txt
orders.csv
```

Ruta local esperada tras ejecutar el script:

```txt
data/orders.csv
```

La carpeta `data/` esta ignorada por Git segun `.gitignore`; por eso los datos no estan versionados en el repositorio actual.

## Alcance actual

Implementado actualmente:

- Descarga del dataset desde Kaggle.
- Carga de `data/orders.csv`.
- Imputacion de valores nulos mediante `KNNImputer`.
- Codificacion ordinal temporal para seleccion de caracteristicas.
- Seleccion de variables por varianza, correlacion e informacion mutua.
- Generacion de un dataset procesado en `data/processed/df_visualizacion.csv`.
- Preparacion de datos de entrenamiento y prueba.
- Entrenamiento de modelos de regresion.
- Evaluacion con `MSE` y `R2`.

No identificado en el repositorio actual:

- API de prediccion.
- Dashboard o interfaz visual.
- Modelos serializados (`.pkl`, `.joblib` u otros).
- Reportes de resultados guardados.
- Graficas exportadas.
- Tests automatizados.
- Dockerfile o configuracion de despliegue.

