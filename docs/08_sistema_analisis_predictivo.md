# Flujo predictivo actual

Este documento resume el pipeline implementado en el repositorio actual.

## Etapas implementadas

1. Obtencion de datos del pipeline.
2. Carga de `data/orders.csv`.
3. Imputacion de valores nulos con `KNNImputer`.
4. Codificacion temporal y seleccion de caracteristicas.
5. Preparacion de conjuntos de entrenamiento y prueba.
6. Entrenamiento de varios modelos de regresion.
7. Evaluacion con `MSE` y `R2` en consola.

## Salidas del pipeline

El pipeline genera como salida principal:

- `data/orders.csv`
- `data/processed/df_visualizacion.csv`

Los resultados numericos de los modelos se imprimen en consola durante la ejecucion de `src/models.py`.

