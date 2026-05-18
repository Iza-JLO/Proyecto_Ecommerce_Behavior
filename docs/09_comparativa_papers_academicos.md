# Comparativa interna de modelos

Este documento resume la comparacion interna de modelos implementada en el pipeline actual.

## Comparacion disponible

El repositorio compara varios modelos de regresion sobre la misma particion de entrenamiento y prueba, usando las mismas variables de entrada y las mismas metricas de evaluacion.

La comparacion actual se centra en:

- Modelos sin escalado.
- Modelos con escalado mediante `MinMaxScaler`.
- Evaluacion con `MSE` y `R2`.
- Impresion de resultados en consola durante la ejecucion de `src/models.py`.

## Alcance actual

La documentacion no registra benchmarks externos ni referencias fuera del pipeline implementado en el repositorio.

## Estado actual

Implementado actualmente como comparacion interna de modelos dentro del flujo local.

