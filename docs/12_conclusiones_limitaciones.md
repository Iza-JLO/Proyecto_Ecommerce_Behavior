# Conclusiones y limitaciones

## Conclusiones sobre el estado actual

El repositorio implementa un flujo local de Machine Learning para datos de e-commerce. La tarea real observada es regresion sobre `total_amount_usd`.

El proyecto cubre las etapas basicas de descarga de datos, preprocesamiento, seleccion de caracteristicas, entrenamiento de varios modelos y evaluacion con MSE y R2.

La comparacion de modelos se realiza en consola, sin persistir resultados ni seleccionar automaticamente un modelo final.

## Fortalezas

- Flujo simple y entendible.
- Uso de varios modelos de regresion.
- Separacion parcial de utilidades de entrenamiento y evaluacion.
- Consideracion explicita de posible `data leakage` en algunas columnas.
- Uso de pipelines para entrenamiento con y sin escalado.

## Limitaciones

- No hay datos versionados en el repositorio.
- No hay resultados numericos guardados.
- No hay modelo final persistido.
- No hay API ni dashboard.
- No hay tests automatizados.
- No hay validacion cruzada identificada.
- No hay busqueda sistematica de hiperparametros identificada.
- No hay diccionario formal de datos.
- No hay comparativa academica formal implementada.
- El notebook exploratorio es breve y no documenta un analisis EDA completo.

## Riesgos metodologicos

- Las metricas dependen de una particion unica entrenamiento/prueba.
- El rendimiento puede variar si el dataset descargado cambia.
- La ausencia de resultados persistidos dificulta auditar comparaciones previas.
- Algunas decisiones de seleccion de caracteristicas dependen de umbrales definidos en el codigo sin reporte experimental asociado.

## Trabajo futuro recomendado

- Guardar metricas en un archivo reproducible, por ejemplo `results/metrics.csv`.
- Persistir el mejor modelo con `joblib`.
- Agregar validacion cruzada.
- Crear un diccionario de datos.
- Documentar resultados numericos despues de una ejecucion controlada.
- Agregar tests para preprocesamiento y entrenamiento.
- Implementar API o dashboard solo si el proyecto requiere inferencia interactiva.
- Agregar una comparativa academica formal con referencias verificables.

