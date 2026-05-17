# Evaluacion y resultados

La evaluacion esta implementada en `src/model_utils.py` y se ejecuta desde `src/models.py`.

## Metricas calculadas

La funcion `evaluar_modelo(modelo, X_test, y_test)` calcula:

```txt
MSE
R2
```

Implementacion usada:

```python
mean_squared_error(y_test, y_pred)
r2_score(y_test, y_pred)
```

## Reporte de resultados

`src/models.py` imprime en consola los resultados de cada modelo con el formato:

```txt
Resultados para <Modelo> sin escalado: MSE=<valor>, R2=<valor>
Resultados para <Modelo> con escalado: MSE=<valor>, R2=<valor>
```

## Resultados persistidos

No se identifican archivos de metricas guardadas, reportes generados, tablas comparativas persistidas ni graficas exportadas en el repositorio actual.

Por esta razon, esta documentacion no incluye valores numericos de desempeno. Para obtenerlos, se debe ejecutar el flujo localmente con los datos descargados.

## Graficas y matrices

No se identifican en el repositorio actual matrices de confusion, curvas ROC, curvas precision-recall, graficas de residuales exportadas ni comparativas visuales guardadas.

Ademas, dado que el problema implementado es de regresion, metricas como matriz de confusion o curva ROC no aplican directamente salvo que se reformule una tarea de clasificacion.

