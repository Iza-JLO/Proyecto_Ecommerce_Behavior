# Flujo de datos

El flujo de datos implementado se basa en archivos CSV locales generados por el pipeline actual.

## Flujo implementado

```txt
Obtencion de datos del pipeline
    |
    |  src/download_data.py
    v
data/orders.csv
    |
    |  src/eda_selection.py
    v
data/processed/df_visualizacion.csv
    |
    |  src/models.py
    v
Entrenamiento y evaluacion en consola
```

## Entrada principal

`src/download_data.py` genera o deja disponible el archivo `orders.csv` del pipeline y lo guarda en:

```txt
data/orders.csv
```

La carpeta `data/` no esta versionada. Para reproducir el flujo, se debe ejecutar primero el script de descarga.

## Dataset procesado

`src/eda_selection.py` crea la carpeta `data/processed` si no existe y guarda un CSV procesado en:

```txt
data/processed/df_visualizacion.csv
```

El nombre impreso por el script indica `df_final.csv`, pero la ruta real usada por `to_csv` es `data/processed/df_visualizacion.csv`.

## Uso en entrenamiento

`src/models.py` carga:

```txt
data/orders.csv
data/processed/df_visualizacion.csv
```

Despues agrega al dataframe procesado las columnas `pages_viewed_before_purchase`, `session_duration_minutes` y `month` desde el dataset original.

## Datos y resultados versionados

No hay datasets versionados en el repositorio actual. Tampoco se identifican archivos de resultados, metricas persistidas, graficas exportadas ni modelos guardados.

