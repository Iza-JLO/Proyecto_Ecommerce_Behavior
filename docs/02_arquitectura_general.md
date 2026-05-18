# Arquitectura general

El repositorio tiene una estructura simple orientada a scripts y un notebook exploratorio. La arquitectura actual corresponde a un pipeline local de descarga, preprocesamiento, seleccion de caracteristicas y entrenamiento.

## Estructura actual

```txt
.
|-- .gitignore
|-- LICENSE
|-- README.md
|-- requirements.txt
|-- src/
|   |-- download_data.py
|   |-- eda.ipynb
|   |-- eda_selection.py
|   |-- model_utils.py
|   `-- models.py
`-- docs/
```

La carpeta `docs/` contiene la documentacion tecnica agregada al repositorio.

## Componentes principales

### `src/download_data.py`

Script responsable de dejar disponible `orders.csv` en `data/orders.csv` para el resto del pipeline.

### `src/eda.ipynb`

Notebook exploratorio breve. Carga `../data/orders.csv`, muestra las primeras filas del dataset y consulta la distribucion de la columna `category`.

El archivo conserva salidas guardadas, pero no documenta un analisis exploratorio completo ni un diccionario formal de columnas.

### `src/eda_selection.py`

Modulo principal de preprocesamiento y seleccion de caracteristicas. Incluye funciones para:

- Cargar `data/orders.csv`.
- Verificar valores nulos.
- Imputar datos con KNN.
- Codificar variables categoricas para seleccion.
- Seleccionar variables por varianza, correlacion e informacion mutua.
- Preparar datos para entrenamiento mediante `encoded_data`.
- Guardar `data/processed/df_visualizacion.csv`.

### `src/model_utils.py`

Modulo de utilidades para entrenamiento y evaluacion:

- Entrenamiento con escalado `MinMaxScaler`.
- Entrenamiento sin escalado.
- Evaluacion con `mean_squared_error` y `r2_score`.

### `src/models.py`

Script de entrenamiento y comparacion de modelos. Carga el dataset original y el dataset procesado, agrega algunas variables desde el dataset original, genera particiones de entrenamiento/prueba y evalua varios modelos de regresion.

## Dependencias

Las dependencias declaradas estan en `requirements.txt` y cubren obtencion de datos, procesamiento, modelado y visualizacion.

## Estado de sistema predictivo

El sistema actual funciona como pipeline local ejecutado por scripts.

