# Arquitectura general

El repositorio tiene una estructura simple orientada a scripts y un notebook exploratorio. No se observa una arquitectura de paquete Python formal ni una aplicacion web/API.

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

Script responsable de descargar `orders.csv` desde Kaggle mediante `kagglehub` y guardar el resultado en `data/orders.csv`.

### `src/eda.ipynb`

Notebook exploratorio breve. Carga `../data/orders.csv`, muestra las primeras filas del dataset y consulta la distribucion de la columna `category`.

El notebook contiene salidas ejecutadas que muestran un dataset con 28 columnas y ejemplos de columnas como `order_id`, `customer_id`, `order_date`, `year`, `month`, `quarter`, `day_of_week`, `product_name`, `category`, `unit_price_usd`, `payment_method`, `device_used`, `delivery_days`, `delivery_date`, `order_status`, `returned`, `customer_rating`, `session_duration_minutes`, `pages_viewed_before_purchase` e `is_repeat_customer`.

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

Las dependencias declaradas estan en `requirements.txt`:

```txt
kagglehub[pandas-datasets]
pandas
scikit-learn
numpy
matplotlib
seaborn
xgboost
```

## Estado de sistema predictivo

No se identifica una capa de servicio, API, dashboard, interfaz de usuario, persistencia de modelos ni despliegue. El sistema actual funciona como pipeline local ejecutado por scripts.

