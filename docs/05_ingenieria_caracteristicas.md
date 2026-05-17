# Ingenieria de caracteristicas

La ingenieria y seleccion de caracteristicas se implementa principalmente en `src/eda_selection.py`.

## Seleccion de variables

La funcion `seleccionar_features(df_enc)` ejecuta tres pasos.

### Eliminacion de constantes

Usa:

```python
VarianceThreshold(threshold=0.0)
```

### Eliminacion por alta correlacion

Calcula una matriz de correlacion absoluta y elimina columnas con correlacion superior a:

```python
CORR_THRESHOLD = 0.95
```

### Informacion mutua contra el target

Usa:

```python
mutual_info_regression(X, y, random_state=42)
```

El script conserva variables con puntaje de informacion mutua en el intervalo:

```txt
0.001 < MI <= 0.3
```

El comentario del codigo indica que este umbral se ajusto para evitar variables con posible `data leakage`, como impuestos o descuentos.

## Encoding para entrenamiento

La funcion `encoded_data(df)` aplica dos estrategias de codificacion.

### One-Hot Encoding

Columnas codificadas con `OneHotEncoder`:

```txt
payment_method
day_of_week
month
```

### Target Encoding

Columnas codificadas con `TargetEncoder`:

```txt
product_name
category
```

El ajuste se realiza con `X_train` y `y_train`, y luego se transforma `X_test`, lo cual reduce riesgo de fuga de informacion desde el conjunto de prueba.

## Variables agregadas en entrenamiento

En `src/models.py`, antes del entrenamiento se agregan al dataframe procesado estas columnas desde `data/orders.csv`:

```txt
pages_viewed_before_purchase
session_duration_minutes
month
```

El comentario del codigo indica que estas variables se agregaron manualmente para intentar mejorar el rendimiento de los modelos.

