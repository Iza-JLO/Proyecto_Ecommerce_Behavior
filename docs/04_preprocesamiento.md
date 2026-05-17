# Preprocesamiento

El preprocesamiento principal esta implementado en `src/eda_selection.py`.

## Carga de datos

La funcion `cargar_datos()` lee:

```txt
data/orders.csv
```

La variable objetivo definida globalmente es:

```python
TARGET = "total_amount_usd"
```

## Verificacion de nulos

La funcion `verificar_nulos(df, etapa)` calcula los valores nulos por columna y muestra en consola columnas con valores faltantes, numero de nulos y porcentaje de nulos sobre el total.

## Imputacion KNN

La funcion `imputar_knn(df)` usa:

```python
KNNImputer(n_neighbors=5)
```

Procedimiento implementado:

1. Se separan columnas numericas y categoricas.
2. La variable objetivo `total_amount_usd` se excluye de las columnas numericas a imputar.
3. Las categoricas se transforman temporalmente con `OrdinalEncoder`.
4. Se concatenan numericas y categoricas codificadas.
5. Se aplica `KNNImputer`.
6. Las categoricas imputadas se redondean, se limitan al rango valido y se invierten a sus categorias originales.

## Columnas eliminadas antes del entrenamiento

La funcion `encoded_data(df)` elimina columnas consideradas como posible fuga de informacion o de bajo aporte:

```txt
discount_pct
tax_pct
shipping_fee_usd
returned
order_status
```

La eliminacion usa `errors="ignore"`, por lo que el flujo no falla si alguna columna no existe.

## Particion entrenamiento/prueba

`encoded_data(df)` divide los datos con:

```python
train_test_split(X, y, test_size=0.2, random_state=97)
```

Esto crea una particion de 80% entrenamiento y 20% prueba.

## Escalado

El escalado no se aplica durante la seleccion de caracteristicas. Se aplica en entrenamiento para algunos modelos mediante `MinMaxScaler` dentro de un `Pipeline`, definido en `src/model_utils.py`.

