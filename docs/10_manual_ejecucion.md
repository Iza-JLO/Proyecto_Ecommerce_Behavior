# Manual de ejecucion

Este manual describe como ejecutar el flujo implementado actualmente.

## Requisitos

Dependencias declaradas:

```txt
pandas
scikit-learn
numpy
matplotlib
seaborn
xgboost
```

## Crear entorno virtual en Windows

```powershell
py -m venv .venv
.\.venv\Scripts\activate
py -m pip install -r requirements.txt
```

## 1. Descargar datos

```powershell
py src/download_data.py
```

Salida esperada:

```txt
data/orders.csv
```

Nota: esta etapa deja disponible el archivo local requerido por el resto del pipeline.

## 2. Ejecutar preprocesamiento y seleccion

```powershell
py src/eda_selection.py
```

Salida esperada:

```txt
data/processed/df_visualizacion.csv
```

## 3. Entrenar y evaluar modelos

```powershell
py src/models.py
```

El script imprime en consola informacion de carga de datos, tamano de conjuntos de entrenamiento/prueba, informacion de `X_train`, MSE y R2 para cada modelo.

## 4. Notebook exploratorio

El notebook disponible es:

```txt
src/eda.ipynb
```

Este notebook espera que exista:

```txt
data/orders.csv
```

## Orden recomendado

```txt
1. py src/download_data.py
2. py src/eda_selection.py
3. py src/models.py
```

## Problemas conocidos de ejecucion

- Los datos no estan versionados porque `data/` esta en `.gitignore`.
- Los modelos entrenados no se guardan; cada ejecucion vuelve a entrenarlos.
- Las metricas no se guardan en archivo; solo se imprimen en consola.

