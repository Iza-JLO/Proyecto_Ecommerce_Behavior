# Dashboard académico del proyecto

Este dashboard en Streamlit presenta de forma interactiva el flujo del proyecto `Proyecto_Ecommerce_Behavior`: preparación de datos, justificación de variables y evaluación de modelos predictivos para estimar `total_amount_usd`.

La app no inventa resultados. Si faltan archivos, muestra advertencias e indica el comando necesario para generarlos.

## Instalación

Desde la raíz del repositorio:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

En Windows, si usas el lanzador `py`:

```bash
py -m venv .venv
.venv\Scripts\activate
py -m pip install -r requirements.txt
```

Si el entorno ya existe, basta con activarlo e instalar dependencias:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Generar datos

El dataset original se descarga con:

```bash
python src/download_data.py
```

El dataset procesado se genera con el archivo real encontrado en el repositorio:

```bash
python src/eda_selection.py
```

Nota técnica: el README principal menciona `src/eda_selection_data.py`, pero en el repositorio actual el script existente es `src/eda_selection.py`.

## Generar scores de variables

Si quieres guardar Mutual Information en CSV:

```bash
python src/generate_feature_scores.py
```

Salida esperada:

```text
data/processed/feature_scores.csv
```

## Generar resultados de modelos

El dashboard no entrena modelos al abrirse. Para guardar métricas reales:

```bash
python src/generate_model_results.py
```

Salidas esperadas:

```text
data/processed/model_results.csv
data/processed/model_predictions.csv
```

El script reproduce los modelos definidos en `src/models.py` y guarda MSE, R², CV R² promedio, CV R² desviación, tamaños de train/test y predicciones.

Nota técnica: `SVR` está definido en `src/models.py`, pero el script auxiliar lo excluye porque en pruebas previas puede tardar más de 40 minutos. Si se necesita evaluarlo, debe agregarse manualmente al script y ejecutarse con tiempo suficiente.

## Ejecutar la app

Desde la raíz del repositorio:

```bash
streamlit run dashboard/app.py
```

## Qué muestra cada sección

### 1. Pipeline del proyecto

Muestra el estado de los archivos principales, resumen del dataset original y procesado, nulos por columna, etapas del pipeline y comparación de columnas antes/después.

### 2. Justificación de variables

Compara variables originales y procesadas, clasifica su estado como conservada, eliminada, target o no determinada, y muestra scores de Mutual Information si existen o si pueden calcularse con los datos disponibles.

### 3. Evaluación de modelos

Lee métricas guardadas en `data/processed/model_results.csv`, compara modelos con y sin escalado, permite seleccionar la métrica principal y muestra predicción vs valor real cuando existe `model_predictions.csv`.

## Guía rápida para exposición

1. Presentar el objetivo del proyecto: estimar el monto total de compra a partir de comportamiento y atributos de pedidos e-commerce.
2. Mostrar el pipeline y confirmar qué archivos están disponibles.
3. Explicar el tratamiento de nulos e imputación KNN según `src/eda_selection.py`.
4. Mostrar variables conservadas y eliminadas usando la comparación original vs procesado.
5. Explicar Mutual Information como criterio de apoyo para selección de variables. Si el CSV de scores no existe, generarlo con `python src/generate_feature_scores.py`.
6. Comparar modelos desde la sección de evaluación.
7. Explicar la métrica principal seleccionada: MSE, R² o CV R².
8. Concluir con el modelo que aparece automáticamente como mejor según la métrica seleccionada. No es necesario memorizar resultados porque dependen de los archivos generados.

## Limitaciones conocidas

- En el repositorio inspeccionado inicialmente no existía la carpeta `data/`, por lo que el dashboard queda preparado para mostrar advertencias hasta que se generen los CSV.
- `src/models.py` ejecuta entrenamiento al importarse. Por eso la app no lo importa directamente y usa `src/generate_model_results.py` como script auxiliar.
- `SVR` se omite del script auxiliar de resultados por tiempo de ejecución elevado.
- Las métricas del script de modelos se calculan sobre la escala usada por `encoded_data()`, donde el target se transforma con `np.log1p`.
- Si el dataset de Kaggle cambia de columnas, la app intentará adaptarse, pero los scripts originales pueden requerir las columnas esperadas por `encoded_data()`.
