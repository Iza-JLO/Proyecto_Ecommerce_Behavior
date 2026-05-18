# Proyecto_Ecommerce_Behavior
Proyecto de Machine Learning enfocado en el análisis del comportamiento de usuarios en e-commerce. Se aplican y comparan distintos modelos predictivos para identificar patrones de navegación e intención de compra, incluyendo preprocesamiento, ingeniería de características y evaluación de desempeño


1 (Ambientes y requerimientos)
Ejecutar en terminal:

    1.-Crear un ambiente: py -m venv .venv
    2.-.venv\Scripts\activate (windows)
    3.-py -m pip install -r requirements.txt

2 (Carga de datos)

Ejecutar en terminal:

    1.-py src/download_data.py

3 Pre-procesamiento inicial de datos: 

Ejecutar en terminal:

    1.-py src/eda_selection.py

4 Entrenamiento y evaluación de modelos:

Ejecutar en terminal:

    1.-py src/models.py

Los artefactos generados quedan en `artifacts/`:

    - artifacts/logs/: salida completa de cada ejecución
    - artifacts/reports/: resúmenes, métricas y features seleccionadas

# Mini-sistema C: Decisor de prioridad de envio

Demo hibrida rule + model para priorizar pedidos de e-commerce. Usa el dataset
existente, entrena rapidamente un modelo de valor de orden y otro de
probabilidad de devolucion, aplica reglas de negocio y escribe la cola demo en
`artifacts/reports/shipping_decisions.csv`.

API + frontend:

    py -m uvicorn src.api:app --reload --host 127.0.0.1 --port 8000

Demo batch:

    py src/integrations/shipping_priority.py

Abrir:

    http://127.0.0.1:8000
    http://127.0.0.1:8000/docs

Endpoints principales:

    - GET  /api/health
    - GET  /api/model-info
    - GET  /api/orders
    - POST /api/orders/simulate
    - POST /api/orders/decide
