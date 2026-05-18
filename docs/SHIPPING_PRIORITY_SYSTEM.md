# Shipping Priority Dashboard

Este apartado explica la extension del proyecto: una mini aplicacion que usa el pipeline de Machine Learning para decidir la prioridad de envio de pedidos.

## Que hace el sistema

El sistema toma pedidos de e-commerce y calcula:

1. Valor esperado del pedido con un modelo de regresion.
2. Probabilidad de devolucion con un modelo de clasificacion.
3. Revenue esperado con esta formula:

```text
revenue esperado = valor predicho * (1 - probabilidad de devolucion) - costo de envio
```

Con ese resultado asigna una prioridad:

- `High`: pedido con revenue esperado alto y riesgo aceptable.
- `Medium`: pedido con revenue esperado moderado.
- `Low`: pedido con bajo revenue esperado o alta probabilidad de devolucion.

Los resultados se guardan como archivos CSV:

- `artifacts/reports/shipping_decisions.csv`: historial de decisiones.
- `artifacts/reports/warehouse_queue.csv`: cola simulada para almacen.

## Como se conecta con el proyecto original

No usa una base nueva. La demo entrena sus modelos con `data/orders.csv`, la misma base principal del proyecto.

Variables usadas:

- `product_name`
- `category`
- `quantity`
- `payment_method`
- `customer_rating`
- `session_duration_minutes`
- `pages_viewed_before_purchase`
- `is_repeat_customer`

Modelos usados:

- El mejor modelo de regresion del proyecto segun `R2` en `data/processed/model_results.csv` para predecir `total_amount_usd`.
- `Logistic Regression` para predecir `returned`.

En la ejecucion actual, el mejor regresor detectado es `MLPRegressor`.

## Como ejecutar la API y el dashboard

Desde la carpeta principal del proyecto:

```bash
.venv/bin/python -m uvicorn src.api:app --reload --host 127.0.0.1 --port 8000
```

Luego abrir:

```text
http://127.0.0.1:8000
```

La documentacion interactiva de la API queda en:

```text
http://127.0.0.1:8000/docs
```

## Endpoints de la API

`GET /api/health`

Sirve para revisar si la API esta viva.

`GET /api/model-info`

Muestra informacion de los modelos entrenados, como filas usadas, metricas y rutas de salida.

`GET /api/orders`

Devuelve las ultimas decisiones guardadas.

`POST /api/orders/simulate`

Toma un pedido aleatorio de `orders.csv`, lo procesa y guarda la decision.

`POST /api/orders/decide`

Permite enviar un pedido manualmente. Ejemplo:

```json
{
  "order_id": "API-DEMO-001",
  "customer_id": "C-DEMO",
  "product_name": "Mechanical Keyboard",
  "category": "Electronics",
  "quantity": 2,
  "payment_method": "Credit Card",
  "customer_rating": 4.5,
  "session_duration_minutes": 18,
  "pages_viewed_before_purchase": 8,
  "is_repeat_customer": 1,
  "shipping_cost_usd": 6.99
}
```

## Demo batch sin frontend

Tambien se puede correr una simulacion rapida por terminal:

```bash
.venv/bin/python src/integrations/shipping_priority.py
```

Eso genera varias decisiones y actualiza los CSV de reportes.
