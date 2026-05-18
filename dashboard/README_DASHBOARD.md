# Shipping Priority Dashboard

Dashboard sencillo servido por la API en `src/api.py`.

Ejecutar:

```bash
.venv/bin/python -m uvicorn src.api:app --reload --host 127.0.0.1 --port 8000
```

Abrir:

```text
http://127.0.0.1:8000
```

El boton **Simular pedido** envia `POST /api/orders/simulate`. La API toma un pedido aleatorio de `data/orders.csv`, predice valor, probabilidad de devolucion, revenue esperado y prioridad de envio. Cada decision se guarda en:

- `artifacts/reports/shipping_decisions.csv`
- `artifacts/reports/warehouse_queue.csv`

La explicacion completa esta en `docs/SHIPPING_PRIORITY_SYSTEM.md`.
