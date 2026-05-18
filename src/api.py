from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .integrations.shipping_priority import (
    ShippingOrder,
    decide_shipping_priority,
    model_info,
    recent_decisions,
    simulate_order,
    train_shipping_models,
)
from .project_paths import PROJECT_ROOT, ensure_project_dirs


class OrderRequest(BaseModel):
    order_id: str = Field(default="API-DEMO-001")
    customer_id: str = Field(default="C-DEMO")
    product_name: str = Field(default="Mechanical Keyboard")
    category: str = Field(default="Electronics")
    quantity: float = Field(default=2, ge=1)
    payment_method: str = Field(default="Credit Card")
    customer_rating: float | None = Field(default=4.5, ge=1, le=5)
    session_duration_minutes: float | None = Field(default=18.0, ge=0)
    pages_viewed_before_purchase: float | None = Field(default=8, ge=0)
    is_repeat_customer: float | None = Field(default=1, ge=0, le=1)
    shipping_cost_usd: float = Field(default=6.99, ge=0)


app = FastAPI(
    title="Shipping Priority Dashboard API",
    description="API sencilla para priorizar pedidos usando modelos entrenados con orders.csv.",
    version="1.0.0",
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.on_event("startup")
def startup() -> None:
    ensure_project_dirs()
    train_shipping_models(force=False)


@app.get("/", response_class=HTMLResponse)
def dashboard() -> str:
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "project_root": str(PROJECT_ROOT)}


@app.get("/api/model-info")
def get_model_info() -> dict[str, Any]:
    return model_info()


@app.get("/api/orders")
def get_orders(limit: int = 25) -> list[dict[str, Any]]:
    return recent_decisions(limit=limit)


@app.post("/api/orders/simulate")
def post_simulate_order() -> dict[str, Any]:
    return decide_shipping_priority(simulate_order(), source="simulation")


@app.post("/api/orders/decide")
def post_decide_order(order: OrderRequest) -> dict[str, Any]:
    payload = order.model_dump() if hasattr(order, "model_dump") else order.dict()
    return decide_shipping_priority(ShippingOrder(**payload), source="api")
