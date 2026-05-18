import json
import logging
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from project_paths import DATA_DIR, LOGS_DIR, MODELS_DIR, PROCESSED_DATA_DIR, REPORTS_DIR, ensure_project_dirs


FEATURE_COLUMNS = [
    "product_name",
    "category",
    "quantity",
    "payment_method",
    "customer_rating",
    "session_duration_minutes",
    "pages_viewed_before_purchase",
    "is_repeat_customer",
]

NUMERIC_FEATURES = [
    "quantity",
    "customer_rating",
    "session_duration_minutes",
    "pages_viewed_before_purchase",
    "is_repeat_customer",
]
CATEGORICAL_FEATURES = ["product_name", "category", "payment_method"]

MODEL_BUNDLE_PATH = MODELS_DIR / "shipping_priority_models.joblib"
DECISIONS_PATH = REPORTS_DIR / "shipping_decisions.csv"
QUEUE_PATH = REPORTS_DIR / "warehouse_queue.csv"
MODEL_INFO_PATH = REPORTS_DIR / "shipping_model_info.json"

RANDOM_STATE = 42
PROJECT_MODEL_RESULTS_PATH = PROCESSED_DATA_DIR / "model_results.csv"


def _project_value_estimators() -> dict[str, Any]:
    return {
        "DecisionTreeRegressor": DecisionTreeRegressor(random_state=RANDOM_STATE),
        "RandomForestRegressor": RandomForestRegressor(n_estimators=150, random_state=RANDOM_STATE),
        "XGBRegressor": XGBRegressor(
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_estimators=1200,
            learning_rate=0.02,
            max_depth=6,
            gamma=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
        ),
        "MLPRegressor": MLPRegressor(
            hidden_layer_sizes=(256, 128, 64),
            solver="adam",
            max_iter=3000,
            batch_size=32,
            early_stopping=True,
            random_state=RANDOM_STATE,
            learning_rate="adaptive",
            learning_rate_init=0.0001,
            alpha=0.0001,
        ),
        "SVR": SVR(C=50, gamma="scale", kernel="poly", degree=3, epsilon=0.1),
        "ElasticNet": ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=10000, random_state=RANDOM_STATE),
        "KNeighborsRegressor": KNeighborsRegressor(n_neighbors=5, weights="distance", metric="minkowski", p=2),
    }


def _select_project_value_model() -> tuple[str, Any, dict[str, Any]]:
    estimators = _project_value_estimators()
    if PROJECT_MODEL_RESULTS_PATH.exists():
        results = pd.read_csv(PROJECT_MODEL_RESULTS_PATH)
        results = results[results["model_name"].isin(estimators)]
        if not results.empty:
            best = results.sort_values("r2", ascending=False).iloc[0].to_dict()
            model_name = str(best["model_name"])
            return model_name, estimators[model_name], {
                "model_name": model_name,
                "r2": float(best["r2"]),
                "mse": float(best["mse"]),
                "scaling": str(best.get("scaling", "")),
                "source_path": str(PROJECT_MODEL_RESULTS_PATH),
            }

    model_name = "RandomForestRegressor"
    return model_name, estimators[model_name], {
        "model_name": model_name,
        "source_path": str(PROJECT_MODEL_RESULTS_PATH),
        "note": "No model_results.csv found; using fallback project regressor.",
    }


def _needs_retrain(bundle: dict[str, Any]) -> bool:
    info = bundle.get("info", {})
    return info.get("value_model_source") != "best_project_model_by_r2"


@dataclass
class ShippingOrder:
    order_id: str
    customer_id: str
    product_name: str
    category: str
    quantity: float
    payment_method: str
    customer_rating: float | None = None
    session_duration_minutes: float | None = None
    pages_viewed_before_purchase: float | None = None
    is_repeat_customer: float | None = None
    shipping_cost_usd: float = 0.0


def _logger() -> logging.Logger:
    ensure_project_dirs()
    logger = logging.getLogger("shipping_priority")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(LOGS_DIR / "shipping_priority.log", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    return logger


def _read_orders() -> pd.DataFrame:
    path = DATA_DIR / "orders.csv"
    if not path.exists():
        raise FileNotFoundError(f"No se encontro {path}. Ejecuta primero src/download_data.py.")
    return pd.read_csv(path)


def _build_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ]
    )


def train_shipping_models(force: bool = False) -> dict[str, Any]:
    ensure_project_dirs()
    if MODEL_BUNDLE_PATH.exists() and not force:
        bundle = joblib.load(MODEL_BUNDLE_PATH)
        if not _needs_retrain(bundle):
            return bundle

    df = _read_orders()
    required = FEATURE_COLUMNS + ["total_amount_usd", "returned"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en orders.csv: {missing}")

    X = df[FEATURE_COLUMNS].copy()
    y_value = np.log1p(df["total_amount_usd"].astype(float))
    y_return = df["returned"].astype(int)

    X_train, X_test, y_value_train, y_value_test, y_return_train, y_return_test = train_test_split(
        X,
        y_value,
        y_return,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_return,
    )

    value_model_name, value_estimator, project_best = _select_project_value_model()
    value_steps: list[tuple[str, Any]] = [("preprocess", _build_preprocessor())]
    if project_best.get("scaling") == "con escalado":
        from sklearn.preprocessing import MinMaxScaler

        value_steps.append(("scaler", MinMaxScaler()))
    value_steps.append(("model", value_estimator))
    value_model = Pipeline(value_steps)
    return_model = Pipeline(
        [
            ("preprocess", _build_preprocessor()),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    value_model.fit(X_train, y_value_train)
    return_model.fit(X_train, y_return_train)

    value_pred_log = value_model.predict(X_test)
    return_pred = return_model.predict(X_test)
    return_prob = return_model.predict_proba(X_test)[:, 1]

    info = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "features": FEATURE_COLUMNS,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "value_r2_log": float(r2_score(y_value_test, value_pred_log)),
        "value_rmse_usd": float(root_mean_squared_error(np.expm1(y_value_test), np.expm1(value_pred_log))),
        "return_accuracy": float(accuracy_score(y_return_test, return_pred)),
        "return_auc": float(roc_auc_score(y_return_test, return_prob)),
        "model_type": f"{value_model_name} + Logistic Regression",
        "value_model_source": "best_project_model_by_r2",
        "project_best_value_model": project_best,
    }

    bundle = {"value_model": value_model, "return_model": return_model, "info": info}
    joblib.dump(bundle, MODEL_BUNDLE_PATH)
    MODEL_INFO_PATH.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    _logger().info("shipping models trained %s", json.dumps(info))
    return bundle


def load_shipping_models() -> dict[str, Any]:
    return train_shipping_models(force=False)


def model_info() -> dict[str, Any]:
    bundle = load_shipping_models()
    return bundle["info"] | {
        "decisions_path": str(DECISIONS_PATH),
        "queue_path": str(QUEUE_PATH),
    }


def _order_to_frame(order: ShippingOrder) -> pd.DataFrame:
    data = asdict(order)
    features = {column: data.get(column) for column in FEATURE_COLUMNS}
    return pd.DataFrame([features])


def _priority(expected_revenue: float, return_probability: float) -> tuple[str, str, str]:
    if return_probability >= 0.45:
        return (
            "Low",
            "high",
            "Low priority: Expected revenue or return risk does not justify fast-track.",
        )
    if expected_revenue >= 150 and return_probability < 0.35:
        return (
            "High",
            "medium" if return_probability >= 0.15 else "low",
            "High priority: High expected revenue after return risk and shipping cost.",
        )
    if expected_revenue >= 80:
        return (
            "Medium",
            "medium",
            "Medium priority: Moderate expected revenue; keep in normal priority lane.",
        )
    return (
        "Low",
        "medium" if return_probability >= 0.15 else "low",
        "Low priority: Expected revenue or return risk does not justify fast-track.",
    )


def decide_shipping_priority(order: ShippingOrder, source: str = "api") -> dict[str, Any]:
    bundle = load_shipping_models()
    X = _order_to_frame(order)

    predicted_value = float(np.expm1(bundle["value_model"].predict(X)[0]))
    return_probability = float(bundle["return_model"].predict_proba(X)[0, 1])
    expected_revenue = predicted_value * (1 - return_probability) - float(order.shipping_cost_usd)
    priority, return_risk, message = _priority(expected_revenue, return_probability)

    decision = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "order_id": order.order_id,
        "customer_id": order.customer_id,
        "category": order.category,
        "product_name": order.product_name,
        "quantity": float(order.quantity),
        "predicted_order_value_usd": round(predicted_value, 2),
        "return_probability": round(return_probability, 4),
        "shipping_cost_usd": round(float(order.shipping_cost_usd), 2),
        "expected_revenue_usd": round(expected_revenue, 2),
        "shipping_priority": priority,
        "return_risk": return_risk,
        "routing_message": message,
        "source": source,
    }
    _append_decision(decision)
    _logger().info("decision queued %s", json.dumps(decision))
    return decision


def _append_decision(decision: dict[str, Any]) -> None:
    ensure_project_dirs()
    row = pd.DataFrame([decision])
    for path in (DECISIONS_PATH, QUEUE_PATH):
        row.to_csv(path, mode="a", header=not path.exists(), index=False)


def recent_decisions(limit: int = 25) -> list[dict[str, Any]]:
    if not DECISIONS_PATH.exists():
        return []
    df = pd.read_csv(DECISIONS_PATH).tail(limit)
    return df.iloc[::-1].to_dict(orient="records")


def simulate_order() -> ShippingOrder:
    df = _read_orders()
    row = df.sample(1).iloc[0]
    return ShippingOrder(
        order_id=str(row["order_id"]),
        customer_id=str(row["customer_id"]),
        product_name=str(row["product_name"]),
        category=str(row["category"]),
        quantity=float(row["quantity"]),
        payment_method=str(row["payment_method"]),
        customer_rating=None if pd.isna(row["customer_rating"]) else float(row["customer_rating"]),
        session_duration_minutes=float(row["session_duration_minutes"]),
        pages_viewed_before_purchase=float(row["pages_viewed_before_purchase"]),
        is_repeat_customer=float(row["is_repeat_customer"]),
        shipping_cost_usd=float(row["shipping_fee_usd"]),
    )


def run_demo_batch(count: int = 10) -> list[dict[str, Any]]:
    train_shipping_models(force=False)
    return [decide_shipping_priority(simulate_order(), source="simulation") for _ in range(count)]


if __name__ == "__main__":
    decisions = run_demo_batch()
    print(f"Demo completada. Decisiones guardadas en {DECISIONS_PATH}")
    print(pd.DataFrame(decisions).to_string(index=False))
