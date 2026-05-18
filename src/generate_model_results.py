from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from eda_selection import encoded_data  # noqa: E402


ORDERS_PATH = ROOT_DIR / "data" / "orders.csv"
PROCESSED_PATH = ROOT_DIR / "data" / "processed" / "df_visualizacion.csv"
RESULTS_PATH = ROOT_DIR / "data" / "processed" / "model_results.csv"
PREDICTIONS_PATH = ROOT_DIR / "data" / "processed" / "model_predictions.csv"


def build_model_specs(random_seed: int = 42):
    return [
        ("DecisionTreeRegressor", "sin escalado", DecisionTreeRegressor(random_state=random_seed)),
        ("RandomForestRegressor", "sin escalado", RandomForestRegressor(n_estimators=150, random_state=random_seed)),
        (
            "XGBRegressor",
            "sin escalado",
            XGBRegressor(
                objective="reg:squarederror",
                random_state=random_seed,
                n_estimators=1200,
                learning_rate=0.02,
                max_depth=6,
                gamma=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
            ),
        ),
        (
            "MLPRegressor",
            "con escalado",
            MLPRegressor(
                hidden_layer_sizes=(256, 128, 64),
                solver="adam",
                max_iter=3000,
                batch_size=32,
                early_stopping=True,
                random_state=random_seed,
                learning_rate="adaptive",
                learning_rate_init=0.0001,
                alpha=0.0001,
            ),
        ),
        ("ElasticNet", "con escalado", ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=10000, random_state=random_seed)),
        ("KNeighborsRegressor", "con escalado", KNeighborsRegressor(n_neighbors=5, weights="distance", metric="minkowski", p=2)),
    ]


def build_pipeline(model, scaling: str) -> Pipeline:
    steps = []
    if scaling == "con escalado":
        steps.append(("scaler", MinMaxScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


def load_model_data() -> pd.DataFrame:
    if not ORDERS_PATH.exists():
        raise FileNotFoundError("No se encontró data/orders.csv. Ejecuta primero: python src/download_data.py")
    if not PROCESSED_PATH.exists():
        raise FileNotFoundError("No se encontró data/processed/df_visualizacion.csv. Ejecuta: python src/eda_selection.py")

    df_inicial = pd.read_csv(ORDERS_PATH)
    df_model = pd.read_csv(PROCESSED_PATH)

    extra_columns = ["pages_viewed_before_purchase", "session_duration_minutes", "month"]
    available_extra_columns = [col for col in extra_columns if col in df_inicial.columns and col not in df_model.columns]
    if available_extra_columns:
        df_model[available_extra_columns] = df_inicial[available_extra_columns]

    return df_model


def main() -> None:
    df_model = load_model_data()
    X_train, X_test, y_train, y_test = encoded_data(df=df_model)

    rows = []
    prediction_frames = []
    cv_folds = 5

    for model_name, scaling, estimator in build_model_specs():
        print(f"Evaluando {model_name} ({scaling})...")
        pipeline_for_cv = build_pipeline(clone(estimator), scaling)
        cv_scores = cross_val_score(pipeline_for_cv, X_train, y_train, cv=cv_folds, scoring="r2")

        pipeline = build_pipeline(clone(estimator), scaling)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rows.append(
            {
                "model_name": model_name,
                "scaling": scaling,
                "mse": mse,
                "r2": r2,
                "cv_r2_mean": float(np.mean(cv_scores)),
                "cv_r2_std": float(np.std(cv_scores)),
                "cv_folds": cv_folds,
                "train_rows": len(X_train),
                "test_rows": len(X_test),
                "target_scale": "log1p(total_amount_usd)",
            }
        )

        prediction_frames.append(
            pd.DataFrame(
                {
                    "model_name": model_name,
                    "scaling": scaling,
                    "y_true": y_test.to_numpy(),
                    "y_pred": y_pred,
                    "residual": y_test.to_numpy() - y_pred,
                    "target_scale": "log1p(total_amount_usd)",
                }
            )
        )

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULTS_PATH, index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(PREDICTIONS_PATH, index=False)
    print(f"Resultados guardados en {RESULTS_PATH.relative_to(ROOT_DIR)}")
    print(f"Predicciones guardadas en {PREDICTIONS_PATH.relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()
