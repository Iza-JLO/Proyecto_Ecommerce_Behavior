from __future__ import annotations

import ast
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
ORDERS_PATH = DATA_DIR / "orders.csv"
PROCESSED_PATH = PROCESSED_DIR / "df_visualizacion.csv"
FEATURE_SCORES_PATH = PROCESSED_DIR / "feature_scores.csv"
MODEL_RESULTS_PATH = PROCESSED_DIR / "model_results.csv"
MODEL_PREDICTIONS_PATH = PROCESSED_DIR / "model_predictions.csv"
EDA_SCRIPT_PATH = ROOT_DIR / "src" / "eda_selection.py"
MODELS_SCRIPT_PATH = ROOT_DIR / "src" / "models.py"


@dataclass(frozen=True)
class FileRecord:
    label: str
    path: Path
    exists: bool
    modified: str
    size_kb: float | None


def relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT_DIR))
    except ValueError:
        return str(path)


def file_record(label: str, path: Path) -> FileRecord:
    if not path.exists():
        return FileRecord(label, path, False, "No disponible", None)
    stat = path.stat()
    modified = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
    return FileRecord(label, path, True, modified, round(stat.st_size / 1024, 2))


def project_file_status() -> pd.DataFrame:
    records = [
        file_record("Dataset original", ORDERS_PATH),
        file_record("Dataset procesado", PROCESSED_PATH),
        file_record("Scores de variables", FEATURE_SCORES_PATH),
        file_record("Resultados de modelos", MODEL_RESULTS_PATH),
        file_record("Predicciones de modelos", MODEL_PREDICTIONS_PATH),
    ]
    return pd.DataFrame(
        [
            {
                "archivo": record.label,
                "ruta": relative_path(record.path),
                "existe": record.exists,
                "modificado": record.modified,
                "tamano_kb": record.size_kb,
            }
            for record in records
        ]
    )


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def dataframe_profile(df: pd.DataFrame) -> dict[str, Any]:
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()
    return {
        "filas": len(df),
        "columnas": df.shape[1],
        "numericas": numeric_cols,
        "categoricas": categorical_cols,
    }


def null_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["columna", "nulos", "porcentaje"])
    nulos = df.isna().sum()
    return (
        pd.DataFrame(
            {
                "columna": nulos.index,
                "nulos": nulos.values,
                "porcentaje": np.where(len(df) > 0, nulos.values / len(df) * 100, 0),
            }
        )
        .sort_values(["nulos", "columna"], ascending=[False, True])
        .reset_index(drop=True)
    )


def _literal_assignments(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    tree = ast.parse(path.read_text(encoding="utf-8"))
    values: dict[str, Any] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    try:
                        values[target.id] = ast.literal_eval(node.value)
                    except Exception:
                        continue
    return values


def detect_target(columns: list[str] | None = None) -> str | None:
    values = _literal_assignments(EDA_SCRIPT_PATH)
    target = values.get("TARGET")
    if isinstance(target, str):
        return target
    columns = columns or []
    candidates = ["total_amount_usd", "total_amount", "target", "y"]
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def detect_thresholds() -> dict[str, Any]:
    values = _literal_assignments(EDA_SCRIPT_PATH)
    return {
        "KNN_NEIGHBORS": values.get("KNN_NEIGHBORS"),
        "CORR_THRESHOLD": values.get("CORR_THRESHOLD"),
        "MI_THRESHOLD": values.get("MI_THRESHOLD"),
        "RANDOM_STATE": values.get("RANDOM_STATE"),
    }


def detect_columns_dropped_in_code() -> list[str]:
    if not EDA_SCRIPT_PATH.exists():
        return []
    tree = ast.parse(EDA_SCRIPT_PATH.read_text(encoding="utf-8"))
    dropped: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "drop":
            continue
        for keyword in node.keywords:
            if keyword.arg == "columns":
                try:
                    value = ast.literal_eval(keyword.value)
                except Exception:
                    value = None
                if isinstance(value, str):
                    dropped.append(value)
                elif isinstance(value, list):
                    dropped.extend([item for item in value if isinstance(item, str)])
    return sorted(set(dropped))


def detect_model_specs_from_source() -> pd.DataFrame:
    if not MODELS_SCRIPT_PATH.exists():
        return pd.DataFrame(columns=["model_name", "scaling", "source_variable"])
    tree = ast.parse(MODELS_SCRIPT_PATH.read_text(encoding="utf-8"))
    model_vars: dict[str, str] = {}
    scaled_vars: set[str] = set()
    unscaled_vars: set[str] = set()

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and isinstance(node.value, ast.Call):
                func = node.value.func
                if isinstance(func, ast.Name):
                    model_vars[target.id] = func.id
            if isinstance(target, ast.Name) and target.id in {"modelos_scalados", "modelos_no_scalados"}:
                try:
                    values = ast.literal_eval(node.value)
                except Exception:
                    values = []
                names = []
                if isinstance(node.value, ast.List):
                    names = [elt.id for elt in node.value.elts if isinstance(elt, ast.Name)]
                if target.id == "modelos_scalados":
                    scaled_vars.update(names or values)
                else:
                    unscaled_vars.update(names or values)

    rows = []
    for var, name in model_vars.items():
        if var in scaled_vars:
            rows.append({"model_name": name, "scaling": "con escalado", "source_variable": var})
        elif var in unscaled_vars:
            rows.append({"model_name": name, "scaling": "sin escalado", "source_variable": var})
    return pd.DataFrame(rows)


def compare_columns(original: pd.DataFrame | None, processed: pd.DataFrame | None, target: str | None) -> pd.DataFrame:
    original_cols = set(original.columns) if original is not None else set()
    processed_cols = set(processed.columns) if processed is not None else set()
    dropped_in_code = set(detect_columns_dropped_in_code())
    rows = []
    for col in sorted(original_cols | processed_cols):
        if target and col == target:
            status = "Target"
        elif col in original_cols and col in processed_cols:
            status = "Conservada"
        elif col in original_cols and col not in processed_cols:
            status = "Eliminada"
        elif col not in original_cols and col in processed_cols:
            status = "Generada/solo procesada"
        else:
            status = "No determinada"

        if col in dropped_in_code:
            comment = "El código la elimina explícitamente en encoded_data."
        elif status == "Eliminada":
            comment = "Motivo no determinado desde los archivos actuales."
        elif status == "Target":
            comment = "Variable objetivo detectada desde el código."
        elif status == "Conservada":
            comment = "Aparece en el dataset original y en el procesado."
        else:
            comment = "Estado derivado por comparación de columnas."

        rows.append(
            {
                "variable": col,
                "estado": status,
                "en_original": col in original_cols,
                "en_procesado": col in processed_cols,
                "comentario": comment,
            }
        )
    return pd.DataFrame(rows)


def compute_feature_scores(df: pd.DataFrame, target: str) -> pd.DataFrame:
    from sklearn.feature_selection import mutual_info_regression
    from sklearn.preprocessing import OrdinalEncoder

    if target not in df.columns:
        raise ValueError(f"No se encontró el target '{target}' en el dataset.")

    work = df.copy()
    y = pd.to_numeric(work[target], errors="coerce")
    valid_y = y.notna()
    work = work.loc[valid_y].drop(columns=[target])
    y = y.loc[valid_y]

    if work.empty:
        return pd.DataFrame(columns=["variable", "mi_score"])

    numeric_cols = work.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = [col for col in work.columns if col not in numeric_cols]

    X_parts = []
    feature_names: list[str] = []
    discrete_features: list[bool] = []

    if numeric_cols:
        num = work[numeric_cols].apply(pd.to_numeric, errors="coerce")
        num = num.fillna(num.median(numeric_only=True)).fillna(0)
        X_parts.append(num.to_numpy())
        feature_names.extend(numeric_cols)
        discrete_features.extend([False] * len(numeric_cols))

    if categorical_cols:
        cat = work[categorical_cols].astype("object").where(work[categorical_cols].notna(), "__MISSING__")
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_parts.append(enc.fit_transform(cat))
        feature_names.extend(categorical_cols)
        discrete_features.extend([True] * len(categorical_cols))

    X = np.hstack(X_parts)
    scores = mutual_info_regression(X, y.to_numpy(), discrete_features=discrete_features, random_state=42)
    return (
        pd.DataFrame({"variable": feature_names, "mi_score": scores})
        .sort_values("mi_score", ascending=False)
        .reset_index(drop=True)
    )


def attach_variable_status(scores: pd.DataFrame, status: pd.DataFrame) -> pd.DataFrame:
    if scores.empty:
        return scores
    merged = scores.merge(status, on="variable", how="left")
    merged["estado"] = merged["estado"].fillna("No determinada")
    merged["comentario"] = merged["comentario"].fillna("Motivo no determinado desde los archivos actuales.")
    return merged


def best_model_row(results: pd.DataFrame, metric: str) -> pd.Series | None:
    if results.empty or metric not in results.columns:
        return None
    values = pd.to_numeric(results[metric], errors="coerce")
    if values.dropna().empty:
        return None
    idx = values.idxmin() if metric == "mse" else values.idxmax()
    return results.loc[idx]
