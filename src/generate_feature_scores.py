from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dashboard.utils import (  # noqa: E402
    FEATURE_SCORES_PATH,
    ORDERS_PATH,
    PROCESSED_PATH,
    compute_feature_scores,
    detect_target,
)

import pandas as pd  # noqa: E402


def main() -> None:
    source_path = PROCESSED_PATH if PROCESSED_PATH.exists() else ORDERS_PATH
    if not source_path.exists():
        raise FileNotFoundError(
            "No se encontraron datos para calcular Mutual Information. "
            "Ejecuta primero: python src/download_data.py y python src/eda_selection.py"
        )

    df = pd.read_csv(source_path)
    target = detect_target(df.columns.tolist())
    if target is None:
        raise ValueError("No fue posible detectar la variable objetivo desde el código o los datos.")

    scores = compute_feature_scores(df, target)
    FEATURE_SCORES_PATH.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(FEATURE_SCORES_PATH, index=False)
    print(f"Scores guardados en {FEATURE_SCORES_PATH.relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()
