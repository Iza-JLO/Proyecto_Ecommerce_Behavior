from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
LOGS_DIR = ARTIFACTS_DIR / "logs"
MODELS_DIR = ARTIFACTS_DIR / "models"


def ensure_project_dirs() -> None:
    for path in (DATA_DIR, PROCESSED_DATA_DIR, REPORTS_DIR, LOGS_DIR, MODELS_DIR):
        path.mkdir(parents=True, exist_ok=True)
