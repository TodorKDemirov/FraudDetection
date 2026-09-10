import joblib
from pathlib import Path

MODELS_DIR = Path("models")

def load_model(model_name: str):
    model_path = MODELS_DIR / f'{model_name}.joblib'

    if not model_path.exists():
        raise FileNotFoundError(f"Model {model_name} does not exist")

    return joblib.load(model_path)