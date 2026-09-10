import joblib
from pathlib import Path

MODELS_DIR = Path("../models")

def save_model(model, model_name: str) -> None:
    MODELS_DIR.mkdir(exist_ok=True)

    model_path = MODELS_DIR / f'{model_name}.joblib'

    joblib.dump(model, model_path)

    print(f"Saved model to {model_path}")