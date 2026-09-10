from pathlib import Path

from src.data.load_data import load_data
from src.data.validate_data import validate_data
from src.features.build_features import build_features

from src.models.train import (
    split_data,
    train_xgboost
)

from src.models.evaluate import evaluate_model
from src.models.save_model import save_model
from src.tracking.mlflow_tracking import track_model

from src.config import (
    XGB_N_ESTIMATORS,
    XGB_MAX_DEPTH,
    XGB_LEARNING_RATE,
    XGB_SUBSAMPLE,
    XGB_COLSAMPLE_BYTREE,
    XGB_MIN_CHILD_WEIGHT,
    XGB_GAMMA,
    XGB_REG_ALPHA,
    XGB_REG_LAMBDA,
    XGB_THRESHOLD
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "creditcard.csv"


def run_pipeline():

    print("=== Starting Fraud Detection Pipeline ===")

    # 1. Load data
    print("\n[1/7] Loading data...")
    df = load_data(DATA_PATH)

    # 2. Validate data
    print("[2/7] Validating data...")
    validate_data(df)

    # 3. Build features
    print("[3/7] Building features...")
    df = build_features(df)

    # 4. Split data
    print("[4/7] Splitting data...")
    X_train, X_test, y_train, y_test = split_data(df)

    # 5. Train model
    print("[5/7] Training XGBoost...")
    model = train_xgboost(X_train, y_train)

    # 6. Evaluate model
    print("[6/7] Evaluating model...")
    metrics = evaluate_model(
        model,
        X_test,
        y_test,
        threshold=XGB_THRESHOLD
    )

    # 7. Save and track model
    print("[7/7] Saving and tracking model...")

    save_model(model, "xgboost_final")

    params = {
        "n_estimators": XGB_N_ESTIMATORS,
        "max_depth": XGB_MAX_DEPTH,
        "learning_rate": XGB_LEARNING_RATE,
        "subsample": XGB_SUBSAMPLE,
        "colsample_bytree": XGB_COLSAMPLE_BYTREE,
        "min_child_weight": XGB_MIN_CHILD_WEIGHT,
        "gamma": XGB_GAMMA,
        "reg_alpha": XGB_REG_ALPHA,
        "reg_lambda": XGB_REG_LAMBDA,
        "threshold": XGB_THRESHOLD
    }

    track_model(
        model,
        "XGBoost",
        X_test,
        y_test,
        params,
        threshold=XGB_THRESHOLD
    )

    print("\n=== Pipeline completed successfully ===")

    return metrics


if __name__ == "__main__":
    run_pipeline()