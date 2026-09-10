import os
from pathlib import Path

import joblib
import mlflow


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_final.joblib"
MODEL_NAME = "FraudDetectionXGBoost"

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://localhost:5000"
)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Fraud Detection")


print(f"Loading model from: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

print("Model loaded successfully.")


with mlflow.start_run(run_name="xgboost_final") as run:

    mlflow.log_param("model_type", "XGBClassifier")

    mlflow.xgboost.log_model(
        model,
        name="model"
    )

    run_id = run.info.run_id

    print(f"Run ID: {run_id}")


model_uri = f"runs:/{run_id}/model"

print("Registering model...")

model_version = mlflow.register_model(
    model_uri=model_uri,
    name=MODEL_NAME
)

print(
    f"Registered model: {model_version.name}, "
    f"version: {model_version.version}"
)


client = mlflow.MlflowClient()

client.set_registered_model_alias(
    name=MODEL_NAME,
    alias="champion",
    version=model_version.version
)

print(
    f"{MODEL_NAME} version {model_version.version} "
    f"is now 'champion'."
)

print("Done!")