import mlflow

mlflow.set_tracking_uri("sqlite:///src/mlflow.db")

RUN_ID = "8783a7b427614fbea01a11daa2c2d2fc"
MODEL_NAME = "FraudDetectionXGBoost"

model_uri = f"runs:/{RUN_ID}/model"

model_version = mlflow.register_model(
    model_uri=model_uri,
    name=MODEL_NAME
)

print(f"Model registered: {model_version.name}")
print(f"Version: {model_version.version}")
