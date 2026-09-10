import mlflow

mlflow.set_tracking_uri("sqlite:///src/mlflow.db")

MODEL_NAME = "FraudDetectionXGBoost"
MODEL_VERSION = 1

client = mlflow.MlflowClient()

client.set_registered_model_alias(
    name=MODEL_NAME,
    alias="champion",
    version=MODEL_VERSION
)

print(f"{MODEL_NAME} version {MODEL_VERSION} is now 'champion'")