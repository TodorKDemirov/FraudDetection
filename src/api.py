import mlflow
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from src.features.build_features import build_features
from src.config import XGB_THRESHOLD

import os

mlflow.set_tracking_uri(
    os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
)

MODEL_NAME = "FraudDetectionXGBoost"
MODEL_ALIAS = "champion"

MODEL_URI = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"

# Load the actual sklearn/XGBoost model
model = mlflow.xgboost.load_model(MODEL_URI)

# Get the exact feature order used during training
FEATURE_COLUMNS = model.feature_names_in_.tolist()


app = FastAPI(
    title="Fraud Detection API",
    description="API for credit card fraud detection",
    version="1.0.0"
)


class Transaction(BaseModel):
    Time: float
    Amount: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float


@app.get("/")
def root():
    return {
        "message": "Fraud Detection API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "alias": MODEL_ALIAS
    }


@app.post("/predict")
def predict(transaction: Transaction):

    # Convert request to DataFrame
    df = pd.DataFrame([transaction.model_dump()])

    # Build the same features used during training
    df = build_features(df)

    # Remove features that were not used by the model
    df = df.drop(
        ["Class", "hour", "Amount"],
        axis=1,
        errors="ignore"
    )

    # Ensure exact feature order
    df = df[FEATURE_COLUMNS]

    # Get fraud probability
    fraud_probability = float(
        model.predict_proba(df)[0, 1]
    )

    # Apply selected threshold
    is_fraud = int(
        fraud_probability >= XGB_THRESHOLD
    )

    return {
        "is_fraud": is_fraud,
        "fraud_probability": fraud_probability,
        "threshold": XGB_THRESHOLD
    }