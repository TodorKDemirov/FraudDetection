import os

import mlflow
import mlflow.xgboost

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)


MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "sqlite:///mlflow.db"
)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def track_model(
    model,
    model_name,
    X_test,
    y_test,
    params,
    threshold=0.6
):
    mlflow.set_experiment("Fraud Detection")

    with mlflow.start_run(run_name=model_name):

        y_prob = model.predict_proba(X_test)[:, 1]

        y_pred = (y_prob >= threshold).astype(int)

        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)

        mlflow.log_params(params)

        mlflow.log_metrics({
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc
        })

        mlflow.xgboost.log_model(
            model,
            name="model"
        )

        print(f"{model_name} tracked in MLflow")