import numpy as np

from src.models.evaluate import evaluate_model


class MockModel:
    def predict_proba(self, X):
        return np.array([
            [0.90, 0.10],
            [0.20, 0.80],
            [0.30, 0.70],
            [0.95, 0.05],
        ])


def test_evaluate_model():
    model = MockModel()

    X_test = [1, 2, 3, 4]
    y_test = [0, 1, 1, 0]

    metrics = evaluate_model(
        model,
        X_test,
        y_test,
        threshold=0.60
    )

    assert isinstance(metrics, dict)

    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "roc_auc" in metrics
    assert "pr_auc" in metrics

    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["pr_auc"] == 1.0