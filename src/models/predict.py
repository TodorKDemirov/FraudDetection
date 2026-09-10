from src.config import XGB_THRESHOLD


def predict(model, X):
    probability = model.predict_proba(X)[:, 1]

    prediction = (probability >= XGB_THRESHOLD).astype(int)

    return prediction, probability