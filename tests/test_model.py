import pandas as pd

from src.features.build_features import build_features
from src.models.train import split_data, train_xgboost


def create_dataframe():
    data = {
        "Time": [
            0.0, 3600.0, 7200.0, 10800.0,
            14400.0, 18000.0, 21600.0, 25200.0,
            28800.0, 32400.0, 36000.0, 39600.0,
        ],
        "Amount": [
            10.0, 20.0, 30.0, 40.0,
            50.0, 60.0, 70.0, 80.0,
            90.0, 100.0, 110.0, 120.0,
        ],
        "Class": [
            0, 0, 0, 0,
            0, 0, 1, 1,
            0, 0, 1, 0,
        ],
    }

    for i in range(1, 29):
        data[f"V{i}"] = [0.0] * 12

    return pd.DataFrame(data)


def test_split_data():
    df = create_dataframe()
    df = build_features(df)

    X_train, X_test, y_train, y_test = split_data(df)

    assert len(X_train) + len(X_test) == len(df)
    assert len(y_train) + len(y_test) == len(df)

    assert "Class" not in X_train.columns
    assert "Class" not in X_test.columns
    assert "hour" not in X_train.columns
    assert "Amount" not in X_train.columns


def test_xgboost_training():
    df = create_dataframe()
    df = build_features(df)

    X_train, X_test, y_train, y_test = split_data(df)

    model = train_xgboost(X_train, y_train)

    assert model is not None
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")


def test_xgboost_prediction():
    df = create_dataframe()
    df = build_features(df)

    X_train, X_test, y_train, y_test = split_data(df)

    model = train_xgboost(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    assert len(predictions) == len(X_test)
    assert probabilities.shape[0] == len(X_test)
    assert probabilities.shape[1] == 2