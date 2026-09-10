import pandas as pd

from src.features.build_features import build_features


def create_dataframe():
    data = {
        "Time": [0.0, 3600.0, 7200.0, 86400.0],
        "Amount": [0.0, 10.0, 100.0, 300.0],
    }

    for i in range(1, 29):
        data[f"V{i}"] = [0.0, 0.0, 0.0, 0.0]

    return pd.DataFrame(data)


def test_hour_features():
    df = create_dataframe()
    result = build_features(df)

    assert "hour" in result.columns
    assert "hour_sin" in result.columns
    assert "hour_cos" in result.columns

    assert result.loc[0, "hour"] == 0
    assert result.loc[1, "hour"] == 1
    assert result.loc[2, "hour"] == 2


def test_amount_features():
    df = create_dataframe()
    result = build_features(df)

    assert "Amount_categorical_zero_amount" in result.columns
    assert "Amount_categorical_small" in result.columns
    assert "Amount_categorical_medium" in result.columns
    assert "Amount_categorical_large" in result.columns

    assert result.loc[0, "Amount_categorical_zero_amount"] == 1
    assert result.loc[1, "Amount_categorical_small"] == 1
    assert result.loc[2, "Amount_categorical_medium"] == 1
    assert result.loc[3, "Amount_categorical_large"] == 1


def test_original_columns_are_preserved():
    df = create_dataframe()
    result = build_features(df)

    assert "Time" in result.columns
    assert "Amount" in result.columns
    assert "V1" in result.columns
    assert "V28" in result.columns