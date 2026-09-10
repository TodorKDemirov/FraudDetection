import pandas as pd
import pytest

from src.data.validate_data import validate_data


def create_valid_dataframe():
    data = {
        "Time": [0.0, 3600.0],
        "Amount": [10.0, 100.0],
        "Class": [0, 1],
    }

    for i in range(1, 29):
        data[f"V{i}"] = [0.0, 0.0]

    return pd.DataFrame(data)


def test_valid_data():
    df = create_valid_dataframe()

    # Should not raise an exception
    validate_data(df)


def test_empty_dataframe():
    df = pd.DataFrame()

    with pytest.raises(ValueError):
        validate_data(df)


def test_missing_required_column():
    df = create_valid_dataframe()

    df = df.drop(columns=["Amount"])

    with pytest.raises(ValueError):
        validate_data(df)


def test_invalid_class_values():
    df = create_valid_dataframe()

    df.loc[0, "Class"] = 2

    with pytest.raises(ValueError):
        validate_data(df)


def test_negative_amount():
    df = create_valid_dataframe()

    df.loc[0, "Amount"] = -10

    with pytest.raises(ValueError):
        validate_data(df)