import pandas as pd
import numpy as np


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Create hour from Time
    df["hour"] = (df["Time"] // 3600) % 24

    # Cyclical encoding of hour
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # Amount categories
    df["Amount_categorical_zero_amount"] = (df["Amount"] <= 0).astype(float)
    df["Amount_categorical_small"] = (
        (df["Amount"] > 0) & (df["Amount"] <= 20)
    ).astype(float)

    df["Amount_categorical_medium"] = (
        (df["Amount"] > 20) & (df["Amount"] <= 200)
    ).astype(float)

    df["Amount_categorical_large"] = (
        df["Amount"] > 200
    ).astype(float)

    return df