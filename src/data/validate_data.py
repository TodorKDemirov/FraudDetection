import pandas as pd

REQUIRED_COLUMNS = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9',
                    'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',
                    'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount', 'Class']

def validate_data(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError('Dataset is empty.')

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]

    if missing_columns:
        raise ValueError(f'Missing required columns: {missing_columns}')

    if df.isnull().values.any():
        missing_values = df.isnull().sum()
        missing_values = missing_values[missing_values > 0]

        raise ValueError(f'Dataset contains missing values: {missing_values}')

    if not all(df[f"V{i}"].dtype == "float64" for i in range(1, 29)):
        raise ValueError("V1-v28 must contain only numeric values")

    if not df["Class"].isin([0, 1]).all():
        raise ValueError("Class must contain only 0 and 1")

    if (sum([val < 0 for val in df["Amount"]])):
        raise ValueError("Amount must not be negative")