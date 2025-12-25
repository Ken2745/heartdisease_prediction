import pandas as pd
from typing import Union

FEATURE_COLUMNS = [
    'age', 'sex', 'trestbps', 'chol', 'fbs', 'thalach',
    'exang', 'oldpeak', 'slope', 'ca',
    'cp_0', 'cp_1', 'cp_2', 'cp_3',
    'restecg_0', 'restecg_1', 'restecg_2',
    'thal_0', 'thal_1', 'thal_2', 'thal_3'
]

COLUMN_RENAME_MAP = {
    'trtbps': 'trestbps',
    'thalachh': 'thalach',
    'exng': 'exang',
    'slp': 'slope',
    'caa': 'ca',
    'thall': 'thal'
}

def preprocess_data(data: Union[pd.DataFrame, dict]):

    # Convert dict input (api) to DataFrame
    if isinstance(data, dict):
        data = pd.DataFrame([data])

    # Rename columns
    data = data.rename(columns=COLUMN_RENAME_MAP)

    # One-hot encode categorical features
    categorical_features = ['cp', 'restecg', 'thal']
    df = pd.get_dummies(data, columns=categorical_features)
    df = df.astype("float32")

    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0

    df = df[FEATURE_COLUMNS]

    return df
