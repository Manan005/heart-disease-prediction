"""Data loading, cleaning and preprocessing for the heart disease model.

Cleaning decisions come from the EDA (notebooks/01_eda.ipynb):
- chol == 0 and trestbps == 0 are missing values in disguise -> NaN
- `id`, `dataset` (hospital) and `num` are not used as features
- `ca` and `thal` are dropped: almost only recorded in Cleveland, so their
  presence/absence mostly encodes the hospital, not the patient
- `slope` is dropped for the same reason (98% of its missing values come from
  Hungary and VA Long Beach)
- missing categorical values are filled with the most frequent category, NOT a
  separate "unknown" category: e.g. 75 of the 90 missing `fbs` values are Swiss
  patients (93% disease rate), so "unknown" would act as a hidden hospital flag
- fbs / exang are stored as "yes" / "no" text so they encode like other categories
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "heart_disease_uci.csv"

NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalch", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "target"

RANDOM_STATE = 42


def load_raw(path=RAW_DATA_PATH):
    """Read the raw CSV exactly as downloaded."""
    return pd.read_csv(path)


def clean(df):
    """Apply the EDA-driven cleaning rules and return features + binary target."""
    df = df.copy()

    # Target: 0 = no disease, 1-4 = disease present -> 0 / 1
    df[TARGET] = (df["num"] > 0).astype(int)

    # Physically impossible zeros are really missing values
    for col in ["chol", "trestbps"]:
        df[col] = df[col].replace(0, np.nan)

    # Booleans with missing values -> "yes" / "no" text (NaN stays NaN)
    for col in ["fbs", "exang"]:
        df[col] = df[col].map({True: "yes", False: "no"})

    return df[FEATURES + [TARGET]]


def split(df, test_size=0.2, random_state=RANDOM_STATE):
    """Stratified train/test split so both sets keep the 55/45 class balance."""
    X = df[FEATURES]
    y = df[TARGET]
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )


def build_preprocessor():
    """Median imputation + scaling for numeric columns,
    most-frequent imputation + one-hot encoding for categorical columns.

    Nothing is fitted here; it learns medians/categories only when it is fitted
    on the training data (inside a model Pipeline), which prevents test-set leakage.
    """
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ])


def load_data():
    """Convenience: raw CSV -> cleaned -> (X_train, X_test, y_train, y_test)."""
    return split(clean(load_raw()))
