from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


TARGET_COL = "class"
TARGET_MAP = {"e": 0, "p": 1}


@dataclass
class DatasetBundle:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    feature_columns: List[str]
    feature_options: Dict[str, List[str]]


def load_raw_dataset(csv_path: str | Path) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Expected target column '{TARGET_COL}' in dataset")
    return df


def preprocess_dataframe(df: pd.DataFrame, drop_unknown_rows: bool = False) -> pd.DataFrame:
    data = df.copy()

    # The original dataset encodes missing values as '?'.
    if drop_unknown_rows:
        data = data.replace("?", pd.NA).dropna().reset_index(drop=True)
    else:
        data = data.replace("?", "unknown")

    if data[TARGET_COL].isin(TARGET_MAP.keys()).all() is False:
        raise ValueError("Target column contains values outside {'e', 'p'}")

    data[TARGET_COL] = data[TARGET_COL].map(TARGET_MAP)
    return data


def split_dataset(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> DatasetBundle:
    feature_columns = [c for c in df.columns if c != TARGET_COL]
    X = df[feature_columns]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    feature_options = {
        c: sorted(X[c].astype(str).unique().tolist())
        for c in feature_columns
    }

    return DatasetBundle(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_columns=feature_columns,
        feature_options=feature_options,
    )
