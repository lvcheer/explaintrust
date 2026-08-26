"""Tabular-data helpers for the Streamlit app's "upload your own data" mode.

These functions are deliberately free of any Streamlit dependency so the
data/model logic can be unit-tested with plain Python and reused elsewhere.
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression

CLASSIFICATION_MODELS = ["Random Forest", "Gradient Boosting", "Logistic Regression"]
REGRESSION_MODELS = ["Random Forest", "Gradient Boosting", "Linear Regression"]


def read_csv_bytes(data: bytes) -> pd.DataFrame:
    """Read a CSV from raw uploaded bytes into a DataFrame."""
    return pd.read_csv(io.BytesIO(data))


def infer_task(y: pd.Series) -> str:
    """Guess ``"classification"`` vs ``"regression"`` from the target column.

    Discrete targets (a handful of unique values) are treated as
    classification; high-cardinality continuous targets as regression. This is
    only a heuristic — the UI lets the user override it.
    """
    y = y.dropna()
    n_unique = int(y.nunique())
    if n_unique <= 2:
        return "classification"
    if pd.api.types.is_numeric_dtype(y):
        # A small number of distinct integer values smells like class labels.
        if n_unique <= 20 and np.allclose(y, np.round(y)):
            return "classification"
        return "regression"
    # Non-numeric target: treat its distinct values as class labels.
    return "classification"


def prepare_tabular(df: pd.DataFrame, target_col: str, task: str):
    """Turn a raw DataFrame into ``(X, y, feature_names)``.

    * ``target_col`` is removed from the feature matrix.
    * Non-numeric feature columns are dropped (v0.1 supports numeric features).
    * Rows containing missing values are dropped.
    * For classification, the target is encoded to integer class labels;
      for regression it is coerced to float.

    Raises
    ------
    ValueError
        If ``target_col`` is missing or no numeric feature column remains.
    """
    if target_col not in df.columns:
        raise ValueError(f"target column {target_col!r} is not in the data")

    y = df[target_col].copy()
    X = df.drop(columns=[target_col]).copy()
    X = X.select_dtypes(include=[np.number])
    if X.shape[1] == 0:
        raise ValueError("no numeric feature columns found (need at least 3)")

    mask = X.notna().all(axis=1) & y.notna()
    X = X.loc[mask].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)

    if task == "classification":
        y = y.astype("category").cat.codes.to_numpy().astype(int)
    else:
        y = y.astype(float).to_numpy()

    return X.to_numpy(dtype=float), y, list(X.columns)


def make_model(task: str, model_name: str, n_estimators: int = 100, max_depth: int = 4, seed: int = 0):
    """Build an *untrained* sklearn model matching ``task`` and ``model_name``."""
    if task == "classification":
        if model_name == "Random Forest":
            return RandomForestClassifier(
                n_estimators=n_estimators, max_depth=max_depth, random_state=seed
            )
        if model_name == "Gradient Boosting":
            return GradientBoostingClassifier(
                n_estimators=n_estimators, max_depth=max_depth, random_state=seed
            )
        return LogisticRegression(max_iter=1000, random_state=seed)

    # regression
    if model_name == "Random Forest":
        return RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth, random_state=seed
        )
    if model_name == "Gradient Boosting":
        return GradientBoostingRegressor(
            n_estimators=n_estimators, max_depth=max_depth, random_state=seed
        )
    return LinearRegression()


def segment_feature(X: np.ndarray) -> int:
    """Index of the numeric feature with the largest spread.

    Used to pick a sensible axis for the distribution-verification segments
    when the user's data has no known "interesting" column.
    """
    std = np.std(np.asarray(X, dtype=float), axis=0)
    return int(np.argmax(std))
