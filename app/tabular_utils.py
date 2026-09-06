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
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score

CLASSIFICATION_MODELS = ["Random Forest", "MLP", "XGBoost", "Gradient Boosting", "Logistic Regression"]
REGRESSION_MODELS = ["Random Forest", "MLP", "XGBoost", "Gradient Boosting", "Linear Regression"]


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
    """Turn a raw DataFrame into ``(X, y, feature_names, metadata)``.

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
    raw_X = df.drop(columns=[target_col]).copy()
    X = raw_X.select_dtypes(include=[np.number])
    if X.shape[1] == 0:
        raise ValueError("no numeric feature columns found")

    metadata = {
        "input_rows": len(df),
        "input_feature_columns": raw_X.shape[1],
        "used_numeric_features": list(X.columns),
        "dropped_non_numeric_features": [c for c in raw_X.columns if c not in X.columns],
    }

    mask = X.notna().all(axis=1) & y.notna()
    X = X.loc[mask].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)
    metadata["usable_rows"] = len(X)
    metadata["dropped_rows_missing"] = int(len(df) - len(X))

    if task == "classification":
        categorical = y.astype("category")
        metadata["class_mapping"] = {
            str(label): int(code) for code, label in enumerate(categorical.cat.categories)
        }
        y = categorical.cat.codes.to_numpy().astype(int)
    else:
        y = y.astype(float).to_numpy()
        metadata["class_mapping"] = None

    return X.to_numpy(dtype=float), y, list(X.columns), metadata


def make_model(task: str, model_name: str, n_estimators: int = 100, max_depth: int = 4, seed: int = 0, *, learning_rate: float = 0.1, max_iter: int = 1000,
               regularization: float = 1.0, hidden_layers=(64, 32), alpha: float = 0.0001,
               fit_intercept: bool = True):
    """Build an *untrained* sklearn model matching ``task`` and ``model_name``."""
    choices = CLASSIFICATION_MODELS if task == "classification" else REGRESSION_MODELS
    if task not in ("classification", "regression") or model_name not in choices:
        raise ValueError(f"unsupported task/model: {task}/{model_name}")
    if model_name == "MLP":
        cls = MLPClassifier if task == "classification" else MLPRegressor
        return make_pipeline(StandardScaler(), cls(hidden_layer_sizes=hidden_layers,
            learning_rate_init=learning_rate, max_iter=max_iter, alpha=alpha,
            random_state=seed, early_stopping=False))
    if model_name == "XGBoost":
        from xgboost import XGBClassifier, XGBRegressor
        cls = XGBClassifier if task == "classification" else XGBRegressor
        return cls(n_estimators=n_estimators, max_depth=max_depth, random_state=seed,
                   n_jobs=1, tree_method="hist", learning_rate=learning_rate)
    if task == "classification":
        if model_name == "Random Forest":
            return RandomForestClassifier(
                n_estimators=n_estimators, max_depth=max_depth, random_state=seed
            )
        if model_name == "Gradient Boosting":
            return GradientBoostingClassifier(
                learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth, random_state=seed
            )
        return LogisticRegression(C=regularization, max_iter=max_iter, random_state=seed, fit_intercept=fit_intercept)

    # regression
    if model_name == "Random Forest":
        return RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth, random_state=seed
        )
    if model_name == "Gradient Boosting":
        return GradientBoostingRegressor(
            learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth, random_state=seed
        )
    return LinearRegression(fit_intercept=fit_intercept)


def segment_feature(X: np.ndarray) -> int:
    """Index of the numeric feature with the largest spread.

    Used to pick a default axis for exploratory subgroup segments
    when the user's data has no known "interesting" column.
    """
    std = np.std(np.asarray(X, dtype=float), axis=0)
    return int(np.argmax(std))


def evaluate_model(model, X, y, task):
    """Held-out metrics; AUC uses scores, never thresholded labels."""
    prediction = model.predict(X)
    if task == "regression":
        return {"R²": float(r2_score(y, prediction)),
                "MAE": float(mean_absolute_error(y, prediction)),
                "RMSE": float(np.sqrt(mean_squared_error(y, prediction)))}
    labels = np.asarray(model.classes_)
    positive = labels[1]
    probabilities = np.asarray(model.predict_proba(X))[:, 1]
    truth = np.asarray(y) == positive
    both = len(np.unique(truth)) == 2
    fpr, tpr, _ = roc_curve(truth, probabilities) if both else ([], [], [])
    return {"Accuracy": float(accuracy_score(y, prediction)),
            "Balanced accuracy": float(balanced_accuracy_score(y, prediction)),
            "Precision": float(precision_score(y, prediction, pos_label=positive, zero_division=0)),
            "Recall": float(recall_score(y, prediction, pos_label=positive, zero_division=0)),
            "F1": float(f1_score(y, prediction, pos_label=positive, zero_division=0)),
            "ROC AUC": float(roc_auc_score(truth, probabilities)) if both else None,
            "confusion_matrix": confusion_matrix(y, prediction, labels=labels).tolist(),
            "classes": labels.tolist(), "fpr": np.asarray(fpr).tolist(), "tpr": np.asarray(tpr).tolist(),
            "positive_class": positive.item() if hasattr(positive, "item") else positive}
