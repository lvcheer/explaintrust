"""Loaders for the real-world tabular datasets used to anchor report thresholds.

Two UCI datasets are supported out of the box (downloaded on first use into
``experiments/data/``, which is gitignored):

* **Adult Income** (UCI id=2) — binary classification (>50K vs <=50K), real
  collinearity (``relationship``/``marital-status``, ``education``/``education-num``),
  mixed categorical/numeric features.
* **Diabetes 130-US Hospitals** (UCI id=296) — binary classification
  (readmitted vs not), demographic columns (race/gender/age) useful for
  cross-subpopulation checks.

Each loader returns ``(X, y, feature_names)`` with categorical columns one-hot
encoded and missing values imputed, ready to feed the trust-metric battery.
"""
from __future__ import annotations

import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
DATA_DIR = HERE / "data"

# --------------------------------------------------------------------------- #
# Adult
# --------------------------------------------------------------------------- #
ADULT_COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
    "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
    "hours-per-week", "native-country", "income",
]
ADULT_NUMERIC = ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
ADULT_CATEGORICAL = [
    "workclass", "marital-status", "occupation", "relationship", "race", "sex",
    "native-country",
]
ADULT_DROP = ["fnlwgt", "education"]  # fnlwgt = sampling weight; education is redundant with education-num

# --------------------------------------------------------------------------- #
# Diabetes 130-US
# --------------------------------------------------------------------------- #
DIABETES_MEDS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
    "examide", "citoglipton", "insulin", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone", "metformin-rosiglitazone",
    "metformin-pioglitazone",
]
DIABETES_NUMERIC = [
    "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
    "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses",
] + DIABETES_MEDS + ["change", "diabetesMed"]
DIABETES_CATEGORICAL = [
    "race", "gender", "age", "admission_type_id", "max_glu_serum", "A1Cresult",
]
# IDs, near-empty columns, ICD codes, and outcome-adjacent administrative codes.
DIABETES_DROP = [
    "encounter_id", "patient_nbr", "weight", "payer_code", "medical_specialty",
    "diag_1", "diag_2", "diag_3", "discharge_disposition_id", "admission_source_id",
]


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    print(f"downloading {url} -> {dest}", flush=True)
    urlretrieve(url, dest)


def _encode(df: pd.DataFrame, numeric: list[str], categorical: list[str]):
    df = df.copy()
    for c in categorical:
        df[c] = df[c].fillna("missing").astype(str)
    for c in numeric:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(df[c].median())

    X_num = df[numeric].to_numpy(dtype=float)
    # z-score numeric columns so scale-sensitive metrics (infidelity, LIME
    # slopes, distribution segmentation) are comparable across datasets with
    # very different units (e.g. capital-gain ranges 0..99999).
    mean = X_num.mean(axis=0)
    std = X_num.std(axis=0)
    X_num = (X_num - mean) / (std + 1e-12)

    dummies = pd.get_dummies(df[categorical], drop_first=True)
    X_cat = dummies.to_numpy(dtype=float)
    X = np.hstack([X_num, X_cat])
    names = list(numeric) + list(dummies.columns)
    return X, names


def load_adult() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load Adult Income as a binary classification task (>50K vs <=50K)."""
    zip_path = DATA_DIR / "adult.zip"
    _download("https://archive.ics.uci.edu/static/public/2/adult.zip", zip_path)
    extracted = DATA_DIR / "adult"
    if not extracted.exists():
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(DATA_DIR)

    train = pd.read_csv(
        extracted / "adult.data", header=None, names=ADULT_COLUMNS,
        skipinitialspace=True, na_values=["?"],
    )
    test = pd.read_csv(
        extracted / "adult.test", header=None, names=ADULT_COLUMNS,
        skipinitialspace=True, na_values=["?"], skiprows=1,
    )
    test["income"] = test["income"].str.rstrip(".")  # "<=50K." -> "<=50K"
    df = pd.concat([train, test], ignore_index=True)

    y = (df["income"] == ">50K").astype(int).to_numpy()
    df = df.drop(columns=ADULT_DROP + ["income"])
    X, names = _encode(df, ADULT_NUMERIC, ADULT_CATEGORICAL)
    return X, y, names


def load_diabetes() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load Diabetes 130-US as a binary classification task (readmitted vs not)."""
    zip_path = DATA_DIR / "diabetes130.zip"
    _download(
        "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip",
        zip_path,
    )
    extracted = DATA_DIR / "diabetes130"
    if not extracted.exists():
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(DATA_DIR)

    df = pd.read_csv(extracted / "diabetic_data.csv", na_values=["?"], low_memory=False)
    df = df.drop(columns=DIABETES_DROP)

    y = (df["readmitted"] != "NO").astype(int).to_numpy()
    df = df.drop(columns=["readmitted"])

    # Medication columns: "No" -> 0, anything else (Steady/Up/Down) -> 1.
    for c in DIABETES_MEDS:
        df[c] = df[c].fillna("No").map(lambda v: 0 if v == "No" else 1)
    df["change"] = (df["change"] == "Ch").astype(int)
    df["diabetesMed"] = (df["diabetesMed"] == "Yes").astype(int)

    X, names = _encode(df, DIABETES_NUMERIC, DIABETES_CATEGORICAL)
    return X, y, names


if __name__ == "__main__":
    for loader in (load_adult, load_diabetes):
        X, y, names = loader()
        print(f"{loader.__name__}: X={X.shape}, y={y.shape}, positive rate={y.mean():.3f}")
        print(f"  first 8 features: {names[:8]}")
