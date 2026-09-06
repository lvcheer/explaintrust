"""Loaders for the real-world tabular datasets used to anchor report thresholds.

Two UCI datasets are supported out of the box (downloaded on first use into
``experiments/data/``, which is gitignored):

* **Adult Income** (UCI id=2) — binary classification (>50K vs <=50K), real
  collinearity (``relationship``/``marital-status``, ``education``/``education-num``),
  mixed categorical/numeric features.
* **Diabetes 130-US Hospitals** (UCI id=296) — binary classification
  (readmitted vs not), demographic columns (race/gender/age) useful for
  cross-subpopulation checks.

Each loader returns raw features and split metadata. ``prepare_split`` splits
first, then fits imputation, scaling and category vocabulary on training rows only.
"""
from __future__ import annotations

import zipfile
import hashlib
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

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


@dataclass
class RawDataset:
    frame: pd.DataFrame
    y: np.ndarray
    numeric: list[str]
    categorical: list[str]
    test_mask: np.ndarray | None = None
    groups: np.ndarray | None = None


def _encode(train: pd.DataFrame, test: pd.DataFrame,
            numeric: list[str], categorical: list[str]):
    """Fit exclusively on train; unseen test categories use an all-zero block."""
    train_num = train[numeric].apply(pd.to_numeric, errors="coerce")
    test_num = test[numeric].apply(pd.to_numeric, errors="coerce")
    # All-missing training columns use a fixed zero fallback, never test values.
    medians = train_num.median().fillna(0.)
    train_num = train_num.fillna(medians)
    mean, std = train_num.mean(), train_num.std(ddof=0)
    # Constant training columns retain unit scale rather than amplifying a
    # held-out difference by 1e12. This scale also depends on training only.
    scale = std.where(std > 1e-12, 1.)
    train_num = ((train_num - mean) / scale).to_numpy(dtype=float)
    test_num = ((test_num.fillna(medians) - mean) / scale).to_numpy(dtype=float)

    train_blocks, test_blocks, names = [train_num], [test_num], list(numeric)
    for column in categorical:
        tr = train[column].fillna("missing").astype(str)
        te = test[column].fillna("missing").astype(str)
        categories = sorted(tr.unique())
        # Same drop-first convention as before, with a training-only vocabulary.
        for category in categories[1:]:
            train_blocks.append((tr == category).to_numpy(dtype=float)[:, None])
            test_blocks.append((te == category).to_numpy(dtype=float)[:, None])
            names.append(f"{column}_{category}")
    return np.hstack(train_blocks), np.hstack(test_blocks), names


def prepare_split(data: RawDataset, seed: int, test_size: float = 0.3):
    """Adult uses its official holdout; Diabetes holds out 30% of patients."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    if (data.test_mask is None) == (data.groups is None):
        raise ValueError("provide exactly one of official test_mask or patient groups")
    if data.test_mask is not None:
        train_idx = np.flatnonzero(~data.test_mask)
        test_idx = np.flatnonzero(data.test_mask)
        protocol = "adult-official-holdout"
        group_context = {}
    else:
        if pd.isna(data.groups).any():
            raise ValueError("patient groups must not contain missing identifiers")
        train_idx, test_idx = next(GroupShuffleSplit(
            n_splits=1, test_size=test_size, random_state=seed,
        ).split(data.frame, data.y, groups=data.groups))
        train_groups, test_groups = set(data.groups[train_idx]), set(data.groups[test_idx])
        protocol = "diabetes-patient-group-holdout"
        group_context = {
            "train_patients": len(train_groups), "test_patients": len(test_groups),
            "patient_overlap": len(train_groups & test_groups),
        }
    if not len(train_idx) or not len(test_idx):
        raise ValueError("both training and test partitions must be nonempty")
    Xtr, Xte, names = _encode(
        data.frame.iloc[train_idx], data.frame.iloc[test_idx], data.numeric, data.categorical,
    )
    context = {
        "protocol": protocol, "seed": seed, "preprocessing_fit": "train_only",
        "train_rows": len(train_idx), "test_rows": len(test_idx),
        "encoded_features": len(names), "unknown_categories": "all_zero",
        "train_indices_sha256": hashlib.sha256(np.asarray(train_idx, dtype="<i8").tobytes()).hexdigest(),
        "test_indices_sha256": hashlib.sha256(np.asarray(test_idx, dtype="<i8").tobytes()).hexdigest(),
        **group_context,
    }
    return Xtr, Xte, data.y[train_idx], data.y[test_idx], names, context


def load_adult() -> RawDataset:
    """Load raw Adult features with the official test-file membership retained."""
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
    test_mask = np.arange(len(df)) >= len(train)
    return RawDataset(df, y, ADULT_NUMERIC, ADULT_CATEGORICAL, test_mask=test_mask)


def load_diabetes() -> RawDataset:
    """Load raw Diabetes features and patient groups (readmitted vs not)."""
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
    groups = df["patient_nbr"].to_numpy(copy=True)
    df = df.drop(columns=DIABETES_DROP)

    y = (df["readmitted"] != "NO").astype(int).to_numpy()
    df = df.drop(columns=["readmitted"])

    # Medication columns: "No" -> 0, anything else (Steady/Up/Down) -> 1.
    for c in DIABETES_MEDS:
        df[c] = df[c].fillna("No").map(lambda v: 0 if v == "No" else 1)
    df["change"] = (df["change"] == "Ch").astype(int)
    df["diabetesMed"] = (df["diabetesMed"] == "Yes").astype(int)

    return RawDataset(df, y, DIABETES_NUMERIC, DIABETES_CATEGORICAL, groups=groups)


if __name__ == "__main__":
    for loader in (load_adult, load_diabetes):
        Xtr, Xte, ytr, yte, names, context = prepare_split(loader(), seed=0)
        print(f"{loader.__name__}: train={Xtr.shape}, test={Xte.shape}, protocol={context}")
        print(f"  first 8 features: {names[:8]}")
