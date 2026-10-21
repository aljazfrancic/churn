"""Download, load and clean the Telco Customer Churn data."""

import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from churn.config import ADDON_COLUMNS, CSV_NAME, CSV_SHA256, DATA_DIR, DATA_URL, ID_COLUMN, TARGET


def verify_sha256(data: bytes, sha256: str) -> None:
    """Raise if `data` is not the expected file: the dataset (D12) or a saved model (D52)."""
    digest = hashlib.sha256(data).hexdigest()
    if digest != sha256:
        raise ValueError(f"Unexpected file: SHA-256 {digest}, expected {sha256}.")


def download_data(data_dir: Path = DATA_DIR, url: str = DATA_URL, sha256: str = CSV_SHA256) -> Path:
    """Return the path of the verified dataset CSV, downloading it on first use (D12).

    Only the expected CSV is read from the zip, in memory, so the archive can never write files
    elsewhere. Nothing is saved unless the checksum matches; an existing file is re-verified.
    """
    csv_path = data_dir / CSV_NAME
    if csv_path.exists():
        verify_sha256(csv_path.read_bytes(), sha256)
        return csv_path

    with urllib.request.urlopen(url, timeout=60) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    data = archive.read(CSV_NAME)
    verify_sha256(data, sha256)
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path.write_bytes(data)
    return csv_path


def load_raw(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Return the dataset exactly as published."""
    return pd.read_csv(download_data(data_dir))


def column_overview(df: pd.DataFrame) -> pd.DataFrame:
    """One row per column: dtype, number of distinct values, missing values, example values."""
    return pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "unique": df.nunique(),
            "missing": df.isna().sum(),
            "examples": [", ".join(map(str, df[col].unique()[:3])) for col in df.columns],
        }
    )


def quality_checks(raw: pd.DataFrame) -> pd.DataFrame:
    """Integrity counts on the raw data: missing values, IDs, blanks, duplicates (D16, D18)."""
    identical = raw[raw.drop(columns=ID_COLUMN).duplicated()]
    simple_plan = (
        identical["tenure"].eq(1)
        & identical["Contract"].eq("Month-to-month")
        & ~identical[ADDON_COLUMNS].eq("Yes").any(axis=1)
    )
    counts = {
        "missing values": raw.isna().sum().sum(),
        "duplicate customer IDs": raw[ID_COLUMN].duplicated().sum(),
        "blank TotalCharges": (raw["TotalCharges"].astype(str).str.strip() == "").sum(),
        "rows repeating an earlier row apart from the ID": len(identical),
        "… of them tenure 1, month-to-month, no add-ons": simple_plan.sum(),
    }
    return pd.Series(counts, name="count").to_frame()


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fix data types and drop the ID column.

    Returns a new frame and is idempotent, so it can safely be re-applied to cleaned data
    (e.g. API input). `Churn` stays Yes/No for readable plots (D37).
    """
    df = df.drop(columns=ID_COLUMN, errors="ignore")

    # D16: TotalCharges is text because brand-new customers (tenure 0) have a blank value;
    # they have not been billed yet, so 0 is the true amount.
    total = df["TotalCharges"].replace(r"^\s*$", np.nan, regex=True)
    df["TotalCharges"] = pd.to_numeric(total).fillna(0.0)

    # D19: SeniorCitizen is the only 0/1 column; use Yes/No like all other binary columns.
    df["SeniorCitizen"] = df["SeniorCitizen"].replace({0: "No", 1: "Yes"}).astype("str")
    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split cleaned data into features X and the binary target y (1 = churn, D37)."""
    y = (df[TARGET] == "Yes").astype(int)
    return df.drop(columns=TARGET), y
