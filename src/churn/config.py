"""Project-wide constants: paths, data source, random seed and column groups (D36)."""

from pathlib import Path

# Repository root (the package is installed in editable mode by `uv sync`).
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"  # gitignored, filled by churn.data.download_data (D12)

# D12: Kaggle's public download endpoint works without login; the checksum pins the exact file.
DATA_URL = "https://www.kaggle.com/api/v1/datasets/download/blastchar/telco-customer-churn"
CSV_NAME = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
CSV_SHA256 = "88be4b93fbe0cc83421af1c503794c97c342eca914c1576db7c276e61d61358a"

RANDOM_STATE = 42  # D13

ID_COLUMN = "customerID"
TARGET = "Churn"
NUMERIC_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges"]
ADDON_COLUMNS = [  # internet add-ons
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]
CATEGORICAL_COLUMNS = [
    # demographics
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    # services
    "PhoneService",
    "MultipleLines",
    "InternetService",
    *ADDON_COLUMNS,
    # account
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

# D20: engineered features, added by churn.features.FeatureEngineer
NEW_NUMERIC_COLUMNS = ["AvgMonthlyCharges", "AbsChargeDelta", "NumAddons"]
NEW_CATEGORICAL_COLUMNS = ["TenureGroup"]
