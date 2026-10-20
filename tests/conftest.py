"""Shared test data: synthetic customers in the published (raw) format."""

import numpy as np
import pandas as pd
import pytest

from churn.config import ADDON_COLUMNS

LEVELS = {
    "gender": ["Female", "Male"],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "PhoneService": ["Yes", "No"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
}


@pytest.fixture
def raw_customers() -> pd.DataFrame:
    """60 random but consistent customers, incl. brand-new ones with a blank TotalCharges."""
    rng = np.random.default_rng(0)
    n = 60
    df = pd.DataFrame({col: rng.choice(levels, n) for col, levels in LEVELS.items()})
    df.insert(0, "customerID", [f"{i:04d}-TEST" for i in range(n)])
    df["SeniorCitizen"] = rng.integers(0, 2, n)
    df["MultipleLines"] = np.where(
        df["PhoneService"] == "No", "No phone service", rng.choice(["Yes", "No"], n)
    )
    for col in ADDON_COLUMNS:
        df[col] = np.where(
            df["InternetService"] == "No", "No internet service", rng.choice(["Yes", "No"], n)
        )
    df["tenure"] = rng.integers(1, 73, n)
    df.loc[:6, "tenure"] = [0, 0, 0, 9, 18, 36, 60]  # 3 brand-new customers; every tenure group
    df["MonthlyCharges"] = rng.uniform(18, 119, n).round(2)
    total = (df["tenure"] * df["MonthlyCharges"] * rng.uniform(0.9, 1.1, n)).round(2)
    df["TotalCharges"] = total.astype(str).where(df["tenure"] > 0, " ")
    df["Churn"] = rng.choice(["Yes", "No"], n)
    return df
