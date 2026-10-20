"""Feature engineering as the stateless first step of the model pipeline (D17, D20, D21)."""

from typing import Self

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from churn.config import ADDON_COLUMNS
from churn.data import clean

# D20: tenure groups in months; the last one is open-ended, as live customers can exceed 72.
TENURE_EDGES = [-1, 6, 12, 24, 48, np.inf]
TENURE_LABELS = ["0–6", "7–12", "13–24", "25–48", "49+"]


class FeatureEngineer(TransformerMixin, BaseEstimator):
    """Type fixes, level collapse and new features; learns nothing from the data (D21).

    Being stateless, it cannot leak information between CV folds. The switches turn the level
    collapse and the new features off for the preprocessing ablation (D32).
    """

    def __init__(self, collapse_levels: bool = True, new_features: bool = True):
        self.collapse_levels = collapse_levels
        self.new_features = new_features

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """Nothing to learn."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return a new frame with the collapsed levels and the new features appended."""
        # D21, D37: clean() is idempotent, so raw records (e.g. from the CRM) work as well.
        X = clean(X)
        if self.collapse_levels:
            # D17: "No internet/phone service" repeats InternetService/PhoneService = "No".
            X[ADDON_COLUMNS] = X[ADDON_COLUMNS].replace("No internet service", "No")
            X["MultipleLines"] = X["MultipleLines"].replace("No phone service", "No")
        if self.new_features:
            # D20: new customers (tenure 0) have no billing history; use their current bill.
            average = (X["TotalCharges"] / X["tenure"]).where(X["tenure"] > 0, X["MonthlyCharges"])
            X["AvgMonthlyCharges"] = average
            X["AbsChargeDelta"] = (X["MonthlyCharges"] - average).abs()  # bill vs its history
            X["NumAddons"] = X[ADDON_COLUMNS].eq("Yes").sum(axis=1)
            X["TenureGroup"] = pd.cut(X["tenure"], TENURE_EDGES, labels=TENURE_LABELS)
        return X

    def __sklearn_is_fitted__(self) -> bool:
        return True  # stateless, like sklearn's FunctionTransformer
