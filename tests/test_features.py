"""Tests for the feature engineering step (D9)."""

import numpy as np
import pandas as pd

from churn.config import ADDON_COLUMNS, NEW_NUMERIC_COLUMNS
from churn.data import clean
from churn.features import FeatureEngineer


def test_new_customers_get_finite_features(raw_customers):
    features = FeatureEngineer().fit_transform(raw_customers)
    new = features[raw_customers["tenure"] == 0]
    assert len(new) == 3
    assert (new["AvgMonthlyCharges"] == new["MonthlyCharges"]).all()
    assert (new["AbsChargeDelta"] == 0).all()
    assert np.isfinite(features[NEW_NUMERIC_COLUMNS].to_numpy(dtype=float)).all()


def test_average_and_delta_from_billing_history(raw_customers):
    customer = raw_customers.iloc[[5]].assign(tenure=10, MonthlyCharges=80.0, TotalCharges="700")
    features = FeatureEngineer().fit_transform(customer)
    assert features["AvgMonthlyCharges"].item() == 70.0
    assert features["AbsChargeDelta"].item() == 10.0


def test_tenure_group_edges(raw_customers):
    tenure = [0, 6, 7, 12, 13, 24, 25, 48, 49, 72, 90]  # 90: open-ended last group
    customers = raw_customers.iloc[: len(tenure)].assign(tenure=tenure)
    groups = FeatureEngineer().fit_transform(customers)["TenureGroup"].astype(str).tolist()
    assert groups == ["0–6"] * 2 + ["7–12"] * 2 + ["13–24"] * 2 + ["25–48"] * 2 + ["49+"] * 3


def test_level_collapse_and_addon_count(raw_customers):
    addons = dict.fromkeys(ADDON_COLUMNS, "No") | {"OnlineSecurity": "Yes", "TechSupport": "Yes"}
    customer = raw_customers.iloc[[0]].assign(InternetService="DSL", **addons)
    assert FeatureEngineer().fit_transform(customer)["NumAddons"].item() == 2

    features = FeatureEngineer().fit_transform(raw_customers)
    for col in [*ADDON_COLUMNS, "MultipleLines"]:
        assert set(features[col]) == {"Yes", "No"}
    assert (features.loc[raw_customers["InternetService"] == "No", "NumAddons"] == 0).all()


def test_switches_off_only_clean(raw_customers):
    features = FeatureEngineer(collapse_levels=False, new_features=False).fit_transform(
        raw_customers
    )
    pd.testing.assert_frame_equal(features, clean(raw_customers))


def test_transform_leaves_input_untouched(raw_customers):
    original = raw_customers.copy()
    FeatureEngineer().fit_transform(raw_customers)
    pd.testing.assert_frame_equal(raw_customers, original)
