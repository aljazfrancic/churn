"""Tests for the EDA statistics helpers (D9)."""

import numpy as np
import pandas as pd
import pytest

from churn.data import clean
from churn.eda import (
    association_with_churn,
    charge_correlations,
    churn_by_bill_change,
    churn_rate_by,
    cramers_v,
    format_p_value,
    observed_vs_expected,
    quantile_bins,
    service_mix,
)


def test_churn_rate_by_sorts_highest_first():
    df = pd.DataFrame({"Contract": ["A", "A", "B", "B"], "Churn": ["Yes", "No", "No", "No"]})
    rates = churn_rate_by(df, "Contract")
    assert rates.index.tolist() == ["A", "B"]
    assert rates["churn_rate"].tolist() == [0.5, 0.0]
    assert rates["customers"].tolist() == [2, 2]


def test_cramers_v_bounds():
    x = pd.Series(["a", "a", "b", "b"] * 25)
    assert cramers_v(x, x) == pytest.approx(1.0)
    independent = pd.Series(["u", "v"] * 50)  # same mix of u/v within a and b
    assert cramers_v(x, independent) == pytest.approx(0.0)


def test_quantile_bins_cover_all_values():
    values = pd.Series(range(100))
    bins = quantile_bins(values, q=4)
    assert bins.notna().all()
    assert bins.cat.categories.tolist() == ["0–25", "25–50", "50–74", "74–99"]


def test_format_p_value():
    assert format_p_value(0.0) == "< 1e-300"
    assert format_p_value(0.487) == "0.49"
    assert format_p_value(0.0312) == "0.031"
    assert format_p_value(5.86e-258) == "5.9e-258"


def test_service_mix_names_every_combination():
    df = pd.DataFrame(
        {
            "PhoneService": ["Yes", "Yes", "Yes", "No", "No"],
            "InternetService": ["No", "DSL", "Fiber optic", "DSL", "No"],
        }
    )
    assert service_mix(df).tolist() == [
        "Phone only",
        "Phone + DSL",
        "Phone + Fiber optic",
        "DSL only",
        "Neither",
    ]


def test_observed_vs_expected_uses_the_overall_churn_rate():
    df = pd.DataFrame({"Contract": ["A", "A", "A", "B"], "Churn": ["Yes", "Yes", "No", "No"]})
    table = observed_vs_expected(df, "Contract")  # overall churn rate: 2 of 4
    assert table["customers"].tolist() == [3, 1]
    assert table["churned"].tolist() == [2, 0]
    assert table["churned_if_unrelated"].tolist() == [1.5, 0.5]


def test_association_uses_the_plain_chi2_of_the_formula(raw_customers):
    df = clean(raw_customers)
    table = association_with_churn(df)
    observed = pd.crosstab(df["gender"], df["Churn"]).to_numpy()  # a 2 × 2 table
    expected = observed.sum(axis=1, keepdims=True) * observed.sum(axis=0) / observed.sum()
    chi2 = ((observed - expected) ** 2 / expected).sum()  # sum of (O − E)² / E, no Yates (D38)
    assert table.loc["gender", "chi2"] == pytest.approx(chi2)
    assert table.loc["gender", "cramers_v"] == pytest.approx(np.sqrt(chi2 / len(df)))


def test_bill_change_keeps_increases_and_decreases_apart():
    features = pd.DataFrame(
        {
            "tenure": [1, 12, 12, 12, 12],  # the first has no billing history yet
            "MonthlyCharges": [50.0, 80.0, 50.0, 50.0, 51.0],
            "AvgMonthlyCharges": [50.0, 70.0, 60.0, 50.0, 50.0],
        }
    )
    y = pd.Series([1, 1, 1, 0, 1])
    table = churn_by_bill_change(features, y)
    assert table.index.tolist() == ["up > USD 2", "within ± USD 2", "down > USD 2"]
    assert table["customers"].tolist() == [1, 2, 1]
    assert table["churn_rate"].tolist() == [1.0, 0.5, 1.0]


def test_charge_correlations_of_consistent_bills():
    tenure = pd.Series([1, 2, 3, 4])
    monthly = pd.Series([20.0, 50.0, 80.0, 110.0])
    features = pd.DataFrame(
        {
            "tenure": tenure,
            "MonthlyCharges": monthly,
            "TotalCharges": tenure * monthly,
            "AvgMonthlyCharges": monthly,
        }
    )
    assert charge_correlations(features).tolist() == pytest.approx([1.0, 1.0])
