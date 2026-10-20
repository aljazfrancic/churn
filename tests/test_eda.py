"""Tests for the EDA statistics helpers (D9)."""

import pandas as pd
import pytest

from churn.eda import churn_rate_by, cramers_v, format_p_value, quantile_bins


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
    assert format_p_value(0.0312) == "3e-02"
