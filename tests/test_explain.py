"""Tests for the model explanations (D9)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from churn.config import TARGET
from churn.explain import explain, input_features, odds_ratios, typical_flagged_churner
from churn.features import TENURE_LABELS
from churn.pipeline import build_pipeline


def _fitted(raw_customers):
    X, y = raw_customers.drop(columns=TARGET), raw_customers[TARGET].eq("Yes").astype(int)
    return build_pipeline(LogisticRegression(max_iter=1000)).fit(X, y), X, y


def test_every_model_input_maps_to_its_original_feature(raw_customers):
    pipeline, X, _ = _fitted(raw_customers)
    features = input_features(pipeline)

    assert features.index.tolist() == pipeline[:-1].transform(X).columns.tolist()
    assert features["feature"].nunique() == 23  # 19 raw features + 4 new ones
    contract = features.loc["Contract_Two year"]
    assert (contract["feature"], contract["level"]) == ("Contract", "Two year")
    assert pd.isna(contract["reference"])  # every contract type is an input of its own
    phone = features.loc["PhoneService_Yes"]
    assert (phone["feature"], phone["level"], phone["reference"]) == ("PhoneService", "Yes", "No")
    assert pd.isna(features.loc["tenure", "level"])


def test_shap_values_add_up_to_the_models_log_odds(raw_customers):
    pipeline, X, _ = _fitted(raw_customers)
    shap_values = explain(pipeline, X, X.iloc[:10])

    assert shap_values.values.index.equals(X.index[:10])
    total = shap_values.base_value + shap_values.values.sum(axis=1)
    assert total.to_numpy() == pytest.approx(pipeline.decision_function(X.iloc[:10]))
    # The base value is the log-odds of a customer with every input at its background mean.
    average = pipeline[:-1].transform(X).mean().to_frame().T
    assert shap_values.base_value == pytest.approx(pipeline[-1].decision_function(average)[0])
    by_feature = shap_values.by_feature()
    assert by_feature.shape == (10, 23)
    assert by_feature.sum(axis=1).to_numpy() == pytest.approx(total - shap_values.base_value)


def test_typical_flagged_churner_has_the_median_risk_among_them():
    y = pd.Series([1, 1, 1, 0, 1], index=list("abcde"))
    proba = pd.Series([0.9, 0.6, 0.4, 0.95, 0.1], index=list("abcde"))
    # Flagged churners: a, b, c (d stays, e is below the threshold).
    assert typical_flagged_churner(y, proba, threshold=0.3) == "b"


def test_odds_ratios_compare_levels_with_the_most_common_one(raw_customers):
    pipeline, X, y = _fitted(raw_customers)
    ratios = odds_ratios(pipeline, X, y, n_resamples=50)
    inputs = pipeline[:-1].transform(X)
    coef = pd.Series(pipeline[-1].coef_[0], index=inputs.columns)

    assert ratios.loc[("tenure", "+1 SD"), "odds_ratio"] == pytest.approx(np.exp(coef["tenure"]))
    phone = ratios.loc[("PhoneService", "Yes vs No"), "odds_ratio"]
    assert phone == pytest.approx(np.exp(coef["PhoneService_Yes"]))
    contracts = [f"Contract_{level}" for level in ["Month-to-month", "One year", "Two year"]]
    reference = inputs[contracts].mean().idxmax()
    for column in set(contracts) - {reference}:
        comparison = f"{column.removeprefix('Contract_')} vs {reference.removeprefix('Contract_')}"
        expected = np.exp(coef[column] - coef[reference])
        assert ratios.loc[("Contract", comparison), "odds_ratio"] == pytest.approx(expected)
    # Tenure groups keep their natural order, not the alphabetical one.
    groups = [comparison.split(" vs ")[0] for comparison in ratios.loc["TenureGroup"].index]
    assert groups == [group for group in TENURE_LABELS if group in groups]
    assert (ratios["ci_low"] < ratios["ci_high"]).all()
    assert ratios.equals(odds_ratios(pipeline, X, y, n_resamples=50))  # fixed seed: reproducible
