"""Tests for the candidate models, the tuning helpers and the ensemble (D9)."""

import pytest
from sklearn.base import clone
from sklearn.model_selection import ParameterSampler

from churn.config import TARGET
from churn.models import best_models, candidates, tune_models, tuning_results


def test_search_spaces_match_their_models():
    for model, space in candidates().values():
        for params in ParameterSampler(space, n_iter=1, random_state=0):
            clone(model).set_params(**params)  # raises on a misspelt parameter name


def test_best_models_give_probabilities(raw_customers):
    X, y = raw_customers.drop(columns=TARGET), raw_customers[TARGET].eq("Yes").astype(int)
    searches = tune_models(X, y, n_iter=1, cv=3)

    results = tuning_results(searches)
    assert results["pr_auc"].is_monotonic_decreasing
    assert results.loc["Dummy (prior)", "pr_auc"] == pytest.approx(y.mean())  # = churn rate

    models = best_models(searches)
    members = [name for name, _ in models["Ensemble"].estimators]
    assert len(members) == 3 and "Dummy (prior)" not in members
    for name, model in models.items():
        proba = model.fit(X, y).predict_proba(X)[:, 1]
        assert ((proba >= 0) & (proba <= 1)).all(), name
