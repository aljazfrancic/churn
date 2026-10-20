"""Tests for the model comparison and the selection rule (D9)."""

import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold

from churn.config import RANDOM_STATE, TARGET
from churn.evaluation import SIMPLEST_FIRST, cross_validate_models, select_model
from churn.models import candidates
from churn.pipeline import build_pipeline


def _table(pr_auc: dict[str, tuple[float, float]]) -> pd.DataFrame:
    """Comparison table with the given PR-AUC (mean, std) per model."""
    return pd.DataFrame(pr_auc, index=["pr_auc", "pr_auc_std"]).T


def test_simpler_model_within_one_std_is_preferred():
    table = _table(
        {"MLP": (0.71, 0.03), "XGBoost": (0.70, 0.02), "Logistic regression": (0.69, 0.02)}
    )
    assert select_model(table) == "Logistic regression"  # 0.69 ≥ 0.71 − 0.03


def test_clearly_better_model_is_kept():
    table = _table(
        {"Dummy (prior)": (0.27, 0.0), "XGBoost": (0.70, 0.02), "Logistic regression": (0.60, 0.02)}
    )
    assert select_model(table) == "XGBoost"


def test_complexity_order_covers_every_candidate():
    assert set(SIMPLEST_FIRST) == set(candidates()) - {"Dummy (prior)"} | {"Ensemble"}


def test_every_model_is_scored_on_the_same_folds(raw_customers):
    X, y = raw_customers.drop(columns=TARGET), raw_customers[TARGET].eq("Yes").astype(int)
    models = {
        "Dummy": build_pipeline(DummyClassifier()),
        "LR": build_pipeline(LogisticRegression(max_iter=1000)),
    }
    scores = cross_validate_models(models, X, y, n_splits=3, n_repeats=2)

    assert scores.groupby("model").size().to_dict() == {"Dummy": 6, "LR": 6}
    assert (scores["brier"] > 0).all()  # sklearn's negated loss is flipped back
    # A constant prediction ranks no one first: its PR-AUC is the churn rate of the fold.
    folds = RepeatedStratifiedKFold(n_splits=3, n_repeats=2, random_state=RANDOM_STATE + 1)
    churn_rates = [y.iloc[test].mean() for _, test in folds.split(X, y)]
    dummy = scores[scores["model"] == "Dummy"]
    assert dummy["pr_auc"].tolist() == pytest.approx(churn_rates)
    assert (dummy["roc_auc"] == 0.5).all()
