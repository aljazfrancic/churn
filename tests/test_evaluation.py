"""Tests for the model comparison and the selection rule (D9)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.utils.validation import check_is_fitted

from churn.config import RANDOM_STATE, TARGET
from churn.evaluation import (
    SIMPLEST_FIRST,
    cross_validate_models,
    decision_metrics,
    holdout_decisions,
    holdout_probabilities,
    holdout_ranking,
    select_model,
)
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


def test_decision_metrics_match_sklearn():
    rng = np.random.default_rng(0)
    y, flagged = rng.integers(0, 2, 200), rng.integers(0, 2, 200)
    assert decision_metrics(y, flagged) == pytest.approx(
        {
            "precision": precision_score(y, flagged),
            "recall": recall_score(y, flagged),
            "f1": f1_score(y, flagged),
            "f2": fbeta_score(y, flagged, beta=2),
            "balanced_accuracy": balanced_accuracy_score(y, flagged),
            "accuracy": accuracy_score(y, flagged),
            "share_flagged": flagged.mean(),
        }
    )


def test_no_offers_leave_precision_undefined():
    metrics = decision_metrics([1, 0, 1], [0, 0, 0])
    assert np.isnan(metrics["precision"])
    assert metrics["recall"] == metrics["f2"] == metrics["share_flagged"] == 0


def test_bootstrap_intervals_bracket_the_test_estimate():
    rng = np.random.default_rng(0)
    y = pd.Series(rng.integers(0, 2, 300))
    proba = np.clip(rng.normal(0.35 + 0.3 * y, 0.15), 0, 1)  # informative but overlapping
    cv_scores = pd.DataFrame({"pr_auc": [0.7, 0.8], "roc_auc": [0.8, 0.9], "brier": [0.2, 0.1]})

    ranking = holdout_ranking(y, proba, cv_scores)
    decisions = holdout_decisions(y, proba, {"low": 0.4, "high": 0.6})

    assert ranking.index.tolist() == ["pr_auc", "roc_auc", "brier"]
    assert ranking["cv_mean"].tolist() == pytest.approx([0.75, 0.85, 0.15])
    assert decisions.columns.unique(level=0).tolist() == ["low", "high"]
    for table in [ranking, decisions["low"], decisions["high"]]:
        assert (table["ci_low"] < table["test"]).all()
        assert (table["test"] < table["ci_high"]).all()
    assert holdout_ranking(y, proba, cv_scores).equals(ranking)  # fixed seed: reproducible


def test_holdout_probabilities_come_from_refitted_copies(raw_customers):
    X, y = raw_customers.drop(columns=TARGET), raw_customers[TARGET].eq("Yes").astype(int)
    models = {
        "Dummy": build_pipeline(DummyClassifier()),
        "LR": build_pipeline(LogisticRegression(max_iter=1000)),
    }
    proba = holdout_probabilities(models, X.iloc[:40], y.iloc[:40], X.iloc[40:])

    assert proba.columns.tolist() == ["Dummy", "LR"]
    assert proba.index.equals(X.index[40:])
    assert proba["Dummy"].eq(y.iloc[:40].mean()).all()  # the training churn rate for everyone
    with pytest.raises(NotFittedError):
        check_is_fitted(models["LR"])  # the given models stay unfitted
