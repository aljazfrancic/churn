"""Tests for the decision threshold (D9)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score

from churn.config import TARGET
from churn.evaluation import comparison_folds
from churn.pipeline import build_pipeline
from churn.threshold import THRESHOLDS, metrics_by_threshold, near_optimal, tune_threshold


def _tuned(raw_customers):
    X, y = raw_customers.drop(columns=TARGET), raw_customers[TARGET].eq("Yes").astype(int)
    model = build_pipeline(LogisticRegression(max_iter=1000))
    return model, X, y, tune_threshold(model, X, y)


@pytest.mark.filterwarnings("ignore:Found unknown categories")  # rare levels in small folds
def test_threshold_maximises_the_mean_f2_over_the_comparison_folds(raw_customers):
    model, X, y, tuned = _tuned(raw_customers)

    f2 = []  # recomputed: F2 at every threshold on each of the 15 comparison folds
    for train, val in comparison_folds().split(X, y):
        proba = clone(model).fit(X.iloc[train], y.iloc[train]).predict_proba(X.iloc[val])[:, 1]
        f2.append(
            [fbeta_score(y.iloc[val], proba >= t, beta=2, zero_division=0) for t in THRESHOLDS]
        )
    mean_f2 = np.mean(f2, axis=0)

    assert len(f2) == 15
    assert tuned.cv_results_["scores"] == pytest.approx(mean_f2)
    assert tuned.best_threshold_ == THRESHOLDS[mean_f2.argmax()]
    assert tuned.estimator_.predict_proba(X).shape == (len(X), 2)  # refitted on all customers


def test_near_optimal_range_holds_the_best_threshold(raw_customers):
    *_, tuned = _tuned(raw_customers)
    low, high = near_optimal(tuned)
    assert low <= tuned.best_threshold_ <= high
    assert near_optimal(tuned, tolerance=1) == (THRESHOLDS[0], THRESHOLDS[-1])


def test_metrics_by_threshold_flag_customers_at_or_above_it():
    y = pd.Series([1, 1, 0, 0])
    proba = np.array([0.9, 0.5, 0.5, 0.1])
    curves = metrics_by_threshold(y, proba, thresholds=np.array([0.1, 0.5, 0.9, 0.95]))

    assert curves["share_flagged"].tolist() == [1, 0.75, 0.25, 0]
    assert curves["recall"].tolist() == [1, 1, 0.5, 0]
    assert curves["precision"].iloc[:3].tolist() == pytest.approx([0.5, 2 / 3, 1])
    assert np.isnan(curves["precision"].iloc[3])  # no one flagged
