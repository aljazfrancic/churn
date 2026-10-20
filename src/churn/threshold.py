"""Decision threshold: the F2-optimal cut-off, tuned by CV on the training set (D31)."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.model_selection import TunedThresholdClassifierCV

from churn.evaluation import comparison_folds, decision_metrics

# D31: recall weighted 2× precision; round thresholds, scored exactly on every fold. Above the
# highest probability no one is flagged: F2 is then 0, without an undefined-precision warning.
F2 = make_scorer(fbeta_score, beta=2, zero_division=0)
THRESHOLDS = np.round(np.arange(0.01, 1, 0.01), 2)


def tune_threshold(
    model: BaseEstimator, X: pd.DataFrame, y: pd.Series
) -> TunedThresholdClassifierCV:
    """`model` refitted on all of X, with the threshold of the best mean F2 on the comparison folds.

    Each fold's model scores its validation customers at every threshold; the threshold with the
    highest F2 averaged over the 15 folds wins (D31, D41).
    """
    tuner = TunedThresholdClassifierCV(
        model,
        scoring=F2,
        thresholds=THRESHOLDS,
        cv=comparison_folds(),
        n_jobs=-1,
        store_cv_results=True,
    )
    return tuner.fit(X, y)


def near_optimal(tuned: TunedThresholdClassifierCV, tolerance: float = 0.01) -> tuple[float, float]:
    """Lowest and highest threshold whose mean CV F2 is within `tolerance` of the best."""
    thresholds, scores = tuned.cv_results_["thresholds"], tuned.cv_results_["scores"]
    close = thresholds[scores >= tuned.best_score_ - tolerance]
    return close.min(), close.max()


def metrics_by_threshold(
    y_true: pd.Series, proba: np.ndarray, thresholds: np.ndarray = THRESHOLDS
) -> pd.DataFrame:
    """Decision metrics at every threshold; one row per threshold (D33)."""
    proba = np.asarray(proba)
    rows = {threshold: decision_metrics(y_true, proba >= threshold) for threshold in thresholds}
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("threshold")
