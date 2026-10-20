"""Model comparison on fresh folds, the choice of the final model and its hold-out evaluation.

Comparison and selection: D26, D28, D41, D42; hold-out test: D30, D47, D48.
"""

from collections.abc import Callable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    RepeatedStratifiedKFold,
    cross_validate,
)

from churn.config import RANDOM_STATE
from churn.models import SCORING as TUNING_SCORING

# D30: the tuning metrics plus the Brier score, now that every model gives probabilities (D40).
SCORING = TUNING_SCORING | {"brier": "neg_brier_score"}
# The same metrics for the hold-out test, as functions of the labels and probabilities.
RANKING_METRICS = {
    "pr_auc": average_precision_score,
    "roc_auc": roc_auc_score,
    "brier": brier_score_loss,
}

# D42: simplest first, by explainability, calibration and serving cost. Models not listed (the
# dummy baseline) are never selected.
SIMPLEST_FIRST = ["Logistic regression", "XGBoost", "Random forest", "SVM (RBF)", "MLP", "Ensemble"]


def comparison_folds(n_splits: int = 5, n_repeats: int = 3) -> RepeatedStratifiedKFold:
    """The folds every model is compared on: new shuffles, not the tuning folds (D26, D41)."""
    return RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=RANDOM_STATE + 1
    )


def cross_validate_models(
    models: dict[str, BaseEstimator],
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    n_repeats: int = 3,
) -> pd.DataFrame:
    """Score every model on the same repeated stratified folds; one row per model and fold.

    The folds are new shuffles, not the tuning folds (D26); sharing them lets the models be
    compared fold by fold (D41).
    """
    folds = comparison_folds(n_splits, n_repeats)
    frames = []
    for name, model in models.items():
        cv = cross_validate(model, X, y, scoring=SCORING, cv=folds, n_jobs=-1)
        frames.append(
            pd.DataFrame(
                {
                    "model": name,
                    "fold": range(n_splits * n_repeats),
                    "pr_auc": cv["test_pr_auc"],
                    "roc_auc": cv["test_roc_auc"],
                    "brier": -cv["test_brier"],  # sklearn negates losses so that higher = better
                    "fit_seconds": cv["fit_time"],
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def comparison_table(
    scores: pd.DataFrame, searches: dict[str, RandomizedSearchCV | GridSearchCV]
) -> pd.DataFrame:
    """Mean scores per model over the folds, best PR-AUC first (D41).

    `tuning_pr_auc` is the score on the tuning folds that chose the setting (none for the untuned
    ensemble); a drop on the fresh folds shows how optimistic it was (D26).
    """
    by_model = scores.groupby("model")
    table = by_model[["pr_auc", "roc_auc", "brier", "fit_seconds"]].mean()
    table.insert(1, "pr_auc_std", by_model["pr_auc"].std())
    tuning = pd.Series({name: search.best_score_ for name, search in searches.items()}, dtype=float)
    table.insert(2, "tuning_pr_auc", tuning)
    return table.sort_values("pr_auc", ascending=False)


def select_model(table: pd.DataFrame, simplest_first: list[str] = SIMPLEST_FIRST) -> str:
    """The simplest model whose mean PR-AUC is within 1 std of the best model's (D28, D42)."""
    eligible = table.loc[[name for name in simplest_first if name in table.index]]
    best = eligible["pr_auc"].idxmax()
    cutoff = eligible.loc[best, "pr_auc"] - eligible.loc[best, "pr_auc_std"]
    return eligible.index[eligible["pr_auc"] >= cutoff][0]


def holdout_probabilities(
    models: dict[str, BaseEstimator],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
) -> pd.DataFrame:
    """Churn probability of every test customer per model, each refitted on all training data."""

    def fit_predict(model: BaseEstimator) -> np.ndarray:
        return clone(model).fit(X_train, y_train).predict_proba(X_test)[:, 1]

    probabilities = Parallel(n_jobs=-1)(delayed(fit_predict)(model) for model in models.values())
    return pd.DataFrame(dict(zip(models, probabilities, strict=True)), index=X_test.index)


def ranking_metrics(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    """Threshold-free metrics of the churn probabilities (D29, D30)."""
    return {name: metric(y_true, proba) for name, metric in RANKING_METRICS.items()}


def decision_metrics(y_true: np.ndarray, flagged: np.ndarray) -> dict[str, float]:
    """Metrics of the offers (flagged customers) from the confusion counts (D30).

    Computed by hand, as the bootstrap calls this thousands of times (D48); precision is NaN
    when no one is flagged.
    """
    churns, flagged = np.asarray(y_true, dtype=bool), np.asarray(flagged, dtype=bool)
    tp, fp = np.sum(churns & flagged), np.sum(~churns & flagged)
    fn, tn = np.sum(churns & ~flagged), np.sum(~churns & ~flagged)

    def f_score(beta: float) -> float:  # (1 + β²)·TP / ((1 + β²)·TP + β²·FN + FP)
        return (1 + beta**2) * tp / ((1 + beta**2) * tp + beta**2 * fn + fp)

    recall = tp / (tp + fn)
    return {
        "precision": tp / (tp + fp) if tp + fp else np.nan,
        "recall": recall,
        "f1": f_score(1),
        "f2": f_score(2),
        "balanced_accuracy": (recall + tn / (tn + fp)) / 2,
        "accuracy": (tp + tn) / len(churns),
        "share_flagged": flagged.mean(),
    }


def _bootstrap(
    metrics: Callable[[np.ndarray, np.ndarray], dict[str, float]],
    y_true: pd.Series,
    values: np.ndarray,
    n_resamples: int = 1000,
) -> pd.DataFrame:
    """`metrics` on resampled test sets (customers drawn with replacement); one row each (D48).

    The seed is fixed, so every call draws the same resamples.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    y_true, values = np.asarray(y_true), np.asarray(values)
    samples = (rng.integers(0, len(y_true), len(y_true)) for _ in range(n_resamples))
    return pd.DataFrame([metrics(y_true[i], values[i]) for i in samples])


def _with_ci(estimates: dict[str, float], resampled: pd.DataFrame) -> pd.DataFrame:
    """Test estimates with the 95 % percentile interval of their bootstrap distribution."""
    bounds = resampled.quantile([0.025, 0.975])
    return pd.DataFrame(
        {"test": pd.Series(estimates), "ci_low": bounds.iloc[0], "ci_high": bounds.iloc[1]}
    )


def holdout_ranking(y_true: pd.Series, proba: np.ndarray, cv_scores: pd.DataFrame) -> pd.DataFrame:
    """Threshold-free test metrics with bootstrap 95 % CIs, next to their CV means (D47, D48).

    `cv_scores` are the model's fold scores from `cross_validate_models`.
    """
    resampled = _bootstrap(ranking_metrics, y_true, proba)
    table = _with_ci(ranking_metrics(y_true, proba), resampled)
    table.insert(0, "cv_mean", cv_scores[list(RANKING_METRICS)].mean())
    return table


def holdout_decisions(
    y_true: pd.Series, proba: np.ndarray, thresholds: dict[str, float]
) -> pd.DataFrame:
    """Test metrics of the offers at each threshold, with bootstrap 95 % CIs (D30, D47, D48).

    Customers with a churn probability ≥ the threshold are flagged, as in scikit-learn.
    """
    tables = {}
    for label, threshold in thresholds.items():
        flagged = np.asarray(proba) >= threshold
        resampled = _bootstrap(decision_metrics, y_true, flagged)
        tables[label] = _with_ci(decision_metrics(y_true, flagged), resampled)
    return pd.concat(tables, axis=1)
