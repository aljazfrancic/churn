"""Model comparison on fresh folds and the choice of the final model (D26, D28, D41, D42)."""

import pandas as pd
from sklearn.base import BaseEstimator
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

# D42: simplest first, by explainability, calibration and serving cost. Models not listed (the
# dummy baseline) are never selected.
SIMPLEST_FIRST = ["Logistic regression", "XGBoost", "Random forest", "SVM (RBF)", "MLP", "Ensemble"]


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
    folds = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=RANDOM_STATE + 1
    )
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
