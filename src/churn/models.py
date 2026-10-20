"""Candidate models, hyperparameter tuning and the soft-voting ensemble (D24, D27)."""

import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.base import BaseEstimator, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from churn.config import RANDOM_STATE
from churn.pipeline import build_pipeline

# D29, D30: threshold-free ranking metrics; PR-AUC picks the best setting.
SCORING = {"pr_auc": "average_precision", "roc_auc": "roc_auc"}


def candidates() -> dict[str, tuple[BaseEstimator, dict]]:
    """Untuned models with their search spaces (D24, D27).

    Models run single-threaded; the searches parallelise over settings and folds instead.
    """
    return {
        "Dummy (prior)": (DummyClassifier(strategy="prior"), {}),
        # D46: scaled inputs need < 100 iterations, the unscaled ablation baseline up to ~5,200.
        "Logistic regression": (LogisticRegression(max_iter=10_000), {"C": loguniform(1e-3, 1e2)}),
        "Random forest": (
            RandomForestClassifier(random_state=RANDOM_STATE),
            {
                "n_estimators": randint(200, 601),
                "max_depth": [4, 6, 8, 12, 16, None],
                "min_samples_leaf": randint(1, 21),
                "max_features": ["sqrt", 0.3, 0.5],
            },
        ),
        "XGBoost": (
            XGBClassifier(n_jobs=1, random_state=RANDOM_STATE),
            {
                "n_estimators": randint(100, 601),
                "learning_rate": loguniform(0.01, 0.3),
                "max_depth": randint(2, 7),
                "subsample": uniform(0.6, 0.4),  # 0.6–1.0
                "colsample_bytree": uniform(0.5, 0.5),  # 0.5–1.0
                "min_child_weight": randint(1, 11),
                "reg_lambda": loguniform(0.1, 10),
            },
        ),
        "SVM (RBF)": (SVC(), {"C": loguniform(0.1, 100), "gamma": loguniform(1e-3, 1)}),
        "MLP": (
            MLPClassifier(early_stopping=True, max_iter=1000, random_state=RANDOM_STATE),
            {
                "hidden_layer_sizes": [(16,), (32,), (64,), (32, 16)],
                "alpha": loguniform(1e-5, 1e-1),
                "learning_rate_init": loguniform(1e-4, 1e-2),
            },
        ),
    }


def tune_models(
    X: pd.DataFrame, y: pd.Series, n_iter: int = 25, cv: int = 5
) -> dict[str, RandomizedSearchCV | GridSearchCV]:
    """Tune every candidate inside the full pipeline on the same stratified folds (D26, D27)."""
    folds = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    settings = {"scoring": SCORING, "refit": "pr_auc", "cv": folds, "n_jobs": -1}
    searches = {}
    for name, (model, space) in candidates().items():
        pipeline = build_pipeline(model)
        params = {f"model__{key}": values for key, values in space.items()}
        if params:
            search = RandomizedSearchCV(
                pipeline, params, n_iter=n_iter, random_state=RANDOM_STATE, **settings
            )
        else:
            search = GridSearchCV(pipeline, {}, **settings)  # nothing to tune: one CV run
        searches[name] = search.fit(X, y)
    return searches


def _format_params(params: dict) -> str:
    """'name=value' pairs without the pipeline prefix; floats to 3 significant digits."""
    pairs = []
    for key, value in sorted(params.items()):
        value = f"{value:.3g}" if isinstance(value, float) else value
        pairs.append(f"{key.removeprefix('model__')}={value}")
    return ", ".join(pairs)


def tuning_results(searches: dict[str, RandomizedSearchCV | GridSearchCV]) -> pd.DataFrame:
    """CV scores of each model's best setting (mean and std over the folds), best first.

    `worst_setting` is the PR-AUC of the worst sampled setting: how much tuning mattered.
    """
    rows = {}
    for name, search in searches.items():
        results, best = search.cv_results_, search.best_index_
        rows[name] = {
            "pr_auc": results["mean_test_pr_auc"][best],
            "pr_auc_std": results["std_test_pr_auc"][best],
            "roc_auc": results["mean_test_roc_auc"][best],
            "worst_setting": results["mean_test_pr_auc"].min(),
            "fit_seconds": results["mean_fit_time"][best],
            "best_params": _format_params(search.best_params_),
        }
    return pd.DataFrame.from_dict(rows, orient="index").sort_values("pr_auc", ascending=False)


def best_models(
    searches: dict[str, RandomizedSearchCV | GridSearchCV], ensemble_size: int = 3
) -> dict[str, BaseEstimator]:
    """Unfitted pipelines with the best settings, plus a soft-voting ensemble of the top ones."""
    models = {}
    for name, search in searches.items():
        pipeline = clone(search.best_estimator_)
        model = pipeline.named_steps["model"]
        if not hasattr(model, "predict_proba"):
            # D40: the SVM only scores customers; Platt scaling turns scores into probabilities.
            pipeline.set_params(model=CalibratedClassifierCV(model, ensemble=False))
        models[name] = pipeline

    # D24: average the probabilities of the best models by CV PR-AUC (never the baseline).
    ranked = sorted(searches, key=lambda name: searches[name].best_score_, reverse=True)
    top = [n for n in ranked if not isinstance(searches[n].best_estimator_[-1], DummyClassifier)]
    members = [(name, models[name]) for name in top[:ensemble_size]]
    models["Ensemble"] = VotingClassifier(members, voting="soft")
    return models
