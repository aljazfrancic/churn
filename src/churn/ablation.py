"""Preprocessing ablation: what each preprocessing step adds to the tuned models (D32, D44)."""

import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from churn.evaluation import cross_validate_models
from churn.pipeline import build_pipeline

# D32: every variant adds one step to the one before; the fifth is the final setup.
VARIANTS = {
    "Baseline": {"scaler": None, "collapse_levels": False, "new_features": False},
    "+ standardisation": {"scaler": "standard", "collapse_levels": False, "new_features": False},
    "+ Yeo-Johnson": {"scaler": "yeo-johnson", "collapse_levels": False, "new_features": False},
    "+ level collapse": {"scaler": "yeo-johnson", "collapse_levels": True, "new_features": False},
    "+ new features": {"scaler": "yeo-johnson", "collapse_levels": True, "new_features": True},
}
CLASS_WEIGHTS = "+ class weights"  # the final setup with churners weighted up; not used (D25)
LADDER = [*VARIANTS, CLASS_WEIGHTS]
MODELS = ["Logistic regression", "XGBoost", "SVM (RBF)"]  # linear, trees, most scale-sensitive


def with_class_weights(model: BaseEstimator, y: pd.Series) -> BaseEstimator:
    """Copy of `model` in which churners weigh as much in total as retained customers (D32)."""
    model = clone(model)
    if isinstance(model, XGBClassifier):
        return model.set_params(scale_pos_weight=(y == 0).sum() / (y == 1).sum())
    if isinstance(model, CalibratedClassifierCV):  # D40: the SVM inside Platt scaling
        return model.set_params(estimator__class_weight="balanced")
    return model.set_params(class_weight="balanced")


def run_ablation(
    models: dict[str, Pipeline], X: pd.DataFrame, y: pd.Series, names: list[str] = MODELS
) -> pd.DataFrame:
    """Score every variant of the named tuned models on the comparison folds (D44).

    Only the preprocessing changes: each model keeps its tuned setting. One row per model, variant
    and fold; `inputs` is the number of model inputs the variant produces.
    """
    frames = []
    for name in names:
        model = models[name].named_steps["model"]
        variants = {v: build_pipeline(clone(model), **switches) for v, switches in VARIANTS.items()}
        variants[CLASS_WEIGHTS] = build_pipeline(with_class_weights(model, y))
        inputs = {v: clone(p)[:-1].fit_transform(X).shape[1] for v, p in variants.items()}
        scores = cross_validate_models(variants, X, y).rename(columns={"model": "variant"})
        frames.append(scores.assign(model=name, inputs=scores["variant"].map(inputs)))
    return pd.concat(frames, ignore_index=True)


def ablation_table(scores: pd.DataFrame) -> pd.DataFrame:
    """Model inputs plus mean PR-AUC and Brier score per variant (rows) and model (columns)."""
    means = {
        model: group.groupby("variant")[["pr_auc", "brier"]].mean()
        for model, group in scores.groupby("model", sort=False)
    }
    table = pd.concat(means, axis=1)
    table.insert(0, ("", "inputs"), scores.groupby("variant")["inputs"].first())
    return table.loc[LADDER]


def step_effects(scores: pd.DataFrame) -> pd.DataFrame:
    """Same-fold PR-AUC change from each variant to the next, per model (D44).

    Columns are (model, step), a step being named after the variant that adds it; rows are folds.
    """
    pr_auc = scores.pivot(index="fold", columns=["model", "variant"], values="pr_auc")
    changes = {
        model: pr_auc[model][LADDER].diff(axis=1).drop(columns=LADDER[0])
        for model in scores["model"].unique()
    }
    return pd.concat(changes, axis=1)
