"""Tests for the preprocessing ablation (D9)."""

import inspect
from itertools import pairwise

import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

from churn.ablation import (
    LADDER,
    VARIANTS,
    ablation_table,
    run_ablation,
    step_effects,
    with_class_weights,
)
from churn.config import TARGET
from churn.evaluation import cross_validate_models
from churn.pipeline import build_pipeline


def test_each_variant_adds_one_step_and_the_last_is_the_final_setup():
    switches = list(VARIANTS.values())
    for before, after in pairwise(switches):
        assert sum(before[key] != after[key] for key in before) == 1
    defaults = {
        name: param.default
        for name, param in inspect.signature(build_pipeline).parameters.items()
        if param.kind is param.KEYWORD_ONLY
    }
    assert switches[-1] == defaults


def test_class_weights_reach_every_model():
    y = pd.Series([0, 0, 0, 1])  # 3 retained customers per churner
    assert with_class_weights(LogisticRegression(), y).class_weight == "balanced"
    assert with_class_weights(XGBClassifier(), y).scale_pos_weight == 3
    svm = CalibratedClassifierCV(SVC())  # D40: the SVM inside Platt scaling
    assert with_class_weights(svm, y).estimator.class_weight == "balanced"
    assert svm.estimator.class_weight is None  # the tuned model itself stays unweighted


def test_step_effects_are_same_fold_changes():
    pr_auc = {  # two folds per variant
        "Baseline": [0.50, 0.40],
        "+ standardisation": [0.60, 0.55],
        "+ Yeo-Johnson": [0.62, 0.60],
        "+ level collapse": [0.61, 0.60],
        "+ new features": [0.65, 0.61],
        "+ class weights": [0.64, 0.62],
    }
    scores = pd.DataFrame(
        [
            {"model": "LR", "variant": variant, "fold": fold, "pr_auc": value}
            for variant, values in pr_auc.items()
            for fold, value in enumerate(values)
        ]
    )
    effects = step_effects(scores)["LR"]
    assert effects.columns.tolist() == LADDER[1:]
    assert effects.loc[0].tolist() == pytest.approx([0.10, 0.02, -0.01, 0.04, -0.01])
    assert effects.loc[1].tolist() == pytest.approx([0.15, 0.05, 0.00, 0.01, 0.01])


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")  # unscaled inputs
def test_ablation_scores_every_variant_on_the_comparison_folds(raw_customers):
    X, y = raw_customers.drop(columns=TARGET), raw_customers[TARGET].eq("Yes").astype(int)
    tuned = {"LR": build_pipeline(LogisticRegression(max_iter=1000))}
    scores = run_ablation(tuned, X, y, names=["LR"])

    assert scores.groupby("variant").size().to_dict() == dict.fromkeys(LADDER, 15)
    table = ablation_table(scores)
    assert table.index.tolist() == LADDER
    assert table[("", "inputs")].tolist() == [40, 40, 40, 26, 34, 34]  # as in test_pipeline
    # The final variant is the tuned model as compared in section 5: same folds, same scores.
    final = scores.loc[scores["variant"] == "+ new features", "pr_auc"]
    assert final.tolist() == cross_validate_models(tuned, X, y)["pr_auc"].tolist()
