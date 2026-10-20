"""Explaining the final model: SHAP values per customer and odds ratios (D6, D49, D50)."""

from collections.abc import Hashable
from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, clone
from sklearn.pipeline import Pipeline

from churn.config import RANDOM_STATE


def input_features(pipeline: Pipeline) -> pd.DataFrame:
    """Original feature and level of every model input, e.g. `Contract_Two year` → Contract.

    Numeric inputs have no level; a yes/no input names its `reference`, the level the encoder
    dropped (D23). Rows follow the order of the model inputs.
    """
    preprocess = pipeline.named_steps["preprocess"]
    rows = {}
    for name, transformer, columns in preprocess.transformers_:
        if name == "numeric":
            rows.update({column: (column, None, None) for column in columns})
        elif name == "categorical":
            names = iter(transformer.get_feature_names_out(columns))
            encoded = zip(columns, transformer.categories_, transformer.drop_idx_, strict=True)
            for column, levels, dropped in encoded:
                reference = None if dropped is None else levels[dropped]
                for level in levels:
                    if level != reference:
                        rows[next(names)] = (column, level, reference)
    table = pd.DataFrame.from_dict(rows, orient="index", columns=["feature", "level", "reference"])
    return table.loc[preprocess.get_feature_names_out()]


@dataclass(frozen=True)
class ShapValues:
    """SHAP values of the logistic regression for a set of customers, in log-odds (D49)."""

    values: pd.DataFrame  # customers × model inputs
    base_value: float  # log-odds of a customer with every input at its training mean
    inputs: pd.DataFrame  # the customers' model inputs (scaled and one-hot encoded)
    features: pd.DataFrame  # the customers' features before encoding, for labels
    groups: pd.Series  # original feature of every model input

    def by_feature(self) -> pd.DataFrame:
        """SHAP values summed per original feature: exact, as SHAP values add up."""
        return self.values.T.groupby(self.groups, sort=False).sum().T


def explain(pipeline: Pipeline, X_background: pd.DataFrame, X: pd.DataFrame) -> ShapValues:
    """Exact SHAP values of the fitted final pipeline for the customers in X (D6, D49).

    The background is the training set: a value says how far an input moves the customer's
    log-odds from those of a customer with every input at its training mean.
    """
    preprocess, model = pipeline[:-1], pipeline[-1]
    background, inputs = preprocess.transform(X_background), preprocess.transform(X)
    masker = shap.maskers.Independent(background, max_samples=len(background))  # exact means
    explanation = shap.LinearExplainer(model, masker)(inputs)
    return ShapValues(
        values=pd.DataFrame(explanation.values, index=inputs.index, columns=inputs.columns),
        base_value=float(explanation.base_values[0]),
        inputs=inputs,
        features=pipeline.named_steps["features"].transform(X),
        groups=input_features(pipeline)["feature"],
    )


def typical_flagged_churner(y_true: pd.Series, proba: pd.Series, threshold: float) -> Hashable:
    """The flagged churner with the median risk: the kind of case marketing would see (D49)."""
    risks = proba[(y_true == 1) & (proba >= threshold)]
    return (risks - risks.median()).abs().idxmin()


def _contrasts(features: pd.DataFrame, inputs: pd.DataFrame, orders: dict) -> pd.DataFrame:
    """Weights on the coefficients behind each odds ratio; one row per (feature, comparison)."""
    rows = {}
    for feature, members in features.groupby("feature", sort=False):
        if members["level"].isna().all():  # numeric: per +1 SD after Yeo-Johnson
            rows[(feature, "+1 SD")] = {feature: 1.0}
        elif len(members) == 1:  # yes/no: the reference level has no input of its own
            ((column, row),) = members.iterrows()
            rows[(feature, f"{row['level']} vs {row['reference']}")] = {column: 1.0}
        else:  # every level is an input: compare each with the most common one
            if feature in orders:
                members = members.loc[members["level"].map(orders[feature]).sort_values().index]
            reference = inputs[members.index].mean().idxmax()
            for column, row in members.drop(index=reference).iterrows():
                comparison = f"{row['level']} vs {members.loc[reference, 'level']}"
                rows[(feature, comparison)] = {column: 1.0, reference: -1.0}
    table = pd.DataFrame.from_dict(rows, orient="index")
    return table.reindex(columns=inputs.columns).fillna(0.0)


def _coefficients(
    model: BaseEstimator, inputs: np.ndarray, y: np.ndarray, rows: np.ndarray
) -> np.ndarray:
    return clone(model).fit(inputs[rows], y[rows]).coef_[0]


def odds_ratios(
    pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, n_resamples: int = 1000
) -> pd.DataFrame:
    """Odds ratios of the fitted logistic regression with bootstrap 95 % intervals (D50).

    Numeric inputs: per +1 standard deviation after Yeo-Johnson; yes/no inputs: Yes vs No; the
    other levels: vs the most common level of their feature, since every level is an input and a
    single coefficient has no reference. The intervals refit only the regression on resampled
    training customers: the preprocessing stays as fitted, so every coefficient keeps its units.
    """
    preprocess, model = pipeline[:-1], pipeline[-1]
    inputs = preprocess.transform(X)
    features = pipeline.named_steps["features"].transform(X)
    categorical = features.select_dtypes("category")  # tenure groups: natural, not A–Z order
    orders = {
        column: {level: i for i, level in enumerate(categorical[column].cat.categories)}
        for column in categorical
    }
    contrasts = _contrasts(input_features(pipeline), inputs, orders)

    rng = np.random.default_rng(RANDOM_STATE)
    samples = (rng.integers(0, len(y), len(y)) for _ in range(n_resamples))
    matrix, target = inputs.to_numpy(), y.to_numpy()
    resampled = Parallel(n_jobs=-1)(
        delayed(_coefficients)(model, matrix, target, rows) for rows in samples
    )
    log_ratios = np.asarray(resampled) @ contrasts.to_numpy().T  # resamples × comparisons
    low, high = np.quantile(log_ratios, [0.025, 0.975], axis=0)
    return pd.DataFrame(
        {
            "odds_ratio": np.exp(contrasts.to_numpy() @ model.coef_[0]),
            "ci_low": np.exp(low),
            "ci_high": np.exp(high),
        },
        index=contrasts.index.set_names(["feature", "comparison"]),
    )
