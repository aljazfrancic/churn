"""Preprocessing and the full model pipeline: raw customer records in, churn predictions out."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, StandardScaler

from churn.config import (
    CATEGORICAL_COLUMNS,
    NEW_CATEGORICAL_COLUMNS,
    NEW_NUMERIC_COLUMNS,
    NUMERIC_COLUMNS,
)
from churn.features import FeatureEngineer


def _scaler(name: str | None) -> BaseEstimator | str:
    """Numeric scaling: Yeo-Johnson (D22); plain standardisation or none for the ablation (D32)."""
    match name:
        case "yeo-johnson":
            return PowerTransformer(method="yeo-johnson")  # also standardises: mean 0, std 1
        case "standard":
            return StandardScaler()
        case None:
            return "passthrough"
        case _:
            raise ValueError(f"Unknown scaler: {name!r}")


def build_preprocessor(
    scaler: str | None = "yeo-johnson", new_features: bool = True
) -> ColumnTransformer:
    """Scale numeric and one-hot encode categorical features (D22, D23).

    Only the listed columns reach the model, any other input column is dropped (D39). The output
    is a DataFrame with readable names such as `Contract_Two year` (for SHAP and coefficients).
    """
    numeric = NUMERIC_COLUMNS + (NEW_NUMERIC_COLUMNS if new_features else [])
    categorical = CATEGORICAL_COLUMNS + (NEW_CATEGORICAL_COLUMNS if new_features else [])
    # D23: Yes/No → one 0/1 column; categories unseen in training are encoded as all zeros.
    encoder = OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False)
    return ColumnTransformer(
        [("numeric", _scaler(scaler), numeric), ("categorical", encoder, categorical)],
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")


def build_pipeline(
    model: BaseEstimator,
    *,
    scaler: str | None = "yeo-johnson",
    collapse_levels: bool = True,
    new_features: bool = True,
) -> Pipeline:
    """Feature engineering → preprocessing → model, fitted as one unit (D21).

    In cross-validation every step is fitted on the training folds only (no leakage). The defaults
    are the final setup; the switches let the ablation vary each step (D32).
    """
    features = FeatureEngineer(collapse_levels=collapse_levels, new_features=new_features)
    return Pipeline(
        [
            ("features", features),
            ("preprocess", build_preprocessor(scaler=scaler, new_features=new_features)),
            ("model", model),
        ]
    )


def compare_skewness(numeric: pd.DataFrame) -> pd.DataFrame:
    """Skewness per column: raw (unchanged by standardisation), log1p and Yeo-Johnson (D22).

    `lambda` is the fitted Yeo-Johnson exponent: 1 = unchanged, 0 = log1p, < 0 = stronger than log.
    """
    yeo_johnson = PowerTransformer(method="yeo-johnson").set_output(transform="pandas")
    transformed = yeo_johnson.fit_transform(numeric)
    return pd.DataFrame(
        {
            "skew_raw": numeric.skew(),
            "skew_log1p": np.log1p(numeric).skew(),
            "skew_yeo_johnson": transformed.skew(),
            "lambda": pd.Series(yeo_johnson.lambdas_, index=numeric.columns),
        }
    )
