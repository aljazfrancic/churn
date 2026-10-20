"""Tests for the preprocessing and model pipeline (D9)."""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from churn.config import TARGET
from churn.pipeline import build_pipeline


def _split(raw):
    return raw.drop(columns=TARGET), raw[TARGET].eq("Yes").astype(int)


@pytest.mark.parametrize(
    ("collapse_levels", "new_features", "n_inputs"),
    [
        # 3 numeric + 6 Yes/No columns + 3-level MultipleLines, InternetService, 6 add-ons,
        # Contract (3 each) + PaymentMethod (4)
        (False, False, 40),
        (True, False, 26),  # MultipleLines + add-ons become Yes/No: 7 × (3 → 1)
        (True, True, 34),  # + 3 numeric + 5 TenureGroup columns
    ],
)
def test_model_inputs(raw_customers, collapse_levels, new_features, n_inputs):
    X, y = _split(raw_customers)
    pipeline = build_pipeline(
        LogisticRegression(), collapse_levels=collapse_levels, new_features=new_features
    )
    prepared = pipeline[:-1].fit_transform(X, y)
    assert prepared.shape == (len(X), n_inputs)
    assert np.isfinite(prepared.to_numpy(dtype=float)).all()
    assert "customerID" not in prepared.columns


@pytest.mark.parametrize("scaler", ["yeo-johnson", "standard", None])
@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")  # unscaled inputs
def test_pipeline_takes_raw_records(raw_customers, scaler):
    X, y = _split(raw_customers)  # blank TotalCharges, SeniorCitizen 0/1, with customerID
    pipeline = build_pipeline(LogisticRegression(max_iter=1000), scaler=scaler).fit(X, y)
    proba = pipeline.predict_proba(X)[:, 1]
    assert proba.shape == (len(X),)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_unseen_category_is_ignored(raw_customers):
    X, y = _split(raw_customers)
    pipeline = build_pipeline(LogisticRegression(max_iter=1000)).fit(X, y)
    with pytest.warns(UserWarning, match="unknown categories"):
        proba = pipeline.predict_proba(X.head(1).assign(PaymentMethod="Crypto"))
    assert np.isfinite(proba).all()


def test_unknown_scaler_raises():
    with pytest.raises(ValueError, match="Unknown scaler"):
        build_pipeline(LogisticRegression(), scaler="minmax")
