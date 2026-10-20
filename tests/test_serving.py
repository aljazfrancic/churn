"""Tests for saving, loading and scoring the model as the REST API would (D9)."""

import hashlib
import json
import re

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from churn.config import CSV_SHA256, TARGET
from churn.pipeline import build_pipeline
from churn.serving import MODEL_FILE, load_model, model_metadata, save_model, score


def _fitted(raw_customers):
    X, y = raw_customers.drop(columns=TARGET), raw_customers[TARGET].eq("Yes").astype(int)
    return build_pipeline(LogisticRegression(max_iter=1000)).fit(X, y), X


def test_reloaded_model_predicts_like_the_pipeline_at_its_threshold(raw_customers, tmp_path):
    pipeline, X = _fitted(raw_customers)
    saved = save_model(pipeline, 0.4, {"model": "LR"}, directory=tmp_path)
    model, metadata = load_model(tmp_path)

    assert metadata == saved
    proba = pipeline.predict_proba(X)[:, 1]
    np.testing.assert_array_equal(model.predict_proba(X)[:, 1], proba)
    assert ((proba >= 0.4) & (proba < 0.5)).any()  # the threshold makes a difference here
    assert (model.predict(X) == (proba >= 0.4)).all()  # flagged at 0.4, not the default 0.5


def test_version_is_the_date_and_the_start_of_the_file_hash(raw_customers, tmp_path):
    pipeline, _ = _fitted(raw_customers)
    metadata = save_model(pipeline, 0.4, {}, directory=tmp_path)

    digest = hashlib.sha256((tmp_path / MODEL_FILE).read_bytes()).hexdigest()
    assert metadata["sha256"] == digest
    assert re.fullmatch(rf"\d{{8}}-{digest[:8]}", metadata["version"])
    assert metadata["version"][:8] == metadata["created_at"][:10].replace("-", "")


def test_tampered_model_file_is_rejected(raw_customers, tmp_path):
    pipeline, _ = _fitted(raw_customers)
    save_model(pipeline, 0.4, {}, directory=tmp_path)
    path = tmp_path / MODEL_FILE
    path.write_bytes(path.read_bytes() + b"\0")

    with pytest.raises(ValueError, match="SHA-256"):
        load_model(tmp_path)


def test_score_answers_raw_records_in_json(raw_customers, tmp_path):
    pipeline, X = _fitted(raw_customers)
    save_model(pipeline, 0.4, {}, directory=tmp_path)
    model, metadata = load_model(tmp_path)
    # As the CRM exports them: blank TotalCharges for the 3 brand-new customers, SeniorCitizen 0/1.
    records = raw_customers.drop(columns=TARGET).head(5).to_dict("records")
    response = json.loads(json.dumps(score(model, {"customers": records}, metadata["version"])))

    assert response["model_version"] == metadata["version"]
    assert response["threshold"] == 0.4
    predictions = response["predictions"]
    assert [p["customerID"] for p in predictions] == raw_customers["customerID"].head(5).tolist()
    proba = pipeline.predict_proba(X.head(5))[:, 1]
    assert [p["churn_probability"] for p in predictions] == pytest.approx(
        np.floor(proba * 1000) / 1000
    )
    assert [p["flagged"] for p in predictions] == (proba >= 0.4).tolist()


def test_shown_risk_reaches_the_threshold_only_when_flagged(raw_customers, tmp_path):
    pipeline, X = _fitted(raw_customers)
    proba = pipeline.predict_proba(X)[:, 1]
    edge = int(np.flatnonzero(np.round(proba, 3) > proba)[0])  # rounding would lift this risk …
    threshold = round(float(proba[edge]), 3)  # … to exactly the threshold
    save_model(pipeline, threshold, {}, directory=tmp_path)
    model, metadata = load_model(tmp_path)
    records = raw_customers.drop(columns=TARGET).to_dict("records")
    predictions = score(model, {"customers": records}, metadata["version"])["predictions"]

    assert predictions[edge]["churn_probability"] < threshold
    assert not predictions[edge]["flagged"]
    for prediction in predictions:
        assert (prediction["churn_probability"] >= threshold) == prediction["flagged"]


def test_metadata_holds_training_data_and_expected_metrics():
    ranking = pd.DataFrame(
        {
            "cv_mean": [0.70, 0.85, 0.13],
            "test": [0.68, 0.84, 0.14],
            "ci_low": [0.63, 0.82, 0.12],
            "ci_high": [0.73, 0.86, 0.15],
        },
        index=["pr_auc", "roc_auc", "brier"],
    )
    rows = ["precision", "recall", "f1", "f2", "balanced_accuracy", "accuracy", "share_flagged"]
    decisions = pd.DataFrame({"test": 0.5, "ci_low": 0.4, "ci_high": 0.6}, index=rows)
    metadata = model_metadata("LR", pd.Series([0, 1, 0, 0]), ranking, decisions)

    assert metadata["training"] == {"customers": 4, "churn_rate": 0.25, "data_sha256": CSV_SHA256}
    assert metadata["metrics"]["cv_pr_auc"] == 0.70
    test = metadata["metrics"]["test"]
    assert list(test) == ["pr_auc", "roc_auc", "brier", "precision", "recall", "share_flagged"]
    assert test["pr_auc"] == {"value": 0.68, "ci_low": 0.63, "ci_high": 0.73}
    assert test["recall"] == {"value": 0.5, "ci_low": 0.4, "ci_high": 0.6}
    json.dumps(metadata)  # JSON types only
