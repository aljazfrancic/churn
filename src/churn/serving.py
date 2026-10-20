"""Saving and loading the final model, and scoring customers like the REST API (D35, D51–D55)."""

import hashlib
import io
import json
import math
from datetime import UTC, datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import FixedThresholdClassifier
from sklearn.pipeline import Pipeline

from churn.config import CSV_SHA256, ID_COLUMN, MODELS_DIR
from churn.data import verify_sha256

MODEL_FILE = "churn_model.joblib"
METADATA_FILE = "churn_model.json"
# D54: the offer metrics saved next to the threshold-free ones (precision, recall, workload).
DECISION_METRICS = ["precision", "recall", "share_flagged"]


def model_metadata(
    name: str, y_train: pd.Series, ranking: pd.DataFrame, decisions: pd.DataFrame
) -> dict:
    """Training data and expected metrics of the final model, saved with it (D54).

    `ranking` comes from churn.evaluation.holdout_ranking, `decisions` from holdout_decisions at
    the chosen threshold. The test values and their 95 % intervals are the baseline that
    monitoring compares live results with.
    """
    test = pd.concat([ranking, decisions.loc[DECISION_METRICS]])
    test = test.rename(columns={"test": "value"})[["value", "ci_low", "ci_high"]]
    return {
        "model": name,
        "training": {
            "customers": len(y_train),
            "churn_rate": round(y_train.mean(), 3),
            "data_sha256": CSV_SHA256,
        },
        "metrics": {
            "cv_pr_auc": round(ranking.loc["pr_auc", "cv_mean"], 3),
            "test": test.round(3).to_dict("index"),
        },
    }


def save_model(
    pipeline: Pipeline, threshold: float, metadata: dict, directory: Path = MODELS_DIR
) -> dict:
    """Save the fitted pipeline with its decision threshold, plus its metadata as JSON (D52).

    The model file holds a FixedThresholdClassifier, so `predict` flags customers at `threshold`,
    never at the default 0.5. The metadata gains the file's SHA-256 and a version: the UTC date
    plus the start of the hash, so it sorts by date and changes with the file (D53). Returns the
    metadata as saved.
    """
    buffer = io.BytesIO()
    joblib.dump(FixedThresholdClassifier(pipeline, threshold=threshold), buffer)
    data = buffer.getvalue()
    sha256 = hashlib.sha256(data).hexdigest()
    created = datetime.now(UTC)
    metadata = {
        "version": f"{created:%Y%m%d}-{sha256[:8]}",
        "created_at": created.isoformat(timespec="seconds"),
        "sha256": sha256,
        "threshold": float(threshold),
        **metadata,
    }
    directory.mkdir(parents=True, exist_ok=True)
    (directory / MODEL_FILE).write_bytes(data)
    (directory / METADATA_FILE).write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def load_model(directory: Path = MODELS_DIR) -> tuple[FixedThresholdClassifier, dict]:
    """The saved model and its metadata; the model file is checked against its SHA-256 first.

    Unpickling can run code, so a corrupted or swapped file is rejected before it is loaded (D52).
    The check guards against accidents, not against someone who can also edit the metadata: only
    load model files from a trusted source.
    """
    metadata = json.loads((directory / METADATA_FILE).read_text())
    data = (directory / MODEL_FILE).read_bytes()
    verify_sha256(data, metadata["sha256"])
    return joblib.load(io.BytesIO(data)), metadata


def score(model: FixedThresholdClassifier, request: dict, version: str) -> dict:
    """Response body of `POST /v1/predictions` for a request body of raw customer records (D55).

    The records hold the CRM's fields as exported (blank TotalCharges, SeniorCitizen 0/1): the
    pipeline cleans them itself (D21). Every record needs its customerID, which is echoed back.
    Probabilities are cut, not rounded, to 3 decimals: a shown risk reaches the threshold (2
    decimals, D31) only if the customer is flagged (D62).
    """
    customers = pd.DataFrame.from_records(request["customers"])
    proba = model.predict_proba(customers)[:, 1]
    flagged = model.predict(customers)  # at the saved threshold (D52)
    return {
        "model_version": version,
        "threshold": float(model.threshold),
        "predictions": [
            {
                "customerID": customer,
                "churn_probability": math.floor(p * 1000) / 1000,
                "flagged": bool(f),
            }
            for customer, p, f in zip(customers[ID_COLUMN], proba, flagged, strict=True)
        ],
    }
