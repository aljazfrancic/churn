"""Offline tests for churn.data; the real download is exercised by the notebook run in CI (D9)."""

import hashlib
import io
import zipfile

import pandas as pd
import pytest

from churn.config import CSV_NAME
from churn.data import clean, download_data, split_features_target


@pytest.fixture
def raw() -> pd.DataFrame:
    """Rows in the published format, incl. a brand-new customer with a blank TotalCharges."""
    return pd.DataFrame(
        {
            "customerID": ["0001-A", "0002-B", "0003-C"],
            "SeniorCitizen": [0, 1, 0],
            "tenure": [0, 12, 40],
            "MonthlyCharges": [20.0, 70.5, 99.9],
            "TotalCharges": [" ", "846.0", "3996.0"],
            "Churn": ["No", "Yes", "No"],
        }
    )


def test_clean_fixes_types_and_drops_id(raw):
    df = clean(raw)
    assert "customerID" not in df.columns
    assert df["TotalCharges"].tolist() == [0.0, 846.0, 3996.0]
    assert pd.api.types.is_float_dtype(df["TotalCharges"])
    assert df["SeniorCitizen"].tolist() == ["No", "Yes", "No"]


def test_clean_is_idempotent_and_leaves_input_untouched(raw):
    original = raw.copy()
    cleaned = clean(raw)
    pd.testing.assert_frame_equal(clean(cleaned), cleaned)
    pd.testing.assert_frame_equal(raw, original)


def test_split_features_target(raw):
    X, y = split_features_target(clean(raw))
    assert "Churn" not in X.columns
    assert y.tolist() == [0, 1, 0]


def _zip_with_csv(tmp_path, content: bytes):
    """Write a zip like Kaggle's (one CSV member) and return its path."""
    path = tmp_path / "dataset.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(CSV_NAME, content)
    return path


def test_download_extracts_verified_csv(tmp_path):
    content = b"a,b\n1,2\n"
    archive = _zip_with_csv(tmp_path, content)
    sha256 = hashlib.sha256(content).hexdigest()

    path = download_data(tmp_path / "data", url=archive.as_uri(), sha256=sha256)

    assert path.read_bytes() == content


def test_download_rejects_wrong_checksum_without_saving(tmp_path):
    archive = _zip_with_csv(tmp_path, b"tampered\n")
    with pytest.raises(ValueError, match="SHA-256"):
        download_data(tmp_path / "data", url=archive.as_uri(), sha256="0" * 64)
    assert not (tmp_path / "data" / CSV_NAME).exists()


def test_existing_file_is_reverified(tmp_path):
    (tmp_path / CSV_NAME).write_bytes(b"corrupted\n")
    with pytest.raises(ValueError, match="SHA-256"):
        download_data(tmp_path, url="https://never.contacted.invalid")


def test_other_archive_members_are_never_extracted(tmp_path):
    # A malicious member path ("zip slip") must not be written anywhere.
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("../evil.txt", b"x")
        archive.writestr(CSV_NAME, b"ok\n")
    path = tmp_path / "dataset.zip"
    path.write_bytes(buffer.getvalue())

    download_data(tmp_path / "data", url=path.as_uri(), sha256=hashlib.sha256(b"ok\n").hexdigest())

    assert not (tmp_path / "evil.txt").exists()
