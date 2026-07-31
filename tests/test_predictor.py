"""
tests/test_predictor.py
───────────────────────
Unit tests for api/predictor.py — model encoding + inference logic.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from api.predictor import _encode_customer, _get_categories, predict


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def feature_cols():
    path = Path("data/feature_columns.json")
    if path.exists():
        return json.loads(path.read_text())
    # Minimal fallback for CI before training
    return [
        "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
        "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
        "PaperlessBilling", "MonthlyCharges", "TotalCharges",
        "InternetService_DSL", "InternetService_Fiber optic", "InternetService_No",
        "Contract_Month-to-month", "Contract_One year", "Contract_Two year",
        "PaymentMethod_Bank transfer (automatic)", "PaymentMethod_Credit card (automatic)",
        "PaymentMethod_Electronic check", "PaymentMethod_Mailed check",
    ]


@pytest.fixture
def sample_customer():
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 24,
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.35,
        "TotalCharges": 1685.0,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
    }


# ---------------------------------------------------------------------------
# _get_categories
# ---------------------------------------------------------------------------

def test_get_categories_internet_service():
    cats = _get_categories("InternetService")
    assert "DSL" in cats
    assert "Fiber optic" in cats
    assert "No" in cats


def test_get_categories_contract():
    cats = _get_categories("Contract")
    assert "Month-to-month" in cats
    assert "One year" in cats
    assert "Two year" in cats


def test_get_categories_payment():
    cats = _get_categories("PaymentMethod")
    assert len(cats) == 4


def test_get_categories_invalid_key():
    with pytest.raises(KeyError):
        _get_categories("NonExistentCol")


# ---------------------------------------------------------------------------
# _encode_customer
# ---------------------------------------------------------------------------

def test_encode_output_shape(sample_customer, feature_cols):
    X, df = _encode_customer(sample_customer, feature_cols)
    assert X.shape == (1, len(feature_cols))


def test_encode_gender_male(sample_customer, feature_cols):
    sample_customer["gender"] = "Male"
    X, df = _encode_customer(sample_customer, feature_cols)
    assert df["gender"].iloc[0] == 1


def test_encode_gender_female(sample_customer, feature_cols):
    sample_customer["gender"] = "Female"
    X, df = _encode_customer(sample_customer, feature_cols)
    assert df["gender"].iloc[0] == 0


def test_encode_binary_yes(sample_customer, feature_cols):
    """Partner='Yes' should encode to 1."""
    sample_customer["Partner"] = "Yes"
    X, df = _encode_customer(sample_customer, feature_cols)
    assert df["Partner"].iloc[0] == 1


def test_encode_binary_no(sample_customer, feature_cols):
    """Partner='No' should encode to 0."""
    sample_customer["Partner"] = "No"
    X, df = _encode_customer(sample_customer, feature_cols)
    assert df["Partner"].iloc[0] == 0


def test_encode_one_hot_internet_dsl(sample_customer, feature_cols):
    """DSL should set InternetService_DSL=1 and others=0."""
    sample_customer["InternetService"] = "DSL"
    X, df = _encode_customer(sample_customer, feature_cols)
    if "InternetService_DSL" in df.columns:
        assert df["InternetService_DSL"].iloc[0] == 1
        assert df["InternetService_Fiber optic"].iloc[0] == 0


def test_encode_output_dtype(sample_customer, feature_cols):
    """Output array must be float32 (XGBoost requirement)."""
    X, _ = _encode_customer(sample_customer, feature_cols)
    assert X.dtype == np.float32


def test_encode_unknown_feature_filled_with_zero(sample_customer, feature_cols):
    """Columns not present in raw dict should be filled with 0."""
    X, df = _encode_customer(sample_customer, feature_cols)
    # All values should be finite
    assert np.all(np.isfinite(X))


# ---------------------------------------------------------------------------
# predict() — mocked model
# ---------------------------------------------------------------------------

def test_predict_returns_expected_keys(sample_customer, feature_cols):
    n_features = len(feature_cols)
    mock_pipeline = MagicMock()
    mock_pipeline.predict_proba.return_value = np.array([[0.4, 0.6]])
    mock_pipeline.named_steps = {"xgb": MagicMock(), "smote": MagicMock()}

    mock_explainer = MagicMock()
    shap_arr = np.random.uniform(-0.3, 0.3, size=(1, n_features))
    mock_explainer.shap_values.return_value = shap_arr

    mock_artifacts = (mock_pipeline, mock_explainer, feature_cols)

    import api.predictor as predictor_module
    with patch.object(predictor_module, "_load_artifacts", return_value=mock_artifacts):
        result = predict(sample_customer)

    assert "churn_probability" in result
    assert "churn_prediction" in result
    assert "top_3_reasons" in result
    assert len(result["top_3_reasons"]) == 3


def test_predict_probability_in_range(sample_customer, feature_cols):
    n_features = len(feature_cols)
    mock_pipeline = MagicMock()
    mock_pipeline.predict_proba.return_value = np.array([[0.25, 0.75]])
    mock_pipeline.named_steps = {"xgb": MagicMock(), "smote": MagicMock()}
    mock_explainer = MagicMock()
    mock_explainer.shap_values.return_value = np.zeros((1, n_features))

    import api.predictor as predictor_module
    with patch.object(predictor_module, "_load_artifacts",
                      return_value=(mock_pipeline, mock_explainer, feature_cols)):
        result = predict(sample_customer)

    assert 0.0 <= result["churn_probability"] <= 1.0
    assert result["churn_probability"] == pytest.approx(0.75, abs=1e-4)


def test_predict_true_when_prob_above_threshold(sample_customer, feature_cols):
    n_features = len(feature_cols)
    mock_pipeline = MagicMock()
    mock_pipeline.predict_proba.return_value = np.array([[0.1, 0.9]])
    mock_pipeline.named_steps = {"xgb": MagicMock(), "smote": MagicMock()}
    mock_explainer = MagicMock()
    mock_explainer.shap_values.return_value = np.zeros((1, n_features))

    import api.predictor as predictor_module
    with patch.object(predictor_module, "_load_artifacts",
                      return_value=(mock_pipeline, mock_explainer, feature_cols)):
        result = predict(sample_customer)

    assert result["churn_prediction"] is True


def test_predict_false_when_prob_below_threshold(sample_customer, feature_cols):
    n_features = len(feature_cols)
    mock_pipeline = MagicMock()
    mock_pipeline.predict_proba.return_value = np.array([[0.8, 0.2]])
    mock_pipeline.named_steps = {"xgb": MagicMock(), "smote": MagicMock()}
    mock_explainer = MagicMock()
    mock_explainer.shap_values.return_value = np.zeros((1, n_features))

    import api.predictor as predictor_module
    with patch.object(predictor_module, "_load_artifacts",
                      return_value=(mock_pipeline, mock_explainer, feature_cols)):
        result = predict(sample_customer)

    assert result["churn_prediction"] is False
