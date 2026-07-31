"""
tests/conftest.py
─────────────────
Shared pytest fixtures.
"""

import json
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Minimal mock model + explainer so tests run without a real .pkl artifact
# ---------------------------------------------------------------------------

def _make_mock_artifacts(feature_cols: list[str]):
    """
    Return (mock_pipeline, mock_explainer, feature_cols).
    The mock pipeline implements .predict_proba() and .named_steps.
    The mock explainer implements .shap_values().
    """
    n_features = len(feature_cols)

    # --- Mock XGB inside named_steps ---
    mock_xgb = MagicMock()

    # --- Mock SMOTE inside named_steps ---
    mock_smote = MagicMock()
    mock_smote.fit_resample = MagicMock(
        return_value=(np.zeros((10, n_features)), np.zeros(10))
    )

    # --- Mock pipeline ---
    mock_pipeline = MagicMock()
    mock_pipeline.predict_proba = MagicMock(
        return_value=np.array([[0.3, 0.7]])
    )
    mock_pipeline.named_steps = {"xgb": mock_xgb, "smote": mock_smote}

    # --- Mock explainer ---
    mock_explainer = MagicMock()
    # Return a list with one array of SHAP values (one value per feature)
    shap_arr = np.random.uniform(-0.5, 0.5, size=(1, n_features))
    mock_explainer.shap_values = MagicMock(return_value=shap_arr)

    return mock_pipeline, mock_explainer


FEATURE_COLS_PATH = Path(__file__).parent.parent / "data" / "feature_columns.json"
TRAIN_CSV_PATH   = Path(__file__).parent.parent / "data" / "train.csv"


def _load_feature_cols() -> list[str]:
    if FEATURE_COLS_PATH.exists():
        return json.loads(FEATURE_COLS_PATH.read_text())
    # Fallback hard-coded minimal feature list for CI (before training)
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


@pytest.fixture(scope="session")
def feature_cols():
    return _load_feature_cols()


@pytest.fixture(scope="session")
def mock_artifacts(feature_cols):
    """Session-scoped mock artifacts — created once for the test session."""
    pipeline, explainer = _make_mock_artifacts(feature_cols)
    return pipeline, explainer, feature_cols


@pytest.fixture(scope="session")
def test_client(mock_artifacts):
    """TestClient with model artifacts patched to mocks."""
    from api import predictor

    with patch.object(predictor, "_load_artifacts", return_value=mock_artifacts):
        from api.main import app
        with TestClient(app) as client:
            yield client


@pytest.fixture
def valid_payload():
    """A well-formed customer payload matching the Pydantic schema."""
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
