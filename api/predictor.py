"""
api/predictor.py
────────────────
Encapsulates model loading and inference logic.
Kept separate from the FastAPI route so it is easily unit-testable.
"""

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

MODEL_DIR = Path(__file__).parent.parent / "model"
DATA_DIR = Path(__file__).parent.parent / "data"

MODEL_PATH = MODEL_DIR / "churn_model.pkl"
EXPLAINER_PATH = MODEL_DIR / "shap_explainer.pkl"
FEATURE_COLS_PATH = DATA_DIR / "feature_columns.json"

# Binary yes/no columns (same list as preprocess.py – must stay in sync)
BINARY_YES_NO = [
    "Partner", "Dependents", "PhoneService", "PaperlessBilling",
    "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]
NOMINAL_COLS = ["InternetService", "Contract", "PaymentMethod"]


@lru_cache(maxsize=1)
def _load_artifacts() -> tuple[Any, Any, list[str]]:
    """
    Load model, SHAP explainer, and feature column list exactly once.
    lru_cache ensures this is cached for the Lambda container lifetime.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {MODEL_PATH}. "
            "Run: python model/train.py"
        )
    model = joblib.load(MODEL_PATH)
    explainer = joblib.load(EXPLAINER_PATH)
    feature_cols = json.loads(FEATURE_COLS_PATH.read_text())
    return model, explainer, feature_cols


def _encode_customer(raw: dict, feature_cols: list[str]) -> np.ndarray:
    """
    Replicate the exact transformations from data/preprocess.py so the
    API receives the same feature vector the model was trained on.
    Returns shape (1, n_features) float32 array.
    """
    row = raw.copy()

    # gender → {0,1}
    row["gender"] = 1 if row["gender"] == "Male" else 0

    # binary yes/no → {0,1}
    for col in BINARY_YES_NO:
        if col in row:
            row[col] = 1 if str(row[col]).lower() == "yes" else 0

    # one-hot encode nominals (get_dummies style, drop_first=False)
    for col in NOMINAL_COLS:
        val = row.pop(col)  # remove original
        # enumerate all categories to produce the same dummy columns
        for cat in _get_categories(col):
            row[f"{col}_{cat}"] = int(val == cat)

    # Build a single-row DataFrame aligned to training feature order
    df = pd.DataFrame([row])
    df = df.reindex(columns=feature_cols, fill_value=0)
    return df.values.astype(np.float32), df


def _get_categories(col: str) -> list[str]:
    """Return the same categories used during training (from preprocess.py)."""
    cats = {
        "InternetService": ["DSL", "Fiber optic", "No"],
        "Contract": ["Month-to-month", "One year", "Two year"],
        "PaymentMethod": [
            "Bank transfer (automatic)",
            "Credit card (automatic)",
            "Electronic check",
            "Mailed check",
        ],
    }
    return cats[col]


def predict(raw_customer: dict) -> dict:
    """
    Run inference on a single customer dict.
    Returns: {churn_probability, churn_prediction, top_3_reasons}
    """
    model, explainer, feature_cols = _load_artifacts()

    X, df_row = _encode_customer(raw_customer, feature_cols)

    # --- Prediction ---
    prob = float(model.predict_proba(X)[0, 1])
    prediction = bool(prob >= 0.5)

    # --- SHAP per-prediction explanation ---
    # Extract XGB model from pipeline for TreeExplainer
    xgb_model = model.named_steps["xgb"]
    shap_vals = explainer.shap_values(X)[0]  # shape (n_features,)

    # Build top-3 reasons sorted by |SHAP value|
    top_idx = np.argsort(np.abs(shap_vals))[::-1][:3]
    top_3 = [
        {
            "feature": feature_cols[i],
            "impact": round(float(shap_vals[i]), 5),
            "value": round(float(X[0, i]), 4),
        }
        for i in top_idx
    ]

    return {
        "churn_probability": round(prob, 5),
        "churn_prediction": prediction,
        "top_3_reasons": top_3,
    }
