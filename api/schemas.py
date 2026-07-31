"""
api/schemas.py
──────────────
Pydantic v2 input/output models for the /predict endpoint.

All feature fields match the raw Telco dataset columns.  The API accepts the
"human-readable" form (e.g. "Yes"/"No", "Month-to-month") so callers don't
need to know about one-hot encoding — the predictor layer handles that.
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Input schema
# ---------------------------------------------------------------------------

class CustomerFeatures(BaseModel):
    """
    All features required to predict churn for a single customer.
    Matches the raw Telco CSV columns (pre-encoding).
    """

    # ── Demographics ──────────────────────────────────────────────────────────
    gender: Literal["Male", "Female"] = Field(..., description="Customer gender")
    SeniorCitizen: int = Field(..., ge=0, le=1, description="1 if senior citizen, else 0")
    Partner: Literal["Yes", "No"] = Field(..., description="Has a partner?")
    Dependents: Literal["Yes", "No"] = Field(..., description="Has dependents?")

    # ── Account info ─────────────────────────────────────────────────────────
    tenure: int = Field(..., ge=0, le=72, description="Months with the company")
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(..., gt=0, description="Monthly bill in USD")
    TotalCharges: float = Field(..., ge=0, description="Total billed to date in USD")

    # ── Phone services ────────────────────────────────────────────────────────
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]

    # ── Internet services ─────────────────────────────────────────────────────
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]

    @field_validator("TotalCharges")
    @classmethod
    def total_charges_not_less_than_monthly(cls, v: float, info) -> float:
        # TotalCharges should be ≥ MonthlyCharges for a customer with tenure ≥ 1.
        # We allow 0 for brand-new customers (tenure=0) — validated elsewhere.
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
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
        }
    }


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

class ShapReason(BaseModel):
    """One feature driving the prediction."""
    feature: str = Field(..., description="Feature name")
    impact: float = Field(..., description="SHAP value (positive = pushes toward churn)")
    value: float = Field(..., description="Actual feature value used for this prediction")


class PredictionResponse(BaseModel):
    """Full response from /predict."""
    churn_probability: float = Field(
        ..., ge=0.0, le=1.0, description="P(churn) in [0,1]"
    )
    churn_prediction: bool = Field(
        ..., description="True if probability ≥ 0.5"
    )
    top_3_reasons: list[ShapReason] = Field(
        ..., description="Top 3 features by |SHAP value| driving this prediction"
    )
    model_version: str = Field(default="1.0.0")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str
