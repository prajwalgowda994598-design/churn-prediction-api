"""
api/main.py
-----------
FastAPI application entry point.

Endpoints:
  GET  /health   - liveness probe
  POST /predict  - churn prediction + SHAP explanation

Mangum wraps the app so it can be deployed to AWS Lambda unchanged.
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from api.schemas import CustomerFeatures, HealthResponse, PredictionResponse
from api.predictor import _load_artifacts, predict

logger = logging.getLogger("uvicorn.error")

# ---------------------------------------------------------------------------
# App lifecycle: pre-load model on startup so first request is fast
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm-start: load model artifacts once when the container starts."""
    try:
        _load_artifacts()
        logger.info("Model artifacts loaded successfully.")
    except FileNotFoundError as exc:
        logger.warning(f"Model not loaded on startup: {exc}")
    yield  # application runs here


app = FastAPI(
    title="Customer Churn Prediction API",
    description=(
        "Predicts telco customer churn probability using XGBoost + SHAP. "
        "Returns probability, binary prediction, and top-3 SHAP-based reasons."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Exception handler: return clean JSON for unhandled errors
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)},
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    tags=["Ops"],
)
async def health() -> HealthResponse:
    """Liveness probe. Also reports whether the model is loaded."""
    try:
        _load_artifacts()
        model_loaded = True
    except FileNotFoundError:
        model_loaded = False
    return HealthResponse(
        status="ok",
        model_loaded=model_loaded,
        version="1.0.0",
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict customer churn",
    tags=["Prediction"],
)
async def predict_churn(customer: CustomerFeatures) -> PredictionResponse:
    """
    Accept a customer feature payload and return:
    - **churn_probability**: float in [0, 1]
    - **churn_prediction**: boolean (True = likely to churn)
    - **top_3_reasons**: list of feature name + SHAP impact + actual value

    All fields are required. See the schema for valid enumerations.
    """
    try:
        result = predict(customer.model_dump())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.error(f"Prediction error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")

    return PredictionResponse(**result, model_version="1.0.0")


# ---------------------------------------------------------------------------
# AWS Lambda handler via Mangum
# ---------------------------------------------------------------------------

# Import here so the module-level `handler` is available for Lambda even when
# the Mangum package is present.  If running locally with uvicorn, this is a
# no-op.
try:
    from mangum import Mangum
    handler = Mangum(app, lifespan="off")
except ImportError:
    handler = None  # local dev without Mangum installed


# ---------------------------------------------------------------------------
# Local dev entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ["PORT"])
    uvicorn.run("api.main:app", host="0.0.0.0", port=port, reload=True)
