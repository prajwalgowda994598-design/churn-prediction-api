# ─────────────────────────────────────────────────────────────────────────────
# Multi-stage Dockerfile for Customer Churn Prediction API
#
# Stage 1 (builder): install all Python deps into a venv
# Stage 2 (runtime): copy only the venv + app code → lean final image
#
# Final image size target: < 600 MB (xgboost + shap are heavy but necessary)
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: builder ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

# System build deps (needed to compile some wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy only requirements first to leverage layer caching
COPY requirements.txt .

# Create isolated venv + install dependencies
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip --quiet \
    && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt


# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="portfolio-project"
LABEL description="Churn Prediction API — XGBoost + SHAP + FastAPI"

# libgomp1 is needed at runtime by XGBoost (OpenMP threading)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the pre-built venv from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Copy application source (preserving package structure)
COPY api/       ./api/
COPY data/      ./data/
COPY model/     ./model/

# Ensure Python finds our packages without pip install -e
ENV PYTHONPATH="/app"

# Non-root user for security
RUN useradd -m -u 1001 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Health check — Docker will poll this every 30 s
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Start with uvicorn; workers=1 is correct for Lambda-style single-process
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
