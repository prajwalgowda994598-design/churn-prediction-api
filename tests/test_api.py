"""
tests/test_api.py
─────────────────
API endpoint tests using FastAPI TestClient (no real HTTP server).
"""

import pytest


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_returns_200(self, test_client):
        response = test_client.get("/health")
        assert response.status_code == 200

    def test_health_body_structure(self, test_client):
        data = test_client.get("/health").json()
        assert "status" in data
        assert "model_loaded" in data
        assert "version" in data
        assert data["status"] == "ok"


# ---------------------------------------------------------------------------
# /predict — happy path
# ---------------------------------------------------------------------------

class TestPredictHappyPath:
    def test_predict_returns_200(self, test_client, valid_payload):
        response = test_client.post("/predict", json=valid_payload)
        assert response.status_code == 200, response.text

    def test_predict_response_schema(self, test_client, valid_payload):
        data = test_client.post("/predict", json=valid_payload).json()
        assert "churn_probability" in data
        assert "churn_prediction" in data
        assert "top_3_reasons" in data
        assert "model_version" in data

    def test_churn_probability_in_range(self, test_client, valid_payload):
        data = test_client.post("/predict", json=valid_payload).json()
        assert 0.0 <= data["churn_probability"] <= 1.0

    def test_top_3_reasons_count(self, test_client, valid_payload):
        data = test_client.post("/predict", json=valid_payload).json()
        assert len(data["top_3_reasons"]) == 3

    def test_top_3_reasons_schema(self, test_client, valid_payload):
        data = test_client.post("/predict", json=valid_payload).json()
        for reason in data["top_3_reasons"]:
            assert "feature" in reason
            assert "impact" in reason
            assert "value" in reason

    def test_churn_prediction_is_bool(self, test_client, valid_payload):
        data = test_client.post("/predict", json=valid_payload).json()
        assert isinstance(data["churn_prediction"], bool)


# ---------------------------------------------------------------------------
# /predict — edge cases & error handling
# ---------------------------------------------------------------------------

class TestPredictEdgeCases:
    def test_missing_required_field_returns_422(self, test_client, valid_payload):
        """Pydantic must reject a payload missing a required field."""
        incomplete = valid_payload.copy()
        del incomplete["tenure"]
        response = test_client.post("/predict", json=incomplete)
        assert response.status_code == 422

    def test_invalid_enum_field_returns_422(self, test_client, valid_payload):
        """Invalid Contract value must fail validation."""
        bad = valid_payload.copy()
        bad["Contract"] = "Daily"  # not in the Literal enum
        response = test_client.post("/predict", json=bad)
        assert response.status_code == 422

    def test_negative_tenure_returns_422(self, test_client, valid_payload):
        """tenure has ge=0 constraint — negative value must fail."""
        bad = valid_payload.copy()
        bad["tenure"] = -5
        response = test_client.post("/predict", json=bad)
        assert response.status_code == 422

    def test_tenure_above_max_returns_422(self, test_client, valid_payload):
        """tenure has le=72 — value > 72 must fail."""
        bad = valid_payload.copy()
        bad["tenure"] = 200
        response = test_client.post("/predict", json=bad)
        assert response.status_code == 422

    def test_zero_monthly_charges_returns_422(self, test_client, valid_payload):
        """MonthlyCharges has gt=0 — zero must fail."""
        bad = valid_payload.copy()
        bad["MonthlyCharges"] = 0.0
        response = test_client.post("/predict", json=bad)
        assert response.status_code == 422

    def test_empty_body_returns_422(self, test_client):
        """Empty JSON must fail validation."""
        response = test_client.post("/predict", json={})
        assert response.status_code == 422

    def test_wrong_gender_value_returns_422(self, test_client, valid_payload):
        """gender must be exactly 'Male' or 'Female'."""
        bad = valid_payload.copy()
        bad["gender"] = "Unknown"
        response = test_client.post("/predict", json=bad)
        assert response.status_code == 422

    def test_extra_fields_are_ignored(self, test_client, valid_payload):
        """Pydantic v2 by default ignores extra fields — must still return 200."""
        extra = valid_payload.copy()
        extra["unknown_column"] = "hacker_value"
        response = test_client.post("/predict", json=extra)
        # Pydantic v2 ignores extras by default; prediction should succeed
        assert response.status_code == 200

    def test_senior_citizen_out_of_range(self, test_client, valid_payload):
        """SeniorCitizen must be 0 or 1."""
        bad = valid_payload.copy()
        bad["SeniorCitizen"] = 5
        response = test_client.post("/predict", json=bad)
        assert response.status_code == 422
