"""
streamlit_app.py
────────────────
Interactive Churn Prediction dashboard.
Calls the local FastAPI backend (or any URL you configure).

Run:
    # Start API first (in a separate terminal):
    python run.py

    # Then start this dashboard:
    streamlit run streamlit_app.py
"""

import json
from pathlib import Path

import requests
import streamlit as st

# ── Config ────────────────────────────────────────────────────────────────────
API_URL = "https://churn-prediction-api-zab6.onrender.com"

st.set_page_config(
    page_title="Churn Predictor",
    page_icon="📉",
    layout="wide",
)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📉 Customer Churn Prediction")
st.caption(
    "XGBoost + SMOTE + SHAP · Telco Customer Dataset · "
    f"API: `{API_URL}`"
)

# ── Health check banner ───────────────────────────────────────────────────────
try:
    health = requests.get(f"{API_URL}/health", timeout=3).json()
    if health.get("model_loaded"):
        st.success("API is online · Model loaded ✓", icon="✅")
    else:
        st.warning("API is online but model not loaded yet.", icon="⚠️")
except Exception:
    st.error(
        f"Cannot reach API at `{API_URL}`. "
        "Run `python run.py` in a terminal first.",
        icon="🔴",
    )

st.divider()

# ── Input form ────────────────────────────────────────────────────────────────
st.subheader("Customer Details")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Demographics**")
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
    partner = st.selectbox("Has Partner?", ["Yes", "No"])
    dependents = st.selectbox("Has Dependents?", ["Yes", "No"])

with col2:
    st.markdown("**Account**")
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    )
    monthly = st.number_input("Monthly Charges ($)", min_value=1.0, max_value=200.0, value=65.0, step=0.5)
    total = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=float(tenure * monthly), step=1.0)

with col3:
    st.markdown("**Services**")
    phone = st.selectbox("Phone Service", ["Yes", "No"])
    multi = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
    security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

# ── Predict button ────────────────────────────────────────────────────────────
st.divider()
predict_clicked = st.button("🔍 Predict Churn", type="primary", use_container_width=True)

if predict_clicked:
    payload = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
        "PhoneService": phone,
        "MultipleLines": multi,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": device,
        "TechSupport": support,
        "StreamingTV": tv,
        "StreamingMovies": movies,
    }

    with st.spinner("Waking up the API (free tier can take up to a minute on first request)..."):
        try:
            resp = requests.post(
                f"{API_URL}/predict",
                json=payload,
                timeout=60,
            )
            resp.raise_for_status()
            result = resp.json()
        except requests.exceptions.ConnectionError:
            st.error("Cannot reach the API. Make sure `python run.py` is running.")
            st.stop()
        except requests.exceptions.HTTPError as e:
            st.error(f"API returned an error: {e.response.text}")
            st.stop()

    # ── Result display ────────────────────────────────────────────────────────
    prob   = result["churn_probability"]
    churn  = result["churn_prediction"]
    top3   = result["top_3_reasons"]

    st.divider()
    st.subheader("Prediction Result")

    res_col1, res_col2 = st.columns([1, 2])

    with res_col1:
        if churn:
            st.error(f"### ⚠️ Likely to Churn\nProbability: **{prob:.1%}**")
        else:
            st.success(f"### ✅ Likely to Stay\nProbability of churn: **{prob:.1%}**")

        # Probability gauge using progress bar
        st.markdown("**Churn Risk**")
        color = "red" if prob > 0.6 else "orange" if prob > 0.4 else "green"
        st.progress(prob, text=f"{prob:.1%}")

    with res_col2:
        st.markdown("**Top 3 Reasons (SHAP)**")
        st.caption("Positive impact = pushes toward churn · Negative = pushes away")

        for i, reason in enumerate(top3, 1):
            feat    = reason["feature"]
            impact  = reason["impact"]
            val     = reason["value"]
            sign    = "🔴 +" if impact > 0 else "🟢 "
            bar_val = min(abs(impact) * 3, 1.0)   # scale for display
            st.markdown(f"**{i}. `{feat}`** = `{val}`")
            st.markdown(f"{sign}`{impact:+.4f}` impact on churn probability")
            st.progress(bar_val)

    # ── Raw JSON expander ─────────────────────────────────────────────────────
    with st.expander("Raw API Response (JSON)"):
        st.json(result)

    # ── Payload expander ─────────────────────────────────────────────────────
    with st.expander("Request Payload sent to API"):
        st.json(payload)

# ── Sidebar: EDA plots ────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Model Info")
    metrics_path = Path("model/metrics.json")
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        st.metric("AUC (ROC)", metrics.get("auc", "—"))
        st.metric("F1 (Churn class)", metrics.get("f1_churn", "—"))
        st.metric("Recall (Churn)", metrics.get("recall_churn", "—"))
        st.metric("Precision (Churn)", metrics.get("precision_churn", "—"))
    else:
        st.info("Run `python model/train.py` to see metrics.")

    st.divider()
    st.header("EDA Plots")
    eda_dir = Path("data/eda_plots")
    plots = {
        "Churn Distribution": eda_dir / "churn_distribution.png",
        "Numeric Features": eda_dir / "numeric_distributions.png",
        "Correlation Heatmap": eda_dir / "correlation_heatmap.png",
    }
    for label, path in plots.items():
        if path.exists():
            with st.expander(label):
                st.image(str(path), use_container_width=True)

    st.divider()
    st.header("SHAP Importance")
    shap_path = Path("model/shap_importance.png")
    if shap_path.exists():
        st.image(str(shap_path), use_container_width=True)
    else:
        st.info("Train the model to see SHAP importance plot.")

    st.divider()
    st.caption("Made with IBM Bob · Portfolio Project")
