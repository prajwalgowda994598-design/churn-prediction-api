"""Re-generate SHAP explainer from already-saved model + training data."""
import json, sys
from pathlib import Path
import joblib, numpy as np, pandas as pd, shap

MODEL_DIR = Path("model")
DATA_DIR  = Path("data")

model     = joblib.load(MODEL_DIR / "churn_model.pkl")
train_df  = pd.read_csv(DATA_DIR / "train.csv")
feat_cols = [c for c in train_df.columns if c != "Churn"]
X_train   = train_df[feat_cols].values.astype(np.float32)

xgb_model = model.named_steps["xgb"]

rng = np.random.default_rng(42)
idx = rng.choice(len(X_train), size=min(500, len(X_train)), replace=False)
X_sample = X_train[idx]

print("[shap] Building TreeExplainer ...")
explainer = shap.TreeExplainer(xgb_model)
shap_vals = explainer.shap_values(X_sample)

joblib.dump(explainer, MODEL_DIR / "shap_explainer.pkl")
print(f"[shap] Saved -> {MODEL_DIR / 'shap_explainer.pkl'}")

# global importance bar chart
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

mean_abs = np.abs(shap_vals).mean(axis=0)
top_idx = np.argsort(mean_abs)[::-1][:15]
top_features = [feat_cols[i] for i in top_idx]
top_vals     = mean_abs[top_idx]

fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(top_features[::-1], top_vals[::-1], color="#3b82d4")
ax.set_xlabel("Mean |SHAP value|")
ax.set_title("Global Feature Importance (SHAP)")
plt.tight_layout()
fig.savefig(MODEL_DIR / "shap_importance.png", dpi=120)
plt.close(fig)
print(f"[shap] Plot saved -> {MODEL_DIR / 'shap_importance.png'}")
print("[shap] Done.")
