"""
model/train.py
──────────────
Full training pipeline:
  1. Load preprocessed train/test splits.
  2. Handle class imbalance via SMOTE on the training set.
  3. Tune XGBoost hyperparameters with RandomizedSearchCV (5-fold stratified CV).
  4. Evaluate on held-out test set → AUC, F1, confusion matrix.
  5. Persist model artifact (joblib) + SHAP global importance plot.

Why SMOTE over class_weight?
  ─ class_weight="balanced" in XGBoost (scale_pos_weight) adjusts the loss but
    keeps the original distribution for every fold in CV — minority class still
    underrepresented during tree-split search.
  ─ SMOTE synthesises new minority samples *before* fitting, giving the model
    a genuinely balanced view during tree construction.  This typically yields
    higher recall on the churned class (the business-critical side) with
    comparable AUC — exactly what a telco wants (false negatives = lost revenue).
  ─ IMPORTANT: SMOTE is applied ONLY inside each CV fold and on the final
    training set.  It is NEVER applied to the test set.
"""

import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent.parent / "data"
MODEL_DIR = Path(__file__).parent
TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"
MODEL_ARTIFACT = MODEL_DIR / "churn_model.pkl"
SHAP_PLOT = MODEL_DIR / "shap_importance.png"
METRICS_JSON = MODEL_DIR / "metrics.json"

RANDOM_STATE = 42
CV_FOLDS = 5
N_ITER = 30  # RandomizedSearchCV iterations (tune up for better accuracy)


# ---------------------------------------------------------------------------
# Hyperparameter search space
# ---------------------------------------------------------------------------
PARAM_DIST = {
    # Learning dynamics
    "xgb__n_estimators": [200, 400, 600, 800],
    "xgb__learning_rate": [0.01, 0.05, 0.1, 0.2],
    "xgb__max_depth": [3, 4, 5, 6, 7],
    # Regularisation (combat overfitting)
    "xgb__subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "xgb__colsample_bytree": [0.5, 0.6, 0.7, 0.8, 1.0],
    "xgb__min_child_weight": [1, 3, 5, 7],
    "xgb__gamma": [0, 0.1, 0.2, 0.5],
    "xgb__reg_alpha": [0, 0.01, 0.1, 1.0],    # L1
    "xgb__reg_lambda": [0.5, 1.0, 2.0, 5.0],  # L2
}


def load_splits() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load CSV splits and return (X_train, X_test, y_train, y_test)."""
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    X_train = train_df.drop(columns=["Churn"]).values.astype(np.float32)
    y_train = train_df["Churn"].values.astype(int)
    X_test = test_df.drop(columns=["Churn"]).values.astype(np.float32)
    y_test = test_df["Churn"].values.astype(int)

    feature_names = [c for c in train_df.columns if c != "Churn"]
    print(f"[train] X_train={X_train.shape}  X_test={X_test.shape}")
    print(f"[train] Train churn rate: {y_train.mean():.3f}")
    return X_train, X_test, y_train, y_test, feature_names


def build_pipeline() -> ImbPipeline:
    """
    SMOTE → XGBClassifier pipeline.
    Using imblearn Pipeline ensures SMOTE is only applied to training folds
    during cross-validation, never to the validation/test fold.
    """
    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
    xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",     # fast for medium datasets
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return ImbPipeline([("smote", smote), ("xgb", xgb)])


def tune(pipeline: ImbPipeline, X_train, y_train) -> RandomizedSearchCV:
    """RandomizedSearchCV over the pipeline."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=PARAM_DIST,
        n_iter=N_ITER,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        random_state=RANDOM_STATE,
        refit=True,  # refit best params on full training data after search
    )
    print(f"[train] Running RandomizedSearchCV ({N_ITER} iters × {CV_FOLDS} folds) …")
    search.fit(X_train, y_train)
    print(f"[train] Best CV AUC: {search.best_score_:.4f}")
    print(f"[train] Best params: {search.best_params_}")
    return search


def evaluate(model, X_test, y_test, feature_names) -> dict:
    """Compute metrics and save confusion matrix plot."""
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    auc = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred, output_dict=True)

    print(f"\n{'='*50}")
    print(f"  Test AUC:  {auc:.4f}")
    print(f"  Precision: {report['1']['precision']:.4f}")
    print(f"  Recall:    {report['1']['recall']:.4f}")
    print(f"  F1-score:  {report['1']['f1-score']:.4f}")
    print(f"{'='*50}\n")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

    if auc < 0.90:
        print(f"[warn] AUC {auc:.4f} < 0.90 target. Consider more iterations or data.")

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(4, 3))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=["No Churn", "Churn"],
        colorbar=False, ax=ax
    )
    plt.tight_layout()
    fig.savefig(MODEL_DIR / "confusion_matrix.png", dpi=120)
    plt.close(fig)

    return {
        "auc": round(auc, 4),
        "precision_churn": round(report["1"]["precision"], 4),
        "recall_churn": round(report["1"]["recall"], 4),
        "f1_churn": round(report["1"]["f1-score"], 4),
        "accuracy": round(report["accuracy"], 4),
    }


def generate_shap_plot(model, X_train, feature_names: list[str]) -> None:
    """
    Compute SHAP values on the training set and save a global bar plot.
    We use the XGBClassifier extracted from the pipeline for SHAP
    (SHAP's TreeExplainer needs the raw model, not the imblearn wrapper).
    """
    print("[shap] Computing SHAP values (this may take ~30 s) ...")
    xgb_model = model.named_steps["xgb"]

    # Use a background sample directly from X_train (no re-SMOTE needed for SHAP)
    sample_size = min(500, len(X_train))
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_train), size=sample_size, replace=False)
    X_sample = X_train[idx]

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_sample)

    # Save explainer for inference-time SHAP (avoid re-fitting)
    joblib.dump(explainer, MODEL_DIR / "shap_explainer.pkl")

    # Global feature importance bar chart
    fig, ax = plt.subplots(figsize=(8, 6))
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs_shap)[::-1][:15]  # top 15
    top_features = [feature_names[i] for i in top_idx]
    top_vals = mean_abs_shap[top_idx]

    ax.barh(top_features[::-1], top_vals[::-1], color="#3b82d4")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Global Feature Importance (SHAP)")
    plt.tight_layout()
    fig.savefig(SHAP_PLOT, dpi=120)
    plt.close(fig)
    print(f"[shap] Plot saved -> {SHAP_PLOT}")


def main() -> None:
    # ── Verify preprocessed data exists ──────────────────────────────────────
    if not TRAIN_CSV.exists():
        sys.exit(
            "[error] Train data not found. Run: python data/preprocess.py"
        )

    X_train, X_test, y_train, y_test, feature_names = load_splits()

    # ── Tune + fit ────────────────────────────────────────────────────────────
    pipeline = build_pipeline()
    search = tune(pipeline, X_train, y_train)
    best_model = search.best_estimator_

    # ── Evaluate ──────────────────────────────────────────────────────────────
    metrics = evaluate(best_model, X_test, y_test, feature_names)

    # ── Persist model + metrics ───────────────────────────────────────────────
    joblib.dump(best_model, MODEL_ARTIFACT)
    METRICS_JSON.write_text(json.dumps(metrics, indent=2))
    print(f"[train] Model saved -> {MODEL_ARTIFACT}")

    # ── SHAP global importance ────────────────────────────────────────────────
    generate_shap_plot(best_model, X_train, feature_names)

    print("\n[train] Training complete.")


if __name__ == "__main__":
    main()
