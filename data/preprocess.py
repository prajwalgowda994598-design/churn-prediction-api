"""
data/preprocess.py
──────────────────
Loads the raw Telco Churn CSV, performs EDA, cleans and encodes features,
and returns train/test splits ready for model training.

Design decisions (interview-defensible):
  • TotalCharges has ~11 blank strings coerced to NaN → filled with median
    (only 0.15% of rows; imputing vs. dropping changes nothing materially).
  • customerID is a surrogate key → dropped.
  • Binary yes/no columns → label-encoded {0,1}.
  • Multi-class nominals (InternetService, Contract, etc.) → pd.get_dummies
    (one-hot). Tree models don't need scaling.
  • Target: Churn → {0: No, 1: Yes}.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless – no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).parent
RAW_CSV = DATA_DIR / "telco_churn.csv"
EDA_DIR = DATA_DIR / "eda_plots"
PROCESSED_TRAIN = DATA_DIR / "train.csv"
PROCESSED_TEST = DATA_DIR / "test.csv"
FEATURE_COLUMNS_JSON = DATA_DIR / "feature_columns.json"

RANDOM_STATE = 42
TEST_SIZE = 0.20

# ---------------------------------------------------------------------------
# Binary columns that are yes/no strings
# ---------------------------------------------------------------------------
BINARY_YES_NO = [
    "Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn",
    "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]

# Multi-class nominal columns → one-hot encoded
NOMINAL_COLS = ["InternetService", "Contract", "PaymentMethod"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_raw(path: Path = RAW_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[preprocess] Loaded {len(df):,} rows × {df.shape[1]} cols")
    return df


def run_eda(df: pd.DataFrame) -> None:
    """Save EDA plots to data/eda_plots/."""
    EDA_DIR.mkdir(exist_ok=True)

    # 1. Churn rate distribution (bar)
    fig, ax = plt.subplots(figsize=(5, 3))
    churn_counts = df["Churn"].value_counts()
    ax.bar(churn_counts.index, churn_counts.values, color=["#3b82d4", "#e74c3c"])
    ax.set_title("Churn Distribution")
    ax.set_ylabel("Customer Count")
    for bar, val in zip(ax.patches, churn_counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
                f"{val}", ha="center", fontsize=10)
    plt.tight_layout()
    fig.savefig(EDA_DIR / "churn_distribution.png", dpi=120)
    plt.close(fig)
    churn_rate = churn_counts.get("Yes", 0) / len(df) * 100
    print(f"[eda] Churn rate: {churn_rate:.1f}%  -> class imbalance detected")

    # 2. Numeric feature distributions by churn label
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, col in zip(axes, num_cols):
        for label, color in [("No", "#3b82d4"), ("Yes", "#e74c3c")]:
            subset = df[df["Churn"] == label][col].dropna()
            ax.hist(subset, bins=40, alpha=0.6, color=color, label=label)
        ax.set_title(col)
        ax.legend()
    plt.suptitle("Numeric Features by Churn Label", y=1.02)
    plt.tight_layout()
    fig.savefig(EDA_DIR / "numeric_distributions.png", dpi=120)
    plt.close(fig)

    # 3. Correlation heatmap (numeric only)
    numeric_df = df[num_cols].apply(pd.to_numeric, errors="coerce")
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Numeric Feature Correlations")
    plt.tight_layout()
    fig.savefig(EDA_DIR / "correlation_heatmap.png", dpi=120)
    plt.close(fig)

    print(f"[eda] Plots saved to {EDA_DIR}/")


def clean_and_encode(df: pd.DataFrame) -> pd.DataFrame:
    """Return a fully numeric DataFrame with target column 'Churn'."""
    df = df.copy()

    # --- Drop surrogate key ---
    df.drop(columns=["customerID"], inplace=True)

    # --- Fix TotalCharges: blank strings → NaN → median impute ---
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    median_tc = df["TotalCharges"].median()
    n_missing = int(df["TotalCharges"].isna().sum())
    # pandas 3.0 Copy-on-Write: assign back instead of inplace=True on a slice
    df["TotalCharges"] = df["TotalCharges"].fillna(median_tc)
    print(f"[preprocess] Imputed {n_missing} missing TotalCharges with median={median_tc:.2f}")

    # --- Binary yes/no → {0, 1} ---
    for col in BINARY_YES_NO:
        if col in df.columns:
            # Some binary cols have a third value "No phone service" / "No internet service"
            # → treat those as 0 (no feature)
            df[col] = df[col].map(lambda x: 1 if str(x).strip().lower() == "yes" else 0)

    # --- gender → {0, 1} ---
    df["gender"] = df["gender"].map({"Male": 1, "Female": 0})

    # --- One-hot encode multi-class nominals ---
    df = pd.get_dummies(df, columns=NOMINAL_COLS, drop_first=False)

    # Ensure boolean dummies become int (avoids XGBoost dtype warnings)
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    print(f"[preprocess] Final shape after encoding: {df.shape}")
    return df


def split_and_save(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified 80/20 split; persist as parquet; return (train_df, test_df)."""
    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df["Churn"]
    )
    train_df.to_csv(PROCESSED_TRAIN, index=False)
    test_df.to_csv(PROCESSED_TEST, index=False)

    # Persist feature column list (needed by API to align input at inference time)
    feature_cols = [c for c in train_df.columns if c != "Churn"]
    FEATURE_COLUMNS_JSON.write_text(json.dumps(feature_cols, indent=2))

    print(f"[preprocess] Train={len(train_df):,}  Test={len(test_df):,}")
    print(f"[preprocess] Feature columns saved -> {FEATURE_COLUMNS_JSON}")
    return train_df, test_df


def run(skip_eda: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_raw = load_raw()
    if not skip_eda:
        run_eda(df_raw)
    df_clean = clean_and_encode(df_raw)
    return split_and_save(df_clean)


if __name__ == "__main__":
    run()
