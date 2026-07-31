"""
tests/test_preprocessing.py
────────────────────────────
Unit tests for data/preprocess.py — clean_and_encode + split logic.
"""

import json
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.preprocess import clean_and_encode, BINARY_YES_NO, NOMINAL_COLS


# ---------------------------------------------------------------------------
# Minimal sample DataFrame mimicking the raw CSV
# ---------------------------------------------------------------------------

SAMPLE_RAW_CSV = """customerID,gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,MultipleLines,InternetService,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,StreamingMovies,Contract,PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges,Churn
7590-VHVEG,Female,0,Yes,No,1,No,No phone service,DSL,No,Yes,No,No,No,No,Month-to-month,Yes,Electronic check,29.85,29.85,No
5575-GNVDE,Male,0,No,No,34,Yes,No,DSL,Yes,No,Yes,No,No,No,One year,No,Mailed check,56.95,1889.5,No
3668-QPYBK,Male,0,No,No,2,Yes,No,DSL,Yes,Yes,No,No,No,No,Month-to-month,Yes,Mailed check,53.85,108.15,Yes
7795-CFOCW,Male,0,No,No,45,No,No phone service,DSL,Yes,No,Yes,Yes,No,No,One year,No,Bank transfer (automatic),42.3,1840.75,No
9237-HQITU,Female,0,No,No,2,Yes,No,Fiber optic,No,No,No,No,No,No,Month-to-month,Yes,Electronic check,70.7,151.65,Yes
"""


@pytest.fixture
def raw_df():
    return pd.read_csv(StringIO(SAMPLE_RAW_CSV))


# ---------------------------------------------------------------------------
# clean_and_encode tests
# ---------------------------------------------------------------------------

def test_customer_id_dropped(raw_df):
    encoded = clean_and_encode(raw_df)
    assert "customerID" not in encoded.columns


def test_churn_column_binary(raw_df):
    encoded = clean_and_encode(raw_df)
    assert set(encoded["Churn"].unique()).issubset({0, 1})


def test_gender_encoded(raw_df):
    encoded = clean_and_encode(raw_df)
    assert set(encoded["gender"].unique()).issubset({0, 1})
    # Male → 1, Female → 0
    assert encoded.loc[encoded.index[1], "gender"] == 1  # row index 1 = Male


def test_binary_yes_no_encoded(raw_df):
    encoded = clean_and_encode(raw_df)
    for col in BINARY_YES_NO:
        if col in encoded.columns:
            assert set(encoded[col].unique()).issubset({0, 1}), \
                f"{col} has non-binary values: {encoded[col].unique()}"


def test_nominal_cols_one_hotted(raw_df):
    encoded = clean_and_encode(raw_df)
    for col in NOMINAL_COLS:
        # Original column must be gone
        assert col not in encoded.columns, f"{col} not one-hot encoded"
        # At least one dummy column must exist
        dummies = [c for c in encoded.columns if c.startswith(col + "_")]
        assert len(dummies) > 0, f"No dummy columns for {col}"


def test_total_charges_nan_imputed():
    """A blank TotalCharges string should be imputed (not NaN) after encoding."""
    csv_data = SAMPLE_RAW_CSV.replace("29.85,No", " ,No", 1)
    df = pd.read_csv(StringIO(csv_data))
    encoded = clean_and_encode(df)
    assert encoded["TotalCharges"].isna().sum() == 0


def test_no_string_columns_remain(raw_df):
    """After encoding, no object dtype columns should remain (all numeric)."""
    encoded = clean_and_encode(raw_df)
    obj_cols = encoded.select_dtypes(include="object").columns.tolist()
    assert obj_cols == [], f"Non-numeric columns remain: {obj_cols}"


def test_output_shape_reasonable(raw_df):
    """Should have more columns than input (due to one-hot) but fewer rows."""
    encoded = clean_and_encode(raw_df)
    # 5 rows in, 5 rows out
    assert len(encoded) == 5
    # More columns due to dummies
    assert encoded.shape[1] > raw_df.shape[1]
