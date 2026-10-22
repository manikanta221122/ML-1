"""Fraud-stage feature engineering.

Key idea: a fraudulent claim is one whose amount is out of line with what the
customer/policy profile predicts.  So the fraud model sees the profile plus
  * predicted_cost   - severity-model estimate (out-of-fold in training)
  * cost_ratio       - claimed / predicted
  * log_gap          - log(claimed) - log(predicted)
  * anomaly_score    - IsolationForest score on the numeric profile
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

ENGINEERED = ["predicted_cost", "cost_ratio", "log_gap", "anomaly_score"]


def numeric_profile(df: pd.DataFrame, cont_cols):
    return df[cont_cols].to_numpy(dtype=float)


def build_fraud_frame(df: pd.DataFrame, feature_cols, predicted_cost, anomaly_score,
                      claimed=None) -> pd.DataFrame:
    X = df[feature_cols].copy()
    X["predicted_cost"] = np.asarray(predicted_cost, dtype=float)
    if claimed is None:
        claimed = X["predicted_cost"].to_numpy()
    claimed = np.asarray(claimed, dtype=float)
    X["cost_ratio"] = claimed / np.maximum(X["predicted_cost"], 1.0)
    X["log_gap"] = np.log1p(claimed) - np.log1p(X["predicted_cost"])
    X["anomaly_score"] = np.asarray(anomaly_score, dtype=float)
    return X


def encode_for_fraud(X: pd.DataFrame, encoder) -> pd.DataFrame:
    """Ordinal-encode categoricals so the fraud model can digest them."""
    Z = X.copy()
    cats = list(encoder.feature_names_in_)
    Z[cats] = encoder.transform(Z[cats])
    return Z


def risk_band(p: float, low: float = 0.30, high: float = 0.60):
    """Map a fraud probability to (band, action) using adjustable thresholds."""
    if p < low:
        return "LOW", "Auto-approve"
    if p < high:
        return "MEDIUM", "Manual review"
    return "HIGH", "Refer to Special Investigation Unit"
