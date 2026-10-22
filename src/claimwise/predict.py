"""Inference: score claims for cost and fraud risk."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import config as C
from .features import build_fraud_frame, encode_for_fraud, risk_band


class ClaimWise:
    def __init__(self, model_path=C.MODEL_DIR / "claimwise.joblib"):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"{model_path} not found - run `python -m claimwise train` first")
        self.b = joblib.load(model_path)

    def _complete(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill any missing feature columns with training defaults."""
        out = df.copy()
        for c in self.b["feature_cols"]:
            if c not in out.columns:
                out[c] = self.b["defaults"][c]
        return out

    def score(self, df: pd.DataFrame, claimed_col: str = C.TARGET,
              low: float = 0.30, high: float = 0.60) -> pd.DataFrame:
        """Return predicted cost, fraud probability, risk band and action.

        `claimed_col` is the amount being claimed; if absent, fraud is judged on
        profile anomaly only (cost ratio defaults to 1).
        """
        d = self._complete(df).reset_index(drop=True)
        feats = self.b["feature_cols"]
        pred = self.b["severity"].predict(d[feats])
        anomaly = -self.b["anomaly"].score_samples(d[self.b["cont_cols"]])
        claimed = d[claimed_col].to_numpy(float) if claimed_col in d.columns else None
        Xf = encode_for_fraud(build_fraud_frame(d, feats, pred, anomaly, claimed),
                              self.b["fraud_encoder"])
        p = self.b["fraud"].predict_proba(Xf)[:, 1]
        res = pd.DataFrame({
            C.ID_COL: d[C.ID_COL] if C.ID_COL in d.columns else np.arange(len(d)),
            "predicted_cost": np.round(pred, 2),
        })
        if claimed is not None:
            res["claimed_amount"] = np.round(claimed, 2)
            res["cost_ratio"] = np.round(claimed / np.maximum(pred, 1), 2)
        res["fraud_probability"] = np.round(p, 4)
        bands = [risk_band(x, low, high) for x in p]
        res["risk_band"] = [b[0] for b in bands]
        res["recommended_action"] = [b[1] for b in bands]
        return res
