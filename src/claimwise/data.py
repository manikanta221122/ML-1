"""Data loading, synthetic Allstate-like generator, and the fraud supplement.

The Kaggle Allstate Claims Severity data has 116 categorical (cat1..cat116) and
14 continuous (cont1..cont14) anonymised features and a `loss` target, but NO
fraud label.  The "fraud supplement" below injects a simulated, documented
fraud label so a supervised detector can be trained and evaluated.  Replace
`add_fraud_supplement` with real SIU labels if you ever get them.
"""
from __future__ import annotations

import string
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C


def _levels(k: int) -> list[str]:
    letters = list(string.ascii_uppercase)
    out = letters + [a + b for a in letters for b in letters]
    return out[:k]


def generate_synthetic(n: int = 20000, seed: int = C.RANDOM_STATE) -> pd.DataFrame:
    """Create an Allstate-shaped dataset (same columns, right-skewed loss)."""
    rng = np.random.default_rng(seed)
    cols = {C.ID_COL: np.arange(1, n + 1)}
    for i in range(1, 117):
        if i <= 72:
            p = rng.uniform(0.05, 0.5)
            cols[f"cat{i}"] = np.where(rng.random(n) < p, "B", "A")
        elif i <= 88:
            lv = _levels(4)
            cols[f"cat{i}"] = rng.choice(lv, n, p=[.7, .2, .07, .03])
        elif i <= 108:
            lv = _levels(int(rng.integers(5, 16)))
            w = rng.dirichlet(np.ones(len(lv)) * 0.5)
            cols[f"cat{i}"] = rng.choice(lv, n, p=w)
        else:
            lv = _levels(int(rng.integers(15, 60)))
            w = rng.dirichlet(np.ones(len(lv)) * 0.3)
            cols[f"cat{i}"] = rng.choice(lv, n, p=w)
    for j in range(1, 15):
        cols[f"cont{j}"] = rng.beta(rng.uniform(1.5, 4), rng.uniform(1.5, 4), n)
    df = pd.DataFrame(cols)

    signal = (
        0.9 * (df["cat80"] == "B").astype(float)
        + 0.7 * (df["cat79"] == "B").astype(float)
        + 0.5 * (df["cat1"] == "B").astype(float)
        + 0.4 * (df["cat12"] == "B").astype(float)
        + 0.35 * df["cat101"].isin(_levels(4)[:2]).astype(float)
        + 1.4 * df["cont2"] + 0.9 * df["cont7"] - 0.8 * df["cont3"]
        + 0.6 * df["cont12"] * df["cont11"]
    )
    log_loss = 6.6 + 0.55 * signal + rng.normal(0, 0.55, n)
    df[C.TARGET] = np.round(np.exp(log_loss), 2)
    return df


def load_allstate(path: str | Path | None = None, n_synthetic: int = 20000):
    """Load Kaggle train.csv if present, else fall back to synthetic data.

    Returns (df, source) where source is 'kaggle' or 'synthetic'.
    """
    path = Path(path) if path else C.TRAIN_CSV
    if path.exists():
        return pd.read_csv(path), "kaggle"
    if C.SYNTH_CSV.exists():
        return pd.read_csv(C.SYNTH_CSV), "synthetic"
    df = generate_synthetic(n_synthetic)
    C.RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(C.SYNTH_CSV, index=False)
    return df, "synthetic"


def add_fraud_supplement(df: pd.DataFrame, rate: float = C.FRAUD_RATE,
                         seed: int = C.RANDOM_STATE) -> pd.DataFrame:
    """Simulate a fraud label and the padded claims it implies.

    Fraud probability depends on a risk profile.  For fraudulent rows the
    claimed amount is inflated 1.5x-3.5x (the classic 'padded claim' pattern),
    which the detector learns to spot as a claim far above what the profile
    predicts.
    """
    rng = np.random.default_rng(seed)
    out = df.copy()
    n = len(out)
    risk = (
        1.0 * (out["cat80"] == "B").astype(float)
        + 0.8 * (out["cat1"] == "B").astype(float) * out["cat100"].isin(["A", "B"]).astype(float)
        + 1.2 * (out["cont14"] > 0.8).astype(float)
        + 0.8 * (out["cont5"] > 0.85).astype(float)
    ).to_numpy()
    w = 0.4 + risk
    p = np.clip(rate * w / w.mean(), 0.003, 0.6)
    fraud = rng.random(n) < p
    out[C.FRAUD_COL] = fraud.astype(int)
    inflate = rng.uniform(1.5, 3.5, n)
    out.loc[fraud, C.TARGET] = np.round(out.loc[fraud, C.TARGET] * inflate[fraud], 2)
    return out


def feature_columns(df: pd.DataFrame):
    cats = [c for c in df.columns if c.startswith("cat")]
    conts = [c for c in df.columns if c.startswith("cont")]
    return cats, conts
