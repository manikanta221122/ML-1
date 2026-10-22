import numpy as np
import pandas as pd
import pytest

from claimwise import config as C
from claimwise.data import add_fraud_supplement, generate_synthetic
from claimwise.features import risk_band


def test_synthetic_shape_and_skew():
    df = generate_synthetic(1000, seed=1)
    assert df.shape[1] == 1 + 116 + 14 + 1
    assert df[C.TARGET].skew() > 0.5


def test_fraud_supplement_rate_and_inflation():
    df = generate_synthetic(5000, seed=2)
    f = add_fraud_supplement(df, rate=0.05, seed=2)
    assert 0.02 < f[C.FRAUD_COL].mean() < 0.10
    ratio = f.loc[f[C.FRAUD_COL] == 1, C.TARGET].mean() / df.loc[f[C.FRAUD_COL] == 1, C.TARGET].mean()
    assert ratio > 1.4


def test_risk_band():
    assert risk_band(0.1)[0] == "LOW"
    assert risk_band(0.45)[0] == "MEDIUM"
    assert risk_band(0.9)[0] == "HIGH"


def test_train_and_predict(tmp_path):
    from claimwise.predict import ClaimWise
    from claimwise.train import train
    m = train(n_synthetic=3000, out_dir=tmp_path, report_dir=tmp_path, make_plots=False, verbose=False)
    assert m["fraud"]["roc_auc"] > 0.7
    cw = ClaimWise(tmp_path / "claimwise.joblib")
    df = add_fraud_supplement(generate_synthetic(200, seed=9)).head(50)
    res = cw.score(df)
    assert len(res) == 50
    assert res["fraud_probability"].between(0, 1).all()
    assert set(res["risk_band"]) <= {"LOW", "MEDIUM", "HIGH"}
