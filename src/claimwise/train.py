"""End-to-end training: severity model -> fraud model -> reports."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, classification_report, confusion_matrix,
                             mean_absolute_error, mean_squared_error, precision_recall_curve,
                             r2_score, roc_auc_score)
from sklearn.model_selection import KFold, cross_val_predict, train_test_split
from sklearn.preprocessing import OrdinalEncoder

from . import config as C
from .data import add_fraud_supplement, feature_columns, load_allstate
from .features import build_fraud_frame, encode_for_fraud
from .models import (build_anomaly_model, build_baseline_model, build_fraud_model,
                     build_severity_model)


def _reg_metrics(y, p):
    return {"MAE": float(mean_absolute_error(y, p)),
            "RMSE": float(np.sqrt(mean_squared_error(y, p))),
            "R2": float(r2_score(y, p)),
            "R2_log": float(r2_score(np.log1p(y), np.log1p(p)))}


def validate_training_df(df):
    if C.TARGET not in df.columns:
        raise ValueError(f"Training CSV needs a '{C.TARGET}' column (the claim cost).")
    cats, conts = feature_columns(df)
    if len(cats) + len(conts) < 3:
        raise ValueError("Training CSV needs Allstate-style feature columns (cat1.., cont1..).")
    return cats, conts


def train(data_path=None, n_synthetic=20000, out_dir=C.MODEL_DIR, report_dir=C.REPORT_DIR,
          make_plots=True, verbose=True, df=None, fraud_rate=C.FRAUD_RATE, max_rows=None,
          test_size=C.TEST_SIZE, max_iter=400, progress=None):
    """Train everything. `df` lets the UI pass an uploaded DataFrame directly;
    `progress(fraction, message)` is called so a UI can show a progress bar."""
    log = print if verbose else (lambda *a, **k: None)
    prog = progress or (lambda f, m: None)
    prog(0.02, "Loading data")
    out_dir, report_dir = Path(out_dir), Path(report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    if df is None:
        df, source = load_allstate(data_path, n_synthetic)
    else:
        source = "uploaded"
    validate_training_df(df)
    df = df.dropna(subset=[C.TARGET])
    df = df[df[C.TARGET] > 0]
    if max_rows and len(df) > max_rows:
        df = df.sample(max_rows, random_state=C.RANDOM_STATE).reset_index(drop=True)
    log(f"[data] source={source} rows={len(df):,}")
    if C.FRAUD_COL not in df.columns:
        df = add_fraud_supplement(df, rate=fraud_rate)
        log(f"[data] fraud supplement added, fraud rate={df[C.FRAUD_COL].mean():.2%}")

    cats, conts = feature_columns(df)
    feats = cats + conts
    tr, te = train_test_split(df, test_size=test_size, random_state=C.RANDOM_STATE,
                              stratify=df[C.FRAUD_COL])
    tr, te = tr.reset_index(drop=True), te.reset_index(drop=True)

    prog(0.10, "Training claim-cost models")
    # ---- Stage 1: claim cost (severity) -------------------------------------------------
    # Train the cost model on clean (non-fraud) claims so it learns what a *legitimate*
    # claim for this profile looks like; padded claims then stand out in the ratio.
    clean = tr[tr[C.FRAUD_COL] == 0]
    base = build_baseline_model(cats, conts).fit(clean[feats], clean[C.TARGET])
    sev = build_severity_model(cats, conts, max_iter).fit(clean[feats], clean[C.TARGET])
    te_clean = te[te[C.FRAUD_COL] == 0]
    metrics = {
        "data_source": source, "n_rows": int(len(df)),
        "baseline_ridge": _reg_metrics(te_clean[C.TARGET], base.predict(te_clean[feats])),
        "gradient_boosting": _reg_metrics(te_clean[C.TARGET], sev.predict(te_clean[feats])),
    }
    log("[severity] ridge  :", {k: round(v, 3) for k, v in metrics["baseline_ridge"].items()})
    log("[severity] boost  :", {k: round(v, 3) for k, v in metrics["gradient_boosting"].items()})

    prog(0.40, "Cross-validated cost predictions for the fraud stage")
    # ---- Stage 2: fraud -----------------------------------------------------------------
    # Out-of-fold predicted cost for ALL training claims (no leakage into the fraud model)
    oof = cross_val_predict(
        build_severity_model(cats, conts, max_iter), tr[feats], tr[C.TARGET],
        cv=KFold(5, shuffle=True, random_state=C.RANDOM_STATE), n_jobs=-1)
    prog(0.65, "Training anomaly + fraud models")
    anomaly = build_anomaly_model().fit(tr[conts])
    a_tr = -anomaly.score_samples(tr[conts])
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit(tr[cats])

    Xf_tr = encode_for_fraud(build_fraud_frame(tr, feats, oof, a_tr, tr[C.TARGET]), enc)
    fraud = build_fraud_model().fit(Xf_tr, tr[C.FRAUD_COL])

    pred_te = sev.predict(te[feats])
    a_te = -anomaly.score_samples(te[conts])
    Xf_te = encode_for_fraud(build_fraud_frame(te, feats, pred_te, a_te, te[C.TARGET]), enc)
    p = fraud.predict_proba(Xf_te)[:, 1]
    y = te[C.FRAUD_COL].to_numpy()
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    best = int(np.nanargmax(f1[:-1]))
    thr_best = float(thr[best])
    yhat = (p >= thr_best).astype(int)
    metrics["fraud"] = {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "fraud_rate_test": float(y.mean()),
        "best_f1_threshold": thr_best,
        "precision_at_best": float(prec[best]), "recall_at_best": float(rec[best]),
        "f1_at_best": float(f1[best]),
        "confusion_matrix": confusion_matrix(y, yhat).tolist(),
    }
    log("[fraud]", {k: (round(v, 3) if isinstance(v, float) else v)
                    for k, v in metrics["fraud"].items()})

    prog(0.85, "Computing feature importance")
    metrics["n_train"], metrics["n_test"] = int(len(tr)), int(len(te))
    metrics["top_features"] = _importance(sev, te_clean, feats)
    # ---- Save ---------------------------------------------------------------------------
    bundle = {"severity": sev, "fraud": fraud, "anomaly": anomaly, "fraud_encoder": enc,
              "cat_cols": cats, "cont_cols": conts, "feature_cols": feats,
              "threshold": thr_best, "defaults": _defaults(tr, cats, conts)}
    joblib.dump(bundle, out_dir / "claimwise.joblib")
    (report_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    if make_plots:
        _plots(report_dir, te_clean[C.TARGET].to_numpy(), sev.predict(te_clean[feats]),
               df[C.TARGET].to_numpy(), y, p, rec, prec)
    prog(1.0, "Done")
    log(f"[done] model -> {out_dir/'claimwise.joblib'}  reports -> {report_dir}")
    return metrics


def _importance(model, df, feats, n=1500, top=15):
    from sklearn.inspection import permutation_importance
    d = df.sample(min(n, len(df)), random_state=C.RANDOM_STATE)
    r = permutation_importance(model, d[feats], d[C.TARGET], n_repeats=2,
                               random_state=C.RANDOM_STATE, n_jobs=-1)
    order = np.argsort(r.importances_mean)[::-1][:top]
    return [{"feature": feats[i], "importance": float(r.importances_mean[i])} for i in order]


def _defaults(df, cats, conts):
    d = {c: df[c].mode().iloc[0] for c in cats}
    d.update({c: float(df[c].median()) for c in conts})
    return d


def _plots(report_dir, y_true, y_pred, all_loss, y_fraud, p, rec, prec):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].hist(all_loss, bins=80, color="#4C78A8"); ax[0].set_title("Loss (skewed)")
    ax[1].hist(np.log1p(all_loss), bins=80, color="#54A24B"); ax[1].set_title("log1p(Loss)")
    fig.tight_layout(); fig.savefig(report_dir / "target_distribution.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    ax.scatter(y_true, y_pred, s=4, alpha=.35)
    m = max(y_true.max(), y_pred.max()); ax.plot([0, m], [0, m], "r--", lw=1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Actual claim cost"); ax.set_ylabel("Predicted claim cost")
    ax.set_title("Predicted vs actual (log axes)")
    fig.tight_layout(); fig.savefig(report_dir / "pred_vs_actual.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.8, 4))
    ax.plot(rec, prec, color="#E45756"); ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Fraud precision-recall"); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(report_dir / "fraud_pr_curve.png", dpi=130); plt.close(fig)
