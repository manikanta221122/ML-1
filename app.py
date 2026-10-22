"""ClaimWise dashboard.   Run:  python -m claimwise app     (or: streamlit run app.py)

Workflow:  1) Train  ->  2) Upload claims CSV  ->  3) Get cost + fraud predictions
"""
import io
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import pandas as pd
import streamlit as st

from claimwise import config as C
from claimwise.data import add_fraud_supplement, feature_columns, generate_synthetic
from claimwise.features import risk_band
from claimwise.predict import ClaimWise
from claimwise.train import train

MODEL_FILE = C.MODEL_DIR / "claimwise.joblib"
METRICS_FILE = C.REPORT_DIR / "metrics.json"

st.set_page_config(page_title="ClaimWise", page_icon="🛡️", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 1.6rem;}
.cw-hero {padding: 1.1rem 1.4rem; border-radius: 14px;
          background: linear-gradient(120deg,#1f3b73,#2b6cb0 60%,#38a3a5); color: #fff;}
.cw-hero h1 {margin: 0; font-size: 2rem;} .cw-hero p {margin: .2rem 0 0; opacity: .9;}
.cw-step {border: 1px solid rgba(128,128,128,.3); border-radius: 12px; padding: .9rem 1rem; height: 100%;}
</style>""", unsafe_allow_html=True)


# ---------------------------------------------------------------- helpers
def model_ready() -> bool:
    return MODEL_FILE.exists()


@st.cache_resource(show_spinner=False)
def load_model(mtime: float):
    return ClaimWise(MODEL_FILE)


def get_model() -> ClaimWise:
    return load_model(MODEL_FILE.stat().st_mtime)


def load_metrics():
    import json
    return json.loads(METRICS_FILE.read_text()) if METRICS_FILE.exists() else None


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode()


def need_model():
    st.warning("No trained model yet. Go to **🧠 Train model** first.")
    st.stop()


def show_metrics(m: dict):
    st.subheader("Claim cost model (held-out legitimate claims)")
    g, b = m["gradient_boosting"], m["baseline_ridge"]
    c = st.columns(4)
    c[0].metric("MAE", f"{g['MAE']:,.0f}", f"{g['MAE']-b['MAE']:+,.0f} vs Ridge", delta_color="inverse")
    c[1].metric("RMSE", f"{g['RMSE']:,.0f}")
    c[2].metric("R² (log scale)", f"{g['R2_log']:.3f}")
    c[3].metric("R² (raw)", f"{g['R2']:.3f}")
    st.caption(f"Ridge baseline: MAE {b['MAE']:,.0f}, R² (log) {b['R2_log']:.3f}")
    f = m["fraud"]
    st.subheader("Fraud model (held-out test set)")
    c = st.columns(5)
    c[0].metric("ROC-AUC", f"{f['roc_auc']:.3f}")
    c[1].metric("PR-AUC", f"{f['pr_auc']:.3f}")
    c[2].metric("Precision", f"{f['precision_at_best']:.1%}")
    c[3].metric("Recall", f"{f['recall_at_best']:.1%}")
    c[4].metric("F1", f"{f['f1_at_best']:.3f}")
    (tn, fp), (fn, tp) = f["confusion_matrix"]
    cm = pd.DataFrame([[tn, fp], [fn, tp]], index=["Actual legit", "Actual fraud"],
                      columns=["Predicted legit", "Predicted fraud"])
    left, right = st.columns([1, 1])
    left.write("**Confusion matrix** (at best-F1 threshold "
               f"{f['best_f1_threshold']:.2f})")
    left.dataframe(cm)
    if m.get("top_features"):
        tf = pd.DataFrame(m["top_features"]).set_index("feature")
        right.write("**Top features driving claim cost** (permutation importance)")
        right.bar_chart(tf, height=260)
    imgs = [("target_distribution", "Target skew & log transform"),
            ("pred_vs_actual", "Predicted vs actual"), ("fraud_pr_curve", "Fraud precision-recall")]
    cols = st.columns(3)
    for col, (name, cap) in zip(cols, imgs):
        p = C.REPORT_DIR / f"{name}.png"
        if p.exists():
            col.image(str(p), caption=cap)


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("## 🛡️ ClaimWise")
    page = st.radio("Navigate", ["🏠 Home", "🧠 Train model", "📂 Predict from CSV",
                                 "🔍 Single claim", "📊 Model report"], label_visibility="collapsed")
    st.divider()
    if model_ready():
        ts = datetime.fromtimestamp(MODEL_FILE.stat().st_mtime).strftime("%d %b %Y %H:%M")
        st.success(f"Model ready\n\nTrained {ts}")
    else:
        st.error("No model trained yet")
    st.markdown("**Fraud risk thresholds**")
    low = st.slider("LOW below", 0.05, 0.90, 0.30, 0.05)
    high = st.slider("HIGH from", 0.10, 0.99, 0.60, 0.05)
    if high <= low:
        st.warning("HIGH should be above LOW")
    st.caption("Probability < LOW → auto-approve · between → manual review · ≥ HIGH → investigate")

st.markdown('<div class="cw-hero"><h1>🛡️ ClaimWise</h1>'
            '<p>AI-based insurance claim cost prediction &amp; fraud detection</p></div>',
            unsafe_allow_html=True)
st.write("")

# ---------------------------------------------------------------- HOME
if page == "🏠 Home":
    c1, c2, c3 = st.columns(3)
    c1.markdown('<div class="cw-step"><h4>① Train</h4>Upload the Allstate <code>train.csv</code> '
                '(or use demo data). The app trains a claim-cost model and a fraud detector.</div>',
                unsafe_allow_html=True)
    c2.markdown('<div class="cw-step"><h4>② Upload claims</h4>Give it a CSV of new claims '
                '(same columns as the dataset). A <code>loss</code> column is treated as the claimed amount.</div>',
                unsafe_allow_html=True)
    c3.markdown('<div class="cw-step"><h4>③ Get decisions</h4>For every claim: fair cost estimate, '
                'fraud probability, risk band and recommended action. Download the results.</div>',
                unsafe_allow_html=True)
    st.write("")
    m = load_metrics()
    if m:
        st.subheader("Current model")
        c = st.columns(4)
        c[0].metric("Data source", m["data_source"])
        c[1].metric("Training rows", f"{m['n_train']:,}")
        c[2].metric("Cost MAE", f"{m['gradient_boosting']['MAE']:,.0f}")
        c[3].metric("Fraud ROC-AUC", f"{m['fraud']['roc_auc']:.3f}")
    else:
        st.info("Start with **🧠 Train model** in the sidebar.")
    with st.expander("About the fraud labels"):
        st.write("The Allstate dataset has no fraud labels, so training adds a **simulated fraud "
                 "supplement** (padded claims). If your CSV already has an `is_fraud` column (0/1), "
                 "it is used instead.")

# ---------------------------------------------------------------- TRAIN
elif page == "🧠 Train model":
    st.header("Train the model")
    src = st.radio("Training data", ["Upload training CSV", "Use data/raw/train.csv",
                                     "Synthetic demo data"], horizontal=True)
    train_df = None
    if src == "Upload training CSV":
        up = st.file_uploader("Allstate train.csv (needs `loss`, cat*, cont*)", type="csv", key="train_up")
        if up:
            train_df = pd.read_csv(up)
    elif src == "Use data/raw/train.csv":
        if C.TRAIN_CSV.exists():
            train_df = pd.read_csv(C.TRAIN_CSV)
        else:
            st.warning("data/raw/train.csv not found - upload a file or use the demo data.")
    else:
        n_syn = st.slider("Synthetic rows", 3000, 60000, 15000, 1000)
        train_df = generate_synthetic(n_syn)
        st.caption("Allstate-shaped synthetic data - good for a quick demo only.")

    if train_df is not None:
        cats, conts = feature_columns(train_df)
        c = st.columns(4)
        c[0].metric("Rows", f"{len(train_df):,}")
        c[1].metric("Categorical cols", len(cats))
        c[2].metric("Continuous cols", len(conts))
        c[3].metric("Has loss", "yes" if C.TARGET in train_df.columns else "NO")
        with st.expander("Preview data"):
            st.dataframe(train_df.head(20), width="stretch")
        if C.TARGET in train_df.columns:
            with st.expander("Claim cost distribution"):
                l = train_df[C.TARGET].clip(upper=train_df[C.TARGET].quantile(.99))
                h, edges = np.histogram(l, bins=40)
                st.bar_chart(pd.DataFrame({"claims": h}, index=np.round(edges[:-1]).astype(int)))
                st.caption(f"Skewness {train_df[C.TARGET].skew():.2f} - this is why the model "
                           "predicts log(loss).")

        with st.expander("⚙️ Training options", expanded=True):
            o1, o2 = st.columns(2)
            max_rows = o1.slider("Max rows to train on (speed vs accuracy)", 2000,
                                 max(len(train_df), 2001), min(len(train_df), 40000), 1000)
            max_iter = o1.slider("Boosting iterations", 100, 800, 300, 50)
            test_size = o2.slider("Test split", 0.1, 0.4, 0.2, 0.05)
            fraud_rate = o2.slider("Simulated fraud rate (if no is_fraud column)", 0.01, 0.15, 0.05, 0.01)

        if st.button("🚀 Train model", type="primary"):
            bar = st.progress(0.0, text="Starting")
            try:
                metrics = train(df=train_df, max_rows=max_rows, max_iter=max_iter,
                                test_size=test_size, fraud_rate=fraud_rate, verbose=False,
                                progress=lambda f, msg: bar.progress(min(f, 1.0), text=msg))
                load_model.clear()
                st.success("Training complete - the model is saved. Next: **📂 Predict from CSV**.")
                show_metrics(metrics)
            except Exception as e:
                bar.empty()
                st.error(f"Training failed: {e}")
    elif src == "Upload training CSV":
        st.info("Upload your training CSV to continue.")

# ---------------------------------------------------------------- PREDICT
elif page == "📂 Predict from CSV":
    st.header("Predict claims from a CSV")
    if not model_ready():
        need_model()
    cw = get_model()
    sample = add_fraud_supplement(generate_synthetic(200, seed=7), rate=0.12, seed=7).drop(
        columns=[C.FRAUD_COL])  # demo claims, some padded
    st.download_button("⬇️ Download a sample input CSV", to_csv_bytes(sample), "sample_claims.csv")
    up = st.file_uploader("Upload claims CSV", type="csv", key="pred_up")
    if up:
        df = pd.read_csv(up)
        st.caption(f"{len(df):,} claims loaded")
        claimed_opts = ["(none)"] + [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
                                     and not c.startswith("cont")]
        default = claimed_opts.index(C.TARGET) if C.TARGET in claimed_opts else 0
        claimed_col = st.selectbox("Column holding the claimed amount", claimed_opts, index=default,
                                   help="Used to compare the claim with the predicted fair cost.")
        if st.button("🔮 Run predictions", type="primary"):
            with st.spinner("Scoring claims..."):
                work = df.copy()
                col = None if claimed_col == "(none)" else claimed_col
                if col and col != C.TARGET:
                    work[C.TARGET] = work[col]
                if col is None and C.TARGET in work.columns:
                    work = work.drop(columns=[C.TARGET])
                res = cw.score(work, low=low, high=high)
                extra = [c for c in df.columns if c not in res.columns
                         and not c.startswith(("cat", "cont"))]
                st.session_state["pred"] = pd.concat([res, df[extra].reset_index(drop=True)], axis=1)
    res = st.session_state.get("pred")
    if res is not None:
        # re-band with current sidebar thresholds so the sliders act live
        res = res.copy()
        bands = [risk_band(p, low, high) for p in res["fraud_probability"]]
        res["risk_band"] = [b[0] for b in bands]
        res["recommended_action"] = [b[1] for b in bands]

        c = st.columns(5)
        c[0].metric("Claims", f"{len(res):,}")
        c[1].metric("Total predicted cost", f"{res['predicted_cost'].sum():,.0f}")
        c[2].metric("🔴 High risk", int((res.risk_band == "HIGH").sum()))
        c[3].metric("🟠 Medium", int((res.risk_band == "MEDIUM").sum()))
        c[4].metric("🟢 Low", int((res.risk_band == "LOW").sum()))
        if "claimed_amount" in res:
            flagged = res[res.risk_band == "HIGH"]
            st.info(f"Claimed amount on HIGH-risk claims: **{flagged.claimed_amount.sum():,.0f}** "
                    f"({flagged.claimed_amount.sum() / max(res.claimed_amount.sum(), 1):.1%} of total).")

        t1, t2, t3 = st.tabs(["Results table", "Charts", "Top suspicious"])
        with t1:
            f1, f2 = st.columns([2, 1])
            sel = f1.multiselect("Risk band", ["HIGH", "MEDIUM", "LOW"], default=["HIGH", "MEDIUM", "LOW"])
            minp = f2.slider("Min fraud probability", 0.0, 1.0, 0.0, 0.05)
            view = res[res.risk_band.isin(sel) & (res.fraud_probability >= minp)]
            st.dataframe(view, width="stretch", height=420, column_config={
                "fraud_probability": st.column_config.ProgressColumn(
                    "fraud_probability", min_value=0.0, max_value=1.0, format="%.2f"),
                "predicted_cost": st.column_config.NumberColumn(format="%.0f"),
            })
            d1, d2 = st.columns(2)
            d1.download_button("⬇️ Download filtered results", to_csv_bytes(view), "claimwise_predictions.csv")
            d2.download_button("⬇️ Download HIGH-risk only",
                               to_csv_bytes(res[res.risk_band == "HIGH"]), "claimwise_high_risk.csv")
        with t2:
            a, b = st.columns(2)
            a.write("**Claims by risk band**")
            a.bar_chart(res["risk_band"].value_counts().reindex(["LOW", "MEDIUM", "HIGH"]).fillna(0))
            h, edges = np.histogram(res["fraud_probability"], bins=20, range=(0, 1))
            b.write("**Fraud probability distribution**")
            b.bar_chart(pd.DataFrame({"claims": h}, index=np.round(edges[:-1], 2)))
            if "claimed_amount" in res:
                st.write("**Claimed vs predicted fair cost** (points far above the diagonal are suspicious)")
                st.scatter_chart(res, x="predicted_cost", y="claimed_amount", color="risk_band", height=380)
        with t3:
            n = st.slider("How many", 5, 50, 10)
            st.dataframe(res.sort_values("fraud_probability", ascending=False).head(n), width="stretch")

# ---------------------------------------------------------------- SINGLE
elif page == "🔍 Single claim":
    st.header("Score a single claim")
    if not model_ready():
        need_model()
    cw = get_model()
    d = dict(cw.b["defaults"])
    st.caption("Tune the main drivers; every other feature uses its training-set typical value.")
    c1, c2, c3 = st.columns(3)
    with c1:
        for k in ["cont2", "cont7", "cont3", "cont12"]:
            d[k] = st.slider(k, 0.0, 1.0, float(d[k]), 0.01)
    with c2:
        for k in ["cont11", "cont14", "cont5"]:
            d[k] = st.slider(k, 0.0, 1.0, float(d[k]), 0.01)
    with c3:
        d["cat1"] = st.selectbox("cat1", ["A", "B"])
        d["cat12"] = st.selectbox("cat12", ["A", "B"])
        d["cat79"] = st.selectbox("cat79", ["A", "B"])
        d["cat80"] = st.selectbox("cat80", ["A", "B", "C", "D"])
    fair = float(cw.score(pd.DataFrame([d]), low=low, high=high)["predicted_cost"].iloc[0])
    mult = st.slider("Claimed amount as a multiple of the fair cost", 0.2, 4.0, 1.0, 0.1)
    d[C.TARGET] = fair * mult
    r = cw.score(pd.DataFrame([d]), low=low, high=high).iloc[0]
    m = st.columns(4)
    m[0].metric("Predicted fair cost", f"{r.predicted_cost:,.0f}")
    m[1].metric("Claimed amount", f"{r.claimed_amount:,.0f}")
    m[2].metric("Fraud probability", f"{r.fraud_probability:.1%}")
    m[3].metric("Risk band", r.risk_band)
    {"LOW": st.success, "MEDIUM": st.warning, "HIGH": st.error}[r.risk_band](
        f"Recommended action: **{r.recommended_action}**")

# ---------------------------------------------------------------- REPORT
elif page == "📊 Model report":
    st.header("Model report")
    m = load_metrics()
    if not m:
        need_model()
    st.caption(f"Data source: {m['data_source']} · {m['n_rows']:,} rows · "
               f"train {m['n_train']:,} / test {m['n_test']:,}")
    show_metrics(m)
