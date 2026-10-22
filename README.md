# ClaimWise
**AI-Based Insurance Claim Cost Prediction and Fraud Detection System**

Two-stage ML system built on the Allstate Claims Severity data
(https://www.kaggle.com/c/allstate-claims-severity/data):

1. **Claim cost model** - gradient boosting on `log1p(loss)` (fixes the heavy right skew),
   trained on legitimate claims so it learns what a *fair* claim for a profile looks like.
   A Ridge model is trained as a baseline for comparison.
2. **Fraud model** - gradient boosting classifier fed the profile plus engineered signals:
   `predicted_cost`, `cost_ratio` (claimed / predicted), `log_gap`, and an
   IsolationForest `anomaly_score`. Predicted cost is **out-of-fold** to avoid leakage.
   Output: fraud probability -> LOW / MEDIUM / HIGH band -> recommended action.

## Important: the fraud supplement
The Allstate dataset has **no fraud labels** ("Need fraud supplement"). `data.add_fraud_supplement`
simulates ~5% fraud: risk-profile-dependent probability, with fraudulent claim amounts padded
1.5x-3.5x. So fraud metrics measure how well the model finds *padded claims in this simulation*,
not real-world fraud. If you obtain real SIU/fraud labels, add an `is_fraud` column to the CSV
and the supplement step is skipped automatically.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .            # or: export PYTHONPATH=src
```

## Data
Download `train.csv` from the Kaggle page (login + accept rules; or
`kaggle competitions download -c allstate-claims-severity`) and place it at
`data/raw/train.csv`. **No data? No problem** - training falls back to a synthetic
Allstate-shaped dataset (same 116 cat + 14 cont columns) so everything runs end to end.

## Run (UI workflow)
```bash
python -m claimwise app
```
1. **🧠 Train model** - upload Kaggle `train.csv` (or use `data/raw/train.csv` / synthetic demo data),
   tune options (rows, boosting iterations, test split, simulated fraud rate), click **Train**.
2. **📂 Predict from CSV** - upload a claims CSV (Allstate columns; pick which column is the claimed
   amount), click **Run predictions**.
3. Filter by risk band / probability, view charts, download all or HIGH-risk-only results.
Extras: **🔍 Single claim** what-if tool, **📊 Model report** (metrics, confusion matrix, feature
importance, plots), and sidebar sliders to set the LOW / HIGH fraud thresholds live.

## Run (command line)
```bash
python -m claimwise train [--data path/train.csv]
python -m claimwise predict claims.csv --output predictions.csv
pytest -q
```
The CSV may omit `loss`; if present it is treated as the claimed amount.

## Layout
```
src/claimwise/  config.py data.py models.py features.py train.py predict.py cli.py
app.py          Streamlit dashboard (single claim, batch CSV, model report)
tests/          pytest suite
reports/        metrics.json + plots written by training
```

## Results on the synthetic demo data (15k rows)
Fraud ROC-AUC ~0.86, PR-AUC ~0.39 (5% base rate). Cost model R2 on log scale ~0.29 - the synthetic
noise floor is high by design; expect different numbers on the real Allstate data
(Kaggle metric there is MAE; top solutions reach ~1100).

## Ideas to extend
Optuna tuning, XGBoost/LightGBM/CatBoost, SHAP explanations, target encoding of high-cardinality
categoricals, FastAPI service, drift monitoring.
