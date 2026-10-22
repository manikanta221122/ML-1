"""Central configuration."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

TRAIN_CSV = RAW_DIR / "train.csv"          # Kaggle Allstate train.csv
SYNTH_CSV = RAW_DIR / "synthetic_train.csv"

ID_COL = "id"
TARGET = "loss"           # claim cost (severity)
FRAUD_COL = "is_fraud"    # supplemental fraud label

RANDOM_STATE = 42
TEST_SIZE = 0.2
FRAUD_RATE = 0.05

# Fraud decision bands on the fraud probability: (low, high, band, action)
BANDS = [(0.0, 0.30, "LOW", "Auto-approve"),
         (0.30, 0.60, "MEDIUM", "Manual review"),
         (0.60, 1.01, "HIGH", "Refer to Special Investigation Unit")]
