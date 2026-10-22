"""Command line: python -m claimwise <command>"""
import argparse
import sys

import pandas as pd

from . import config as C


def main(argv=None):
    ap = argparse.ArgumentParser(prog="claimwise", description="ClaimWise claim cost + fraud")
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate-data", help="write a synthetic Allstate-like CSV")
    g.add_argument("--rows", type=int, default=20000)

    t = sub.add_parser("train", help="train severity + fraud models")
    t.add_argument("--data", help="path to Kaggle train.csv (default data/raw/train.csv)")
    t.add_argument("--rows", type=int, default=20000, help="synthetic rows if no real data")

    p = sub.add_parser("predict", help="score a CSV of claims")
    p.add_argument("input"); p.add_argument("--output", default="predictions.csv")

    sub.add_parser("app", help="launch the Streamlit dashboard")

    a = ap.parse_args(argv)
    if a.cmd == "generate-data":
        from .data import generate_synthetic
        C.RAW_DIR.mkdir(parents=True, exist_ok=True)
        generate_synthetic(a.rows).to_csv(C.SYNTH_CSV, index=False)
        print("wrote", C.SYNTH_CSV)
    elif a.cmd == "train":
        from .train import train
        train(a.data, a.rows)
    elif a.cmd == "predict":
        from .predict import ClaimWise
        res = ClaimWise().score(pd.read_csv(a.input))
        res.to_csv(a.output, index=False)
        print(res.head(10).to_string(index=False)); print("saved ->", a.output)
    elif a.cmd == "app":
        import subprocess
        app = C.ROOT / "app.py"
        sys.exit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(app)]))


if __name__ == "__main__":
    main()
