#!/usr/bin/env python3
"""One command for the complete study: EDA -> train -> evaluate -> federated learning -> report.

The pipeline automatically detects the canonical UNSW-NB15 train/test partitions by row
count when the two filenames have accidentally been swapped. The default research
protocol also removes exact training duplicates and exact train/test overlaps before
model fitting; use --protocol benchmark for a literature-comparison run that preserves
all resolved partition rows.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src import eda, evaluate, federated, report, train  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--tune", action="store_true", help="randomized search for XGBoost")
    ap.add_argument("--permutation", action="store_true", help="permutation importance")
    ap.add_argument("--shap", action="store_true", help="SHAP importance (pip install shap)")
    ap.add_argument("--update-readme", action="store_true", help="write this run's results into README.md")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--protocol", choices=["research", "benchmark"], default="research")
    ap.add_argument("--corr-threshold", type=float, default=0.95)
    ap.add_argument("--target-fpr", type=float, default=0.05, help="OOF FPR cap for primary binary threshold selection.")
    ap.add_argument("--no-threshold-tuning", action="store_true", help="Use threshold 0.5 instead of OOF FPR-constrained threshold.")
    ap.add_argument("--train-file", default=None)
    ap.add_argument("--test-file", default=None)
    a = ap.parse_args()

    common = [
        "--seed", str(a.seed),
        "--protocol", a.protocol,
        "--corr-threshold", str(a.corr_threshold),
        "--target-fpr", str(a.target_fpr),
    ]
    if a.demo:
        common.append("--demo")
    if a.quick:
        common.append("--quick")
    if a.no_threshold_tuning:
        common.append("--no-threshold-tuning")
    if a.data_dir:
        common += ["--data-dir", a.data_dir]
    if a.train_file:
        common += ["--train-file", a.train_file]
    if a.test_file:
        common += ["--test-file", a.test_file]

    steps = [
        ("1/5 dataset analysis", lambda: eda.main(common)),
        ("2/5 training", lambda: train.main(common + (["--tune"] if a.tune else []))),
        ("3/5 evaluation", lambda: evaluate.main(common + (["--permutation"] if a.permutation else [])
                                                + (["--shap"] if a.shap else []))),
        ("4/5 federated learning experiment", lambda: federated.main(common)),
        ("5/5 report", lambda: report.main(common + (["--update-readme"] if a.update_readme else []))),
    ]
    for title, fn in steps:
        print(f"\n===== {title} =====", flush=True)
        fn()


if __name__ == "__main__":
    main()
