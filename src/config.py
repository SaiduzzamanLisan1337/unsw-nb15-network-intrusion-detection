"""Central experiment configuration and CLI helpers."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SEED = 42
TRAIN_FILE_STEM = "UNSW_NB15_training-set"
TEST_FILE_STEM = "UNSW_NB15_testing-set"

# Official predefined UNSW-NB15 partition sizes.  Some mirrors/users accidentally
# swap the filenames; data_loader.py resolves roles by these sizes when both match.
EXPECTED_TRAIN_ROWS = 175_341
EXPECTED_TEST_ROWS = 82_332

MODEL_NAMES = ["logistic_regression", "random_forest", "xgboost"]
PRETTY = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "xgboost_tuned": "XGBoost (tuned)",
}

TARGET_BINARY = "label"          # 0 = Normal, 1 = Attack
TARGET_MULTI = "attack_cat"
NORMAL_CLASS = "Normal"
CATEGORICAL = ["proto", "service", "state"]
LABEL_COLUMNS = [TARGET_BINARY, TARGET_MULTI]
IDENTIFIER_COLUMNS = ["id", "srcip", "dstip", "sport", "dsport", "stime", "ltime"]

LOG_COLUMNS = [
    "dur", "spkts", "dpkts", "sbytes", "dbytes", "rate", "sload", "dload",
    "sloss", "dloss", "sinpkt", "dinpkt", "sjit", "djit", "stcpb", "dtcpb",
    "smean", "dmean", "response_body_len",
]

MIN_FREQUENCY = 100
DEFAULT_PROTOCOL = "research"
DEFAULT_CORR_THRESHOLD = 0.95
DEFAULT_TARGET_FPR = 0.05

# Federated-learning defaults. The FL experiment uses disjoint, non-IID client partitions.
DEFAULT_FED_CLIENTS = 4
DEFAULT_FED_ROUNDS = 6
DEFAULT_FED_LOCAL_EPOCHS = 2
DEFAULT_FED_DIRICHLET_ALPHA = 0.35


@dataclass
class Paths:
    data_dir: Path
    results_dir: Path
    models_dir: Path
    report_dir: Path
    demo: bool

    def make(self) -> "Paths":
        for p in (self.results_dir, self.models_dir, self.report_dir):
            p.mkdir(parents=True, exist_ok=True)
        return self


def add_common_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    p.add_argument(
        "--demo", action="store_true",
        help="Use auto-generated SYNTHETIC data (smoke test only; results are meaningless).",
    )
    p.add_argument(
        "--data-dir", type=Path, default=None,
        help="Folder with UNSW-NB15 partition files (default: data/raw).",
    )
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument(
        "--quick", action="store_true",
        help="Smaller models and fewer CV folds, for a fast run.",
    )
    p.add_argument(
        "--protocol", choices=["research", "benchmark"], default=DEFAULT_PROTOCOL,
        help=("research = remove exact training duplicates and train/test overlaps before modelling; "
              "benchmark = preserve the resolved official partitions. research is recommended."),
    )
    p.add_argument(
        "--corr-threshold", type=float, default=DEFAULT_CORR_THRESHOLD,
        help=("Absolute Spearman threshold for train-fitted correlation filtering. "
              "Use 0 to disable; default 0.95."),
    )
    p.add_argument(
        "--target-fpr", type=float, default=DEFAULT_TARGET_FPR,
        help="Target FPR cap used for OOF binary threshold selection (default 0.05).",
    )
    p.add_argument(
        "--no-threshold-tuning", action="store_true",
        help="Use threshold 0.5 instead of OOF FPR-constrained threshold.",
    )
    p.add_argument(
        "--train-file", type=Path, default=None,
        help="Optional explicit training file. Overrides automatic partition-role detection.",
    )
    p.add_argument(
        "--test-file", type=Path, default=None,
        help="Optional explicit test file. Overrides automatic partition-role detection.",
    )
    return p


def resolve_paths(args: argparse.Namespace) -> Paths:
    if args.demo:
        return Paths(
            data_dir=args.data_dir or ROOT / "data" / "synthetic",
            results_dir=ROOT / "results" / "demo",
            models_dir=ROOT / "models_demo",
            report_dir=ROOT / "reports" / "demo",
            demo=True,
        ).make()
    return Paths(
        data_dir=args.data_dir or ROOT / "data" / "raw",
        results_dir=ROOT / "results",
        models_dir=ROOT / "models",
        report_dir=ROOT / "reports",
        demo=False,
    ).make()
