"""Train ML baselines with leakage-safe threshold selection.

Research design:
- UNSW-NB15 train/test roles are resolved by canonical row counts.
- Research mode removes exact duplicate training rows and train/test overlaps from TRAIN only.
- Binary decision thresholds are selected from out-of-fold (OOF) training predictions only.
- The default threshold policy maximizes F1 subject to OOF FPR <= 5%.
- The final model is then refit on all cleaned training data; TEST is never used for fitting
  or threshold selection.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(__file__).replace("/src/train.py", ""))
from src import config, data_loader, synthetic  # noqa: E402
from src.models import build_pipeline, fit_params, positive_weight  # noqa: E402
from src.thresholding import select_threshold_for_fpr  # noqa: E402

try:
    import psutil
except ImportError:
    psutil = None

CV_SCORING = {
    "binary": {"f1": "f1", "recall": "recall", "precision": "precision", "pr_auc": "average_precision"},
    "multiclass": {"f1_macro": "f1_macro", "f1_weighted": "f1_weighted", "recall_macro": "recall_macro"},
}

TUNE_SPACE = {
    "model__n_estimators": [200, 300, 500],
    "model__max_depth": [4, 6, 8, 10],
    "model__learning_rate": [0.03, 0.05, 0.1],
    "model__subsample": [0.7, 0.8, 1.0],
    "model__colsample_bytree": [0.6, 0.8, 1.0],
    "model__min_child_weight": [1, 3, 5],
}


def ensure_data(paths: config.Paths, quick: bool = False) -> None:
    if paths.demo:
        have = all((paths.data_dir / f"{s}.csv").exists() for s in (config.TRAIN_FILE_STEM, config.TEST_FILE_STEM))
        if not have:
            print(">> --demo: generating SYNTHETIC data (not real; smoke-testing only)")
            synthetic.generate(paths.data_dir, n_train=(3000 if quick else 8000), n_test=(1500 if quick else 4000))


def _rss_mb() -> float:
    return psutil.Process().memory_info().rss / 1e6 if psutil else float("nan")


def _binary_cv_with_oof(pipe, X, y, folds, seed):
    """One CV pass that returns fold metrics AND OOF probabilities for threshold selection."""
    cv = StratifiedKFold(folds, shuffle=True, random_state=seed)
    oof = np.full(len(y), np.nan, dtype=float)
    fold_rows = []
    for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y), start=1):
        model = clone(pipe)
        model.fit(X.iloc[tr_idx], y[tr_idx])
        p = model.predict_proba(X.iloc[va_idx])[:, 1]
        pred = (p >= 0.5).astype(int)
        yt = y[va_idx]
        tp = int(((yt == 1) & (pred == 1)).sum())
        fp = int(((yt == 0) & (pred == 1)).sum())
        fn = int(((yt == 1) & (pred == 0)).sum())
        tn = int(((yt == 0) & (pred == 0)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        ap = float(__import__("sklearn.metrics", fromlist=["average_precision_score"]).average_precision_score(yt, p))
        oof[va_idx] = p
        fold_rows.append({"fold": fold, "f1": f1, "recall": recall, "precision": precision, "pr_auc": ap})
    metrics = {}
    for key in ("f1", "recall", "precision", "pr_auc"):
        vals = np.array([r[key] for r in fold_rows], dtype=float)
        metrics[key] = {"mean": float(vals.mean()), "std": float(vals.std())}
    return metrics, oof


def train_one(name, task, X, y, args, spw, pipe=None, do_cv=True):
    if pipe is None:
        pipe = build_pipeline(name, task, args.seed, args.quick, spw, args.corr_threshold)
    info = {}
    if do_cv and args.cv_folds and args.cv_folds > 1:
        if task == "binary":
            cv_metrics, oof = _binary_cv_with_oof(pipe, X, y, args.cv_folds, args.seed)
            info["cv"] = cv_metrics
            info["cv_folds"] = args.cv_folds
            if not args.no_threshold_tuning:
                info["decision_threshold"] = select_threshold_for_fpr(y, oof, args.target_fpr)
        else:
            min_class = int(np.bincount(y).min()) if np.issubdtype(np.asarray(y).dtype, np.integer) else int(pd.Series(y).value_counts().min())
            effective_folds = min(args.cv_folds, min_class)
            if effective_folds < 2:
                info["cv"] = {}
                info["cv_folds"] = 0
                info["cv_note"] = "Skipped multiclass CV because the rarest class has fewer than 2 training samples."
            else:
                cv = StratifiedKFold(effective_folds, shuffle=True, random_state=args.seed)
                res = cross_validate(pipe, X, y, cv=cv, scoring=CV_SCORING[task], n_jobs=1,
                                     params=fit_params(name, task, y) or None)
                info["cv"] = {k.replace("test_", ""): {"mean": float(np.mean(v)), "std": float(np.std(v))}
                              for k, v in res.items() if k.startswith("test_")}
                info["cv_folds"] = effective_folds

    rss0 = _rss_mb()
    t0 = time.perf_counter()
    pipe.fit(X, y, **fit_params(name, task, y))
    info["train_time_s"] = time.perf_counter() - t0
    info["train_rss_delta_mb"] = max(_rss_mb() - rss0, 0.0)
    info["n_features_in"] = int(X.shape[1])
    info["n_features_out"] = int(len(pipe.named_steps["prep"].get_feature_names_out()))
    if task == "binary" and "decision_threshold" not in info:
        info["decision_threshold"] = {"threshold": 0.5, "selection": "disabled_or_no_cv", "source": "default"}
    return pipe, info


def tune_xgboost(task, X, y, args, spw):
    pipe = build_pipeline("xgboost", task, args.seed, args.quick, spw, args.corr_threshold)
    search = RandomizedSearchCV(
        pipe, TUNE_SPACE, n_iter=3 if args.quick else 15, scoring="f1", refit=True,
        cv=StratifiedKFold(3, shuffle=True, random_state=args.seed), random_state=args.seed, n_jobs=1,
    )
    t0 = time.perf_counter()
    search.fit(X, y)
    info = {
        "tuning_time_s": time.perf_counter() - t0,
        "best_params": {k.replace("model__", ""): (v.item() if hasattr(v, "item") else v)
                        for k, v in search.best_params_.items()},
        "tuning_cv_f1": float(search.best_score_),
    }
    best = build_pipeline("xgboost", task, args.seed, args.quick, spw, args.corr_threshold)
    best.set_params(**search.best_params_)
    return best, info


def main(argv=None) -> None:
    ap = config.add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--tasks", nargs="+", default=["binary", "multiclass"], choices=["binary", "multiclass"])
    ap.add_argument("--models", nargs="+", default=config.MODEL_NAMES, choices=config.MODEL_NAMES)
    ap.add_argument("--cv-folds", type=int, default=None)
    ap.add_argument("--tune", action="store_true", help="Also run randomized XGBoost tuning for binary classification.")
    args = ap.parse_args(argv)
    if args.cv_folds is None:
        args.cv_folds = 3 if args.quick else 5
    if not 0 < args.target_fpr < 1:
        raise ValueError("--target-fpr must be between 0 and 1")

    paths = config.resolve_paths(args)
    ensure_data(paths, args.quick)
    train, _, load_meta = data_loader.load_split(paths.data_dir, protocol=args.protocol,
                                                  train_file=args.train_file, test_file=args.test_file,
                                                  return_meta=True)
    print(f"Train: {len(train):,} model rows | seed={args.seed} | cv_folds={args.cv_folds} | protocol={args.protocol} | demo={args.demo}")

    for task in args.tasks:
        X, y_raw = data_loader.split_xy(train, task)
        meta = {
            "seed": args.seed, "task": task, "n_train": int(len(X)), "quick": args.quick,
            "corr_threshold": args.corr_threshold, "protocol": args.protocol, "demo": args.demo,
            "partition_resolution": load_meta["partition"], "hygiene": load_meta["hygiene"],
            "threshold_policy": ({"enabled": not args.no_threshold_tuning, "target_fpr": args.target_fpr,
                                  "source": "out_of_fold_training_predictions"} if task == "binary" else None),
        }
        if task == "multiclass":
            le = LabelEncoder().fit(y_raw)
            y = le.transform(y_raw)
            meta["classes"] = [str(c) for c in le.classes_]
        else:
            y = y_raw.to_numpy()
        spw = positive_weight(y) if task == "binary" else 1.0
        summary = {"_meta": meta}

        for name in args.models:
            print(f"[{task}] training {name} ...", flush=True)
            try:
                pipe, info = train_one(name, task, X, y, args, spw)
            except ImportError as e:
                print(f"   skipped {name}: {e} (install required dependency)")
                continue
            joblib.dump(pipe, paths.models_dir / f"{task}_{name}.joblib", compress=3)
            summary[name] = info
            cv_txt = ", ".join("%s=%.4f" % (k, v["mean"]) for k, v in info.get("cv", {}).items())
            if task == "binary":
                t = info.get("decision_threshold", {}).get("threshold", 0.5)
                cal = info.get("decision_threshold", {})
                print("   train %.1fs%s | primary threshold=%.4f | OOF FPR=%.4f | OOF F1=%.4f" % (
                    info["train_time_s"], (" | CV " + cv_txt) if cv_txt else "", t,
                    float(cal.get("oof_fpr", 0.0)), float(cal.get("oof_f1", 0.0))))
            else:
                print("   train %.1fs%s" % (info["train_time_s"], (" | CV " + cv_txt) if cv_txt else ""))

        if args.tune and task == "binary" and "xgboost" in args.models:
            print("[binary] tuning xgboost (randomized search) ...", flush=True)
            try:
                best, tinfo = tune_xgboost(task, X, y, args, spw)
                no_cv = copy.copy(args)
                no_cv.cv_folds = 0
                pipe, info = train_one("xgboost_tuned", task, X, y, no_cv, spw, pipe=best, do_cv=False)
                # Re-select threshold from OOF predictions of the selected estimator.
                if not args.no_threshold_tuning:
                    _, oof = _binary_cv_with_oof(best, X, y, args.cv_folds, args.seed)
                    info["decision_threshold"] = select_threshold_for_fpr(y, oof, args.target_fpr)
                else:
                    info["decision_threshold"] = {"threshold": 0.5, "selection": "disabled", "source": "default"}
                tinfo.update(info)
                joblib.dump(pipe, paths.models_dir / f"binary_xgboost_tuned.joblib", compress=3)
                summary["xgboost_tuned"] = tinfo
                print(f"   tuned XGBoost CV F1={tinfo['tuning_cv_f1']:.4f} threshold={tinfo['decision_threshold']['threshold']:.4f}")
            except ImportError as e:
                print(f"   skipped tuning: {e}")

        (paths.models_dir / f"{task}_train_info.json").write_text(json.dumps(summary, indent=2))
    print("Training completed successfully. The primary binary threshold was selected without using the held-out test partition.")


if __name__ == "__main__":
    main()
