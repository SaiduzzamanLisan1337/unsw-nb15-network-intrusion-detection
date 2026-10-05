"""Evaluate trained models on the untouched TEST partition.

Binary results use the leakage-safe OOF-selected decision threshold stored by train.py as
the primary operating point. Threshold=0.5 results are also reported as a secondary baseline.
"""
from __future__ import annotations

import argparse
import json
import pickle
import platform
import sys
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_recall_curve, precision_recall_fscore_support, roc_auc_score, roc_curve,
)
from sklearn.preprocessing import LabelEncoder

from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config, data_loader, visualization as viz  # noqa: E402
from src.preprocessing import original_feature_name  # noqa: E402
from src.train import ensure_data  # noqa: E402

ORDER = config.MODEL_NAMES + ["xgboost_tuned"]


def _timed_batch(pipe, X, repeats=3):
    pipe.predict_proba(X.iloc[: min(1000, len(X))])
    vals = []
    for _ in range(repeats):
        t0 = time.perf_counter(); pipe.predict_proba(X); vals.append(time.perf_counter() - t0)
    return float(np.median(vals))


def _single_flow_latency(pipe, X, n=200, seed=config.SEED):
    est = pipe.named_steps["model"]
    old = getattr(est, "n_jobs", None)
    if old is not None:
        est.set_params(n_jobs=1)
    rng = np.random.default_rng(seed)
    rows = rng.choice(len(X), size=min(n, len(X)), replace=False)
    pipe.predict_proba(X.iloc[[rows[0]]])
    lat = []
    for i in rows:
        t0 = time.perf_counter(); pipe.predict_proba(X.iloc[[i]]); lat.append((time.perf_counter() - t0) * 1e3)
    if old is not None:
        est.set_params(n_jobs=old)
    return float(np.median(lat)), float(np.percentile(lat, 95))


def _recall_at_fpr(fpr, tpr, cap):
    ok = fpr <= cap
    return float(tpr[ok].max()) if ok.any() else 0.0


def binary_metrics(y, proba, thr=0.5):
    y = np.asarray(y, dtype=int); p = np.asarray(proba, dtype=float)
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    div = lambda a, b: float(a / b) if b else 0.0  # noqa: E731
    fpr, tpr, _ = roc_curve(y, p)
    return {
        "precision": div(tp, tp + fp), "recall": div(tp, tp + fn),
        "f1": div(2 * tp, 2 * tp + fp + fn), "fpr": div(fp, fp + tn),
        "fnr": div(fn, fn + tp), "accuracy": div(tp + tn, len(y)),
        "roc_auc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p)),
        "recall_at_fpr_1pct": _recall_at_fpr(fpr, tpr, 0.01),
        "recall_at_fpr_5pct": _recall_at_fpr(fpr, tpr, 0.05),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp), "threshold": float(thr),
    }


def _load(paths, task):
    info_path = paths.models_dir / f"{task}_train_info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"{info_path} not found - run `python3 run_all.py` first.")
    info = json.loads(info_path.read_text())
    models = {n: joblib.load(paths.models_dir / f"{task}_{n}.joblib")
              for n in ORDER if n in info and (paths.models_dir / f"{task}_{n}.joblib").exists()}
    return models, info


def _cost_columns(name, pipe, X, info):
    single_med, single_p95 = _single_flow_latency(pipe, X)
    batch = _timed_batch(pipe, X)
    mi = info.get(name, {})
    row = {
        "train_time_s": mi.get("train_time_s", np.nan),
        "train_rss_delta_mb": mi.get("train_rss_delta_mb", np.nan),
        "batch_inference_s": batch, "throughput_flows_per_s": len(X) / batch,
        "single_flow_latency_ms": single_med, "single_flow_latency_p95_ms": single_p95,
        "model_size_mb": len(pickle.dumps(pipe)) / 1e6,
        "n_features_in": mi.get("n_features_in", X.shape[1]),
        "n_features_out": mi.get("n_features_out", np.nan),
    }
    for k, v in mi.get("cv", {}).items():
        row[f"cv_{k}_mean"], row[f"cv_{k}_std"] = v["mean"], v["std"]
    return row


def eval_binary(paths, test, args):
    models, info = _load(paths, "binary")
    X, y_series = data_loader.split_xy(test, "binary")
    y = y_series.to_numpy()
    rows, cms_primary, cms_default, curves = {}, {}, {}, {}
    threshold_rows = []
    for name, pipe in models.items():
        print(f"[binary] evaluating {name} ...", flush=True)
        proba = pipe.predict_proba(X)[:, 1]
        meta = info.get(name, {})
        threshold_info = meta.get("decision_threshold", {})
        threshold = float(threshold_info.get("threshold", 0.5))
        tuned = binary_metrics(y, proba, threshold)
        default = binary_metrics(y, proba, 0.5)
        row = {**tuned, **_cost_columns(name, pipe, X, info),
               "threshold_selection": threshold_info.get("selection", "default"),
               "threshold_target_fpr": threshold_info.get("target_fpr", np.nan),
               "threshold_oof_fpr": threshold_info.get("oof_fpr", np.nan),
               "threshold_oof_recall": threshold_info.get("oof_recall", np.nan),
               "threshold_oof_f1": threshold_info.get("oof_f1", np.nan)}
        for key in ("precision", "recall", "f1", "fpr", "fnr", "accuracy"):
            row[f"default_{key}"] = default[key]
        rows[name] = row
        cms_primary[name] = [[tuned["tn"], tuned["fp"]], [tuned["fn"], tuned["tp"]]]
        cms_default[name] = [[default["tn"], default["fp"]], [default["fn"], default["tp"]]]
        threshold_rows.append({"model": name, **threshold_info})
        fpr, tpr, _ = roc_curve(y, proba); prec, rec, _ = precision_recall_curve(y, proba)
        curves[name] = {"fpr": fpr, "tpr": tpr, "prec": prec, "rec": rec,
                        "roc_auc": tuned["roc_auc"], "pr_auc": tuned["pr_auc"]}

    df = pd.DataFrame(rows).T; df.index.name = "model"
    df.to_csv(paths.results_dir / "metrics.csv", float_format="%.6g")
    pd.DataFrame(threshold_rows).to_csv(paths.results_dir / "decision_thresholds.csv", index=False, float_format="%.6g")
    viz.confusion_matrices(cms_primary, paths.results_dir / "confusion_matrix.png", paths.demo)
    viz.confusion_matrices(cms_default, paths.results_dir / "confusion_matrix_threshold_0_5.png", paths.demo)
    viz.roc_pr(curves, paths.results_dir / "roc_pr_curves.png", paths.demo)
    viz.model_comparison(df, paths.results_dir / "model_comparison.png", paths.demo)
    print(df[["threshold", "precision", "recall", "f1", "fpr", "default_f1", "default_fpr", "pr_auc"]].round(4).to_string())
    return models, X, y


def eval_multiclass(paths, test, args):
    models, info = _load(paths, "multiclass")
    classes = info["_meta"]["classes"]
    le = LabelEncoder().fit(classes)
    X, y_raw = data_loader.split_xy(test, "multiclass")
    keep = y_raw.isin(classes).to_numpy()
    if not keep.all():
        print(f"   note: dropping {int((~keep).sum())} test rows with classes unseen in training")
    X, y = X[keep], le.transform(y_raw[keep])
    rows, per_class, cms = {}, [], {}
    for name, pipe in models.items():
        print(f"[multiclass] evaluating {name} ...", flush=True)
        pred = pipe.predict(X).astype(int)
        p, r, f, s = precision_recall_fscore_support(y, pred, labels=range(len(classes)), zero_division=0)
        rows[name] = {"accuracy": accuracy_score(y, pred), "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
                      "weighted_f1": f1_score(y, pred, average="weighted", zero_division=0),
                      "macro_recall": float(np.mean(r)), "macro_precision": float(np.mean(p)),
                      **_cost_columns(name, pipe, X, info)}
        for i, c in enumerate(classes):
            per_class.append({"model": name, "class": c, "precision": p[i], "recall": r[i], "f1": f[i], "support": int(s[i])})
        cms[name] = confusion_matrix(y, pred, labels=range(len(classes)))
    df = pd.DataFrame(rows).T; df.index.name = "model"
    df.to_csv(paths.results_dir / "metrics_multiclass.csv", float_format="%.6g")
    pd.DataFrame(per_class).to_csv(paths.results_dir / "per_class_metrics.csv", index=False, float_format="%.6g")
    viz.multiclass_confusions(cms, classes, paths.results_dir / "confusion_matrix_multiclass.png", paths.demo)
    print(df[["accuracy", "macro_f1", "weighted_f1", "macro_recall"]].round(4).to_string())


def tree_importance(models, paths, X, y, args):
    imps = {}
    for name in ("random_forest", "xgboost", "xgboost_tuned"):
        if name not in models: continue
        pipe = models[name]
        names = pipe.named_steps["prep"].get_feature_names_out()
        raw = pd.Series(pipe.named_steps["model"].feature_importances_, index=names)
        agg = raw.groupby([original_feature_name(n) for n in raw.index]).sum().sort_values(ascending=False)
        imps[name] = agg
        agg.to_csv(paths.results_dir / f"feature_importance_{name}.csv", header=["importance"])
    if imps:
        shown = {k: v for k, v in imps.items() if k != "xgboost_tuned"} or imps
        viz.feature_importance(shown, paths.results_dir / "feature_importance.png", "Feature importance (one-hot levels summed)", demo=paths.demo)
    if args.permutation and imps:
        n = min(5000, len(X)); idx = np.random.default_rng(args.seed).choice(len(X), n, replace=False); perm = {}
        for name in [k for k in imps if k != "xgboost_tuned"]:
            print(f"[binary] permutation importance: {name} ...", flush=True)
            r = permutation_importance(models[name], X.iloc[idx], y[idx], scoring="f1", n_repeats=3, random_state=args.seed, n_jobs=1)
            s = pd.Series(r.importances_mean, index=X.columns).sort_values(ascending=False)
            s.to_csv(paths.results_dir / f"permutation_importance_{name}.csv", header=["f1_drop"]); perm[name] = s
        viz.feature_importance(perm, paths.results_dir / "permutation_importance.png", "Mean F1 drop", demo=paths.demo)
    if args.shap and "xgboost" in models:
        try:
            import shap
        except ImportError:
            print("   --shap requested but shap is not installed; skipping."); return
        pipe = models["xgboost"]; n = min(2000, len(X)); idx = np.random.default_rng(args.seed).choice(len(X), n, replace=False)
        Xt = pipe.named_steps["prep"].transform(X.iloc[idx]); names = pipe.named_steps["prep"].get_feature_names_out()
        sv = shap.TreeExplainer(pipe.named_steps["model"]).shap_values(Xt)
        raw = pd.Series(np.abs(sv).mean(axis=0), index=names)
        agg = raw.groupby([original_feature_name(c) for c in raw.index]).sum().sort_values(ascending=False)
        agg.to_csv(paths.results_dir / "shap_importance_xgboost.csv", header=["mean_abs_shap"])
        viz.feature_importance({"xgboost": agg}, paths.results_dir / "shap_importance.png", "Mean |SHAP value|", demo=paths.demo)


def write_run_info(paths, train, test, args, load_meta):
    info = {"demo_synthetic_data": paths.demo, "seed": args.seed, "protocol": args.protocol,
            "target_fpr": args.target_fpr, "partition_resolution": load_meta["partition"],
            "hygiene": load_meta["hygiene"], "n_train": int(len(train)), "n_test": int(len(test)),
            "python": platform.python_version(), "sklearn": sklearn.__version__,
            "numpy": np.__version__, "pandas": pd.__version__}
    try:
        import xgboost; info["xgboost"] = xgboost.__version__
    except ImportError: pass
    (paths.results_dir / "run_info.json").write_text(json.dumps(info, indent=2))


def main(argv=None) -> None:
    ap = config.add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--permutation", action="store_true")
    ap.add_argument("--shap", action="store_true")
    ap.add_argument("--tasks", nargs="+", default=["binary", "multiclass"], choices=["binary", "multiclass"])
    args = ap.parse_args(argv)
    paths = config.resolve_paths(args); ensure_data(paths, args.quick)
    train, test, load_meta = data_loader.load_split(paths.data_dir, protocol=args.protocol,
                                                     train_file=args.train_file, test_file=args.test_file, return_meta=True)
    print(f"Test: {len(test):,} rows | protocol={args.protocol} | demo={paths.demo}")
    if "binary" in args.tasks:
        models, X, y = eval_binary(paths, test, args); tree_importance(models, paths, X, y, args)
    if "multiclass" in args.tasks: eval_multiclass(paths, test, args)
    write_run_info(paths, train, test, args, load_meta)
    print(f"Results written to {paths.results_dir}")


if __name__ == "__main__":
    main()
