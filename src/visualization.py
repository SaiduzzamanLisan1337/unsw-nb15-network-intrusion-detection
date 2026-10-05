"""Matplotlib-only plotting helpers (headless-safe)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import PRETTY  # noqa: E402

COLORS = {"logistic_regression": "#4C72B0", "random_forest": "#55A868",
          "xgboost": "#C44E52", "xgboost_tuned": "#8172B2"}
DEMO_NOTE = "SYNTHETIC DATA - smoke test only, not a result"


def _finish(fig, path: Path, demo: bool = False):
    if demo:
        fig.text(0.5, 0.005, DEMO_NOTE, ha="center", va="bottom", fontsize=8, color="firebrick")
    fig.tight_layout(rect=(0, 0.02 if demo else 0, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _heat(ax, M, xlabels, ylabels, fmt="{:.0f}", cmap="Blues", vmin=None, vmax=None, fs=9):
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(xlabels)), xlabels, rotation=45, ha="right")
    ax.set_yticks(range(len(ylabels)), ylabels)
    thr = (np.nanmax(M) + np.nanmin(M)) / 2 if vmax is None else (vmax + (vmin or 0)) / 2
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center", fontsize=fs,
                    color="white" if M[i, j] > thr else "black")
    return im


def confusion_matrices(cms: dict, path: Path, demo=False):
    fig, axes = plt.subplots(1, len(cms), figsize=(4.6 * len(cms), 4.2), squeeze=False)
    for ax, (name, cm) in zip(axes[0], cms.items()):
        _heat(ax, np.asarray(cm), ["Benign", "Attack"], ["Benign", "Attack"], fmt="{:,.0f}", fs=11)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(PRETTY.get(name, name))
    _finish(fig, path, demo)


def multiclass_confusions(cms: dict, classes, path: Path, demo=False):
    fig, axes = plt.subplots(1, len(cms), figsize=(6.4 * len(cms), 6.0), squeeze=False)
    for ax, (name, cm) in zip(axes[0], cms.items()):
        cm = np.asarray(cm, dtype=float)
        norm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
        _heat(ax, norm, classes, classes, fmt="{:.2f}", vmin=0, vmax=1, fs=7)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title(PRETTY.get(name, name) + " (row-normalised)")
    _finish(fig, path, demo)


def roc_pr(curves: dict, path: Path, demo=False):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6))
    for name, c in curves.items():
        col = COLORS.get(name)
        a1.plot(c["fpr"], c["tpr"], label=f"{PRETTY.get(name, name)} (AUC {c['roc_auc']:.4f})", color=col)
        a2.plot(c["rec"], c["prec"], label=f"{PRETTY.get(name, name)} (AP {c['pr_auc']:.4f})", color=col)
    a1.plot([0, 1], [0, 1], "k:", lw=1)
    a1.set(xlabel="False positive rate", ylabel="True positive rate (recall)", title="ROC curve")
    a2.set(xlabel="Recall", ylabel="Precision", title="Precision-recall curve")
    a1.legend(loc="lower right"); a2.legend(loc="lower left")
    for a in (a1, a2):
        a.grid(alpha=.3)
    _finish(fig, path, demo)


def model_comparison(df: pd.DataFrame, path: Path, demo=False):
    """Left: detection metrics. Right: cost (log scale)."""
    models = list(df.index)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8))
    metrics = ["precision", "recall", "f1", "pr_auc"]
    w = 0.8 / len(models)
    for i, m in enumerate(models):
        a1.bar(np.arange(len(metrics)) + i * w, df.loc[m, metrics].astype(float), w,
               label=PRETTY.get(m, m), color=COLORS.get(m))
    a1.set_xticks(np.arange(len(metrics)) + w * (len(models) - 1) / 2, ["Precision", "Recall", "F1", "PR-AUC"])
    lo = max(float(df[metrics].astype(float).min().min()) - 0.05, 0)
    a1.set_ylim(lo, 1.0); a1.set_title("Detection quality (higher is better)"); a1.legend(); a1.grid(axis="y", alpha=.3)

    costs = [("train_time_s", "Train time (s)"), ("batch_inference_s", "Batch inference (s)"),
             ("single_flow_latency_ms", "Single-flow latency (ms)")]
    for i, m in enumerate(models):
        a2.bar(np.arange(len(costs)) + i * w, [max(float(df.loc[m, c]), 1e-4) for c, _ in costs], w,
               label=PRETTY.get(m, m), color=COLORS.get(m))
    a2.set_xticks(np.arange(len(costs)) + w * (len(models) - 1) / 2, [l for _, l in costs])
    a2.set_yscale("log"); a2.set_title("Computational cost (log scale, lower is better)"); a2.grid(axis="y", alpha=.3)
    _finish(fig, path, demo)


def feature_importance(imps: dict, path: Path, title: str, top: int = 10, demo=False):
    fig, axes = plt.subplots(1, len(imps), figsize=(6.2 * len(imps), 4.6), squeeze=False)
    for ax, (name, s) in zip(axes[0], imps.items()):
        s = s.sort_values(ascending=False).head(top)[::-1]
        ax.barh(s.index, s.values, color=COLORS.get(name, "#4C72B0"))
        ax.set_title(f"{PRETTY.get(name, name)}: top {top}"); ax.set_xlabel(title); ax.grid(axis="x", alpha=.3)
    _finish(fig, path, demo)


def class_distribution(binary: pd.Series, multi: pd.Series, path: Path, demo=False):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4))
    binary.plot.bar(ax=a1, color=["#55A868", "#C44E52"]); a1.set_title("Binary label (training set)")
    a1.set_xticklabels(["Benign (0)", "Attack (1)"], rotation=0)
    for i, v in enumerate(binary.values):
        a1.text(i, v, f"{v:,}\n({v / binary.sum():.1%})", ha="center", va="bottom", fontsize=9)
    multi.sort_values().plot.barh(ax=a2, color="#4C72B0"); a2.set_title("Attack category (training set)")
    a2.set_xscale("log"); a2.set_xlabel("records (log scale)")
    _finish(fig, path, demo)


def corr_heatmap(corr: pd.DataFrame, path: Path, demo=False):
    fig, ax = plt.subplots(figsize=(11, 9.5))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(corr)), corr.columns, fontsize=7)
    fig.colorbar(im, ax=ax, shrink=.8, label="Spearman rho")
    ax.set_title("Numeric feature correlation (training set)")
    _finish(fig, path, demo)


def hetero_heatmaps(mats: dict, rows, cols, path: Path, demo=False):
    fig, axes = plt.subplots(1, len(mats), figsize=(6.4 * len(mats), 4.6), squeeze=False)
    for ax, (title, M) in zip(axes[0], mats.items()):
        _heat(ax, np.asarray(M, dtype=float), cols, rows, fmt="{:.3f}", cmap="Blues", vmin=0, vmax=1, fs=10)
        ax.set_title(title); ax.set_xlabel("Evaluated on"); ax.set_ylabel("Trained on")
    _finish(fig, path, demo)
