"""Leakage-safe operating-point selection for binary intrusion detection.

The decision threshold is selected from out-of-fold (OOF) training predictions only.
It is never selected using the held-out test partition. The default research policy is:

    maximize F1 subject to OOF FPR <= target_fpr

This is an operating-point optimization, not a claim that the model itself has become
intrinsically more accurate. The final test-set FPR is reported to quantify whether the
selected threshold generalizes under distribution shift.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix


def _grouped_threshold_candidates(y_true: np.ndarray, proba: np.ndarray):
    order = np.argsort(-proba, kind="mergesort")
    scores = proba[order]
    labels = y_true[order]
    total_pos = max(int(labels.sum()), 1)
    total_neg = max(int(len(labels) - labels.sum()), 1)

    tp_cum = np.cumsum(labels, dtype=np.int64)
    fp_cum = np.cumsum(1 - labels, dtype=np.int64)
    change = np.r_[scores[1:] < scores[:-1], True]
    idx = np.flatnonzero(change)

    tp = tp_cum[idx].astype(float)
    fp = fp_cum[idx].astype(float)
    fn = total_pos - tp
    tn = total_neg - fp
    fpr = fp / total_neg
    recall = tp / total_pos
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(precision), where=(precision + recall) > 0)
    thresholds = scores[idx]
    # Add an explicit all-negative operating point so an FPR cap is always feasible.
    thresholds = np.r_[np.nextafter(scores[0], np.inf), thresholds]
    fpr = np.r_[0.0, fpr]
    recall = np.r_[0.0, recall]
    precision = np.r_[0.0, precision]
    f1 = np.r_[0.0, f1]
    tn = np.r_[total_neg, tn]
    fp = np.r_[0.0, fp]
    fn = np.r_[total_pos, fn]
    tp = np.r_[0.0, tp]
    return thresholds, fpr, recall, precision, f1, tn, fp, fn, tp


def select_threshold_for_fpr(
    y_true,
    proba,
    target_fpr: float = 0.05,
) -> dict:
    """Select a probability threshold from OOF predictions.

    Among thresholds whose false-positive rate is <= target_fpr on the OOF predictions,
    select the one with maximum F1, then maximum recall, then the lowest threshold.
    """
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(proba, dtype=float)
    if len(y) != len(p) or len(y) == 0:
        raise ValueError("y_true and proba must have the same non-zero length")
    if not 0 < target_fpr < 1:
        raise ValueError("target_fpr must be between 0 and 1")
    if not np.isfinite(p).all():
        raise ValueError("Probability array contains non-finite values")

    thr, fpr, recall, precision, f1, tn, fp, fn, tp = _grouped_threshold_candidates(y, p)
    valid = np.flatnonzero(fpr <= target_fpr + 1e-12)
    if len(valid) == 0:
        # This is theoretically unreachable because an all-negative prediction has FPR=0,
        # but keep the fallback explicit for numerical robustness.
        idx = int(np.argmin(fpr))
        reason = "minimum_fpr_fallback"
    else:
        # Maximize F1, then recall, then choose the highest threshold among exact ties
        # (more conservative false-alarm behavior).
        vf = valid[np.argmax(f1[valid] + recall[valid] * 1e-12)]
        best_f1 = f1[vf]
        tie = valid[np.isclose(f1[valid], best_f1, rtol=1e-12, atol=1e-12)]
        if len(tie) > 1:
            best_recall = recall[tie].max()
            tie = tie[np.isclose(recall[tie], best_recall, rtol=1e-12, atol=1e-12)]
        idx = int(tie[np.argmax(thr[tie])])
        reason = "max_f1_under_fpr_cap"

    threshold = float(thr[idx])
    return {
        "threshold": threshold,
        "target_fpr": float(target_fpr),
        "oof_fpr": float(fpr[idx]),
        "oof_recall": float(recall[idx]),
        "oof_precision": float(precision[idx]),
        "oof_f1": float(f1[idx]),
        "oof_tn": int(tn[idx]),
        "oof_fp": int(fp[idx]),
        "oof_fn": int(fn[idx]),
        "oof_tp": int(tp[idx]),
        "selection": reason,
        "source": "out_of_fold_training_predictions",
    }


def threshold_confusion(y_true, proba, threshold: float) -> tuple[int, int, int, int]:
    pred = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(np.asarray(y_true, dtype=int), pred, labels=[0, 1]).ravel()
    return int(tn), int(fp), int(fn), int(tp)
