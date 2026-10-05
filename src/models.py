"""The three baseline models, built as full preprocessing+model pipelines.

Identical preprocessing, seed and class-imbalance handling (class weights) for all three,
so differences in results are attributable to the learning algorithm.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

from src.preprocessing import build_preprocessor


def make_estimator(name: str, task: str, seed: int, quick: bool = False, scale_pos_weight: float = 1.0):
    n_trees = 20 if quick else 300
    if name == "logistic_regression":
        return LogisticRegression(C=1.0, class_weight="balanced", solver="lbfgs",
                                  max_iter=2000, random_state=seed)
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=n_trees, max_depth=None, class_weight="balanced",
                                      n_jobs=-1, random_state=seed)
    if name in ("xgboost", "xgboost_tuned"):
        from xgboost import XGBClassifier  # imported lazily so the other models work without it
        kw = dict(n_estimators=n_trees, max_depth=(6 if quick else 8), learning_rate=0.1, subsample=0.8,
                  colsample_bytree=0.8, tree_method="hist", n_jobs=-1, random_state=seed)
        if task == "binary":
            kw.update(eval_metric="logloss", scale_pos_weight=scale_pos_weight)
        else:
            kw.update(eval_metric="mlogloss")  # objective inferred (multi:softprob)
        return XGBClassifier(**kw)
    raise ValueError(f"unknown model: {name}")


def build_pipeline(name: str, task: str, seed: int, quick: bool = False,
                   scale_pos_weight: float = 1.0, corr_threshold: float | None = None) -> Pipeline:
    return Pipeline([
        ("prep", build_preprocessor(corr_threshold)),
        ("model", make_estimator(name, task, seed, quick, scale_pos_weight)),
    ])


def fit_params(name: str, task: str, y) -> dict:
    """Extra fit() kwargs. XGBoost multiclass has no class_weight, so use balanced sample weights."""
    if name.startswith("xgboost") and task == "multiclass":
        return {"model__sample_weight": compute_sample_weight("balanced", y)}
    return {}


def positive_weight(y) -> float:
    """neg/pos ratio for XGBoost's scale_pos_weight (a label-only hyperparameter)."""
    y = np.asarray(y)
    pos = max(int((y == 1).sum()), 1)
    return float((y == 0).sum() / pos)
