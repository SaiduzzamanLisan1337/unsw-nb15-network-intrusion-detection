"""Leakage-safe preprocessing.

Every learned step (imputer medians, scaler mean/std, one-hot vocabulary, optional
correlation filter) is part of an sklearn Pipeline, so it is fitted on the training
data ONLY -- including inside each cross-validation fold -- and merely applied to
validation/test data.

Selector functions are module-level (not lambdas) so fitted pipelines can be pickled.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src import config


# ----------------------------------------------------------------- selectors
def select_log_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in config.LOG_COLUMNS if c in df.columns]


def select_plain_numeric(df: pd.DataFrame) -> list[str]:
    skip = set(config.CATEGORICAL) | set(config.LOG_COLUMNS)
    return [c for c in df.columns if c not in skip]


def select_categorical(df: pd.DataFrame) -> list[str]:
    return [c for c in config.CATEGORICAL if c in df.columns]


def _log1p_nonneg(x):
    return np.log1p(np.clip(x, 0, None))


# ------------------------------------------------------- optional corr. filter
class CorrelatedFeatureDropper(BaseEstimator, TransformerMixin):
    """Drops one feature from every numeric pair with |Spearman rho| >= threshold.

    The pairs are discovered on the TRAINING data in fit(); transform() just removes the
    same columns from any frame. Spearman is used because the features are heavy-tailed.
    """

    def __init__(self, threshold: float | None = None):
        self.threshold = threshold

    def fit(self, X: pd.DataFrame, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.drop_cols_: list[str] = []
        if self.threshold is not None:
            num = X.select_dtypes(include="number")
            corr = num.corr(method="spearman").abs()
            upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
            self.drop_cols_ = [c for c in upper.columns if (upper[c] >= self.threshold).any()]
        return self

    def transform(self, X: pd.DataFrame):
        return X.drop(columns=[c for c in self.drop_cols_ if c in X.columns])

    def get_feature_names_out(self, input_features=None):
        names = self.feature_names_in_ if input_features is None else input_features
        return np.asarray([c for c in names if c not in self.drop_cols_], dtype=object)


# ----------------------------------------------------------------- builder
def build_preprocessor(corr_threshold: float | None = None) -> Pipeline:
    log_branch = Pipeline([
        ("log1p", FunctionTransformer(_log1p_nonneg, feature_names_out="one-to-one")),
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
    ])
    plain_branch = Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
    ])
    cat_branch = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="-")),
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                 min_frequency=config.MIN_FREQUENCY, sparse_output=False)),
    ])
    columns = ColumnTransformer(
        [("log", log_branch, select_log_cols),
         ("num", plain_branch, select_plain_numeric),
         ("cat", cat_branch, select_categorical)],
        sparse_threshold=0.0,
    )
    return Pipeline([("corr", CorrelatedFeatureDropper(corr_threshold)), ("columns", columns)])


def original_feature_name(transformed_name: str) -> str:
    """'cat__onehot_proto_tcp' / 'log__dur' -> 'proto' / 'dur' (for aggregated importances)."""
    name = transformed_name.split("__", 1)[-1]
    for c in config.CATEGORICAL:
        if name.startswith(f"{c}_") or name == c:
            return c
    return name
