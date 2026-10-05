"""UNSW-NB15 loading, automatic train/test role resolution, and stateless cleaning.

Important design choice:
The code does NOT trust the filenames alone.  When both resolved files have the
canonical UNSW-NB15 row counts (175,341 and 82,332), the larger file is assigned
the training role and the smaller file the test role.  This fixes the common case
where the two filenames were accidentally swapped.

The default ``research`` protocol removes exact duplicate training rows and removes
training rows that are exact duplicates of a test row.  The test partition itself is
never modified.  ``benchmark`` preserves the resolved partitions for literature-style
comparison, while clearly reporting the potential overlap.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src import config

DATA_HELP = f"""
Could not resolve the UNSW-NB15 train/test files in: {{d}}

Expected predefined partition sizes:
  training: {config.EXPECTED_TRAIN_ROWS:,} rows
  testing : {config.EXPECTED_TEST_ROWS:,} rows

The two files may be named in either order; when both canonical row counts are found,
the larger file is automatically treated as training and the smaller as test.

Expected filenames (recommended):
  {config.TRAIN_FILE_STEM}.csv
  {config.TEST_FILE_STEM}.csv

Official dataset page:
  https://research.unsw.edu.au/projects/unsw-nb15-dataset
"""


def _preferred_path(data_dir: Path, stem: str) -> Path | None:
    candidates = [data_dir / f"{stem}.csv", data_dir / f"{stem}.parquet"]
    for p in candidates:
        if p.exists():
            return p
    # Case-insensitive fallback for user/mirror naming differences.
    for p in data_dir.iterdir() if data_dir.exists() else []:
        if p.is_file() and p.suffix.lower() in {".csv", ".parquet"} and p.stem.lower() == stem.lower():
            return p
    return None


def _read(path: Path, **kwargs) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path, **kwargs)
    return pd.read_csv(path, low_memory=False, **kwargs)


def _row_count(path: Path) -> int:
    """Count rows without loading the full table when possible."""
    if path.suffix.lower() == ".parquet":
        try:
            import pyarrow.parquet as pq
            return int(pq.ParquetFile(path).metadata.num_rows)
        except Exception:
            return int(len(pd.read_parquet(path, columns=None)))
    with path.open("rb") as f:
        n_lines = sum(1 for _ in f)
    return max(n_lines - 1, 0)


def resolve_partition_paths(
    data_dir: Path,
    train_file: Path | None = None,
    test_file: Path | None = None,
) -> tuple[Path, Path, dict[str, Any]]:
    """Resolve the actual train/test files and return a manifest.

    Priority:
      1) explicit --train-file/--test-file;
      2) canonical filenames, corrected automatically when canonical row counts are swapped;
      3) canonical filenames by name if custom/synthetic row counts do not match the official sizes.
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(DATA_HELP.format(d=data_dir))

    if (train_file is None) != (test_file is None):
        raise ValueError("Provide both --train-file and --test-file, or neither.")

    if train_file is not None and test_file is not None:
        tr, te = Path(train_file), Path(test_file)
        if not tr.is_absolute():
            tr = data_dir / tr
        if not te.is_absolute():
            te = data_dir / te
        if not tr.exists() or not te.exists():
            raise FileNotFoundError(f"Explicit train/test files not found: {tr} | {te}")
        meta = {
            "mode": "explicit",
            "train_file": str(tr),
            "test_file": str(te),
            "train_rows": _row_count(tr),
            "test_rows": _row_count(te),
            "filename_roles_swapped": False,
        }
        return tr, te, meta

    named_train = _preferred_path(data_dir, config.TRAIN_FILE_STEM)
    named_test = _preferred_path(data_dir, config.TEST_FILE_STEM)
    if named_train is None or named_test is None:
        raise FileNotFoundError(DATA_HELP.format(d=data_dir))

    if named_train.resolve() == named_test.resolve():
        raise ValueError("The train and test paths resolve to the same file.")

    train_rows_by_name = _row_count(named_train)
    test_rows_by_name = _row_count(named_test)

    canonical = {config.EXPECTED_TRAIN_ROWS, config.EXPECTED_TEST_ROWS}
    if {train_rows_by_name, test_rows_by_name} == canonical:
        if train_rows_by_name == config.EXPECTED_TRAIN_ROWS:
            train_path, test_path = named_train, named_test
            swapped = False
            mode = "canonical_filename_and_row_count"
        else:
            # The exact problem this project must solve: filenames were swapped.
            train_path, test_path = named_test, named_train
            swapped = True
            mode = "auto_corrected_by_row_count"
    else:
        # Synthetic data or a custom mirror may use different sizes. In that case
        # preserve the filename roles, but make the ambiguity visible in the manifest.
        train_path, test_path = named_train, named_test
        swapped = False
        mode = "filename_fallback_noncanonical_row_counts"

    meta = {
        "mode": mode,
        "filename_train_candidate": str(named_train),
        "filename_test_candidate": str(named_test),
        "resolved_train_file": str(train_path),
        "resolved_test_file": str(test_path),
        "train_rows_by_name": int(train_rows_by_name),
        "test_rows_by_name": int(test_rows_by_name),
        "train_rows": int(_row_count(train_path)),
        "test_rows": int(_row_count(test_path)),
        "filename_roles_swapped": bool(swapped),
        "expected_train_rows": config.EXPECTED_TRAIN_ROWS,
        "expected_test_rows": config.EXPECTED_TEST_ROWS,
    }
    return train_path, test_path, meta


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Stateless normalisation of one partition."""
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    if config.TARGET_BINARY not in df.columns:
        raise ValueError(f"Missing required target column: {config.TARGET_BINARY}")
    if config.TARGET_MULTI not in df.columns:
        raise ValueError(f"Missing required target column: {config.TARGET_MULTI}")

    cat = df[config.TARGET_MULTI].astype("string").str.strip().fillna(config.NORMAL_CLASS)
    cat = cat.replace({"Backdoors": "Backdoor", "Backdoors ": "Backdoor", "": config.NORMAL_CLASS})
    df[config.TARGET_MULTI] = cat.astype(str)
    df[config.TARGET_BINARY] = pd.to_numeric(df[config.TARGET_BINARY], errors="coerce")
    if df[config.TARGET_BINARY].isna().any():
        raise ValueError("Found non-numeric/missing values in binary label column 'label'.")
    df[config.TARGET_BINARY] = df[config.TARGET_BINARY].astype(int)

    for c in config.CATEGORICAL:
        if c in df.columns:
            df[c] = df[c].astype("string").str.strip().str.lower().fillna("-").astype(str)

    feat_cols = feature_columns(df)
    num_cols = [c for c in feat_cols if c not in config.CATEGORICAL]
    if num_cols:
        df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    drop = set(config.LABEL_COLUMNS) | set(config.IDENTIFIER_COLUMNS)
    return [c for c in df.columns if c not in drop]


def _row_keys(df: pd.DataFrame, feature_cols: list[str] | None = None) -> pd.Series:
    cols = feature_cols or (feature_columns(df) + config.LABEL_COLUMNS)
    return pd.util.hash_pandas_object(df[cols], index=False)


def clean_training_overlap(
    train: pd.DataFrame,
    test: pd.DataFrame,
    protocol: str,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Apply research/benchmark partition hygiene.

    Research protocol:
      - remove exact duplicate training rows (features + labels);
      - remove training rows whose features + labels are also present in test.
    The test partition is NEVER modified.
    """
    if protocol not in {"research", "benchmark"}:
        raise ValueError("protocol must be 'research' or 'benchmark'")

    cols = feature_columns(train) + config.LABEL_COLUMNS
    train_before = len(train)
    dup_count = int(train[cols].duplicated().sum())
    overlap_keys = set(_row_keys(test, cols).tolist())
    overlap_mask = _row_keys(train, cols).isin(overlap_keys)
    shared_distinct = int(_row_keys(train, cols).loc[overlap_mask].nunique())

    if protocol == "benchmark":
        clean_train = train.reset_index(drop=True)
        return clean_train, {
            "train_rows_before_cleanup": train_before,
            "train_duplicates_before_cleanup": dup_count,
            "train_test_shared_distinct_rows_before_cleanup": shared_distinct,
            "train_duplicate_rows_removed": 0,
            "train_test_overlap_rows_removed": 0,
            "train_rows_after_cleanup": len(clean_train),
        }

    # Remove duplicate training rows first, then remove any remaining exact overlap.
    clean_train = train.drop_duplicates(subset=cols, keep="first").copy()
    keys_after_dedup = _row_keys(clean_train, cols)
    overlap_mask_after = keys_after_dedup.isin(overlap_keys)
    overlap_removed = int(overlap_mask_after.sum())
    clean_train = clean_train.loc[~overlap_mask_after].reset_index(drop=True)

    return clean_train, {
        "train_rows_before_cleanup": train_before,
        "train_duplicates_before_cleanup": dup_count,
        "train_test_shared_distinct_rows_before_cleanup": shared_distinct,
        "train_duplicate_rows_removed": int(train_before - len(train.drop_duplicates(subset=cols, keep="first"))),
        "train_test_overlap_rows_removed": overlap_removed,
        "train_rows_after_cleanup": len(clean_train),
    }


def load_split(
    data_dir: Path,
    protocol: str = config.DEFAULT_PROTOCOL,
    train_file: Path | None = None,
    test_file: Path | None = None,
    return_meta: bool = False,
):
    """Return cleaned train/test frames; optionally also return resolution/cleanup metadata."""
    train_path, test_path, manifest = resolve_partition_paths(data_dir, train_file, test_file)
    train = clean(_read(train_path))
    test = clean(_read(test_path))
    train, hygiene = clean_training_overlap(train, test, protocol)

    meta = {
        "protocol": protocol,
        "partition": manifest,
        "hygiene": hygiene,
        "test_rows_after_cleanup": len(test),
        "remaining_train_test_shared_distinct_rows": int(
            _row_keys(train, feature_columns(train) + config.LABEL_COLUMNS)
            .isin(set(_row_keys(test, feature_columns(test) + config.LABEL_COLUMNS).tolist()))
            .sum()
        ),
    }
    if manifest.get("filename_roles_swapped"):
        print(
            "WARNING: UNSW-NB15 filenames appear swapped. "
            f"Automatically using {Path(manifest['resolved_train_file']).name} as TRAIN "
            f"({manifest['train_rows']:,}) and {Path(manifest['resolved_test_file']).name} as TEST "
            f"({manifest['test_rows']:,}).",
            flush=True,
        )
    if protocol == "research" and (hygiene["train_duplicate_rows_removed"] or hygiene["train_test_overlap_rows_removed"]):
        print(
            "Research hygiene: removed "
            f"{hygiene['train_duplicate_rows_removed']:,} duplicate training rows and "
            f"{hygiene['train_test_overlap_rows_removed']:,} training rows overlapping the test partition.",
            flush=True,
        )
    if return_meta:
        return train, test, meta
    return train, test


def split_xy(df: pd.DataFrame, task: str):
    if task not in {"binary", "multiclass"}:
        raise ValueError("task must be 'binary' or 'multiclass'")
    X = df[feature_columns(df)]
    y = df[config.TARGET_BINARY] if task == "binary" else df[config.TARGET_MULTI]
    return X, y
