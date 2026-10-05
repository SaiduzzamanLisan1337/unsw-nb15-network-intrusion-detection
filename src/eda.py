"""Dataset analysis: partition resolution, hygiene, distributions, and correlations."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config, data_loader, visualization as viz  # noqa: E402
from src.train import ensure_data  # noqa: E402


def raw_inf_count(train_path: Path, test_path: Path) -> dict[str, int]:
    out = {}
    for role, path in (("resolved_train_file", train_path), ("resolved_test_file", test_path)):
        df = data_loader._read(path)
        num = df.select_dtypes(include="number")
        out[role] = int(np.isinf(num.to_numpy(dtype=float)).sum()) if not num.empty else 0
    return out


def main(argv=None) -> None:
    ap = config.add_common_args(argparse.ArgumentParser(description=__doc__))
    args = ap.parse_args(argv)
    paths = config.resolve_paths(args)
    ensure_data(paths, args.quick)

    tr_path, te_path, manifest = data_loader.resolve_partition_paths(
        paths.data_dir, args.train_file, args.test_file
    )
    train_raw = data_loader.clean(data_loader._read(tr_path))
    test_raw = data_loader.clean(data_loader._read(te_path))
    train, test, load_meta = data_loader.load_split(
        paths.data_dir, protocol=args.protocol, train_file=args.train_file, test_file=args.test_file, return_meta=True
    )

    feats = data_loader.feature_columns(train_raw)
    cat = [c for c in feats if c in config.CATEGORICAL]
    num = [c for c in feats if c not in cat]
    excluded = [c for c in train_raw.columns if c not in feats]
    cols_with_labels = feats + config.LABEL_COLUMNS

    def part(df):
        key = df[cols_with_labels]
        return {
            "records": int(len(df)),
            "missing_values_in_features": int(df[feats].isna().sum().sum()),
            "duplicate_rows": int(key.duplicated().sum()),
            "label_counts": {("Benign" if k == 0 else "Attack"): int(v)
                             for k, v in df[config.TARGET_BINARY].value_counts().sort_index().items()},
            "label_pct": {("Benign" if k == 0 else "Attack"): round(float(v) * 100, 2)
                          for k, v in df[config.TARGET_BINARY].value_counts(normalize=True).sort_index().items()},
            "attack_category_counts": {str(k): int(v) for k, v in df[config.TARGET_MULTI].value_counts().items()},
        }

    h_train = pd.util.hash_pandas_object(train_raw[cols_with_labels], index=False)
    h_test = pd.util.hash_pandas_object(test_raw[cols_with_labels], index=False)
    overlap = int(len(set(h_train.tolist()) & set(h_test.tolist())))

    corr = train_raw[num].corr(method="spearman")
    pairs = [
        (a, b, float(corr.loc[a, b]))
        for i, a in enumerate(num)
        for b in num[i + 1:]
        if pd.notna(corr.loc[a, b]) and abs(corr.loc[a, b]) >= args.corr_threshold
    ] if args.corr_threshold > 0 else []
    pairs_df = pd.DataFrame(pairs, columns=["feature_a", "feature_b", "spearman"])
    if not pairs_df.empty:
        pairs_df = pairs_df.sort_values("spearman", key=lambda s: s.abs(), ascending=False)
    pairs_df.to_csv(paths.results_dir / "highly_correlated_pairs.csv", index=False)

    summary = {
        "protocol": args.protocol,
        "partition_resolution": manifest,
        "load_meta": load_meta,
        "model_input": {
            "n_input_features_after_dropping_identifiers_and_labels": len(feats),
            "numerical_features": len(num),
            "categorical_features": cat,
            "columns_excluded_from_model_input": excluded,
        },
        "infinite_values_in_raw_files": raw_inf_count(tr_path, te_path),
        "raw_train": part(train_raw),
        "raw_test": part(test_raw),
        "model_train": part(train),
        "model_test": part(test),
        "raw_distinct_rows_shared_by_train_and_test": overlap,
        "correlated_pairs": {
            "threshold": args.corr_threshold,
            "count": len(pairs),
        },
        "categorical_levels_seen_in_test_but_not_train": {
            c: sorted(set(test_raw[c]) - set(train_raw[c])) for c in cat if c in train_raw and c in test_raw
        },
    }
    (paths.results_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2))

    stats = train_raw[num].describe(percentiles=[.25, .5, .75, .99]).T
    stats["skew"] = train_raw[num].skew()
    stats.to_csv(paths.results_dir / "feature_statistics.csv", float_format="%.6g")

    viz.class_distribution(train[config.TARGET_BINARY].value_counts().sort_index(),
                           train[config.TARGET_MULTI].value_counts(),
                           paths.results_dir / "class_distribution.png", paths.demo)
    viz.corr_heatmap(corr.fillna(0), paths.results_dir / "correlation_heatmap.png", paths.demo)

    print(f"Resolved TRAIN: {tr_path.name} | raw rows={len(train_raw):,} | model rows={len(train):,}")
    print(f"Resolved TEST : {te_path.name} | raw rows={len(test_raw):,} | model rows={len(test):,}")
    print(f"Partition mode: {manifest['mode']} | protocol={args.protocol}")
    print(f"Train model label %: {summary['model_train']['label_pct']} | raw train duplicates={len(train_raw[cols_with_labels]) - len(train_raw[cols_with_labels].drop_duplicates()):,}")
    print(f"Raw distinct train/test shared rows: {overlap:,} | correlated pairs >= {args.corr_threshold}: {len(pairs)}")
    print(
        "Hygiene removed: "
        f"{load_meta['hygiene']['train_duplicate_rows_removed']:,} duplicates + "
        f"{load_meta['hygiene']['train_test_overlap_rows_removed']:,} train/test overlaps"
    )
    print(f"Wrote {paths.results_dir}/dataset_summary.json and plots")


if __name__ == "__main__":
    main()
