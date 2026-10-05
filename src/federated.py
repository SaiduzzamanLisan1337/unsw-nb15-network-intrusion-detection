"""Federated-learning baseline for heterogeneous network intrusion detection.

This is a real FedAvg optimization experiment, not merely independent local models.
Client data are partitioned disjointly and remain local during each federated round;
the server receives only local model parameters and performs weighted averaging.

Important privacy note: this is an algorithmic FL simulation. Raw examples are not sent
through the aggregation step, but there is NO formal differential privacy or secure
aggregation implementation in this repository. Model updates can leak information in
principle. That is deliberately left as a future research stage.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config, data_loader  # noqa: E402
from src.models import positive_weight  # noqa: E402
from src.preprocessing import build_preprocessor  # noqa: E402
from src.train import ensure_data  # noqa: E402


CLIENT_NAMES = ["Client A", "Client B", "Client C", "Client D"]


def sigmoid(z):
    z = np.clip(z, -35, 35)
    return 1.0 / (1.0 + np.exp(-z))


def logreg_metrics(y, proba, threshold=0.5):
    y = np.asarray(y, dtype=int)
    p = np.asarray(proba, dtype=float)
    pred = (p >= threshold).astype(int)
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tp = int(((y == 1) & (pred == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "fpr": fpr,
            "tn": tn, "fp": fp, "fn": fn, "tp": tp}


def local_sgd(w, b, X, y, epochs, lr, batch_size, l2, clip_norm, seed):
    rng = np.random.default_rng(seed)
    w_local = w.copy()
    b_local = float(b)
    n = len(y)
    for epoch in range(epochs):
        order = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            xb = X[idx]
            yb = y[idx]
            p = sigmoid(xb @ w_local + b_local)
            err = p - yb
            grad_w = (xb.T @ err) / len(idx) + l2 * w_local
            grad_b = float(err.mean())
            grad_norm = float(np.linalg.norm(grad_w))
            if grad_norm > clip_norm:
                grad_w *= clip_norm / max(grad_norm, 1e-12)
            grad_b = float(np.clip(grad_b, -clip_norm, clip_norm))
            w_local -= lr * grad_w
            b_local -= lr * grad_b
    return w_local, b_local


def weighted_fedavg(client_states, sizes):
    total = float(sum(sizes))
    w = sum((n / total) * state[0] for state, n in zip(client_states, sizes))
    b = float(sum((n / total) * state[1] for state, n in zip(client_states, sizes)))
    return w, b


def _dirichlet_split(indices, n_clients, alpha, rng):
    indices = np.asarray(indices, dtype=int).copy()
    rng.shuffle(indices)
    proportions = rng.dirichlet(np.full(n_clients, alpha, dtype=float))
    raw_counts = proportions * len(indices)
    counts = np.floor(raw_counts).astype(int)
    remainder = int(len(indices) - counts.sum())
    if remainder:
        order = np.argsort(-(raw_counts - counts))
        counts[order[:remainder]] += 1
    chunks = []
    start = 0
    for c in counts:
        chunks.append(indices[start:start + c])
        start += c
    return chunks


def make_disjoint_noniid_clients(df: pd.DataFrame, n_clients=4, alpha=0.35, seed=config.SEED):
    """Create disjoint non-IID client indices.

    Normal traffic is distributed relatively evenly; each attack family is allocated via
    a low-alpha Dirichlet draw so clients receive different attack mixtures.
    Every training row belongs to at most one client.
    """
    rng = np.random.default_rng(seed)
    client_idx = [[] for _ in range(n_clients)]
    cats = df[config.TARGET_MULTI].astype(str)

    # Evenly distribute normal traffic.
    normal_idx = np.flatnonzero(cats.to_numpy() == config.NORMAL_CLASS)
    for k, chunk in enumerate(np.array_split(rng.permutation(normal_idx), n_clients)):
        client_idx[k].extend(chunk.tolist())

    # Allocate each attack family across clients with non-IID Dirichlet proportions.
    attack_df = df[cats != config.NORMAL_CLASS]
    for category, group in attack_df.groupby(config.TARGET_MULTI, sort=True):
        chunks = _dirichlet_split(group.index.to_numpy(), n_clients, alpha, rng)
        for k, chunk in enumerate(chunks):
            client_idx[k].extend(chunk.tolist())

    arrays = [np.array(sorted(v), dtype=int) for v in client_idx]
    all_ids = np.concatenate(arrays) if arrays else np.array([], dtype=int)
    if len(np.unique(all_ids)) != len(df) or len(all_ids) != len(df):
        raise AssertionError("Federated client partition is not disjoint and complete")
    return arrays


def _plot_rounds(round_df: pd.DataFrame, results_dir: Path, demo=False):
    for col, ylabel, title, filename in [
        ("f1", "F1-score", "FedAvg global test F1 by round", "federated_f1_by_round.png"),
        ("fpr", "False positive rate", "FedAvg global test FPR by round", "federated_fpr_by_round.png"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(round_df["round"], round_df[col], marker="o")
        ax.set_xlabel("Federated round")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        if demo:
            fig.text(0.5, 0.01, "SYNTHETIC DATA - smoke test only", ha="center", fontsize=8)
        fig.tight_layout(rect=(0, 0.03 if demo else 0, 1, 1))
        fig.savefig(results_dir / filename, dpi=160)
        plt.close(fig)


def main(argv=None) -> None:
    ap = config.add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--clients", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=None, help="FedAvg communication rounds. Default: 6, or 3 with --quick.")
    ap.add_argument("--local-epochs", type=int, default=None, help="Local SGD epochs. Default: 2, or 1 with --quick.")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--l2", type=float, default=1e-4)
    ap.add_argument("--clip-norm", type=float, default=5.0)
    ap.add_argument("--dirichlet-alpha", type=float, default=0.35)
    args = ap.parse_args(argv)
    args.rounds = args.rounds if args.rounds is not None else (3 if args.quick else 6)
    args.local_epochs = args.local_epochs if args.local_epochs is not None else (1 if args.quick else 2)

    if args.clients < 2:
        raise ValueError("--clients must be >= 2")

    paths = config.resolve_paths(args)
    ensure_data(paths, args.quick)
    train, test, load_meta = data_loader.load_split(
        paths.data_dir, protocol=args.protocol, train_file=args.train_file,
        test_file=args.test_file, return_meta=True
    )
    X_raw, y_series = data_loader.split_xy(train, "binary")
    X_test_raw, y_test_series = data_loader.split_xy(test, "binary")

    # The fixed feature map is fitted once for this FL algorithmic baseline. This avoids
    # mismatched one-hot dimensions while keeping raw client examples local during rounds.
    prep = build_preprocessor(args.corr_threshold).fit(X_raw, y_series)
    X = prep.transform(X_raw).astype(np.float32)
    X_test = prep.transform(X_test_raw).astype(np.float32)
    y = y_series.to_numpy(dtype=np.float32)
    y_test = y_test_series.to_numpy(dtype=np.int32)
    joblib.dump(prep, paths.models_dir / "federated_preprocessor.joblib", compress=3)

    client_indices = make_disjoint_noniid_clients(train.reset_index(drop=True), args.clients,
                                                  alpha=args.dirichlet_alpha, seed=args.seed)
    dist_rows = []
    for name, idx in zip(CLIENT_NAMES[:args.clients], client_indices):
        subset = train.reset_index(drop=True).iloc[idx]
        counts = subset[config.TARGET_MULTI].value_counts()
        row = {"client": name, "rows": len(idx), "attack_rate": float(subset[config.TARGET_BINARY].mean())}
        for cat, count in counts.items():
            row[f"attack_{cat}"] = int(count) if cat != config.NORMAL_CLASS else int(count)
        dist_rows.append(row)
    dist_df = pd.DataFrame(dist_rows).fillna(0)
    dist_df.to_csv(paths.results_dir / "federated_client_distribution.csv", index=False, float_format="%.6g")

    w = np.zeros(X.shape[1], dtype=np.float32)
    b = 0.0
    round_rows = []
    start_all = time.perf_counter()
    for rnd in range(1, args.rounds + 1):
        client_states = []
        sizes = []
        round_start = time.perf_counter()
        for cid, idx in enumerate(client_indices):
            cw, cb = local_sgd(
                w, b, X[idx], y[idx], epochs=args.local_epochs, lr=args.learning_rate,
                batch_size=args.batch_size, l2=args.l2, clip_norm=args.clip_norm,
                seed=args.seed + rnd * 1000 + cid,
            )
            client_states.append((cw, cb))
            sizes.append(len(idx))
        w, b = weighted_fedavg(client_states, sizes)
        proba = sigmoid(X_test @ w + b)
        met = logreg_metrics(y_test, proba, threshold=0.5)
        round_rows.append({"round": rnd, "f1": met["f1"], "recall": met["recall"],
                           "precision": met["precision"], "fpr": met["fpr"],
                           "round_time_s": time.perf_counter() - round_start,
                           "n_clients": args.clients})
        print(f"[fedavg] round {rnd}/{args.rounds}: F1={met['f1']:.4f} recall={met['recall']:.4f} FPR={met['fpr']:.4f}", flush=True)

    round_df = pd.DataFrame(round_rows)
    round_df.to_csv(paths.results_dir / "federated_round_metrics.csv", index=False, float_format="%.6g")
    _plot_rounds(round_df, paths.results_dir, paths.demo)

    final_proba = sigmoid(X_test @ w + b)
    final = logreg_metrics(y_test, final_proba, threshold=0.5)
    total_s = time.perf_counter() - start_all

    np.savez_compressed(paths.models_dir / "federated_logreg_state.npz", coef=w, intercept=np.array([b], dtype=np.float32))
    summary = {
        "protocol": args.protocol,
        "seed": args.seed,
        "n_clients": args.clients,
        "rounds": args.rounds,
        "local_epochs": args.local_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "l2": args.l2,
        "clip_norm": args.clip_norm,
        "dirichlet_alpha": args.dirichlet_alpha,
        "n_features_out": int(X.shape[1]),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "total_training_time_s": total_s,
        "final_test_metrics": final,
        "data_partition": load_meta,
        "privacy_scope": "raw examples remain local during FedAvg aggregation; no formal DP or secure aggregation is implemented",
        "algorithm": "weighted FedAvg over logistic-regression parameters updated by local mini-batch SGD",
    }
    (paths.results_dir / "federated_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[fedavg] final test: F1={final['f1']:.4f} recall={final['recall']:.4f} precision={final['precision']:.4f} FPR={final['fpr']:.4f}")
    print("Wrote federated_round_metrics.csv, federated_client_distribution.csv, federated_summary.json and convergence plots.")


if __name__ == "__main__":
    main()
