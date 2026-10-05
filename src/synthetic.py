"""Synthetic stand-in with the same schema as the official UNSW-NB15 train/test CSVs.

PURPOSE: let anyone run the whole pipeline end-to-end (CI, smoke test, first look at
the outputs) without downloading the dataset. The numbers it produces say NOTHING
about intrusion detection and must never be reported as results.

Usage:  python3 src/synthetic.py [--out data/synthetic] [--n-train 20000] [--n-test 10000]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402

# approximate class proportions of the official UNSW-NB15 training partition
CLASS_PRIORS = {
    "Normal": 0.319, "Generic": 0.224, "Exploits": 0.190, "Fuzzers": 0.104, "DoS": 0.070,
    "Reconnaissance": 0.060, "Analysis": 0.011, "Backdoor": 0.010, "Shellcode": 0.0065,
    "Worms": 0.0007,
}

CONTINUOUS = [
    "dur", "spkts", "dpkts", "sbytes", "dbytes", "rate", "sload", "dload", "sloss", "dloss",
    "sinpkt", "dinpkt", "sjit", "djit", "stcpb", "dtcpb", "tcprtt", "synack", "ackdat",
    "smean", "dmean", "response_body_len",
]
COUNTS = [
    "trans_depth", "ct_srv_src", "ct_state_ttl", "ct_dst_ltm", "ct_src_dport_ltm",
    "ct_dst_sport_ltm", "ct_dst_src_ltm", "ct_ftp_cmd", "ct_flw_http_mthd", "ct_src_ltm",
    "ct_srv_dst",
]
PROTOS = ["tcp", "udp", "icmp", "arp", "ospf", "sctp", "gre", "ipv6-icmp", "rsvp"]
SERVICES = ["-", "http", "dns", "ftp", "ftp-data", "smtp", "ssh", "pop3", "ssl"]
STATES = ["FIN", "INT", "CON", "REQ", "RST", "ACC"]


def _make(n: int, rng: np.random.Generator, class_params: dict, shift: float) -> pd.DataFrame:
    classes = list(CLASS_PRIORS)
    p = np.array([CLASS_PRIORS[c] for c in classes])
    cats = rng.choice(classes, size=n, p=p / p.sum())
    df = pd.DataFrame({"attack_cat": cats})
    df["label"] = (df["attack_cat"] != config.NORMAL_CLASS).astype(int)

    for col in CONTINUOUS:
        mu = np.array([class_params["mu"][(c, col)] for c in cats]) + rng.normal(0, shift, n)
        df[col] = np.exp(rng.normal(mu, 1.1)) - 1.0
    for col in COUNTS:
        lam = np.array([class_params["lam"][(c, col)] for c in cats])
        df[col] = rng.poisson(lam * np.exp(rng.normal(0, shift, n)))

    for col, choices in (("proto", PROTOS), ("service", SERVICES), ("state", STATES)):
        probs = np.stack([class_params["cat"][(c, col)] for c in cats])
        u = rng.random(n)[:, None]
        df[col] = np.array(choices)[(probs.cumsum(axis=1) < u).sum(axis=1).clip(0, len(choices) - 1)]

    for col in ("sttl", "dttl"):
        opts = np.array([31, 62, 254])
        probs = np.stack([class_params["ttl"][(c, col)] for c in cats])
        u = rng.random(n)[:, None]
        df[col] = opts[(probs.cumsum(axis=1) < u).sum(axis=1).clip(0, 2)]
    for col in ("swin", "dwin"):
        df[col] = np.where(rng.random(n) < 0.7, 255, 0)
    df["is_ftp_login"] = (rng.random(n) < 0.01).astype(int)
    df["is_sm_ips_ports"] = (rng.random(n) < np.where(df["label"] == 1, 0.02, 0.003)).astype(int)

    df.insert(0, "id", np.arange(1, n + 1))
    order = [
        "id", "dur", "proto", "service", "state", "spkts", "dpkts", "sbytes", "dbytes", "rate",
        "sttl", "dttl", "sload", "dload", "sloss", "dloss", "sinpkt", "dinpkt", "sjit", "djit",
        "swin", "stcpb", "dtcpb", "dwin", "tcprtt", "synack", "ackdat", "smean", "dmean",
        "trans_depth", "response_body_len", "ct_srv_src", "ct_state_ttl", "ct_dst_ltm",
        "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm", "is_ftp_login", "ct_ftp_cmd",
        "ct_flw_http_mthd", "ct_src_ltm", "ct_srv_dst", "is_sm_ips_ports", "attack_cat", "label",
    ]
    return df[order]


def _class_params(seed: int) -> dict:
    """Class-conditional distributions: sparse, overlapping signal (classes are separable
    but not trivially so)."""
    rng = np.random.default_rng(seed)
    params = {"mu": {}, "lam": {}, "cat": {}, "ttl": {}}
    base_mu = {c: rng.normal(3.0, 1.0) for c in CONTINUOUS}
    base_lam = {c: rng.uniform(1, 12) for c in COUNTS}
    for cls in CLASS_PRIORS:
        for c in CONTINUOUS:
            bump = rng.normal(0, 0.9) if rng.random() < 0.45 else 0.0
            params["mu"][(cls, c)] = base_mu[c] + (0.0 if cls == "Normal" else bump)
        for c in COUNTS:
            mult = rng.uniform(0.4, 2.5) if (cls != "Normal" and rng.random() < 0.5) else 1.0
            params["lam"][(cls, c)] = base_lam[c] * mult
        for col, choices in (("proto", PROTOS), ("service", SERVICES), ("state", STATES)):
            d = rng.dirichlet(np.ones(len(choices)) * (0.6 if cls != "Normal" else 1.5))
            params["cat"][(cls, col)] = d
        for col in ("sttl", "dttl"):
            params["ttl"][(cls, col)] = rng.dirichlet(np.ones(3) * 1.2)
    return params


def generate(out_dir: Path, n_train: int = 20000, n_test: int = 10000, seed: int = config.SEED) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    params = _class_params(seed)
    train = _make(n_train, np.random.default_rng(seed + 1), params, shift=0.10)
    test = _make(n_test, np.random.default_rng(seed + 2), params, shift=0.25)  # mild shift
    train.to_csv(out_dir / f"{config.TRAIN_FILE_STEM}.csv", index=False)
    test.to_csv(out_dir / f"{config.TEST_FILE_STEM}.csv", index=False)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=config.ROOT / "data" / "synthetic")
    ap.add_argument("--n-train", type=int, default=20000)
    ap.add_argument("--n-test", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=config.SEED)
    a = ap.parse_args(argv)
    generate(a.out, a.n_train, a.n_test, a.seed)
    print(f"Synthetic UNSW-NB15-like files written to {a.out}  (NOT real data)")


if __name__ == "__main__":
    main()
