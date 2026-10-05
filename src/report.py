"""Generate a concise research report from the latest experiment outputs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config

START = "<!-- RESULTS:START -->"
END = "<!-- RESULTS:END -->"
P = config.PRETTY


def _fmt(v, digits=4):
    if pd.isna(v): return "n/a"
    return f"{float(v):.{digits}f}"


def md_table(df: pd.DataFrame, index_name="Model") -> str:
    d = df.copy()
    d.columns = [str(c) for c in d.columns]
    heads = [index_name] + list(d.columns)
    rows = ["| " + " | ".join(heads) + " |", "| " + " | ".join(["---"] * len(heads)) + " |"]
    for idx, row in d.iterrows():
        cells = [P.get(idx, idx) if isinstance(idx, str) else str(idx)]
        for v in row:
            cells.append(str(v))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def eda_block(res: Path) -> str:
    p = res / "dataset_summary.json"
    if not p.exists(): return "_Dataset summary not available._"
    d = json.loads(p.read_text())
    part = d["partition_resolution"]; h = d["load_meta"]["hygiene"]
    return (f"- Partition resolution: **{part['mode']}**; filename swap corrected: **{bool(part.get('filename_roles_swapped'))}**.\n"
            f"- TRAIN: `{Path(part['resolved_train_file']).name}` ({part['train_rows']:,} raw rows); "
            f"TEST: `{Path(part['resolved_test_file']).name}` ({part['test_rows']:,} raw rows).\n"
            f"- Research protocol training rows after hygiene: **{d['model_train']['records']:,}**; "
            f"test rows (untouched): **{d['model_test']['records']:,}**.\n"
            f"- Removed from TRAIN: **{h['train_duplicate_rows_removed']:,}** exact duplicates + "
            f"**{h['train_test_overlap_rows_removed']:,}** train/test overlaps.\n"
            f"- Correlated numeric pairs at threshold {d.get('correlation', {}).get('threshold', 'configured')}: **{len(d.get('correlated_pairs', {}))}**.")


def binary_block(res: Path) -> tuple[str, str]:
    p = res / "metrics.csv"
    if not p.exists(): return "_Binary metrics unavailable._", ""
    m = pd.read_csv(p, index_col=0)
    cols = ["threshold", "precision", "recall", "f1", "fpr", "fnr", "roc_auc", "pr_auc", "recall_at_fpr_1pct", "recall_at_fpr_5pct"]
    view = m[[c for c in cols if c in m.columns]].copy()
    for c in view.columns:
        if c == "threshold": view[c] = view[c].map(lambda x: _fmt(x, 4))
        else: view[c] = view[c].map(lambda x: _fmt(x, 4))
    view.columns = [c.upper().replace("_", " ") for c in view.columns]
    tuned = md_table(view)

    default_cols = ["default_precision", "default_recall", "default_f1", "default_fpr"]
    dv = m[[c for c in default_cols if c in m.columns]].copy()
    if not dv.empty:
        dv.columns = ["Precision", "Recall", "F1", "FPR"][: len(dv.columns)]
        dv = dv.map(lambda x: _fmt(x, 4))
        default = md_table(dv, index_name="Model / default 0.5")
    else:
        default = ""
    return tuned, default


def findings_block(res: Path) -> str:
    p = res / "metrics.csv"
    if not p.exists(): return "_No binary results._"
    m = pd.read_csv(p, index_col=0)
    best_fpr = m["fpr"].idxmin(); best_f1 = m["f1"].idxmax(); best_recall = m["recall"].idxmax()
    return (f"- Highest tuned test F1: **{P.get(best_f1, best_f1)}** ({m.loc[best_f1, 'f1']:.4f}).\n"
            f"- Highest tuned test recall: **{P.get(best_recall, best_recall)}** ({m.loc[best_recall, 'recall']:.4f}).\n"
            f"- Lowest tuned test FPR: **{P.get(best_fpr, best_fpr)}** ({m.loc[best_fpr, 'fpr']:.4f}).\n"
            "- The tuned threshold is selected from OOF training predictions only; the independent TEST partition is never used to choose it.\n"
            "- A test FPR above the training target is interpreted as an operating-point/generalization gap, not hidden by the report.")


def federated_block(res: Path) -> str:
    p = res / "federated_summary.json"
    if not p.exists(): return "_Federated experiment unavailable._"
    d = json.loads(p.read_text()); m = d["final_test_metrics"]
    dist = pd.read_csv(res / "federated_client_distribution.csv") if (res / "federated_client_distribution.csv").exists() else None
    out = (f"**Algorithm:** {d['algorithm']}. Raw client examples remain local during the aggregation step; "
           "this baseline does not implement formal differential privacy or secure aggregation.\n\n"
           f"**Clients:** {d['n_clients']} disjoint clients; **rounds:** {d['rounds']}; **Dirichlet alpha:** {d['dirichlet_alpha']}.\n\n"
           f"**Final global TEST metrics at threshold 0.5:** precision={m['precision']:.4f}, recall={m['recall']:.4f}, "
           f"F1={m['f1']:.4f}, FPR={m['fpr']:.4f}.")
    if dist is not None:
        show = dist[[c for c in ["client", "rows", "attack_rate"] if c in dist.columns]].copy()
        show["attack_rate"] = show["attack_rate"].map(lambda x: f"{x:.2%}")
        out += "\n\n" + md_table(show.set_index("client"), index_name="Client")
    return out


def build_report(res: Path, info: dict) -> str:
    tuned, default = binary_block(res)
    multi = "_Multiclass results unavailable._"
    mm = res / "metrics_multiclass.csv"
    if mm.exists():
        d = pd.read_csv(mm, index_col=0)[["accuracy", "macro_f1", "weighted_f1", "macro_recall"]]
        d.columns = ["Accuracy", "Macro-F1", "Weighted-F1", "Macro-recall"]
        d = d.map(lambda x: _fmt(x, 4)); multi = md_table(d)
    return f"""# Experiment Report: ML-Based Network Intrusion Detection

## 1. Dataset and integrity

{eda_block(res)}

## 2. Binary intrusion detection

The primary operating point uses an FPR-constrained decision threshold selected from out-of-fold training predictions. Threshold `0.5` is retained as a secondary baseline.

{tuned}

### Secondary threshold-0.5 baseline

{default or '_Not available._'}

## 3. Findings

{findings_block(res)}

## 4. Multiclass attack identification

{multi}

## 5. Federated-learning experiment

{federated_block(res)}

## 6. Limitations

- Threshold tuning optimizes an operating point; it does not change the underlying class-conditional distributions or guarantee a target FPR on unseen data.
- The federated component is a simulation over public data, not a deployment across independently managed organizations.
- The FL feature map is fitted once over the cleaned training partition so that clients share a common feature space; therefore this repository should **not** claim formal end-to-end privacy.
- No Byzantine attack or Byzantine-resilient aggregation is implemented yet; that is the next research stage.
- Timing and throughput depend on hardware and software versions.

## 7. Reproducibility

The exact seed, protocol, resolved partitions, cleanup counts, package versions, and threshold-selection metadata are stored under `results/` and `models/`.
"""


def update_readme(info, args):
    readme = config.ROOT / "README.md"
    text = readme.read_text()
    if START not in text or END not in text:
        raise RuntimeError("README.md is missing RESULTS markers.")
    res = config.resolve_paths(args).results_dir
    tuned, default = binary_block(res)
    f = findings_block(res)
    fed = federated_block(res)
    block = f"""{START}

### Latest experiment snapshot

**Protocol:** `{info.get('protocol', args.protocol)}`  
**Seed:** `{info.get('seed', config.SEED)}`  
**Primary threshold:** OOF-selected under target FPR `{info.get('target_fpr', config.DEFAULT_TARGET_FPR):.2%}`  
**Test set:** untouched during fitting and threshold selection.

#### Binary IDS — primary tuned operating point

{tuned}

#### Binary IDS — default threshold 0.5

{default or '_Not available._'}

#### Findings

{f}

#### Federated-learning baseline

{fed}

Full report: [`reports/experiment_report.md`](reports/experiment_report.md)

{END}"""
    pre, rest = text.split(START, 1); _, post = rest.split(END, 1)
    readme.write_text(pre + block + post)


def main(argv=None):
    ap = config.add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--update-readme", action="store_true")
    args = ap.parse_args(argv)
    paths = config.resolve_paths(args); res = paths.results_dir
    if not (res / "metrics.csv").exists():
        raise SystemExit("metrics.csv not found. Run the pipeline first.")
    info = json.loads((res / "run_info.json").read_text()) if (res / "run_info.json").exists() else {}
    out = paths.report_dir / "experiment_report.md"
    out.write_text(build_report(res, info))
    print(f"Wrote {out}")
    if args.update_readme:
        if paths.demo: raise SystemExit("Refusing to write synthetic demo results into README.md.")
        update_readme(info, args); print("README.md results block updated.")


if __name__ == "__main__":
    main()
