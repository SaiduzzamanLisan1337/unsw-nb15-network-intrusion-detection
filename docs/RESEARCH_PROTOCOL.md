# Research Protocol

## Primary protocol: `research`

The default experiment is designed to reduce obvious record-level leakage and to separate model development from independent test evaluation.

### Step 1 — Resolve partitions

When both predefined UNSW-NB15 files are present, canonical row counts determine their TRAIN/TEST roles even if filenames are accidentally swapped.

### Step 2 — Clean partitions

- normalize column names and categorical values;
- convert invalid numeric values to missing;
- remove labels/identifiers from model features.

### Step 3 — Research hygiene

Only the training partition is modified:

- exact duplicate training rows are removed;
- exact training rows also present in TEST are removed.

TEST remains untouched.

### Step 4 — Baseline CV

Binary models use stratified folds and produce out-of-fold probabilities.

### Step 5 — Threshold selection

The threshold is selected from OOF probabilities only:

> maximize F1 subject to OOF FPR <= target FPR.

The default target is `0.05`.

### Step 6 — Final fitting

After threshold selection, each final model is fit on the full cleaned TRAIN partition.

### Step 7 — Independent evaluation

The stored threshold is applied to the untouched TEST partition. Both tuned and threshold-0.5 results are reported.

### Step 8 — Federated experiment

The cleaned TRAIN partition is split into disjoint non-IID clients. During federated rounds, clients locally update a logistic-regression state and the server combines states using weighted FedAvg.

The current FL implementation is a simulation: raw examples remain local during the optimization step, but no formal privacy guarantee is claimed.

## Benchmark protocol: `benchmark`

Benchmark mode preserves the resolved predefined partitions, including any duplicate/overlap records, so published results can be reproduced when necessary. All such integrity information is still reported.
