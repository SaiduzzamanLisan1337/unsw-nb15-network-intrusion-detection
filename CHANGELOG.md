# Changelog

## 2.0.0 — 2026-10-04

- Reworked binary evaluation around an out-of-fold, FPR-constrained operating-point selection policy.
- Added separate threshold-0.5 baseline reporting so threshold tuning cannot hide the original model behavior.
- Added explicit decision-threshold metadata and threshold comparison outputs.
- Replaced the old independent-client heterogeneity experiment with a genuine weighted-FedAvg federated-learning simulation.
- Added disjoint, non-IID client partitioning using a Dirichlet allocation across attack families.
- Added round-by-round federated F1/FPR tracking and client-distribution outputs.
- Clarified that the current FL component provides no formal differential-privacy or secure-aggregation guarantee.
- Added tests for threshold selection, disjoint federated clients, and weighted FedAvg.
- Updated research documentation, architecture, report generation, and GitHub instructions.

## 1.0.0 — 2026-10-04

- Initial GitHub-ready ML-based network intrusion detection baseline.
