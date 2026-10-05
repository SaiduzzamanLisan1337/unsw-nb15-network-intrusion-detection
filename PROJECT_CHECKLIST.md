# Project Quality Checklist

## Dataset integrity

- [x] Detect canonical UNSW-NB15 train/test row counts.
- [x] Correct swapped filenames automatically.
- [x] Report duplicate and train/test-overlap counts.
- [x] Remove duplicate/overlap rows only from training in research mode.
- [x] Keep the test partition untouched.

## ML correctness

- [x] Common preprocessing design across baseline models.
- [x] Train-fitted preprocessing.
- [x] Explicit class-aware training.
- [x] Logistic Regression baseline.
- [x] Random Forest baseline.
- [x] XGBoost baseline.

## False-positive control

- [x] Out-of-fold probability generation.
- [x] Threshold selection without test-set access.
- [x] FPR-constrained primary operating point.
- [x] Secondary threshold-0.5 comparison.
- [x] Test-set FPR explicitly reported even when it exceeds the training target.

## Evaluation

- [x] Precision.
- [x] Recall.
- [x] F1.
- [x] FPR/FNR.
- [x] ROC-AUC.
- [x] PR-AUC.
- [x] Confusion matrices.
- [x] Multiclass per-class results.
- [x] Runtime/latency/throughput.

## Federated learning

- [x] Real FedAvg parameter aggregation.
- [x] Disjoint client datasets.
- [x] Non-IID client distributions.
- [x] Round-by-round global evaluation.
- [x] Client-distribution report.
- [x] Explicit privacy-scope limitation.
- [ ] Secure aggregation.
- [ ] Differential privacy.
- [ ] Byzantine attacks/robust aggregation.
- [ ] Continual learning/concept-drift adaptation.

## GitHub hygiene

- [x] Raw datasets ignored.
- [x] Virtual environments ignored.
- [x] Generated model artifacts ignored.
- [x] CI workflow included.
- [x] Responsible-use policy included.
- [x] Contribution guidelines included.
- [x] Reproducible run instructions included.
