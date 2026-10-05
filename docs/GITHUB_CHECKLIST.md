# GitHub Upload Checklist

Before the first public push:

- [ ] `.venv/` is ignored.
- [ ] Raw UNSW-NB15 files are ignored.
- [ ] Generated `.joblib`/`.npz` models are ignored.
- [ ] `README.md` accurately describes threshold tuning and the FedAvg experiment.
- [ ] `python3 -m pytest -q` passes.
- [ ] `python3 run_all.py --demo --quick` passes.
- [ ] No absolute local filesystem paths appear in committed documentation.
- [ ] No API keys, passwords, credentials, or private datasets are committed.
- [ ] Repository description clearly states that this is an offline ML NIDS research prototype.
- [ ] README results are generated from the exact protocol/seed and are not manually copied from another run.
- [ ] The FL section does not claim formal differential privacy or secure aggregation.
