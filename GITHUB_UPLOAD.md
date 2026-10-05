# GitHub Upload Checklist

1. Open this folder in PyCharm.
2. Create a local virtual environment; do not commit `.venv`.
3. Install dependencies:
   `python3 -m pip install -r requirements.txt`
4. Run tests:
   `python3 -m pytest -q`
5. Run the smoke test:
   `python3 run_all.py --demo --quick`
6. Download UNSW-NB15 separately and place its CSV files in `data/raw/`.
7. Run the full research workflow:
   `python3 run_all.py --protocol research --target-fpr 0.05 --update-readme`
8. Initialize Git and push the repository to GitHub.

Raw datasets and generated/local artifacts are intentionally excluded from the GitHub package.
