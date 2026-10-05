# Dataset

This repository does **not** redistribute the UNSW-NB15 dataset.

## 1. Download the dataset

Obtain the predefined UNSW-NB15 training/testing CSV files from the official UNSW project page:

<https://research.unsw.edu.au/projects/unsw-nb15-dataset>

Place the two files in:

```text
data/raw/
├── UNSW_NB15_training-set.csv
└── UNSW_NB15_testing-set.csv
```

The loader also accepts matching `.parquet` files when `pyarrow` is installed.

## 2. Filename-swap protection

The project does **not** blindly trust filenames. When the two canonical row counts are present:

```text
175,341 rows -> TRAIN
 82,332 rows -> TEST
```

the loader assigns roles by row count. If the two filenames are swapped, the run continues and records the correction in `results/dataset_summary.json`.

For custom or non-canonical datasets, pass explicit files:

```bash
python3 run_all.py \
  --train-file path/to/train.csv \
  --test-file path/to/test.csv
```

## 3. Research protocol vs benchmark protocol

### Research protocol (recommended)

```bash
python3 run_all.py --protocol research
```

Before model fitting, it:

- removes exact duplicate training rows;
- removes training rows that exactly overlap the resolved test partition;
- never modifies the test partition.

### Benchmark protocol

```bash
python3 run_all.py --protocol benchmark
```

This keeps the resolved predefined partitions intact so that you can compare against papers that explicitly used the raw files. The report still records duplicate and overlap counts.

## 4. Data are never committed to GitHub

Raw datasets are intentionally excluded by `.gitignore`. Keep only the placeholder:

```text
data/raw/.gitkeep
```

This makes the repository reproducible without redistributing third-party dataset files.
