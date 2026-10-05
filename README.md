# ML-Based Network Intrusion Detection

A reproducible, research-oriented **Network Intrusion Detection System (NIDS) prototype** that uses machine learning to classify network-flow records as **Benign** or **Attack**. The project compares **Logistic Regression, Random Forest, and XGBoost**, evaluates binary and multiclass detection, studies heterogeneous client behavior, and includes a baseline **Federated Averaging (FedAvg)** experiment.

> **Research context:** This repository is designed as a baseline for future work on **concept drift, continual learning, privacy-preserving federated learning, and Byzantine-robust intrusion detection**.

---

## 1. Project Overview & Classification

### IDS Type

**Network Intrusion Detection System (NIDS)** — offline, flow/feature-based research prototype.

### Detection Methodology

**Machine-learning-based intrusion classification** using supervised learning.

Current models:

- Logistic Regression
- Random Forest
- XGBoost

### Primary Detection Tasks

**Binary classification**

```text
Benign <-> Attack
```

**Multiclass classification**

Attack categories represented by the UNSW-NB15 benchmark are evaluated individually in addition to the binary task.

### Core Capabilities

The system learns from network-flow features associated with attack traffic represented in the selected benchmark dataset, including categories such as:

- DoS
- Exploits
- Reconnaissance
- Fuzzers
- Generic
- Analysis
- Backdoors
- Shellcode
- Worms

> **Important:** detection capability is limited to behaviors represented by the training data and experimental setup. This is not a signature engine and does not claim comprehensive detection of all real-world threats.

---

# 2. Architecture & Threat Model

## High-Level Architecture

```text
                    OFFLINE NETWORK FLOW DATA
                              |
                              v
                    +----------------------+
                    | Dataset Integrity    |
                    | & Hygiene Checks     |
                    | - duplicates         |
                    | - train/test overlap |
                    | - missing/Inf        |
                    +----------+-----------+
                               |
                               v
                    +----------------------+
                    | Preprocessing        |
                    | - feature selection  |
                    | - categorical encode |
                    | - scaling            |
                    | - imbalance handling |
                    +----------+-----------+
                               |
                 +-------------+-------------+
                 |             |             |
                 v             v             v
          +-----------+  +-----------+  +-----------+
          | Logistic  |  | Random    |  | XGBoost   |
          | Regression|  | Forest    |  |           |
          +-----------+  +-----------+  +-----------+
                 |             |             |
                 +-------------+-------------+
                               |
                               v
                    +----------------------+
                    | Evaluation           |
                    | Precision / Recall   |
                    | F1 / FPR / PR-AUC    |
                    | Confusion Matrix     |
                    +----------+-----------+
                               |
                               v
                    +----------------------+
                    | Research Extensions  |
                    | - Heterogeneity      |
                    | - FedAvg baseline    |
                    | - Concept drift      |
                    | - Continual learning |
                    +----------------------+
```

## Packet Capture Method

This repository **does not perform live packet capture**.

It operates on pre-generated network-flow features from public cybersecurity datasets.

Therefore it does **not currently use**:

- `libpcap`
- AF_PACKET
- PF_RING
- DPDK

These technologies are relevant to production/high-throughput NIDS sensors, but live packet acquisition is outside the scope of this research baseline.

## Processing Pipeline

```text
Network-flow records
        |
        v
Data validation
        |
        v
Leakage-safe preprocessing
        |
        v
ML classifier
        |
        v
Probability / class prediction
        |
        v
Threshold-based binary decision
        |
        v
Metrics + plots + report
```

## Threat Model

The current baseline assumes:

- Network traffic has already been converted into tabular flow/features.
- Training data may contain benign and malicious network activity.
- The detector is intended to identify malicious/anomalous traffic patterns.
- The final test partition is held out from model fitting.
- Federated experiments model multiple virtual organizations/clients with heterogeneous local data.

The current repository does **not** implement:

- Byzantine client attacks or defenses
- model poisoning defenses
- secure aggregation
- differential privacy
- continual learning
- live packet interception
- automated blocking/prevention

These are future research extensions.

## Deployment Limitations

This is an **IDS research prototype**, not an IPS.

It:

- detects/classifies traffic;
- does **not** block packets;
- does **not** modify firewall rules;
- does **not** perform automated containment.

It is also **offline and flow-based**, so it does not inspect raw application payloads. It does not decrypt TLS traffic; predictions are based only on the available flow/connection features.

---

# 3. Prerequisites & Dependencies

## Recommended Environment

The project is designed for Linux/macOS/Windows systems with Python 3.9+.

Suggested environment for UNSW-NB15 experiments:

- 4+ CPU cores
- 8 GB RAM minimum
- 16 GB RAM recommended
- 5+ GB free disk space for dataset/results
- GPU optional

> No 100-Mbps/1-Gbps/10-Gbps throughput claim is made because this repository is not a live packet-capture engine. Any future real-time deployment must be benchmarked separately.

## Software Dependencies

Core libraries include:

- Python
- NumPy
- pandas
- scikit-learn
- XGBoost
- imbalanced-learn
- matplotlib
- joblib
- pytest

Install with:

```bash
python3 -m pip install -r requirements.txt
```

Development dependencies:

```bash
python3 -m pip install -r requirements-dev.txt
```

---

# 4. Installation & Deployment

## Clone

```bash
git clone https://github.com/YOUR-USERNAME/ml-network-intrusion-detection.git
cd ml-network-intrusion-detection
```

## Create Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

## Install Requirements

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

## Verify Installation

```bash
python3 -m pytest -q
```

Run the included smoke test:

```bash
python3 run_all.py --demo --quick
```

## Containerization

The current project is intended primarily as a Python research workflow. It does not require Docker, and it does not provide a privileged live-NIC container deployment.

A future production-oriented version could provide Docker/Compose images for a packet/flow acquisition layer plus the ML inference service.

## Deployment Placement Guidance

Because the current system is offline, it has **no live sensor placement requirement**.

A future live NIDS deployment would typically place a sensor on a SPAN/mirror port, network TAP, or another authorized observation point and feed an extracted flow schema into the ML engine.

---

# 5. Configuration & Rulesets

## Configuration

The project is configured primarily through Python modules and command-line options rather than a Snort/Suricata-style rules engine.

Key experimental settings include:

- protocol (`research` / `benchmark`)
- random seed
- cross-validation folds
- target FPR for threshold selection
- demo/quick mode
- federated client/round settings

## Signature Rulesets

This project is **not** compatible with Snort or Suricata signature/rule formats.

It does not use:

- Snort rules
- Suricata rules
- Sigma rules
- YARA rules

Detection is learned from network-flow features using ML models.

## False-Positive Tuning

The research protocol does not optimize solely for accuracy. For the binary task, an operating threshold is selected from **training-only out-of-fold predictions** subject to a target false-positive rate constraint.

The held-out test set is not used to select that threshold.

This makes it possible to analyze the gap between an intended operating point during model development and actual performance on an independent distribution.

---

# 6. Usage & Testing

## Run the Full Research Pipeline

```bash
python3 run_all.py --protocol research --target-fpr 0.05 --update-readme
```

The pipeline performs:

```text
1. Dataset analysis
2. Model training
3. Held-out evaluation
4. Federated baseline experiment
5. Experiment report generation
```

## Demo / Smoke Test

```bash
python3 run_all.py --demo --quick
```

## Run Tests

```bash
python3 -m pytest -q
```

## Dataset Location

Place the UNSW-NB15 CSV files under:

```text
data/raw/
```

The project includes safeguards against accidentally swapped filenames. It resolves the training/testing roles using expected partition sizes and records the resolved roles in the generated metadata.

## Offline PCAP Testing

The current project does **not** parse PCAP files directly.

A future ingestion layer could use tools such as:

- Zeek
- Suricata
- Argus
- CICFlowMeter
- custom flow extraction pipelines

The intended future pipeline is:

```text
PCAP / NIC
   |
   v
Flow extraction
   |
   v
Feature schema
   |
   v
ML IDS
```

## Safe Testing

Use the included demo data, benchmark data, pre-recorded traffic, or an isolated laboratory network. Do not actively scan or test systems without explicit authorization.

Because the current repository is offline, it does not provide an active attack trigger.

---

# 7. Logging & SIEM Integration

## Current Outputs

Research artifacts are written under:

```text
results/
reports/
```

Typical outputs include:

- dataset summary JSON
- model metrics CSV
- confusion matrices
- ROC/PR curves
- feature-importance plots
- federated convergence metrics
- federated client-distribution metadata
- experiment report

## Alert Format

The current system produces **research prediction/metric artifacts**, not a production alert stream such as EVE JSON or syslog.

It therefore does not directly integrate with:

- Splunk
- Elasticsearch/Kibana
- Wazuh
- Microsoft Sentinel
- other SIEMs

A future deployment layer could expose model predictions as JSON events and forward them to a SIEM/message broker.

Example future event schema:

```json
{
  "timestamp": "...",
  "source": "ml-nids",
  "prediction": "attack",
  "score": 0.93,
  "model": "xgboost"
}
```

---

# 8. Performance Metrics & Benchmarks

## Binary Classification — Held-Out Test

Results from the current UNSW-NB15 research-protocol run after training-data hygiene and training-only OOF threshold selection:

| Model | Precision | Recall | F1 | FPR | PR-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8861 | 0.8685 | 0.8772 | 13.68% | 0.9648 |
| Random Forest | 0.8640 | 0.9648 | 0.9116 | 18.61% | 0.9822 |
| XGBoost | 0.8584 | 0.9663 | 0.9092 | 19.52% | 0.9869 |

## Multiclass Classification — Held-Out Test

| Model | Accuracy | Macro-F1 | Weighted-F1 | Macro Recall |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.6129 | 0.3443 | 0.6795 | 0.5593 |
| Random Forest | 0.6836 | 0.5150 | 0.7366 | 0.6347 |
| XGBoost | 0.6900 | 0.5284 | 0.7426 | 0.6534 |

## Interpretation

The results show strong attack recall but a substantial false-alarm problem on the independent test partition.

For XGBoost:

```text
Recall ≈ 96.6%
F1     ≈ 90.9%
FPR    ≈ 19.5%
```

Although the primary operating threshold was selected from training OOF predictions at approximately **5% OOF FPR**, the independent test FPR was substantially higher. This is reported as a **distribution/generalization limitation**, not hidden through test-set tuning.

## Training Performance

The current run used 5-fold stratified cross-validation. Representative binary CV results:

| Model | CV F1 | CV Recall | CV Precision | CV PR-AUC |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.9131 | 0.9713 | 0.8615 | 0.9662 |
| Random Forest | 0.9379 | 0.9518 | 0.9244 | 0.9870 |
| XGBoost | **0.9411** | 0.9502 | **0.9322** | **0.9884** |

The lower independent-test performance compared with cross-validation is an important generalization signal and motivates research into distribution shift and concept drift.

---

# 9. Federated Learning Baseline

The repository includes a baseline **Federated Averaging (FedAvg)** experiment using simulated clients created from public benchmark data.

```text
Client A ─┐
Client B ─┤
Client C ─┼──> Local training → FedAvg → Global model
Client D ─┘
```

The experiment is intended to study the effects of distributed and heterogeneous data. It is **not** a claim that the project already provides a privacy-preserving or Byzantine-robust FL system.

## Current FedAvg Result

Six communication rounds were evaluated:

| Round | F1 | Recall | FPR |
|---:|---:|---:|---:|
| 1 | 0.8064 | 0.9471 | 49.24% |
| 2 | 0.8263 | 0.9595 | 44.45% |
| 3 | 0.8314 | 0.9662 | 43.88% |
| 4 | 0.8347 | 0.9727 | 43.87% |
| 5 | 0.8361 | 0.9748 | 43.74% |
| 6 | **0.8372** | **0.9760** | 43.58% |

Final baseline:

```text
F1        = 0.8372
Precision = 0.7329
Recall    = 0.9760
FPR       = 43.58%
```

The substantial degradation relative to centralized learning under the current heterogeneous simulation motivates further work on non-IID learning, personalization, concept drift, robust aggregation, and privacy mechanisms.

> **Evaluation note:** future publication-grade federated experiments should reserve a dedicated federated validation set for model/round selection and use the held-out test set only for final evaluation.

---

# 10. Limitations

- **Offline flow processing:** no live packet capture or packet-throughput benchmarking.
- **Benchmark dependence:** public datasets may not fully represent operational networks.
- **Distribution shift:** independent-test FPR is substantially higher than development/OOF FPR.
- **Federated simulation:** clients are simulated rather than real organizations.
- **No Byzantine robustness yet:** FedAvg is only a baseline.
- **No formal privacy guarantee yet:** no secure aggregation or differential privacy is implemented.
- **No continual-learning mechanism yet:** the current model does not automatically adapt online to concept drift.
- **No IPS capability:** the system detects but does not block or contain traffic.

---

# 11. Research Roadmap

This repository is intentionally the baseline stage of a larger PhD-oriented research program:

```text
Stage 1
ML-Based Network IDS
        |
        v
Stage 2
Concept Drift Detection
        |
        v
Stage 3
Continual Learning IDS
        |
        v
Stage 4
Federated Intrusion Detection
        |
        v
Stage 5
Non-IID / Personalized Federated IDS
        |
        v
Stage 6
Byzantine Attack Modeling
        |
        v
Stage 7
Byzantine-Robust Aggregation
        |
        v
Stage 8
Privacy-Preserving Federated Learning
        |
        v
Stage 9
Integrated Byzantine-Robust Federated
Continual Intrusion Detection
```

This progression aligns with the broader research direction:

> **Byzantine-Robust Federated Intrusion Detection: Privacy-Preserving Continual Adaptation to Concept Drift in Distributed Networks**

---

# 12. Project Structure

```text
ml-network-intrusion-detection/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── data/
│   ├── raw/
│   │   └── .gitkeep
│   └── README.md
│
├── docs/
│   ├── ARCHITECTURE.md
│   ├── GITHUB_CHECKLIST.md
│   └── RESEARCH_PROTOCOL.md
│
├── models/
│   ├── .gitkeep
│   └── README.md
│
├── reports/
├── results/
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── eda.py
│   ├── evaluate.py
│   ├── heterogeneity.py
│   ├── models.py
│   ├── preprocessing.py
│   ├── report.py
│   ├── synthetic.py
│   ├── train.py
│   └── visualization.py
│
├── tests/
│   └── test_pipeline.py
│
├── .gitignore
├── CITATION.cff
├── LICENSE
├── README.md
├── requirements.txt
├── requirements-dev.txt
└── run_all.py
```

---

# 13. Reproducibility

A valid experiment should document:

- dataset source/version;
- train/test role resolution;
- dataset-cleaning protocol;
- random seed;
- feature-processing steps;
- model configuration;
- cross-validation settings;
- threshold-selection procedure;
- evaluation metrics;
- software dependencies.

Raw datasets, virtual environments, generated caches, and local model artifacts should remain outside the public Git history unless their licenses and project goals explicitly allow distribution.

---

# 14. Citation

If you use this repository or UNSW-NB15 in academic work, cite the original dataset source and the relevant research papers.

**UNSW-NB15 official dataset:**

https://research.unsw.edu.au/projects/unsw-nb15-dataset

---

# 15. Responsible Use

This project is intended for:

- academic research;
- defensive cybersecurity;
- controlled testing;
- machine-learning experimentation;
- reproducible IDS research.

Use security datasets and any active-network testing only on systems for which you have explicit authorization.

Do not deploy the experimental classifier as an automated blocking system without appropriate validation, monitoring, fail-safe mechanisms, and operational security controls.

---

## Research Summary

**Problem:** Centralized IDS can require sensitive traffic sharing, while static models can degrade under changing network behavior.

**Current baseline:** ML-based centralized IDS + heterogeneous client analysis + FedAvg.

**Observed challenge:** high held-out FPR and significant degradation under simulated federated heterogeneity.

**Next research question:** How can federated intrusion detection adapt to concept drift while preserving privacy and resisting malicious clients?

**Long-term goal:** A trustworthy, privacy-preserving, continual, Byzantine-robust federated IDS for distributed networks.
