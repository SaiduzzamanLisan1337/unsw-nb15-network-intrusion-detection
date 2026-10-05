# Architecture

## End-to-end baseline

```text
UNSW-NB15 partitions
        │
        ▼
Partition-role resolver
(canonical row counts)
        │
        ▼
Research hygiene
(duplicate + exact overlap removal from TRAIN only)
        │
        ├──────────────────────────────┐
        ▼                              ▼
TRAIN                         untouched TEST
        │                              │
        ▼                              │
Train-fitted preprocessing             │
        │                              │
        ├──────┬──────────┐            │
        ▼      ▼          ▼            │
       LR      RF       XGBoost        │
        │      │          │            │
        └──────┴──────────┘            │
                 │                     │
                 ▼                     ▼
          OOF threshold selection → final test evaluation
                 │
                 ▼
             Metrics/plots
                 │
                 ▼
        Federated non-IID partition
                 │
      ┌──────────┼──────────┐
      ▼          ▼          ▼
   Client A   Client B   Client C ...
      │          │          │
      └──── local SGD ─────┘
                 │
                 ▼
            weighted FedAvg
                 │
                 ▼
           global FL model
                 │
                 ▼
          round-by-round metrics
```

## Security properties of the current prototype

- No raw dataset is committed to the repository.
- The held-out test partition is not used for fitting or threshold selection.
- The research protocol removes exact duplicate and train/test-overlap records from TRAIN.
- The FL aggregator receives parameter vectors, not raw examples.

## What is deliberately not implemented yet

- Live packet capture (`libpcap`, AF_PACKET, PF_RING, DPDK).
- TLS decryption/proxy inspection.
- IPS traffic blocking.
- Secure aggregation.
- Differential privacy.
- Byzantine-robust aggregation.
- Continual learning under real temporal concept drift.
