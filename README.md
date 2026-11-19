# junyueli-ds-project

Predicting late delivery legs so dispatchers can reassign work or notify customers before a committed delivery window is missed.

**Question:** will a delivery leg that is currently in progress arrive within the customer-committed time window?

## Progress

| Week | Deliverable | Link |
|---|---|---|
| 1 | Problem statement: target, decision context, success criteria, assumptions | [docs/problem_statement.md](docs/problem_statement.md) |
| 1 | Data source inventory: sources, join keys, coverage gaps, quality concerns | [docs/data_source_inventory.md](docs/data_source_inventory.md) |
| 1 | Framing critique: load-bearing vs revisable assumptions | [docs/framing_critique.md](docs/framing_critique.md) |
| 1 | Follow-up: class imbalance and temporal-ordering proxy | [docs/week1_followup.md](docs/week1_followup.md) |
| 2 | Label construction & class balance audit | [docs/week2/label_audit_report.md](docs/week2/label_audit_report.md) · [scripts/build_labels.py](scripts/build_labels.py) |
| 2 | EDA & anomaly detection | [notebooks/02_eda_anomaly_detection.ipynb](notebooks/02_eda_anomaly_detection.ipynb) |
| 2 | Naive baselines & scorecard | [docs/week2/baseline_scorecard.md](docs/week2/baseline_scorecard.md) · [scripts/baselines.py](scripts/baselines.py) |
| 3 | Feature candidates & leakage audit | [docs/week3/feature_leakage_audit.md](docs/week3/feature_leakage_audit.md) · [scripts/leakage_audit.py](scripts/leakage_audit.py) |
| 3 | Feature engineering pipeline (v1.0.0) | [docs/week3/feature_pipeline.md](docs/week3/feature_pipeline.md) · [scripts/features.py](scripts/features.py) · [tests/](tests/test_features.py) |

## Structure

- `docs/` – written deliverables
- `scripts/` – reproducible Python scripts
- `notebooks/` – Jupyter notebooks
- `tests/` – pytest tests for the feature pipeline
- `data/` – local dataset (**not tracked in git**)

## Setup

Python 3.13; package versions are pinned in `requirements.txt`.

```bash
pip install -r requirements.txt
```

Place `delivery_leg_telemetry_train.csv` in a local `data/` folder, then run:

```bash
python scripts/profile_train.py        # week 1 data profiling
python scripts/build_labels.py         # week 2 label + segment audit
python scripts/baselines.py            # week 2 baseline scorecard
python scripts/leakage_audit.py        # week 3 leakage audit
python scripts/features.py             # week 3 feature pipeline -> data/processed/
python -m pytest tests -q              # tests
```

The holdout file is sealed until week 4 and is not used by any code in this repo.
