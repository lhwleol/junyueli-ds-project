# junyueli-ds-project

Predicting late delivery legs so dispatchers can reassign work or notify customers before a committed delivery window is missed.

**Question:** will a delivery leg that is currently in progress arrive within the customer-committed time window?

## Progress

| Week | Deliverable | Link |
|---|---|---|
| 1 | Problem statement: target, decision context, success criteria, assumptions | [docs/problem_statement.md](docs/problem_statement.md) |
| 1 | Data source inventory: sources, join keys, coverage gaps, quality concerns | [docs/data_source_inventory.md](docs/data_source_inventory.md) |

## Structure

- `docs/` – written deliverables
- `scripts/` – Python scripts (e.g. `profile_train.py` reproduces the data profiling numbers)
- `notebooks/` – Jupyter notebooks
- `data/` – local dataset (**not tracked in git**)

## Setup

```bash
pip install -r requirements.txt
```

Place `delivery_leg_telemetry_train.csv` in a local `data/` folder, then run:

```bash
python scripts/profile_train.py
```

The holdout file is sealed until week 4 and is not used by any code in this repo.
