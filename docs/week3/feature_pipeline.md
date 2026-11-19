# Feature Engineering Pipeline (v1.1.0)

**Author:** Junyue Li
**Stage:** Week 3 · Task 2
**Code:** `scripts/build_features.py` (single-command entry point) · `scripts/features.py` (pipeline logic) · **Tests:** `tests/test_features.py` (9 tests, all passing)
**Builds on:** `docs/week3/feature_leakage_audit.md`. Every feature here is an "Include" or "Include with guard" item from that audit. Cap indicators are only produced with `--include-sensitivity`.

---

## 1. How to run

```bash
pip install -r requirements.txt
python scripts/build_features.py     # ONE command: reads data/processed/train_labeled.csv, writes both matrices
python -m pytest tests -q            # 9 tests
```

`build_features.py` is importable (`from build_features import build`) and falls back to the raw training file if `train_labeled.csv` is missing. `features.py` can also be run on its own (`--include-sensitivity` for the at-cap set).

| Output | What it is |
|---|---|
| `data/processed/features_v1.1.0.csv` | 24 row-wise features + `leg_id` + `is_late`. **NaNs kept**, missing indicators added. |
| `data/processed/model_matrix_v1.1.0.csv` | **Model-ready MAIN set**: 46 columns (imputed, one-hot encoded, 2 vehicle-class aggregates), **no nulls**, plus `leg_id` key and `is_late` target |
| `data/processed/model_matrix_v1.1.0_sens.csv` | Same + 6 **at-cap indicators** (52 columns), for sensitivity runs |
| `data/processed/preprocessor_v1.1.0.json` | Fitted imputation values and category lists, to apply unchanged to the holdout in week 4 |
| `docs/week3/feature_manifest_v1.1.0.json` | Version, SHA-256 of the input and every output, full feature dictionary, missing rates, excluded columns |

`_sens` versions of every file are written by the same command (30 row-wise features, 52 model columns).

**About "raw logs":** the brief asks for features "from raw logs". This extract has no raw GPS pings or manifest records (Week 1 inventory, gaps G2 and G4), so the pipeline starts from the leg-level extract. `build_features()` is the single entry point, so a future raw-log loader only has to produce the same 10 input columns.

## 2. Design: two layers, matching the leakage cutoff rules

| Layer | Function | What it does | Safe to run on all rows? |
|---|---|---|---|
| 1. Row-wise | `build_features(raw)` | Deterministic features computed from **that leg's own columns only**. No statistics, no label. NaNs preserved. | **Yes.** A test checks that each row's features are identical whether it's processed alone or with all other rows. |
| 2. Fitted | `make_preprocessor()` | Median / mode imputation, one-hot encoding, **vehicle-class aggregates**, optional scaling. Returns an **unfitted** sklearn `ColumnTransformer`. | **No.** Must be fit inside training folds (audit §4.3). |

Typical CV use, so nothing is fit on the validation fold:

```python
import features as F
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

f = F.build_features(raw);  y = f.pop("is_late")
model = make_pipeline(F.make_preprocessor(scale_numeric=True), LogisticRegression(class_weight="balanced"))
cross_validate(model, f.drop(columns="leg_id"), y, cv=...)
```

The exported `model_matrix` is fit on the full training file, which is correct for training the **final** model. For cross-validation, use the pipeline object above instead.

## 3. Features (24 row-wise, three required families plus support)

### Route-level
| Column | Type | Definition | Missing before imputation |
|---|---|---|---|
| `route__leg_distance_km` | numeric | Straight-line depot → address distance | 7.0% |
| `route__stop_position` | numeric | Position in the driver's daily route (1 = first) | 7.0% |
| `route__is_late_route_stop` | binary | stop position ≥ 15 | 7.0% |
| `route__window_min` | numeric | Committed window width | 7.3% |
| `route__km_per_window_hour` | numeric | distance ÷ window hours (planning difficulty) | 13.8% |
| `route__region` | categorical | Operational region | 0% |

### Time-of-day
| Column | Type | Definition | Missing |
|---|---|---|---|
| `time__departure_hour` | numeric | 0–23 (time zone still unconfirmed) | 6.7% |
| `time__hour_sin`, `time__hour_cos` | numeric | Cyclical encoding, so hour 23 sits next to hour 0 | 6.7% |
| `time__is_night` | binary | hour 0–5 | 6.7% |
| `time__period` | categorical | night (0–5) / morning (6–11) / afternoon (12–17) / evening (18–23) | 6.7% |

### Vehicle-level
| Column | Type | Definition | Missing |
|---|---|---|---|
| `veh__class` | categorical | Vehicle class | 7.8% |
| `veh__is_heavy` | binary | Articulated or Heavy Rigid | 7.8% |
| `veh__early_speed_kmh` | numeric | Speed over the first 20% of the leg; **set to missing for legs < 2 km** (ML engineer's warning) | 6.8% |
| `veh__implied_travel_min` | numeric | distance ÷ early speed, **clipped to [0, 1440] min** (EDA found up to ~51 h) | 13.5% |
| `veh__window_slack_min` | numeric | window − implied travel time | 19.8% |

### Vehicle-level aggregates (fitted, layer 2, v1.1.0)
These compare each leg with **other legs of the same vehicle class**, so they are learned from data. They are fit on the training rows only, inside each CV fold (`VehicleClassAggregates`), and use feature values, never the label.

| Column | Definition |
|---|---|
| `veh__speed_vs_class_median` | early speed ÷ median early speed of that vehicle class (1.0 = typical for the class) |
| `veh__distance_vs_class_median` | leg distance ÷ median leg distance of that vehicle class |

Missing value or unseen class → 1.0 ("typical"). The raw gap is already recorded by `miss__early_speed` / `miss__leg_distance`.

### Weather, operations and missing indicators
| Column | Type | Definition | Missing |
|---|---|---|---|
| `wx__condition` | categorical | NOAA weather category | 6.8% |
| `wx__severity` | numeric | Clear 0 · Fog/Light Rain 1 · Heavy Rain/Snow 2 · Severe Storm 3 | 6.8% |
| `wx__bad_weather_x_heavy` | binary | severity > 0 and heavy vehicle | 13.9% |
| `ops__demand_spike` | binary | Capacity-planning surge flag | 6.5% |
| `miss__departure_hour`, `miss__early_speed`, `miss__weather`, `miss__leg_distance` | binary | 1 if the **raw** value was missing | 0% |

### Route-level at-cap indicators (in the `_sens` matrix)
`sens__at_cap_leg_distance_km`, `sens__at_cap_avg_speed_kmh`, `sens__at_cap_scheduled_window_minutes`, `sens__at_cap_stop_sequence_position`, `sens__at_cap_departure_hour`, `sens__n_extreme_flags`. These are produced by `build_features.py` in the `_sens` matrix. They are **not** in the main matrix, because they are suspected leakage (audit §3.3: `n_extreme_flags` alone predicts lateness better than the known leak). Task 3 measures their importance directly.

### Never produced
`carrier_close_code` (post-outcome leak) and `on_time_delivery` (target source) are never read by `build_features()`. `leg_id` is kept only as a key column. `validate_matrix()` fails the run if any of these appear among the model columns.

## 4. Column naming convention

`<family>__<name>[_<unit>]`, all lower-case snake_case.
- Families: `route`, `time`, `veh`, `wx`, `ops`, `miss`, `sens`.
- Units as suffixes: `_km`, `_kmh`, `_min`.
- One-hot outputs: `<family>__<name>__<level>`, for example `veh__class__light_van`, `wx__condition__heavy_rain`, `route__region__west_coast`.
- Missing category for categoricals: `__missing`, for example `wx__condition__missing`.

All columns are defined once in `REGISTRY` inside `features.py`. The pipeline asserts that its output matches the registry exactly, so a column can't be added or renamed silently.

## 5. Null handling

| Situation | Treatment | Why |
|---|---|---|
| Missing numeric value | Kept as NaN in layer 1; **median** imputation in layer 2 (fit in-fold) | Robust to the capped spikes |
| Missing binary value | **Most-frequent** imputation (fit in-fold) | Keeps the column 0/1 |
| Missing value flagged **MNAR** in week 2 (`departure_hour`: +7 pp late rate when missing) | Imputed **and** kept as an explicit indicator `miss__departure_hour` | The fact that it's missing is itself informative |
| Missing categorical value | Separate **`missing`** category, one-hot encoded | Missingness may carry signal (departure hour: +7 pp late rate) |
| Value present but unreliable (speed on legs < 2 km) | Set to NaN in layer 1; `miss__early_speed` still records the *raw* value | Distinguishes "dropout" from "masked on purpose" |
| Derived feature with a missing input | Result is NaN (no silent zero), then imputed | e.g. `veh__window_slack_min` is 19.8% missing because it needs three inputs |
| Implausible values | Clipped (`veh__implied_travel_min` to one day) | EDA §6.3 |
| Category unseen at fit time (e.g. holdout) | `handle_unknown="ignore"`, giving all-zero one-hot columns | Tested |
| Any null left after layer 2 | **Run fails** (`validate_matrix`) | Guarantees a model-ready matrix |

## 6. Reproducibility and versioning

- **`FEATURE_VERSION = "1.1.0"`** is written into every output filename and the manifest. Rule: **major** = column removed or redefined, **minor** = column added, **patch** = bug fix that doesn't change any definition.
- **Deterministic:** running twice gives byte-identical outputs. The manifest stores SHA-256 hashes (`model_matrix_v1.1.0.csv` → `5ed90329…`), so anyone can verify they have the same matrix.
- **Input fingerprint:** the manifest stores the SHA-256 of the raw training file. If the extract changes, the change is visible.
- **Pinned packages** in `requirements.txt`. These are the exact versions used to produce every number in this repo, and they were installed from PyPI with `pip install -r requirements.txt` on the author's Windows / Python 3.13 machine. The pipeline itself uses no randomness.

## 7. Tests (`python -m pytest tests -q`)

| Test | Checks |
|---|---|
| `test_columns_follow_registry_and_never_use_post_outcome` | Output columns = registry; no `carrier_close_code` |
| `test_sensitivity_features_only_when_requested` | `sens__*` appear only with the flag; a row with all five extremes counts 5 |
| `test_row_wise_values` | Hand-checked values: night period, travel time 120 min, slack 0, speed masked for a 1.5 km leg, missing indicators, interaction flag, label polarity |
| `test_features_are_row_wise_independent_of_other_rows` | Each row's features are identical when computed alone or in a batch, so layer 1 can't leak |
| `test_preprocessor_outputs_clean_matrix_and_handles_unseen_categories` | Fit on 3 rows, transform a 4th with unseen region, vehicle and weather; no nulls, same columns |
| `test_validate_matrix_rejects_nulls_and_forbidden_columns` | The safety check actually fails when it should |
| `test_bad_raw_input_is_rejected` | Missing required column or duplicate `leg_id` raises an error |
| `test_vehicle_aggregates_are_fit_on_training_rows_only` | Class medians come from the fit rows only; unseen class / missing value → 1.0 |
| `test_build_features_entry_point_is_deterministic` | Two runs of `build_features.build()` give identical output hashes |

## 8. Smoke test (v1.0.0 features): the pipeline works end-to-end in CV

Logistic regression (balanced class weights) + `make_preprocessor(scale_numeric=True)`, 5-fold stratified CV, late class positive. This only checks that the pipeline plugs into a model. Proper model selection is week 4.

| Feature set | F1 | Recall | Precision | PR-AUC | F1 on non-extreme rows |
|---|---|---|---|---|---|
| Main (no cap indicators) | 0.678 | 0.807 | 0.585 | 0.777 | 0.11 |
| Sensitivity (+ `sens__*`) | 0.751 | 0.786 | 0.720 | 0.780 | 0.11 |
| *Week 2 benchmarks* | *B3 rule: 0.751* | | | | *B2 rule: 0.12* |

These first numbers already show the week 2 pattern. With the cap indicators, a model only matches the one-line cap rule. Without them, it loses precision. On legs without extreme values, neither does better than the simple rules. The features that can separate those legs still have to be found, which is the job of task 3 (feature importance) and week 4.

## 9. Changelog

| Version | Change |
|---|---|
| 1.0.0 | First release: 24 row-wise features (route, time, vehicle, weather/ops, missing indicators) + 6 optional sensitivity features; two-layer design; manifest; 7 tests |
| 1.1.0 | Added `build_features.py` (single command, reads `train_labeled.csv`, writes main + at-cap matrices); added 2 fitted vehicle-class aggregates; 9 tests |
