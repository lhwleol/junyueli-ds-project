# Feature Candidate Brainstorm & Leakage Audit

**Author:** Junyue Li
**Stage:** Week 3 · Task 1
**Reproduce:** `python scripts/leakage_audit.py` → `docs/week3/single_feature_auc.csv`
**Data:** training file only (2,400 legs). The holdout is still sealed.

---

## 1. Summary

- **33 candidates** were reviewed: 12 raw columns (§3.1) and 21 derived features (§3.2). Decisions:

  | Decision | Count | Items |
  |---|---|---|
  | Include | 7 | R2–R5, D1, D6, D12 |
  | Include with guard (availability to be confirmed) | 14 | R6–R10, D2–D5, D7–D11 |
  | Include only if fit inside CV folds | 2 | D18 target encoding, Isolation-Forest score |
  | Sensitivity runs only | 5 | D13–D17 (cap / hour-23 indicators) |
  | Exclude | 3 | R1 `leg_id`, R11 `carrier_close_code`, D19 label-derived flags |
  | Not buildable without timestamps | 1 | rolling / lag features |
  | Target | 1 | R12 `on_time_delivery` |

- **Confirmed leak, excluded:** `carrier_close_code`. It is entered after the outcome, and it agrees with the label on 92.5% of rows. Any feature built from the label (for example the close-code disagreement flag from week 2) is excluded for the same reason.
- **Suspected leak, sensitivity runs only:** the **cap / hour-23 indicators**. On its own, `n_extreme_flags` predicts lateness *better* than the known leak (CV AUC **0.865** vs **0.857**). A feature that is supposed to be available at the prediction point should not out-predict a field that is written after the outcome. Until Carlos confirms where the capped values come from, these features run only in a side-by-side sensitivity model.
- **Cutoff logic.** There are no timestamps, so look-ahead can't be prevented by filtering on time. It is enforced instead at four levels: column availability, row scope, fitting scope and data split (§4). A test with target encoding shows the fitting rule matters: encoding `leg_id`, which is pure noise, on all rows gives a "perfect" AUC of **1.000**. Fit inside the training folds, it gives **0.500**.

## 1a. Per-column leakage rating (every column in the training file)

Rating scale: **safe** = known before t₀ and recorded before t₀ · **conditional** = known before t₀ in principle, but recording or availability must be confirmed (cutoff rule stated) · **leaked** = only known after the outcome. The same table is available as `docs/week3/feature_leakage_table.csv`.

| Column | Description | Rating | Cutoff logic / exclusion rationale | Decision |
|---|---|---|---|---|
| `leg_id` | Unique record id | safe | Identifier only; target-encoding it gives a fake AUC of 1.0 | Excluded as feature (key only) |
| `region` | Operational region | safe | Fixed before the leg starts | Include |
| `scheduled_window_minutes` | Committed window width | safe | Set at order creation; the 480 cap is a separate at-cap indicator | Include |
| `leg_distance_km` | Straight-line distance | safe | From the route plan; the 420 cap is a separate at-cap indicator | Include |
| `vehicle_class` | Assigned vehicle class | safe | Fixed at assignment | Include |
| `departure_hour` | Departure hour | conditional | Event precedes t₀, but hour 23 may be post-midnight times clipped by the midnight join (runs after the leg); time zone unconfirmed | Include with guard |
| `weather_condition` | NOAA weather at departure | conditional | Fact precedes t₀; value arrives via an **overnight** join, so a live feed must exist at t₀ | Include with guard |
| `avg_speed_kmh` | Speed over first 20% of leg | conditional | Must use pings up to the 20% mark only; records are written at close-out | Include with guard |
| `stop_sequence_position` | Stop order in the daily route | conditional | Safe only if it is the *planned* order, not the actual order written at close-out | Include with guard |
| `demand_spike_flag` | Region × day surge flag | conditional | Must be set in advance, not after the day | Include with guard |
| **`carrier_close_code`** | Close-out status code | **leaked / post-outcome** | Entered after the outcome (data dictionary); agrees with the label on 92.5% of rows | **Excluded** |
| `on_time_delivery` | Target source | leaked | It is the label | Target only |

## 2. The prediction point (t₀)

From the Week 1 problem statement: **t₀ = the moment the vehicle has covered the first 20% of the leg distance.** A feature is legitimate only if its value is **known at or before t₀ in production**. Two different times matter:

| Time | Question | Example |
|---|---|---|
| **Event time** | When does the real-world fact happen? | Weather at departure happens *before* t₀ |
| **Availability time** | When does the value reach the system that will score the leg? | The NOAA weather join runs *overnight*, which is **after** t₀ |

A feature must pass **both**. Several fields in this dataset pass the first and fail, or are unverified on, the second. Those are the "with guard" rows below.

## 3. Feature candidate table

**Risk scale.**
- **None:** fixed before the leg starts and set by planning.
- **Low:** known before t₀, small doubt about how it is recorded.
- **Medium:** known before t₀ in principle, but the recorded value or its availability may come from after t₀.
- **High:** likely to encode post-t₀ information.
- **Leak:** written after the outcome.

**Single-feature AUC** = 5-fold CV ROC-AUC of a small tree that uses only that feature (0.5 = no signal). A pre-t₀ feature scoring close to the known leak is a red flag.

### 3.1 Raw columns

| # | Feature | Source system | Event time | Available at t₀ live? | Leakage risk | Why | Single-feature AUC | Decision |
|---|---|---|---|---|---|---|---|---|
| R1 | `leg_id` | Dispatch | creation | yes | **Exclude** (not a leak, but no signal) | Identifier; order is random (Week 1 follow-up). Target-encoding it gives a fake AUC of 1.0 (§4.3). | — | **Exclude** |
| R2 | `region` | Dispatch | before departure | yes | None | Fixed operational zone | 0.509 | Include |
| R3 | `scheduled_window_minutes` | Order mgmt | order creation | yes | Low | Set when the order is placed. Risk: the 480 cap (see R13–R17). | 0.753 | Include |
| R4 | `leg_distance_km` | Dispatch plan | before departure | yes | Low | Straight-line depot → address, computed from the plan. Risk: the 420 cap. | 0.733 | Include |
| R5 | `vehicle_class` | Dispatch | assignment | yes | None | Fixed when the truck is assigned | 0.753 | Include |
| R6 | `departure_hour` | Telematics / dispatch | departure (< t₀) | yes | **Medium** | Real departure precedes t₀, but hour 23 is 2× over-represented and hour 0 under-represented. That looks like post-midnight times clipped by the **midnight join**, a step that runs after the leg. Time zone is also unconfirmed. | 0.762 | **Include with guard:** keep `hour_23` separate (R17) |
| R7 | `weather_condition` | NOAA, **joined overnight** | departure (< t₀) | **unverified** | **Medium** | The fact is pre-t₀, but the value arrives after the overnight batch. Offline it's always present; live it may not be. A "nearest station, nearest time" join could also pick an observation from *after* departure. | 0.739 | **Include with guard:** confirm a live weather feed; otherwise drop and report the cost |
| R8 | `avg_speed_kmh` | Telematics | **exactly t₀** (first 20% of leg) | **unverified** | **Medium–High** | Defined to end at t₀, but records are **written at leg close-out**. If the aggregate is computed at close-out over more than the first 20%, or only once the leg is complete, it contains post-t₀ driving. The 105 cap is also suspicious. | 0.762 | **Include with guard:** confirm with Carlos when it's computed; departure-time model without speed as a fallback (Week 1 Risk 1) |
| R9 | `stop_sequence_position` | Dispatch | route plan | yes, if *planned* | **Medium** | Fine if it's the *planned* stop order. If it's the *actual* completed order (re-sequenced during the day and written at close-out), it reflects what happened later on the route. The 22 cap is suspicious too. | 0.746 | **Include with guard:** confirm planned vs actual |
| R10 | `demand_spike_flag` | Capacity planning | "for that region and **day**" | **unverified** | **Medium** | A day-level flag can be set *after* the day, using that day's actual volume. It is True for ~50% of legs and unrelated to lateness (AUC 0.49). | 0.488 | **Include with guard:** low value; confirm whether it's set in advance |
| R11 | `carrier_close_code` | TMS close-out | **after the outcome** | no | **Leak** | Data dictionary: "after the outcome is known". It agrees with the label on **92.5%** of rows. | 0.857 | **Exclude** |
| R12 | `on_time_delivery` | outcome | after the outcome | no | **Leak (target)** | It is the label | — | Label only |

### 3.2 Derived features

| # | Feature | Built from | Leakage risk | Why | Single-feature AUC | Decision |
|---|---|---|---|---|---|---|
| D1 | `missing_departure_hour` | R6 | Low | Missingness is a property of the record at t₀. It was informative in the EDA (+7 pp late rate), but weak on its own here. | 0.515 | Include |
| D2 | `missing_avg_speed_kmh`, `missing_weather_condition`, `missing_leg_distance_km` | R4, R7, R8 | **Medium** | If a value is missing because the join *failed at close-out*, the indicator records a post-t₀ process. Live, the same value might be missing for a different reason (feed latency). | 0.49–0.51 | Include with guard: re-check once live data exists |
| D3 | `hour_sin`, `hour_cos` | R6 | Medium (inherits R6) | Cyclical hour encoding | 0.755 / 0.680 | Include with guard (as R6) |
| D4 | `is_night_departure` (00–05) | R6 | Medium (inherits R6) | — | 0.597 | Include with guard |
| D5 | `late_route_stop` (stop ≥ 15) | R9 | Medium (inherits R9) | — | 0.681 | Include with guard |
| D6 | `is_heavy_vehicle` | R5 | None | — | 0.654 | Include |
| D7 | `weather_severity` (0–3 ordinal) | R7 | Medium (inherits R7) | — | 0.738 | Include with guard |
| D8 | `bad_weather_x_heavy` | R5 × R7 | Medium (inherits R7) | The week 2 B2 rule as a feature | 0.628 | Include with guard |
| D9 | `speed_masked_short_legs` | R4, R8 | Medium–High (inherits R8) | Speed set to missing for legs < 2 km (ML engineer's warning) | 0.742 | Include with guard; replaces raw R8 in the pipeline |
| D10 | `implied_travel_min` = distance / speed | R4, R8 | Medium–High (inherits R8) | Unrealistic values (351 legs > 8 h, from the EDA) | 0.688 | Include with guard; clip to a plausible range |
| D11 | `window_slack_min` = window − implied travel | R3, R4, R8 | Medium–High | Inherits speed risk, and "window" is a delivery slot, not a travel budget | 0.683 | Include with guard |
| D12 | `km_per_window_hour` | R3, R4 | Low | Planning-time difficulty ratio | 0.680 | Include |
| D13 | `at_cap_leg_distance_km` | R4 | **High (suspected)** | See §3.3 | 0.639 | **Sensitivity only** |
| D14 | `at_cap_avg_speed_kmh` | R8 | **High (suspected)** | See §3.3 | 0.670 | **Sensitivity only** |
| D15 | `at_cap_scheduled_window_minutes` | R3 | **High (suspected)** | See §3.3 | 0.646 | **Sensitivity only** |
| D16 | `at_cap_stop_sequence_position` | R9 | **High (suspected)** | See §3.3 | 0.647 | **Sensitivity only** |
| D17 | `at_cap_departure_hour` (hour 23) / `n_extreme_flags` | R3, R4, R6, R8, R9 | **High (suspected)** | **Out-predicts the known leak** (0.865 vs 0.857) | 0.659 / **0.865** | **Sensitivity only** |
| D18 | Target encoding of `region` / `vehicle_class` (category → late rate) | R2, R5 + **label** | **High if fit on all rows**, None if fit in-fold | Uses the label by construction (§4.3) | — | Include **only** inside CV folds |
| D19 | Label-derived flags (e.g. `flag_close_code_disagrees` from week 2) | R11, R12 | **Leak** | Built from the outcome | — | **Exclude** (data-quality use only) |
| — | Rolling / lag features (driver's previous leg late, region late rate today) | needs timestamps + vehicle/route IDs | **High** without a time order | Can't be built without keys. Built naively, they would mix future legs into past ones. | — | **Not buildable now**; revisit in week 5 if timestamps are recovered |
| — | Isolation-Forest anomaly score | all numeric | Medium if fit on all rows | An unsupervised model fit on the test rows still sees them | — | Only if fit in-fold (and it found nothing useful in week 2) |

### 3.3 Why the cap indicators are treated as suspected leakage

The evidence collected so far:

| Evidence | Value | What it suggests |
|---|---|---|
| Single-feature AUC of `n_extreme_flags` | **0.865**, higher than the known leak (0.857) | A field that is "available at t₀" carries outcome-level information |
| Late rate in capped / hour-23 rows | 74% vs 5% elsewhere | A distinct cluster, not a smooth effect |
| Numeric features' correlation with the label once those rows are removed | 0.32–0.35 → ≤ 0.04 (week 2 EDA) | Almost all numeric signal comes from the cluster |
| Single-feature AUCs on non-extreme rows | 0.46–0.54 for every feature | Outside the cluster, nothing predicts on its own |
| README | the "midnight join" runs after the leg and is known to produce mismatched rows | A plausible post-t₀ mechanism for both the caps and hour 23 |

None of this *proves* leakage. The caps could be real recording limits on genuinely extreme legs. But the cost of getting it wrong is lopsided:
- If the caps are a **post-t₀ artifact** and we train on them, offline scores look excellent (F1 ≈ 0.75 from one rule), and the model fails as soon as it sees live data, where the artifact doesn't exist yet.
- If they are **real** and we leave them out of the main model, we lose some offline score but still know the size of the gap from the sensitivity run.

**Decision:** the main model is trained **without** D13–D17. A sensitivity model is trained **with** them, and both are reported. The raw columns R3, R4, R6, R8 and R9 stay in, because their *non-capped* values are legitimate. **What would change this:** Carlos confirms the caps are real recording limits that exist at t₀. Then D13–D17 move to "Include".

## 4. Cutoff logic used to prevent look-ahead

There is no timestamp in the data, so a `WHERE feature_time <= t0` filter isn't possible. Look-ahead is prevented at four levels instead.

### 4.1 Column level: availability allow-list
Every column has an **availability time** (event and availability, §2). The pipeline reads features **only from an explicit allow-list** (`Include` and `Include with guard` rows in §3), never by "all columns except the target". A new column added upstream is therefore excluded by default until it has been audited.

```python
ALLOWED_AT_T0 = ["region", "scheduled_window_minutes", "leg_distance_km", "vehicle_class",
                 "departure_hour", "weather_condition", "avg_speed_kmh",
                 "stop_sequence_position", "demand_spike_flag"]
POST_OUTCOME   = ["carrier_close_code", "on_time_delivery"]   # never features
SENSITIVITY_ONLY = ["at_cap_*", "n_extreme_flags"]             # side-by-side run only
```

### 4.2 Row level: a leg's features use only that leg's own values
Every derived feature in §3.2 (except D18) is computed from the same row's columns. **No cross-row aggregates** (region late rate today, driver's previous leg), because without timestamps there's no way to guarantee they only use *earlier* legs.

### 4.3 Fitting level: every fitted transform is fit inside the training fold
Anything that *learns from data* (imputation medians, category encoders, scalers, target encoding, outlier models) is wrapped in an sklearn `Pipeline` and fit on the training folds only, then applied to the held-out fold.

Test of why this matters (`leakage_audit.py`, target encoding):

| Encoded column | AUC, fit on all rows (leaky) | AUC, fit inside folds (correct) |
|---|---|---|
| `region` | 0.533 | 0.498 |
| **`leg_id` (pure noise)** | **1.000** | **0.500** |

`leg_id` has no relationship to lateness. Encoded leakily, each row's encoding *is* its own label, which gives a perfect score. Encoded correctly, it is exactly random. The same mechanism inflates real features more quietly, as `region` shows (0.53 vs 0.50).

### 4.4 Data level: split and holdout
- **Now (no timestamps):** stratified CV on `is_late` × `region`, with all metrics marked *optimistic*. The holdout stays sealed until week 4.
- **If timestamps are recovered:** a time-based split (train on earlier dates, validate on later ones) with a **1-day embargo gap**, so legs from the same day and route don't land on both sides. Features would then have to satisfy `feature_event_time ≤ t₀` and `feature_available_time ≤ t₀ + scoring latency`.

### 4.5 Ongoing detection rule
Before any feature enters a model:
1. **Single-feature AUC check.** Any feature scoring **≥ 0.85**, or above the known leak, is investigated before use.
2. **With/without comparison.** If adding one feature raises validation F1 by more than 0.10, it is investigated for leakage before it is accepted as a real signal.

## 5. Open questions (these decide the "with guard" rows)

| To | Question | Affects |
|---|---|---|
| Carlos | When is `avg_speed_kmh` computed: at the 20% mark, or at close-out over the recorded track? | R8, D9–D11 |
| Carlos | What produces values exactly at 420 / 105 / 480 / 22 / hour 23? | R6, D13–D17 |
| Carlos | Is there a live weather feed, or only the overnight NOAA join? | R7, D7, D8 |
| Carlos | Is `stop_sequence_position` the planned or the actual stop order? | R9, D5 |
| Fatima | Is `demand_spike_flag` set in advance, or after the day? | R10 |
