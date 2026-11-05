# Framing Review & Assumption Stress-Test

**Author:** Junyue Li
**Project:** junyueli-ds-project
**Stage:** Week 1 · Framing Review & Assumption Stress-Test
**Builds on:** [problem_statement.md](problem_statement.md), [data_source_inventory.md](data_source_inventory.md)

---

## 1. The framing being stress-tested

> Predict **`is_late` = 1 − `on_time_delivery`** for each delivery leg, at the moment the vehicle has covered **20% of the leg distance**, using only fields available at that point. Evaluate on the late class (recall primary, precision as a constraint), against a majority-class baseline and a rule-based baseline, broken down per region.

The framing rests on several assumptions. This document separates the **load-bearing** ones, where being wrong means the model definition itself has to change, from the **easily revisable** ones, which can be fixed inside the current framing. For each framing risk it states **what would trigger a redefinition** and **what the redefinition would be**.

## 2. Assumption classification

| Assumption | Load-bearing or revisable | Why |
|---|---|---|
| The label correctly records whether the window was met | **Load-bearing** | Every metric and every model is measured against it |
| `avg_speed_kmh` really exists at the 20% mark in production | **Load-bearing** | It is the only reason for choosing t₀ = 20% |
| Legs can be ordered in time | **Load-bearing** | Without it no evaluation can be called honest for future legs |
| "Not delivered" can be grouped with "late" | Load-bearing *if* it is a large share | It decides whether this is one prediction problem or two |
| Weather is available live | Revisable | Drop it or swap in a forecast; the framing is unchanged |
| `departure_hour` means local time-of-day | Revisable | Re-encode or re-derive; the framing is unchanged |
| Short-leg speed noise can be handled by filtering | Revisable | A feature-engineering fix |
| Intervention capacity (alert budget k) | Revisable | Only moves the threshold |
| No grace period on the window | Revisable | Sensitivity check; re-label in minutes |

## 3. Framing risks

### Risk 1: The prediction point may not exist in production *(load-bearing)*

**The concern.** The README says records are **written at leg close-out**, when the driver marks the stop complete. `avg_speed_kmh` is labelled "at collection", but that may only mean the value *describes* the first 20% of the leg, not that it is *computed and available* at that moment. If the telematics aggregate is only produced at close-out, the model is trained on a feature that is never present when a prediction is actually needed.

**Conditions that would change the model definition:**
- Carlos confirms `avg_speed_kmh` is computed only at close-out or in a batch job → **move t₀ to departure** and drop speed.
- The departure-time variant (no speed) reaches comparable late-class PR-AUC → **move t₀ to departure** anyway, because the earlier prediction gives dispatchers more options (re-sequencing, reassignment before the vehicle leaves). "Comparable" means within the variance across validation folds.
- Only if speed is available live *and* adds clearly more than that variance does t₀ = 20% stay.

### Risk 2: The label may not measure what we think it measures *(load-bearing)*

**The concern.** `on_time_delivery` depends on two things that are known to be imperfect:
1. **Arrival time comes from the driver tapping "stop complete"** in the app, not from a GPS geofence. A driver who taps late or early shifts the label.
2. **The telematics-to-dispatch join** (vehicle ID + 15-minute window) produces duplicate or mismatched rows on midnight-crossing shifts. The training data has no keys to detect them, and hour 23 is about 2× over-represented, which may be a symptom.

If a meaningful share of labels is wrong, the model learns noise, and no feature engineering can push it past the ceiling that noise sets.

**Conditions that would change the model definition:**
- In week 2, legs at hour 23 or at capped values show label rates or feature profiles that are clearly inconsistent with other legs → treat them as suspect: exclude them, or down-weight them and report results with and without them.
- If an independent arrival signal can be obtained (GPS geofence entry time) and it disagrees with the app-based label often → **redefine the label from GPS arrival time**, not the app tap.
- If validation performance plateaus well below the baselines plus a margin, despite reasonable models → treat label noise as the likely ceiling and report it, rather than tuning further.

### Risk 3: There is no way to order legs in time *(load-bearing)*

**The concern.** The extract has no date, timestamp, vehicle ID or route ID. A random train/validation split assumes legs are independent and that the future looks like the past. Neither can be checked. Legs on the same route and day share conditions (the same driver, weather and surge), so a random split can leak information between train and validation and **overstate performance**.

**Conditions that would change the model definition:**
- **Timestamps can be recovered via `leg_id`** → use a time-based split (week 4 as planned) and add the lag/rolling features of week 5.
- **Timestamps cannot be recovered** → the deliverable changes from "a model with an honest forward-looking estimate" to "a model with a *best-case* estimate". Every reported metric must then be labelled optimistic. Week 5's temporal work must be re-scoped, for example by checking whether `leg_id` order behaves like time.
- **Route grouping can be recovered** → use grouped cross-validation (whole routes in one fold), even without dates.

### Risk 4: "Late" and "not delivered" may be different problems *(load-bearing if large)*

**The concern.** The label puts *arrived late* and *never arrived within the window* (failed attempt, customer absent, refused) into the same class. They have different causes and different dispatcher responses: a traffic delay calls for a revised ETA, while a failed attempt calls for re-booking. If failed deliveries are a large share of the late class, the model has to learn two unrelated mechanisms, and its alerts will be harder to act on.

**Conditions that would change the model definition:**
- If `carrier_close_code` or another disposition field can separate the two outcomes, **used only to analyse the label, never as a feature**, and non-delivery turns out to be a substantial share of `is_late = 1`, either:
  - narrow the target to **late arrival only** and treat non-delivery separately, or
  - move to a **three-class target** (on time / late / not delivered).
- If the two can't be separated, keep the combined label, but state in the model card that alerts mix both causes.

### Risk 5: "Available at collection" may not mean "available live" *(revisable)*

**The concern.** Several fields reach the warehouse later than the moment they describe. `weather_condition` is joined **overnight** from NOAA. `demand_spike_flag` comes from a separate capacity-planning system with an unknown refresh rate. `departure_hour` may be in a different time zone from what its name implies; the expected West Coast +3 h shift is not visible in the data. A feature that is accurate offline but late or different online creates a training/serving mismatch.

**Conditions that would change the model:**
- No live weather feed → train without `weather_condition`, or with a forecast-based substitute. Report the performance cost.
- Surge flag only set after the fact → drop it or replace it with a live demand signal.
- `departure_hour` turns out to be UTC or depot time → re-derive local time-of-day before using it, and revisit the region/hour confound.

None of these change the target or t₀. They change the feature set, so this risk is revisable.

## 4. Summary: what would make me redefine the model

| Trigger | Change to the model definition |
|---|---|
| Speed is not available live, or adds little over a departure-time model | Move t₀ to **departure**; drop `avg_speed_kmh` |
| A GPS arrival time exists and disagrees with app-based labels | Rebuild the **label** from GPS arrival |
| Hour-23 or capped rows look like join artifacts | Exclude or down-weight them; report both versions |
| Timestamps can't be recovered | All metrics labelled **optimistic**; re-scope week 5 |
| Non-delivery is a large share of `is_late` | Narrow the target or move to **3 classes** |
| Weather, surge or hour not live / differently defined | Change the **feature set** only |

## 5. Questions to settle at the framing review

1. *(Carlos)* When exactly is `avg_speed_kmh` computed in production: at the 20% mark, or at close-out?
2. *(Carlos)* Can `leg_id` be joined back to vehicle ID, route ID and a timestamp?
3. *(Carlos)* Is there a GPS geofence arrival time that could check the app-based label?
4. *(Priya)* Do we know the share of `on_time_delivery = 0` that are non-deliveries rather than late arrivals?
5. *(Fatima)* Would dispatchers act differently on "likely late" and "likely not delivered"? If so, the target should separate them.
