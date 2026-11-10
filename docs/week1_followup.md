# Week 1 Follow-up: Class Imbalance & a Temporal-Ordering Proxy

**Author:** Junyue Li
**Responds to mentor feedback:** (1) quantify the class imbalance from the 2,400 rows now; (2) propose a concrete proxy for temporal ordering before the week 2 split decisions.
**Reproduce:** `python scripts/week1_followup_checks.py` (training file only; holdout untouched)

---

## 1. Class imbalance

Positive class = **late** (`is_late = 1 − on_time_delivery`).

| | Value |
|---|---|
| Late legs | **415 / 2,400 = 17.3%** (95% CI 15.8%–18.8%) |
| On-time : late | **4.8 : 1** |
| Accuracy of "always on time" | **82.7%**. This is the floor any accuracy number must be compared against. |

**By region**

| Region | n | Late | Late rate | ± 95% CI |
|---|---|---|---|---|
| Southwest | 456 | 93 | 20.4% | ±3.7 pp |
| Southeast | 442 | 78 | 17.6% | ±3.6 pp |
| West Coast | 475 | 82 | 17.3% | ±3.4 pp |
| Midwest | 458 | 76 | 16.6% | ±3.4 pp |
| Northeast | 435 | 67 | 15.4% | ±3.4 pp |
| **Mountain** | **134** | **19** | **14.2%** | **±5.9 pp** |

**What this means for the framing**
- The imbalance is **moderate, not extreme**: about 1 late leg in 6. No resampling is needed to train a model. Class weights and threshold tuning are the first tools, and accuracy is dropped as a headline metric.
- **Mountain has only 19 late legs.** Any per-region recall for Mountain will rest on fewer than 20 positives. In a 5-fold split that is about 4 per fold, which is too few for a stable estimate. I'll report Mountain with confidence intervals and pool folds (out-of-fold predictions) rather than quote per-fold numbers.
- The regional rates overlap within their CIs. Southwest (20.4%) vs Northeast (15.4%) is the only gap worth a second look in week 2.
- This is the *observed* rate. If some labels are noisy (framing critique, Risk 2), the true rate may differ.

## 2. Is there a usable proxy for temporal ordering?

### 2.1 Candidate: `leg_id` order
`leg_id` runs sequentially (`TR-00001` … `TR-02400`). If it was assigned in creation order, it would be a free time proxy. **I tested this before relying on it.** If `leg_id` tracks time, consecutive legs should share conditions (same day → same weather and region mix, the same surge period) and should drift over the sequence.

| Test | Result | If `leg_id` were time-ordered we'd expect |
|---|---|---|
| Late rate across 8 `leg_id` blocks | 13.3%–19.3%, χ² p = **0.60** | Shifts between blocks |
| Next leg has the same weather | 0.495 vs 0.497 by chance | Well above chance (weather persists within a day) |
| Next leg has the same region | 0.175 vs 0.182 by chance | Above chance |
| Lag-1 autocorrelation (late, speed, hour, distance) | −0.035 to +0.008 | Positive |
| Runs test on `is_late` | z = +0.82 | Negative z (clustered outcomes) |

**Conclusion: `leg_id` order is statistically indistinguishable from a random shuffle.** It is not a time proxy, and a "split by `leg_id`" would just be a random split under another name. I'm stating this explicitly so nobody later mistakes a `leg_id` split for a temporal one.

### 2.2 Other columns considered
| Column | Usable as a time proxy? |
|---|---|
| `departure_hour` + `stop_sequence_position` | Gives **within-day** order only. There is no day boundary, so different days can't be ordered. |
| `demand_spike_flag` | Marks region × day surges, but about 50% of legs are flagged and there is no day key, so days can't be reconstructed. |
| `weather_condition` | Seasonal in principle (snow ≈ winter), but snow appears at similar rates in every region (inventory Q8), so it's unreliable as a season marker. |

None of them gives cross-day ordering.

### 2.3 Proposal for week 2 split decisions

1. **First choice: recover real timestamps.** Ask Carlos to join `leg_id` back to the dispatch record's departure date (inventory G1/G2). If that's possible, everything below is replaced by a proper forward-in-time split.

2. **If timestamps can't be recovered, use the following:**
   - **Primary evaluation:** repeated stratified 5-fold CV (5 × 5 repeats) on the training file, stratified on `is_late` × `region` so Mountain's 19 positives are spread across folds. Report the mean ± standard deviation across folds.
   - **Shift stress tests as a stand-in for temporal shift**, since the model will meet conditions it hasn't seen:
     - *Leave-one-region-out*: train on five regions, test on the sixth.
     - *Surge shift*: train on `demand_spike_flag = False`, test on `True`. This mimics a quieter period followed by a busy one.
     - *Weather shift*: train on Clear, test on severe weather.
     The drop from random-CV performance to these splits gives a **rough range for how far performance might fall in deployment**.
   - **Labelling rule:** every random-CV metric is reported as *"optimistic: no temporal validation possible"*. The 600-row holdout (opened in week 4) is the only independent check, and it has the same lack of ordering.

3. **Decision trigger:** if the shift stress tests lose more than about a third of the late-class PR-AUC compared with random CV, the model is too sensitive to conditions to be deployed without real temporal validation. I would escalate that rather than tune around it.
