# Baseline Scorecard

**Author:** Junyue Li
**Stage:** Week 2 · Task 3 (Naive Baseline Model & Performance Benchmarking)
**Reproduce:** `python scripts/baselines.py` → `docs/week2/baseline_scorecard.csv`, `docs/week2/baseline_slices.csv`
**Data:** training file only (2,400 legs, 415 late). The holdout is still sealed.

---

## 1. Setup

- **Positive class = late** (`is_late = 1 − on_time_delivery`). Every precision, recall and F1 below is for the **late** class.
- **Evaluation:** 5-fold stratified CV repeated 5 times (25 folds), stratified on `is_late` × `region` so the 19 Mountain late legs are spread across folds. The scorecard reports mean ± standard deviation over the 25 folds. Rules with a fitted part (B4) are fit on the training folds only.
- **Why not a time-based split:** the data has no timestamp, and `leg_id` order behaves like a random shuffle (Week 1 follow-up). All numbers are therefore **optimistic** in the sense that they assume future legs look like these.
- Missing values: a rule that needs a missing value does **not** fire, so missing never counts as evidence.

## 2. The baselines

| ID | Baseline | Type | What it does |
|---|---|---|---|
| B0 | Majority class | naive | Always predicts "on time" |
| B1 | Stratified random | naive (no-skill reference) | Flags a random 17% of legs as late |
| B2 | Bad weather **and** non-van vehicle | hand-written rule | Uses only operational conditions, no capped values |
| B3 | Any capped value **or** hour 23 | hand-written rule (from EDA) | Flags legs with distance = 420, speed = 105, window = 480, stop = 22 or departure hour 23 |
| B4 | One-split decision stump | fitted rule | Learns a single threshold on one numeric feature (fit per fold) |

## 3. Scorecard

Mean (± standard deviation) across 25 CV folds.

| Baseline | Precision | Recall | **F1** | Accuracy | Share of legs flagged |
|---|---|---|---|---|---|
| B0 Majority class | 0.000 | 0.000 | 0.000 | **0.827** | 0% |
| B1 Stratified random | 0.162 ± 0.026 | 0.162 ± 0.031 | 0.161 ± 0.026 | 0.709 | 17.4% |
| B2 Bad weather & non-van | 0.679 ± 0.036 | 0.463 ± 0.040 | 0.549 ± 0.034 | 0.869 | 11.8% |
| **B3 Capped value or hour 23** | 0.745 ± 0.038 | **0.759 ± 0.041** | **0.751 ± 0.030** | 0.913 | 17.7% |
| B4 Decision stump | **0.875 ± 0.044** | 0.344 ± 0.049 | 0.492 ± 0.053 | 0.878 | 6.8% |

**What the decision stump learned:** in 24 of 25 folds it split on **`avg_speed_kmh > 104.5`**, i.e. "speed is at its 105 cap". In 1 fold it chose `departure_hour > 22.5` (hour 23). Without being told about the caps, the fitted rule found the same cap signal that the EDA found by hand. This is useful independent confirmation that B3 isn't just a product of me picking a rule after looking at the data.

## 4. Slices: where the baselines work and where they don't

Out-of-fold predictions from one full CV pass.

| Baseline | All rows (415 late) | **Non-extreme rows** (1,976 rows, 100 late) | Extreme rows (424, 315 late) | Mountain (134, 19 late) |
|---|---|---|---|---|
| | P / R / F1 | P / R / F1 | P / R / F1 | P / R / F1 |
| B0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| B1 | .18 / .18 / .18 | .06 / .21 / **.09** | .72 / .17 / .28 | .10 / .11 / .10 |
| B2 | .68 / .46 / .55 | .13 / .12 / **.12** | .96 / .57 / .72 | .46 / .26 / .33 |
| B3 | .74 / .76 / .75 | 0 / 0 / **0** | .74 / 1.00 / .85 | .65 / .58 / .61 |
| B4 | .84 / .32 / .46 | 0 / 0 / **0** | .84 / .42 / .56 | .50 / .16 / .24 |

**Reading this table**
- **Every baseline that looks good overall gets its score from the extreme-value rows.** On the 1,976 non-extreme legs, which hold 100 of the 415 late legs, the best baseline (B2) reaches F1 = 0.12, barely above random (0.09). B3 and B4 catch **none** of them by construction.
- B2 uses no capped values at all, yet 57% of its recall comes from the extreme rows. Bad weather and heavy vehicles are concentrated there (week 2 task 1, §4).
- **Mountain** results rest on 19 late legs, so a single leg moves recall by about 5 points. Treat them as indicative only.

## 5. Which metric matters most, and why

**Primary metric: recall on the late class, subject to a minimum precision.**
**Reported summary: F1 on the late class, always shown for all rows *and* for non-extreme rows.**

The reasoning:

1. **Accuracy is ruled out.** "Always on time" (B0) scores 82.7% accuracy and catches no late legs. B3 scores 91.3%. The 8.6-point gap hides the difference between a useless baseline and one that catches 3 out of 4 late legs. With a 17% positive rate, accuracy is dominated by the easy on-time majority.
2. **Recall is the business-critical quantity.** A missed late leg (false negative) means a customer who isn't warned and a dispatcher who doesn't reassign. That is the failure the project exists to prevent (Week 1 problem statement, §4.1).
3. **But recall needs a precision floor.** Every alert costs dispatcher attention. If precision drops too low, dispatchers stop trusting alerts and the recall becomes worthless in practice. I propose **precision ≥ 0.50** as a working floor (at most one false alarm per true alarm), to be confirmed with Fatima against actual dispatcher capacity.
4. **F1 is the single number for comparison tables**, because it combines both and is what the brief asks for. It isn't the decision criterion on its own: a model with higher F1 but precision under the floor is not acceptable.
5. **The non-extreme slice must always be reported alongside.** The overall score can be bought almost entirely with the cap rule. If a model can't beat **F1 ≈ 0.12 on non-extreme legs**, it is not doing anything a one-line rule can't.
6. **For probabilistic models (week 4 onward)**, PR-AUC on the late class will be added. It measures ranking quality across all thresholds, and the threshold can then be set to meet the precision floor.

## 6. The numbers every future model must beat

| Scope | Baseline to beat | F1 | Recall | Precision |
|---|---|---|---|---|
| **All rows** | B3 (capped value or hour 23) | **0.751** | 0.759 | 0.745 |
| **Non-extreme rows** | B2 (bad weather & non-van) | **0.12** | 0.12 | 0.13 |
| Sanity floor | B1 (random) | 0.16 | 0.16 | 0.16 |

**Caveat on B3.** B3's rule came out of the EDA on this same training file, so its score is somewhat optimistic: the rule was chosen after seeing the data. Two things limit this: it has no fitted thresholds, and B4 independently rediscovered the same cap signal under proper CV. If the capped values turn out to be a pipeline artifact (open question to Carlos), B3's score reflects the pipeline rather than delivery risk. The non-extreme benchmark then becomes the one that matters.

## 7. Message for Carlos

Carlos asked to hear about the baseline as soon as it exists, with the metric stated before the score:

> **Baseline is in.** Metric: F1 on the *late* class (positive = late, 17.3% of legs), 5×5 stratified CV on the training file. The best naive baseline, "flag any leg with a capped value (420 km / 105 km/h / 480 min / stop 22) or hour-23 departure", gets **F1 0.75** (P 0.75, R 0.76). On legs *without* those values, no baseline beats **F1 0.12**, which is close to random. So most of the easy signal comes from the caps. Question for you: what produces values exactly at those caps? If they're pipeline artifacts, the realistic benchmark is the 0.12.
