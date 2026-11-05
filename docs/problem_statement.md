# Problem Statement: Predicting Delayed Deliveries to Support Dispatch Decisions

**Author:** Junyue Li
**Project:** junyueli-ds-project
**Stage:** Week 1 · Orientation & Problem Framing

---

## 1. Background

Late deliveries are expensive. Customers complain, re-delivery and support costs go up, and the company loses trust. Most of the time the delay only becomes visible once a promised window has already been missed, and by then the dispatcher has nothing useful left to do. The aim of this project is to move that visibility earlier: flag deliveries that are **likely to be late while there is still time to act**.

## 2. Decision Context

**Who uses the prediction:** dispatchers and route planners at the delivery depot.

**When they use it:** at **dispatch time**, when an order has been assigned to a driver/route and is about to leave the depot.

**What they can do with it:**

| Dispatcher action | Needs to happen before... |
|---|---|
| Re-sequence stops on a route so at-risk orders go earlier | Driver leaves the depot |
| Reassign an order to a less-loaded driver or route | Driver leaves the depot |
| Add capacity (extra driver/vehicle) on high-risk days | Start of the shift |
| Proactively notify the customer with a revised ETA | Start of the promised window |

Each of these actions has a cost and a limited capacity. A dispatcher can realistically intervene on only a small share of the day's orders. So the model's job is not "predict every delay". Its job is to **rank today's orders by risk so the limited interventions go where they help most.**

## 3. Prediction Target

### 3.1 Unit of prediction
One **delivery order** (a single shipment to a single address), evaluated once at the moment it is dispatched.

### 3.2 Definition of "delayed"
An order is labelled **delayed = 1** if:

> **actual delivery timestamp > end of the promised delivery window + 15-minute grace period**

Otherwise **delayed = 0**.

- *Promised delivery window*: the time window (or promised date/time) communicated to the customer at checkout or booking.
- *Grace period*: 15 minutes. It absorbs small timestamp inaccuracies (scanner sync, GPS lag) and matches the point where customers usually start to notice. This value is an assumption, and Section 6 says how it will be checked.

### 3.3 Time horizon
- **Prediction time (t₀):** the dispatch event, when the order is assigned to a route and loaded.
- **Outcome window:** from t₀ to the end of the promised window plus grace. For same-day last-mile delivery this is typically **2–10 hours**.
- **Feature cut-off:** only information available **at or before t₀** may be used as input. This includes the order details, route plan, driver assignment, distance, number of stops ahead, time of day, day of week, weather forecast, and historical performance of the route/zone. Anything recorded after t₀ is excluded, for example actual departure time, in-transit scans or the actual delivery time, because using it would leak the answer.

### 3.4 Edge cases
| Case | Treatment |
|---|---|
| Delivered early or on time | delayed = 0 |
| Failed attempt (customer absent), delivered on a later attempt | delayed = 1. From the customer's view the promise was missed. Flagged separately for analysis. |
| Order cancelled before delivery | Excluded from training and evaluation |
| Missing delivery timestamp | Excluded; the missing rate is reported |
| Promised window changed after dispatch (customer reschedule) | Use the window **in effect at t₀**; customer-initiated reschedules are excluded |

### 3.5 Why this framing serves dispatch decisions
- **Binary classification, not regression on minutes late.** Dispatchers make yes/no decisions ("intervene on this order or not"), and the business impact is mostly the jump from *on time* to *late*. A calibrated probability also lets us rank orders and choose a threshold that matches daily intervention capacity. Predicting minutes of delay can be added later as a secondary target.
- **Predict at dispatch, not at order creation or mid-route.** At order creation the route and driver aren't known yet, so the strongest signals are missing. Mid-route, most interventions are no longer possible. Dispatch is the last point where all the levers in Section 2 are still available.
- **Defined against the customer's promise, not an internal average.** The cost of lateness comes from breaking a promise to the customer, so the label should measure exactly that.

## 4. Success Criteria

### 4.1 Model metrics
Delays are expected to be a minority class (an assumed 10–20% of orders), so accuracy alone would be misleading.

| Metric | Why | Target (initial) |
|---|---|---|
| **PR-AUC** (primary) | Measures ranking quality on the minority class | Clearly above the baseline (Section 4.3) |
| **Recall @ top-k%** | Share of all delays caught if dispatchers review only the top k% riskiest orders (k = daily capacity, assumed 10%) | ≥ 2× the delay rate captured by random selection |
| **Precision @ top-k%** | Share of flagged orders that are really delayed; low precision wastes interventions | Reported alongside recall |
| **Calibration** (Brier score, reliability plot) | Dispatchers should be able to trust "70% risk" as meaning roughly 70% | Well-calibrated across risk deciles |

### 4.2 Business criteria
- Fewer late deliveries when the model guides interventions, compared with current practice. This would be tested later in a pilot or a simulation.
- Alert volume stays within dispatcher capacity, with no alert fatigue.
- Predictions are available within seconds of the dispatch event.
- Predictions can be explained: for each flagged order, the top risk drivers (e.g., "14 stops ahead, rush hour, rainy forecast") should be shown.

### 4.3 Baselines to beat
1. **Rule-based:** flag orders on routes whose historical late rate is above X%.
2. **Simple model:** logistic regression on a small set of features (distance, stops ahead, hour, weekday).

The project is worthwhile only if the final model clearly beats both baselines on recall @ top-k%.

## 5. Key Assumptions

1. Orders have a **recorded promised window** and an **accurate actual delivery timestamp**.
2. A clear **dispatch event** with a timestamp exists, so the feature cut-off can be enforced.
3. Delays depend partly on factors **known at dispatch** (route load, distance, timing, weather). Delays caused only by random events, such as a sudden road closure, will stay hard to predict.
4. Dispatcher intervention capacity is about **10% of daily orders**. This sets k.
5. Historical patterns are **reasonably stable** over time. This will be tested with a time-based split (train on earlier months, test on later months) rather than a random split.
6. A missed delivery (false negative) costs more than an unnecessary intervention (false positive). The threshold should therefore lean toward catching more delays.

## 6. Risks and Open Questions

- **Data availability:** does the dataset contain the promised window, or only an estimated delivery date? If it only has an estimate, the label definition must be adapted, and that change will be documented.
- **Grace period:** test sensitivity to 0, 15 and 30 minutes and check how much the label distribution changes.
- **Feedback loop:** once dispatchers act on predictions, flagged orders may arrive on time *because of* the intervention. Future retraining must account for this.
- **Class imbalance:** the actual delay rate must be measured in the data. Re-weighting or threshold tuning may be needed.
- **Fairness and zone bias:** check that some neighbourhoods are not always flagged as high-risk in a way that lowers their service levels.

## 7. Out of Scope (for now)

- Predicting the exact number of minutes late
- Real-time re-prediction during the route
- Automated route optimisation (the model informs dispatchers but does not re-plan routes itself)
