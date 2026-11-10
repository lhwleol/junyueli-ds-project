"""Week 2 · Naive baselines & performance benchmarking.

Positive class = late (is_late = 1 - on_time_delivery). Training file only; holdout sealed.
Every baseline is scored with 5 x 5 repeated stratified CV (stratified on is_late x region)
so that rules with fitted parts (the decision stump) are never scored on rows they were fit on.

Usage:  python scripts/baselines.py
Outputs: docs/week2/baseline_scorecard.csv, docs/week2/baseline_slices.csv
Full write-up: docs/week2/baseline_scorecard.md

Required baselines (code review):
  B0 majority class  -> always predicts on time (is_late = 0); late-class recall is exactly 0.0.
  B3 rule-based      -> is_late = 1 when flag_any_cap == 1 OR flag_hour_23 == 1
                        (the rule from docs/week2/label_audit_report.md).
  B1, B2, B4 are extra references (random, operational rule, fitted one-split stump).

Which metric matters most for dispatch decisions
------------------------------------------------
* Recall on the LATE class is primary. A missed late leg (false negative) means no proactive
  customer notification and no reassignment, which is exactly the failure this project exists to prevent.
* Precision on the late class is the constraint. Every alert costs dispatcher attention; below a
  working floor of 0.50 (at most one false alarm per true alarm) alerts stop being trusted (alert fatigue).
* F1 on the late class is the single summary number for comparison tables; PR-AUC is added once
  models output probabilities (week 4+).
* Accuracy is NOT a decision metric: with 17.3% late legs, "always on time" already scores 82.7%
  accuracy while catching zero late legs.
* Every score is also reported on non-extreme rows (no capped value, not hour 23), because the cap
  rule alone explains most of the overall score.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "delivery_leg_telemetry_train.csv"
OUT = ROOT / "docs" / "week2"
NUM = ["scheduled_window_minutes", "leg_distance_km", "departure_hour",
       "avg_speed_kmh", "stop_sequence_position"]
CAPS = {"leg_distance_km": 420, "avg_speed_kmh": 105, "scheduled_window_minutes": 480,
        "stop_sequence_position": 22, "departure_hour": 23}
SEED = 42


# ---------- baselines: each takes (train_df, test_df) and returns 0/1 predictions for test ----------
def majority_class(tr, te):
    """Always predict the majority class (on time)."""
    return np.full(len(te), int(tr.is_late.mean() > 0.5))


def stratified_random(tr, te, rng=np.random.default_rng(SEED)):
    """Predict late at random with the training late rate. A 'no skill' reference."""
    return (rng.random(len(te)) < tr.is_late.mean()).astype(int)


def rule_operational(tr, te):
    """Domain rule without the caps: bad weather AND a non-van vehicle -> late."""
    bad_weather = te.weather_condition.notna() & (te.weather_condition != "Clear")
    heavy = te.vehicle_class.notna() & (te.vehicle_class != "Light Van")
    return (bad_weather & heavy).astype(int).values


def rule_extreme_values(tr, te):
    """EDA rule: any value at its cap, or departure hour 23 -> late. Missing = not flagged."""
    hit = np.zeros(len(te), dtype=bool)
    for c, v in CAPS.items():
        hit |= (te[c] == v).values
    return hit.astype(int)


def stump_one_feature(tr, te):
    """Fitted rule: a depth-1 decision tree on the numeric features (median-imputed),
    fit on the training folds only. Threshold and feature are learned, not hand-picked."""
    med = tr[NUM].median()
    clf = DecisionTreeClassifier(max_depth=1, class_weight="balanced", random_state=SEED)
    clf.fit(tr[NUM].fillna(med), tr.is_late)
    return clf.predict(te[NUM].fillna(med))


BASELINES = {
    "B0 Majority class (always on time)": majority_class,
    "B1 Stratified random (no-skill reference)": stratified_random,
    "B2 Rule: bad weather AND non-van vehicle": rule_operational,
    "B3 Rule: any capped value or hour 23": rule_extreme_values,
    "B4 Fitted rule: one-split decision stump": stump_one_feature,
}


def scores(y, p):
    return {"precision": precision_score(y, p, zero_division=0),
            "recall": recall_score(y, p, zero_division=0),
            "f1": f1_score(y, p, zero_division=0),
            "accuracy": accuracy_score(y, p),
            "flag_rate": p.mean()}


def main() -> None:
    df = pd.read_csv(DATA)
    df["is_late"] = 1 - df["on_time_delivery"]
    df["extreme"] = rule_extreme_values(None, df).astype(bool)
    strat = df.is_late.astype(str) + "_" + df.region

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED)
    fold_rows, oof = [], {name: [] for name in BASELINES}
    for i, (a, b) in enumerate(cv.split(df, strat)):
        tr, te = df.iloc[a], df.iloc[b]
        for name, fn in BASELINES.items():
            p = fn(tr, te)
            fold_rows.append({"baseline": name, "fold": i, **scores(te.is_late.values, p)})
            if i < 5:  # first repeat = each row predicted exactly once -> out-of-fold predictions
                oof[name].append(pd.Series(p, index=te.index))

    folds = pd.DataFrame(fold_rows)
    card = folds.groupby("baseline", sort=False).agg(["mean", "std"])
    card.columns = [f"{m}_{s}" for m, s in card.columns]
    card = card.drop(columns=["fold_mean", "fold_std"])
    OUT.mkdir(parents=True, exist_ok=True)
    card.round(4).to_csv(OUT / "baseline_scorecard.csv")

    # slices on out-of-fold predictions (one full pass)
    slices = {"all rows": df.index,
              "extreme-value rows": df.index[df.extreme],
              "non-extreme rows": df.index[~df.extreme],
              "Mountain": df.index[df.region == "Mountain"],
              "non-Mountain": df.index[df.region != "Mountain"]}
    rows = []
    for name in BASELINES:
        p = pd.concat(oof[name]).sort_index()
        for s, idx in slices.items():
            y = df.loc[idx, "is_late"].values
            rows.append({"baseline": name, "slice": s, "n": len(idx), "late": int(y.sum()),
                         **scores(y, p.loc[idx].values)})
    sl = pd.DataFrame(rows)
    sl.round(4).to_csv(OUT / "baseline_slices.csv", index=False)

    # what did the stump learn?
    med = df[NUM].median()
    st = DecisionTreeClassifier(max_depth=1, class_weight="balanced", random_state=SEED).fit(df[NUM].fillna(med), df.is_late)
    feat, thr = NUM[st.tree_.feature[0]], st.tree_.threshold[0]

    pd.set_option("display.width", 200)
    print("== Scorecard: mean (std) over 5x5 repeated stratified CV, positive class = late ==")
    print(card.round(3).to_string())
    print("\n== Slices (out-of-fold predictions, first CV repeat) ==")
    print(sl[["baseline", "slice", "n", "late", "precision", "recall", "f1"]].round(3).to_string(index=False))
    print(f"\nStump fit on all training rows splits on: {feat} > {thr:.2f}")
    fold_feats = []
    for a, _ in RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED).split(df, strat):
        tr = df.iloc[a]; m = tr[NUM].median()
        t = DecisionTreeClassifier(max_depth=1, class_weight="balanced", random_state=SEED).fit(tr[NUM].fillna(m), tr.is_late)
        fold_feats.append(f"{NUM[t.tree_.feature[0]]} > {t.tree_.threshold[0]:.1f}")
    print("Stump splits across the 25 folds:", pd.Series(fold_feats).value_counts().to_dict())


if __name__ == "__main__":
    main()
