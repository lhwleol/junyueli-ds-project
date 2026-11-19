"""Week 3 · Feature candidate brainstorm & leakage audit.

1. Builds every candidate feature listed in docs/week3/feature_leakage_audit.md.
2. Scores each feature ON ITS OWN with 5-fold stratified CV (ROC-AUC for the late class).
   A single pre-outcome feature that predicts the label almost perfectly is a leakage red flag.
3. Demonstrates the cutoff rule for fitted transforms: target encoding fit on ALL rows
   vs. fit inside the training folds only.

Usage:   python scripts/leakage_audit.py
Output:  docs/week3/single_feature_auc.csv
Training file only; holdout stays sealed.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "delivery_leg_telemetry_train.csv"
OUT = ROOT / "docs" / "week3"
SEED = 42
CAPS = {"leg_distance_km": 420, "avg_speed_kmh": 105, "scheduled_window_minutes": 480,
        "stop_sequence_position": 22, "departure_hour": 23}
WEATHER_SEVERITY = {"Clear": 0, "Fog": 1, "Light Rain": 1, "Heavy Rain": 2, "Snow": 2, "Severe Storm": 3}


def build_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise features only: each value depends on the leg's own columns, never on other rows
    or on the label. (Fitted encodings are handled separately, inside CV folds.)"""
    f = pd.DataFrame(index=df.index)
    # raw columns
    for c in ["scheduled_window_minutes", "leg_distance_km", "departure_hour",
              "avg_speed_kmh", "stop_sequence_position"]:
        f[c] = df[c]
    for c in ["region", "weather_condition", "vehicle_class"]:
        f[c] = df[c].astype("category").cat.codes.replace(-1, np.nan)
    f["demand_spike_flag"] = df["demand_spike_flag"].map({True: 1, False: 0, "True": 1, "False": 0})
    # missing indicators
    for c in ["departure_hour", "avg_speed_kmh", "weather_condition", "leg_distance_km"]:
        f[f"missing_{c}"] = df[c].isna().astype(int)
    # cap / extreme indicators
    for c, v in CAPS.items():
        f[f"at_cap_{c}"] = (df[c] == v).astype(int)
    f["n_extreme_flags"] = f[[f"at_cap_{c}" for c in CAPS]].sum(axis=1)
    # time of day
    f["hour_sin"] = np.sin(2 * np.pi * df.departure_hour / 24)
    f["hour_cos"] = np.cos(2 * np.pi * df.departure_hour / 24)
    f["is_night_departure"] = df.departure_hour.between(0, 5).astype(float).where(df.departure_hour.notna())
    # route / vehicle
    f["late_route_stop"] = (df.stop_sequence_position >= 15).astype(float).where(df.stop_sequence_position.notna())
    f["is_heavy_vehicle"] = df.vehicle_class.isin(["Articulated", "Heavy Rigid"]).astype(float).where(df.vehicle_class.notna())
    f["weather_severity"] = df.weather_condition.map(WEATHER_SEVERITY)
    f["bad_weather_x_heavy"] = ((f.weather_severity > 0) & (f.is_heavy_vehicle == 1)).astype(float)
    # speed-derived
    speed_ok = df.avg_speed_kmh.where(df.leg_distance_km >= 2)          # mask speed on legs < 2 km
    f["speed_masked_short_legs"] = speed_ok
    f["implied_travel_min"] = df.leg_distance_km / df.avg_speed_kmh * 60
    f["window_slack_min"] = df.scheduled_window_minutes - f.implied_travel_min
    f["km_per_window_hour"] = df.leg_distance_km / (df.scheduled_window_minutes / 60)
    # POST-OUTCOME (for the audit only - never model inputs)
    f["LEAK_carrier_close_code"] = df["carrier_close_code"]
    return f


def single_feature_auc(x: pd.Series, y: pd.Series) -> float:
    """CV AUC of a small tree on ONE feature (missing -> its own value). Fit per fold."""
    x = x.fillna(-999).to_frame()
    scores = []
    for a, b in StratifiedKFold(5, shuffle=True, random_state=SEED).split(x, y):
        clf = DecisionTreeClassifier(max_depth=3, min_samples_leaf=20, random_state=SEED)
        clf.fit(x.iloc[a], y.iloc[a])
        scores.append(roc_auc_score(y.iloc[b], clf.predict_proba(x.iloc[b])[:, 1]))
    return float(np.mean(scores))


def target_encoding_demo(df: pd.DataFrame, y: pd.Series, col: str = "region", noise_col: str = "leg_id"):
    """Show why encodings must be fit inside the training folds.
    noise_col = leg_id: unique per row, so a leaky target encoding of it reproduces the label."""
    out = {}
    for name, c in [("region", col), ("leg_id (pure noise)", noise_col)]:
        # LEAKY: means computed on all rows, including the row being scored
        leaky = df[c].map(y.groupby(df[c]).mean())
        # CORRECT: means computed on the training folds only, applied to the held-out fold
        correct = pd.Series(np.nan, index=df.index)
        for a, b in StratifiedKFold(5, shuffle=True, random_state=SEED).split(df, y):
            means = y.iloc[a].groupby(df[c].iloc[a]).mean()
            correct.iloc[b] = df[c].iloc[b].map(means).fillna(y.iloc[a].mean()).values
        out[name] = {"AUC_leaky_full_data": roc_auc_score(y, leaky),
                     "AUC_in_fold": roc_auc_score(y, correct)}
    return pd.DataFrame(out).T


def main() -> None:
    df = pd.read_csv(DATA)
    y = 1 - df["on_time_delivery"]
    feats = build_candidates(df)

    rows = []
    for c in feats.columns:
        rows.append({"feature": c, "single_feature_cv_auc": single_feature_auc(feats[c], y),
                     "pct_missing": feats[c].isna().mean() * 100})
    res = pd.DataFrame(rows).sort_values("single_feature_cv_auc", ascending=False)
    cc = df.carrier_close_code.notna()
    agree = (df.loc[cc, "carrier_close_code"] == df.loc[cc, "on_time_delivery"]).mean()

    OUT.mkdir(parents=True, exist_ok=True)
    res.round(4).to_csv(OUT / "single_feature_auc.csv", index=False)
    pd.set_option("display.width", 160)
    print(res.round(3).to_string(index=False))
    print(f"\ncarrier_close_code agrees with on_time_delivery on {agree:.1%} of the {cc.sum()} rows where present")

    ext = feats.n_extreme_flags > 0
    print("\nSingle-feature AUC on NON-extreme rows only (no capped value, not hour 23):")
    sub = []
    for c in ["scheduled_window_minutes", "leg_distance_km", "departure_hour", "avg_speed_kmh",
              "stop_sequence_position", "weather_severity", "is_heavy_vehicle", "window_slack_min",
              "missing_departure_hour", "LEAK_carrier_close_code"]:
        sub.append({"feature": c, "auc_non_extreme": single_feature_auc(feats.loc[~ext, c], y[~ext])})
    print(pd.DataFrame(sub).round(3).to_string(index=False))

    print("\nTarget-encoding demo (why fitted transforms must stay inside training folds):")
    print(target_encoding_demo(df, y).round(3).to_string())


if __name__ == "__main__":
    main()
