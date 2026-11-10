"""Week 1 follow-up checks (mentor feedback).

1. Class imbalance of the late class, overall and per region, with 95% CIs.
2. Is leg_id order a usable proxy for time? Tests for drift across leg_id
   blocks and for serial dependence between consecutive leg_ids.

Usage:  python scripts/week1_followup_checks.py
Reads only data/delivery_leg_telemetry_train.csv (holdout stays sealed).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

DATA = Path(__file__).resolve().parents[1] / "data" / "delivery_leg_telemetry_train.csv"


def wald_ci(p: float, n: int) -> float:
    return 1.96 * np.sqrt(p * (1 - p) / n)


def main() -> None:
    df = pd.read_csv(DATA)
    df["is_late"] = 1 - df["on_time_delivery"]

    # 1. Class imbalance
    n, k = len(df), int(df.is_late.sum())
    p = k / n
    print("== Class imbalance (late = positive) ==")
    print(f"Late: {k}/{n} = {p:.1%}  (95% CI {p - wald_ci(p, n):.1%} to {p + wald_ci(p, n):.1%})")
    print(f"On-time : late ratio = {(n - k) / k:.2f} : 1")
    print(f"Majority-class accuracy = {1 - p:.1%}\n")

    by_region = df.groupby("region").is_late.agg(n="count", late="sum", rate="mean")
    by_region["ci95"] = [wald_ci(r, c) for r, c in zip(by_region.rate, by_region.n)]
    print(by_region.round(3), "\n")

    # 2. leg_id as a time proxy
    df["seq"] = df.leg_id.str.extract(r"(\d+)$", expand=False).astype(int)
    df = df.sort_values("seq").reset_index(drop=True)
    df["block"] = pd.qcut(df.seq, 8, labels=False)

    print("== Drift across 8 leg_id blocks ==")
    print(df.groupby("block").agg(late=("is_late", "mean"),
                                  speed=("avg_speed_kmh", "mean"),
                                  dist=("leg_distance_km", "mean"),
                                  hour=("departure_hour", "mean")).round(3))
    chi2_p = stats.chi2_contingency(pd.crosstab(df.block, df.is_late))[1]
    print(f"Chi-square test, late rate vs block: p = {chi2_p:.3f}\n")

    print("== Serial dependence between consecutive leg_ids ==")
    for col in ["region", "weather_condition", "vehicle_class"]:
        a, b = df[col].values[:-1], df[col].values[1:]
        valid = pd.notna(a) & pd.notna(b)
        same = (a[valid] == b[valid]).mean()
        chance = (df[col].value_counts(normalize=True) ** 2).sum()
        print(f"{col}: P(next leg has same value) = {same:.3f}  vs chance {chance:.3f}")
    for col in ["is_late", "avg_speed_kmh", "departure_hour", "leg_distance_km"]:
        print(f"{col}: lag-1 autocorrelation = {df[col].astype(float).autocorr(1):+.3f}")

    x = df.is_late.values
    runs = 1 + int((x[1:] != x[:-1]).sum())
    n1, n0 = int(x.sum()), int(len(x) - x.sum())
    mu = 2 * n1 * n0 / len(x) + 1
    var = (mu - 1) * (mu - 2) / (len(x) - 1)
    print(f"Runs test on is_late: {runs} runs vs {mu:.1f} expected, z = {(runs - mu) / np.sqrt(var):+.2f}")


if __name__ == "__main__":
    main()
