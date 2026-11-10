"""Week 2 · Label construction & class balance audit.

Builds the binary delay label `is_late` from the training file, adds label-quality
flags, and writes delay-rate breakdowns by segment.

Usage:  python scripts/build_labels.py
Inputs : data/delivery_leg_telemetry_train.csv   (holdout stays sealed until week 4)
Outputs: data/processed/train_labeled.csv
         docs/week2/segment_delay_rates.csv
         docs/week2/figures/*.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "delivery_leg_telemetry_train.csv"
OUT_DATA = ROOT / "data" / "processed" / "train_labeled.csv"
OUT_DOCS = ROOT / "docs" / "week2"
OUT_FIG = OUT_DOCS / "figures"

CAPS = {"leg_distance_km": 420, "avg_speed_kmh": 105,
        "scheduled_window_minutes": 480, "stop_sequence_position": 22}
MIN_POSITIVES = 30  # below this, a segment's delay rate is flagged as unreliable


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval: stays sensible for small n and rates near 0/1."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return centre - half, centre + half


def build_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Target: late = positive class. Source column is on_time_delivery (1 = within window).
    df["is_late"] = (1 - df["on_time_delivery"]).astype(int)

    # Label-quality flags: the label is NOT changed, suspect rows are only marked.
    for col, cap in CAPS.items():
        df[f"flag_cap_{col}"] = (df[col] == cap).astype(int)
    df["flag_any_cap"] = df[[f"flag_cap_{c}" for c in CAPS]].max(axis=1)
    df["flag_hour_23"] = (df["departure_hour"] == 23).astype(int)  # possible midnight-join artefact
    df["n_missing_features"] = df.drop(
        columns=["leg_id", "on_time_delivery", "is_late", "carrier_close_code"]
        + [c for c in df.columns if c.startswith("flag_")]).isna().sum(axis=1)

    # Cross-check against carrier_close_code. It is a post-outcome field, so it is used
    # ONLY to sanity-check the label and is never a model feature.
    agree = df["carrier_close_code"].notna()
    df["flag_close_code_disagrees"] = 0
    df.loc[agree, "flag_close_code_disagrees"] = (
        df.loc[agree, "carrier_close_code"] != df.loc[agree, "on_time_delivery"]).astype(int)
    return df


def add_segments(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["seg_departure_period"] = pd.cut(df.departure_hour, [-1, 5, 11, 17, 22, 23],
                                        labels=["00-05 night", "06-11 morning", "12-17 afternoon",
                                                "18-22 evening", "23 (spike hour)"])
    df["seg_window_width"] = pd.cut(df.scheduled_window_minutes, [0, 60, 120, 240, 479, 480],
                                    labels=["<=1h", "1-2h", "2-4h", "4-8h", "480 (cap)"])
    df["seg_route_position"] = pd.cut(df.stop_sequence_position, [0, 5, 10, 15, 21, 22],
                                      labels=["stops 1-5", "stops 6-10", "stops 11-15",
                                              "stops 16-21", "stop 22 (cap)"])
    df["seg_distance"] = pd.cut(df.leg_distance_km, [0, 10, 100, 250, 419.99, 420],
                                labels=["<10 km", "10-100 km", "100-250 km", "250-420 km", "420 (cap)"])
    df["seg_early_speed"] = pd.cut(df.avg_speed_kmh, [0, 30, 60, 90, 104.99, 105],
                                   labels=["<30", "30-60", "60-90", "90-105", "105 (cap)"])
    return df


SEGMENTS = {
    "region": "Region",
    "seg_departure_period": "Time window: departure period",
    "seg_window_width": "Time window: committed window width",
    "seg_route_position": "Route: stop position",
    "seg_distance": "Route: leg distance",
    "vehicle_class": "Route: vehicle class",
    "weather_condition": "Weather",
    "demand_spike_flag": "Demand spike",
    "seg_early_speed": "Early-leg speed (km/h)",
}


def segment_table(df: pd.DataFrame) -> pd.DataFrame:
    base = df.is_late.mean()
    rows = []
    for col, name in SEGMENTS.items():
        s = df[col].astype("object").where(df[col].notna(), "(missing)")
        ct = pd.crosstab(s, df.is_late)
        p_val = stats.chi2_contingency(ct)[1]
        for level, g in df.groupby(s, observed=True):
            n, k = len(g), int(g.is_late.sum())
            lo, hi = wilson(k, n)
            rows.append({"segment_type": name, "segment": str(level), "n": n, "late": k,
                         "late_rate": k / n, "ci_low": lo, "ci_high": hi,
                         "lift_vs_overall": (k / n) / base, "share_of_rows": n / len(df),
                         "reliable": k >= MIN_POSITIVES, "chi2_p_for_type": p_val})
    return pd.DataFrame(rows)


def plot_segments(table: pd.DataFrame, base: float) -> None:
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    for seg_type, g in table.groupby("segment_type", sort=False):
        fig, ax = plt.subplots(figsize=(7, 0.45 * len(g) + 1.2))
        y = np.arange(len(g))
        colors = ["#2a6fdb" if r else "#9aa5b1" for r in g.reliable]
        ax.barh(y, g.late_rate * 100, color=colors)
        ax.errorbar(g.late_rate * 100, y,
                    xerr=[(g.late_rate - g.ci_low) * 100, (g.ci_high - g.late_rate) * 100],
                    fmt="none", ecolor="#333", capsize=3, lw=1)
        ax.axvline(base * 100, color="#d1495b", ls="--", lw=1)
        ax.set_yticks(y, [f"{s}  (n={n})" for s, n in zip(g.segment, g.n)])
        ax.invert_yaxis()
        ax.set_xlabel("Late rate (%) with 95% Wilson CI")
        ax.set_title(f"{seg_type}   (red dashed line = overall {base:.1%}; grey = fewer than {MIN_POSITIVES} late legs)",
                     fontsize=9, loc="left")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        fig.tight_layout()
        fname = seg_type.lower().replace(":", "").replace(" ", "_").replace("(", "").replace(")", "").replace("/", "")
        fig.savefig(OUT_FIG / f"late_rate_{fname}.png", dpi=150)
        plt.close(fig)


def main() -> None:
    df = build_label(pd.read_csv(RAW))
    df = add_segments(df)

    OUT_DATA.parent.mkdir(parents=True, exist_ok=True)
    keep = [c for c in df.columns if not c.startswith("seg_")]
    df[keep].to_csv(OUT_DATA, index=False)

    n, k = len(df), int(df.is_late.sum())
    lo, hi = wilson(k, n)
    print(f"Rows {n} | late {k} ({k/n:.1%}, 95% CI {lo:.1%}-{hi:.1%}) | ratio {(n-k)/k:.2f}:1")
    print(f"Majority-class accuracy {1-k/n:.1%}")

    cc = df.carrier_close_code.notna()
    print(f"carrier_close_code present: {cc.sum()} | disagrees with label: "
          f"{df.loc[cc,'flag_close_code_disagrees'].sum()} "
          f"({df.loc[cc,'flag_close_code_disagrees'].mean():.1%})")
    for f in ["flag_any_cap", "flag_hour_23", "flag_close_code_disagrees"]:
        g = df.groupby(f).is_late.agg(["count", "mean"])
        print(f, g.round(3).to_dict())

    # Regime check: how much of the late class sits in rows at a cap value or at hour 23?
    extreme = (df.flag_any_cap == 1) | (df.flag_hour_23 == 1)
    print("\n== Extreme-value regime (any cap or hour 23) ==")
    print(df.groupby(extreme.rename("extreme")).is_late.agg(n="count", late="sum", rate="mean").round(3))
    print(f"Share of all late legs inside the regime: {df[extreme].is_late.sum() / k:.1%}")
    outside = df[~extreme]
    for col in ["weather_condition", "vehicle_class", "region"]:
        print(f"\nLate rate OUTSIDE the regime by {col}:")
        print(outside.groupby(col).is_late.agg(n="count", rate="mean").round(3).to_string())
    print()

    table = segment_table(df)
    OUT_DOCS.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(OUT_DOCS / "segment_delay_rates.csv", index=False)
    plot_segments(table, k / n)
    pd.set_option("display.width", 200)
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
