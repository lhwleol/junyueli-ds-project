"""Profile the training file for the week 1 data source inventory.

Usage:  python scripts/profile_train.py
Reads only data/delivery_leg_telemetry_train.csv. The holdout file stays sealed until week 4.
"""
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data" / "delivery_leg_telemetry_train.csv"
NUMERIC = ["scheduled_window_minutes", "leg_distance_km", "departure_hour",
           "avg_speed_kmh", "stop_sequence_position"]
CATEGORICAL = ["region", "weather_condition", "vehicle_class",
               "demand_spike_flag", "carrier_close_code", "on_time_delivery"]


def main() -> None:
    df = pd.read_csv(DATA)
    print(f"Rows: {len(df)}  Columns: {df.shape[1]}\n")

    print("== Missing values (count, %) ==")
    miss = pd.DataFrame({"n": df.isna().sum(), "pct": (df.isna().mean() * 100).round(1)})
    print(miss, "\n")
    print(f"Rows with at least one missing value: {df.isna().any(axis=1).sum()}\n")

    print("== Duplicates ==")
    print(f"Duplicate leg_id: {df.leg_id.duplicated().sum()}")
    print(f"Duplicate rows ignoring leg_id: {df.drop(columns='leg_id').duplicated().sum()}\n")

    print("== Numeric ranges ==")
    print(df[NUMERIC].describe().T.round(2), "\n")

    print("== Values sitting exactly at the column maximum (possible caps) ==")
    for c in NUMERIC:
        mx = df[c].max()
        print(f"{c}: max={mx}, rows at max={(df[c] == mx).sum()}")
    print()

    print("== departure_hour counts ==")
    print(df.departure_hour.value_counts().sort_index().to_dict(), "\n")

    print("== departure_hour by region (checks the West Coast +3h note) ==")
    print(df.groupby("region").departure_hour.describe().round(2), "\n")

    print("== avg_speed_kmh by leg distance band ==")
    bands = pd.cut(df.leg_distance_km, [0, 2, 10, 50, 200, 421])
    print(df.groupby(bands, observed=True).avg_speed_kmh.agg(["count", "mean", "std"]).round(1), "\n")

    print("== Weather by region ==")
    print(pd.crosstab(df.region, df.weather_condition), "\n")

    travel_min = df.leg_distance_km / df.avg_speed_kmh * 60
    longer = (travel_min > df.scheduled_window_minutes).sum()
    print(f"Legs whose implied travel time exceeds the window width: {longer} of {travel_min.notna().sum()}\n")

    for c in CATEGORICAL:
        print(df[c].value_counts(dropna=False), "\n")


if __name__ == "__main__":
    main()
