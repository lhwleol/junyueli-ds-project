"""Week 3 · Versioned feature engineering pipeline.

Turns the leg-level extract into a clean, model-ready feature matrix with route-level,
time-of-day and vehicle-level features. Feature choices follow docs/week3/feature_leakage_audit.md.

Two layers, deliberately separated (cutoff logic, audit §4):
  1. build_features(raw)   - deterministic, ROW-WISE features. Each value uses only that leg's own
                             columns, so it is safe to compute on any set of rows. NaNs are kept and
                             explicit missing indicators are added.
  2. make_preprocessor()   - FITTED steps (median imputation, one-hot encoding). Returns an unfitted
                             sklearn ColumnTransformer that must be fit inside training folds only.

Usage
-----
  python scripts/features.py                         # main feature set (no cap indicators)
  python scripts/features.py --include-sensitivity   # also adds sens__* cap indicators

Outputs (data/ is git-ignored)
  data/processed/features_v{VERSION}.csv              row-wise features, NaNs kept
  data/processed/model_matrix_v{VERSION}.csv          imputed + one-hot, no nulls (preprocessor fit on the training file)
  data/processed/preprocessor_v{VERSION}.json         fitted medians / categories, to apply unchanged to the holdout in week 4
  docs/week3/feature_manifest_v{VERSION}.json         version, input/output hashes, schema
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FEATURE_VERSION = "1.0.0"

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "delivery_leg_telemetry_train.csv"
OUT_DATA = ROOT / "data" / "processed"
OUT_DOCS = ROOT / "docs" / "week3"

# ---- contract with the raw extract -------------------------------------------------------------
ID_COL = "leg_id"
TARGET_SOURCE = "on_time_delivery"
TARGET = "is_late"
POST_OUTCOME = ["carrier_close_code", "on_time_delivery"]          # never features (audit R11, R12)
RAW_REQUIRED = ["leg_id", "region", "scheduled_window_minutes", "leg_distance_km", "departure_hour",
                "weather_condition", "avg_speed_kmh", "vehicle_class", "stop_sequence_position",
                "demand_spike_flag"]

# ---- constants (documented assumptions) --------------------------------------------------------
CAPS = {"leg_distance_km": 420, "avg_speed_kmh": 105, "scheduled_window_minutes": 480,
        "stop_sequence_position": 22, "departure_hour": 23}
SHORT_LEG_KM = 2.0             # first 20% < 400 m -> speed unreliable (ML engineer note)
LATE_ROUTE_STOP = 15
MAX_TRAVEL_MIN = 24 * 60       # clip implied travel time to one day (EDA: max ~51 h is not plausible)
HEAVY = {"Articulated", "Heavy Rigid"}
WEATHER_SEVERITY = {"Clear": 0, "Fog": 1, "Light Rain": 1, "Heavy Rain": 2, "Snow": 2, "Severe Storm": 3}
PERIODS = [(-1, 5, "night"), (5, 11, "morning"), (11, 17, "afternoon"), (17, 23, "evening")]

# ---- feature registry: name -> (family, dtype, description) -------------------------------------
NUMERIC, BINARY, CATEGORICAL = "numeric", "binary", "categorical"
REGISTRY: dict[str, tuple[str, str, str]] = {
    # route-level
    "route__leg_distance_km":      ("route", NUMERIC, "Straight-line depot-to-address distance (km)"),
    "route__stop_position":        ("route", NUMERIC, "Position of the leg in the driver's daily route (1 = first)"),
    "route__is_late_route_stop":   ("route", BINARY, f"1 if stop position >= {LATE_ROUTE_STOP}"),
    "route__window_min":           ("route", NUMERIC, "Width of the customer-committed window (minutes)"),
    "route__km_per_window_hour":   ("route", NUMERIC, "Distance divided by window width (km per window hour): planning difficulty"),
    "route__region":               ("route", CATEGORICAL, "Operational region"),
    # time-of-day
    "time__departure_hour":        ("time", NUMERIC, "Departure hour 0-23 (time zone unconfirmed)"),
    "time__hour_sin":              ("time", NUMERIC, "sin(2*pi*hour/24), cyclical encoding"),
    "time__hour_cos":              ("time", NUMERIC, "cos(2*pi*hour/24), cyclical encoding"),
    "time__is_night":              ("time", BINARY, "1 if departure hour 0-5"),
    "time__period":                ("time", CATEGORICAL, "night / morning / afternoon / evening"),
    # vehicle-level
    "veh__class":                  ("vehicle", CATEGORICAL, "Vehicle class"),
    "veh__is_heavy":               ("vehicle", BINARY, "1 if Articulated or Heavy Rigid"),
    "veh__early_speed_kmh":        ("vehicle", NUMERIC, f"Avg speed over first 20% of leg; set to missing for legs < {SHORT_LEG_KM} km"),
    "veh__implied_travel_min":     ("vehicle", NUMERIC, f"distance / early speed (minutes), clipped to [0, {MAX_TRAVEL_MIN}]"),
    "veh__window_slack_min":       ("vehicle", NUMERIC, "window width minus implied travel time (minutes)"),
    # weather / operations
    "wx__condition":               ("weather", CATEGORICAL, "NOAA weather category at departure"),
    "wx__severity":                ("weather", NUMERIC, "Ordinal weather severity 0-3"),
    "wx__bad_weather_x_heavy":     ("weather", BINARY, "1 if severity > 0 and heavy vehicle"),
    "ops__demand_spike":           ("ops", BINARY, "Capacity-planning surge flag"),
    # missing indicators
    "miss__departure_hour":        ("missing", BINARY, "1 if departure_hour missing (informative, EDA)"),
    "miss__early_speed":           ("missing", BINARY, "1 if avg_speed_kmh missing (telematics dropout)"),
    "miss__weather":               ("missing", BINARY, "1 if weather_condition missing"),
    "miss__leg_distance":          ("missing", BINARY, "1 if leg_distance_km missing"),
}
SENSITIVITY_REGISTRY: dict[str, tuple[str, str, str]] = {
    **{f"sens__at_cap_{c}": ("sensitivity", BINARY, f"1 if {c} == {v} (suspected leakage, audit D13-D17)")
       for c, v in CAPS.items()},
    "sens__n_extreme_flags": ("sensitivity", NUMERIC, "Count of capped values incl. hour 23 (suspected leakage)"),
}


# =============================== layer 1: row-wise features ======================================
def _validate_raw(raw: pd.DataFrame) -> None:
    missing = [c for c in RAW_REQUIRED if c not in raw.columns]
    if missing:
        raise ValueError(f"raw extract is missing required columns: {missing}")
    if raw[ID_COL].duplicated().any():
        raise ValueError("leg_id is not unique")
    hours = raw["departure_hour"].dropna()
    if not hours.between(0, 23).all():
        raise ValueError("departure_hour outside 0-23")


def _to_bool(s: pd.Series) -> pd.Series:
    return s.map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0, "true": 1.0, "false": 0.0})


def build_features(raw: pd.DataFrame, include_sensitivity: bool = False) -> pd.DataFrame:
    """Row-wise feature construction. Never reads POST_OUTCOME columns. NaNs are preserved."""
    _validate_raw(raw)
    r = raw
    f = pd.DataFrame(index=r.index)

    dist, stop, win = r["leg_distance_km"], r["stop_sequence_position"], r["scheduled_window_minutes"]
    hour, speed = r["departure_hour"], r["avg_speed_kmh"]

    # route-level
    f["route__leg_distance_km"] = dist
    f["route__stop_position"] = stop
    f["route__is_late_route_stop"] = (stop >= LATE_ROUTE_STOP).astype(float).where(stop.notna())
    f["route__window_min"] = win
    f["route__km_per_window_hour"] = dist / (win / 60)
    f["route__region"] = r["region"]

    # time-of-day
    f["time__departure_hour"] = hour
    f["time__hour_sin"] = np.sin(2 * np.pi * hour / 24)
    f["time__hour_cos"] = np.cos(2 * np.pi * hour / 24)
    f["time__is_night"] = hour.between(0, 5).astype(float).where(hour.notna())
    f["time__period"] = pd.cut(hour, [p[0] for p in PERIODS] + [PERIODS[-1][1]],
                               labels=[p[2] for p in PERIODS]).astype("object")

    # vehicle-level
    vclass = r["vehicle_class"]
    f["veh__class"] = vclass
    f["veh__is_heavy"] = vclass.isin(HEAVY).astype(float).where(vclass.notna())
    speed_ok = speed.where(~(dist < SHORT_LEG_KM))
    f["veh__early_speed_kmh"] = speed_ok
    travel = (dist / speed_ok * 60).clip(0, MAX_TRAVEL_MIN)
    f["veh__implied_travel_min"] = travel
    f["veh__window_slack_min"] = win - travel

    # weather / operations
    wx = r["weather_condition"]
    f["wx__condition"] = wx
    f["wx__severity"] = wx.map(WEATHER_SEVERITY)
    f["wx__bad_weather_x_heavy"] = ((f["wx__severity"] > 0) & (f["veh__is_heavy"] == 1)).astype(float) \
        .where(f["wx__severity"].notna() & f["veh__is_heavy"].notna())
    f["ops__demand_spike"] = _to_bool(r["demand_spike_flag"])

    # missing indicators (from the RAW columns, before any masking)
    f["miss__departure_hour"] = hour.isna().astype(int)
    f["miss__early_speed"] = speed.isna().astype(int)
    f["miss__weather"] = wx.isna().astype(int)
    f["miss__leg_distance"] = dist.isna().astype(int)

    if include_sensitivity:
        for c, v in CAPS.items():
            f[f"sens__at_cap_{c}"] = (r[c] == v).astype(int)
        f["sens__n_extreme_flags"] = f[[f"sens__at_cap_{c}" for c in CAPS]].sum(axis=1)

    f.insert(0, ID_COL, r[ID_COL].values)
    if TARGET_SOURCE in r.columns:
        f[TARGET] = (1 - r[TARGET_SOURCE]).astype(int)

    expected = [ID_COL] + list(REGISTRY) + (list(SENSITIVITY_REGISTRY) if include_sensitivity else [])
    expected += [TARGET] if TARGET in f.columns else []
    assert list(f.columns) == expected, "feature columns drifted from REGISTRY"
    return f


def feature_columns(include_sensitivity: bool = False) -> dict[str, list[str]]:
    reg = {**REGISTRY, **(SENSITIVITY_REGISTRY if include_sensitivity else {})}
    return {kind: [n for n, (_, t, _) in reg.items() if t == kind] for kind in (NUMERIC, BINARY, CATEGORICAL)}


# =============================== layer 2: fitted preprocessing ===================================
def make_preprocessor(include_sensitivity: bool = False, scale_numeric: bool = False) -> ColumnTransformer:
    """UNFITTED. Fit inside training folds only (audit §4.3).
    scale_numeric=True adds standardisation (use for linear models; trees don't need it).
    The exported model matrix is left unscaled so values stay interpretable."""
    cols = feature_columns(include_sensitivity)
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scale", StandardScaler()))
    numeric = Pipeline(num_steps)
    binary = Pipeline([("impute", SimpleImputer(strategy="most_frequent"))])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    ct = ColumnTransformer([("num", numeric, cols[NUMERIC]),
                            ("bin", binary, cols[BINARY]),
                            ("cat", categorical, cols[CATEGORICAL])],
                           remainder="drop", verbose_feature_names_out=False)
    ct.set_output(transform="pandas")
    return ct


def _clean_names(cols) -> list[str]:
    """Consistent snake_case for one-hot outputs, e.g. 'veh__class_Light Van' -> 'veh__class__light_van'."""
    out = []
    for c in cols:
        for base in (n for n, (_, t, _) in {**REGISTRY, **SENSITIVITY_REGISTRY}.items() if t == CATEGORICAL):
            if c.startswith(base + "_") and not c.startswith(base + "__"):
                c = base + "__" + c[len(base) + 1:]
                break
        out.append(c.lower().replace(" ", "_").replace("-", "_"))
    return out


def preprocessor_params(ct: ColumnTransformer) -> dict:
    p = {}
    for name, pipe, cols in ct.transformers_:
        if name == "remainder":
            continue
        imp = pipe.named_steps["impute"]
        p[name] = {"columns": list(cols), "impute_strategy": imp.strategy,
                   "impute_values": [v if isinstance(v, str) else float(v) for v in imp.statistics_]}
        if "onehot" in pipe.named_steps:
            p[name]["categories"] = [list(map(str, c)) for c in pipe.named_steps["onehot"].categories_]
    return p


# =============================== validation & versioning =========================================
def validate_matrix(X: pd.DataFrame, n_rows: int) -> None:
    assert len(X) == n_rows, "row count changed"
    assert not X.isna().any().any(), "model-ready matrix contains nulls"
    assert np.isfinite(X.select_dtypes("number").to_numpy()).all(), "non-finite values"
    assert X.columns.is_unique, "duplicate column names"
    bad = [c for c in X.columns if any(k in c for k in ("carrier_close_code", "on_time_delivery", ID_COL))]
    assert not bad, f"forbidden columns leaked into the matrix: {bad}"
    assert all(c == c.lower() and " " not in c for c in X.columns), "non snake_case column name"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(raw_path: Path = RAW, include_sensitivity: bool = False) -> dict:
    raw = pd.read_csv(raw_path)
    feats = build_features(raw, include_sensitivity)

    ct = make_preprocessor(include_sensitivity)
    X = ct.fit_transform(feats)  # fit on the full TRAINING file = the final training matrix (holdout untouched)
    X.columns = _clean_names(X.columns)
    validate_matrix(X, len(raw))
    X.insert(0, ID_COL, feats[ID_COL].values)
    X[TARGET] = feats[TARGET].values

    tag = f"v{FEATURE_VERSION}" + ("_sens" if include_sensitivity else "")
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    OUT_DOCS.mkdir(parents=True, exist_ok=True)
    f_path, m_path = OUT_DATA / f"features_{tag}.csv", OUT_DATA / f"model_matrix_{tag}.csv"
    p_path = OUT_DATA / f"preprocessor_{tag}.json"
    feats.to_csv(f_path, index=False, float_format="%.6g")
    X.to_csv(m_path, index=False, float_format="%.6g")
    p_path.write_text(json.dumps(preprocessor_params(ct), indent=2))

    reg = {**REGISTRY, **(SENSITIVITY_REGISTRY if include_sensitivity else {})}
    manifest = {
        "feature_version": FEATURE_VERSION,
        "include_sensitivity": include_sensitivity,
        "input": {"path": str(raw_path.relative_to(ROOT)), "sha256": _sha256(raw_path), "rows": len(raw)},
        "outputs": {p.name: _sha256(p) for p in (f_path, m_path, p_path)},
        "row_wise_features": {n: {"family": fam, "type": t, "description": d,
                                  "pct_missing_before_imputation": round(float(feats[n].isna().mean() * 100), 2)}
                              for n, (fam, t, d) in reg.items()},
        "model_matrix_columns": [c for c in X.columns if c not in (ID_COL, TARGET)],
        "excluded_columns": {"carrier_close_code": "post-outcome (leak)", "on_time_delivery": "target source",
                             "leg_id": "identifier (kept as key, not a feature)"},
    }
    (OUT_DOCS / f"feature_manifest_{tag}.json").write_text(json.dumps(manifest, indent=2))
    return {"features": feats, "matrix": X, "manifest": manifest}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--include-sensitivity", action="store_true", help="add sens__* cap indicators")
    ap.add_argument("--raw", type=Path, default=RAW)
    a = ap.parse_args()
    out = run(a.raw, a.include_sensitivity)
    m = out["manifest"]
    print(f"feature version {m['feature_version']} | rows {m['input']['rows']} | "
          f"row-wise features {len(m['row_wise_features'])} | model-matrix columns {len(m['model_matrix_columns'])}")
    for k, v in m["outputs"].items():
        print(f"  {k}  sha256={v[:12]}")


if __name__ == "__main__":
    main()
