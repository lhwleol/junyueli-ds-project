"""Tests for scripts/features.py.   Run:  python -m pytest tests -q"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import features as F  # noqa: E402


def raw_rows():
    return pd.DataFrame({
        "leg_id": ["A", "B", "C", "D"],
        "region": ["Midwest", "Mountain", None, "West Coast"],
        "scheduled_window_minutes": [120, 480, np.nan, 60],
        "leg_distance_km": [100.0, 420.0, 1.5, np.nan],
        "departure_hour": [3, 23, np.nan, 12],
        "weather_condition": ["Clear", "Snow", None, "Fog"],
        "avg_speed_kmh": [50.0, 105.0, 30.0, np.nan],
        "vehicle_class": ["Light Van", "Articulated", None, "Heavy Rigid"],
        "stop_sequence_position": [1, 22, 15, np.nan],
        "demand_spike_flag": ["True", "False", None, "True"],
        "carrier_close_code": [1, 0, np.nan, 1],
        "on_time_delivery": [1, 0, 1, 1],
    })


def test_columns_follow_registry_and_never_use_post_outcome():
    f = F.build_features(raw_rows())
    assert list(f.columns) == ["leg_id"] + list(F.REGISTRY) + ["is_late"]
    assert not any("carrier_close_code" in c for c in f.columns)


def test_sensitivity_features_only_when_requested():
    assert not any(c.startswith("sens__") for c in F.build_features(raw_rows()).columns)
    f = F.build_features(raw_rows(), include_sensitivity=True)
    assert f.loc[1, "sens__n_extreme_flags"] == 5          # 420 km, 105 km/h, 480 min, stop 22, hour 23


def test_row_wise_values():
    f = F.build_features(raw_rows())
    assert f.loc[0, "time__is_night"] == 1 and f.loc[0, "time__period"] == "night"
    assert f.loc[1, "time__period"] == "evening"
    assert f.loc[0, "veh__implied_travel_min"] == pytest.approx(120.0)
    assert f.loc[0, "veh__window_slack_min"] == pytest.approx(0.0)
    assert np.isnan(f.loc[2, "veh__early_speed_kmh"])       # leg < 2 km -> speed masked
    assert f.loc[2, "miss__early_speed"] == 0              # ...but the raw value was present
    assert f.loc[3, "miss__early_speed"] == 1 and f.loc[3, "miss__leg_distance"] == 1
    assert f.loc[2, "route__is_late_route_stop"] == 1
    assert f.loc[1, "wx__bad_weather_x_heavy"] == 1
    assert list(f["is_late"]) == [0, 1, 0, 0]


def test_features_are_row_wise_independent_of_other_rows():
    full = F.build_features(raw_rows())
    for i in range(4):
        one = F.build_features(raw_rows().iloc[[i]])
        pd.testing.assert_frame_equal(one.reset_index(drop=True), full.iloc[[i]].reset_index(drop=True),
                                      check_dtype=False)


def test_preprocessor_outputs_clean_matrix_and_handles_unseen_categories():
    tr, te = raw_rows().iloc[:3], raw_rows().iloc[3:]
    ct = F.make_preprocessor()
    Xtr = ct.fit_transform(F.build_features(tr))
    Xte = ct.transform(F.build_features(te))           # 'West Coast' / 'Heavy Rigid' / 'Fog' unseen
    for X in (Xtr, Xte):
        X.columns = F._clean_names(X.columns)
        F.validate_matrix(X, len(X))
    assert list(Xtr.columns) == list(Xte.columns)


def test_validate_matrix_rejects_nulls_and_forbidden_columns():
    with pytest.raises(AssertionError):
        F.validate_matrix(pd.DataFrame({"a": [1.0, np.nan]}), 2)
    with pytest.raises(AssertionError):
        F.validate_matrix(pd.DataFrame({"carrier_close_code": [1.0]}), 1)


def test_bad_raw_input_is_rejected():
    bad = raw_rows().drop(columns=["region"])
    with pytest.raises(ValueError):
        F.build_features(bad)
    dup = raw_rows(); dup.loc[1, "leg_id"] = "A"
    with pytest.raises(ValueError):
        F.build_features(dup)
