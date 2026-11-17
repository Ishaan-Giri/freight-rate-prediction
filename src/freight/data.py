"""Loading and column-specific data checks."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

FEATURE_COLUMNS = [
    "pickup", "delivery", "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
    "distance", "equipment", "weight", "date", "market_index", "quote_signal",
]


def _read(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    return df


def load_train() -> pd.DataFrame:
    return _read(C.TRAIN_FILE)


def load_validation() -> pd.DataFrame:
    return _read(C.VALIDATION_FILE)


def load_december() -> pd.DataFrame:
    return _read(C.DECEMBER_FILE)


def load_template() -> pd.DataFrame:
    return pd.read_csv(C.TEMPLATE_FILE)


def check_raw(df: pd.DataFrame, name: str, has_target: bool) -> dict:
    """Column-specific validity checks on supplied (raw) feature files.

    Hard failures raise; known data-quality issues are counted, not "fixed" here.
    Negative longitudes are valid (western hemisphere) and are not flagged.
    """
    expected = (["load_id"] + FEATURE_COLUMNS + ([C.TARGET] if has_target else []))
    assert list(df.columns) == expected, f"{name}: unexpected columns {list(df.columns)}"
    assert df["load_id"].is_unique, f"{name}: duplicate load_id"
    assert df["date"].notna().all(), f"{name}: unparseable dates"
    for col in ["pickup", "delivery", "equipment"]:
        assert df[col].notna().all(), f"{name}: missing {col}"
    assert set(df["equipment"]) <= set(C.EQUIPMENT), f"{name}: unknown equipment"
    assert (df["pickup"] != df["delivery"]).all(), f"{name}: same-city loads"
    assert (df["distance"] > 0).all() and np.isfinite(df["distance"]).all(), f"{name}: bad distance"
    for col in ["pickup_lat", "delivery_lat"]:
        assert df[col].between(24, 50).all(), f"{name}: {col} outside contiguous US"
    for col in ["pickup_lon", "delivery_lon"]:
        assert df[col].between(-125, -66).all(), f"{name}: {col} outside contiguous US"
    mi = df["market_index"].dropna()
    assert (mi > 0).all(), f"{name}: non-positive market_index"
    assert df["quote_signal"].notna().all(), f"{name}: missing quote_signal"
    w = df["weight"]
    report = {
        "rows": int(len(df)),
        "date_min": str(df["date"].min().date()),
        "date_max": str(df["date"].max().date()),
        "weight_missing": int(w.isna().sum()),
        "weight_negative": int((w < 0).sum()),
        "weight_zero": int((w == 0).sum()),
        "weight_at_47500": int((w.abs() == C.WEIGHT_CAP).sum()),
        "weight_abs_max": float(w.abs().max()),
        "market_index_missing": int(df["market_index"].isna().sum()),
        "cities": int(pd.concat([df["pickup"], df["delivery"]]).nunique()),
    }
    if has_target:
        y = df[C.TARGET]
        assert y.notna().all() and (y > 0).all(), f"{name}: invalid target"
    return report


def check_december_raw(df: pd.DataFrame) -> None:
    expected = ["pickup", "delivery", "distance", "equipment", "weight", "date", "predicted_rate"]
    assert list(df.columns) == expected, f"december: unexpected columns {list(df.columns)}"
    assert len(df) == 31 and df["date"].is_unique


def city_lookup(*frames: pd.DataFrame) -> pd.DataFrame:
    """City -> (lat, lon) from supplied feature rows. Asserts one unique point per city."""
    parts = []
    for f in frames:
        for end in ["pickup", "delivery"]:
            parts.append(f[[end, f"{end}_lat", f"{end}_lon"]].set_axis(["city", "lat", "lon"], axis=1))
    pts = pd.concat(parts).drop_duplicates()
    counts = pts.groupby("city").size()
    bad = counts[counts > 1]
    assert bad.empty, f"cities with multiple coordinates: {list(bad.index)}"
    return pts.set_index("city").sort_index()
