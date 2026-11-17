"""Feature matrices shared by training, validation and December prediction (one code path)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

EQUIP_CODE = {e: i for i, e in enumerate(C.EQUIPMENT)}


def tree_matrix(df: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """Features for gradient-boosted trees. `df` must come from Preprocessor.transform."""
    X = pd.DataFrame(index=df.index)
    X["distance"] = df["distance"].astype(float)
    X["equipment"] = df["equipment"].map(EQUIP_CODE).astype(int)
    X["weight"] = df["weight_clean"]
    X["weight_missing"] = df["weight_missing"]
    X["weight_invalid"] = df["weight_invalid"]
    if spec.get("at_cap"):
        X["weight_at_cap"] = df["weight_at_cap"]
    X["market_index"] = df["mi"]
    if spec.get("dow"):
        X["dow"] = df["date"].dt.dayofweek
    if spec.get("coords"):
        for c in ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]:
            X[c] = df[c].astype(float)
        X["dlat"] = X["delivery_lat"] - X["pickup_lat"]
        X["dlon"] = X["delivery_lon"] - X["pickup_lon"]
    if spec.get("qs_raw"):
        X["quote_signal"] = df["quote_signal"].astype(float)
    if spec.get("qs_aligned"):
        X["qs_aligned"] = df["qs_aligned"]
        X["qs_rho"] = df["qs_rho"]
    return X


def linear_matrix(df: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """Design matrix for the log-linear model (log rate target)."""
    X = pd.DataFrame(index=df.index)
    ld = np.log(df["distance"].astype(float))
    X["log_distance"] = ld
    X["log_distance_sq"] = (ld - 6.8) ** 2
    for e in C.EQUIPMENT[1:]:
        ind = (df["equipment"] == e).astype(float)
        X[f"eq_{e}"] = ind
        X[f"eq_{e}_x_logd"] = ind * (ld - 6.8)
    X["log_mi"] = np.log(df["mi"])
    X["weight_10k"] = df["weight_clean"] / 1e4
    X["weight_missing"] = df["weight_missing"]
    X["weight_invalid"] = df["weight_invalid"]
    if spec.get("at_cap"):
        X["weight_at_cap"] = df["weight_at_cap"]
    if spec.get("dow"):
        dow = df["date"].dt.dayofweek
        for d in range(1, 7):
            X[f"dow_{d}"] = (dow == d).astype(float)
    if spec.get("qs_raw"):
        X["quote_signal"] = df["quote_signal"].astype(float).fillna(0.0)
    if spec.get("qs_aligned"):
        X["qs_aligned"] = df["qs_aligned"]
    return X
