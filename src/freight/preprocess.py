"""Fold-fitted preprocessing and the December-mode feature reconstruction.

Every learned statistic (imputation medians, fallback means) is fitted on the training
fold only via `Preprocessor.fit`, then applied with `transform`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .qs_regime import regime_features


class Preprocessor:
    def __init__(self, weight_negative: str = "abs"):
        assert weight_negative in {"abs", "nan"}
        self.weight_negative = weight_negative
        self.weight_median_: dict[str, float] = {}
        self.weight_median_global_: float = np.nan
        self.mi_mean_: float = np.nan

    def _weight_raw(self, df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
        w = df["weight"].astype(float)
        missing = w.isna()
        invalid = w <= 0  # negative (or zero) weights are physically invalid
        if self.weight_negative == "abs":
            w = w.where(~invalid, w.abs())
            w = w.where(w > 0)  # zeros (none observed) become missing
        else:
            w = w.where(~invalid)
        return w, missing, invalid

    def fit(self, train: pd.DataFrame) -> "Preprocessor":
        w, _, _ = self._weight_raw(train)
        self.weight_median_ = w.groupby(train["equipment"]).median().to_dict()
        self.weight_median_global_ = float(w.median())
        self.mi_mean_ = float(train["market_index"].mean())
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        w, missing, invalid = self._weight_raw(df)
        fill = df["equipment"].map(self.weight_median_).fillna(self.weight_median_global_)
        out["weight_missing"] = missing.astype(int)
        out["weight_invalid"] = invalid.astype(int)
        out["weight_at_cap"] = (df["weight"].abs() == C.WEIGHT_CAP).astype(int)
        out["weight_clean"] = w.fillna(fill)
        # market_index is a date-level signal: impute from the same file's feature rows on
        # that date (label-free), falling back to the training-fold mean.
        day_mean = df.groupby("date")["market_index"].transform("mean")
        out["mi_missing"] = df["market_index"].isna().astype(int)
        out["mi"] = df["market_index"].fillna(day_mean).fillna(self.mi_mean_)
        out[["qs_rho", "qs_aligned"]] = regime_features(df)
        return out


def december_mode(features: pd.DataFrame, lookup: pd.DataFrame, mi_reference: pd.DataFrame) -> pd.DataFrame:
    """Rebuild the columns the December chart file lacks, using only feature data.

    - coordinates: from the supplied city lookup (unknown city -> error, no silent imputation)
    - market_index: per-date mean of `mi_reference` feature rows (validation.csv for December)
    - quote_signal: absent (NaN)
    The held-out target must never be passed in.
    """
    assert C.TARGET not in features.columns, "December-mode reconstruction must not see the target"
    assert C.TARGET not in mi_reference.columns, "market_index reference must be feature-only"
    out = features.drop(columns=[c for c in C.DECEMBER_MISSING if c in features.columns]).copy()
    for end in ["pickup", "delivery"]:
        unknown = set(out[end]) - set(lookup.index)
        if unknown:
            raise KeyError(f"no coordinates for {end} cities: {sorted(unknown)}")
        out[f"{end}_lat"] = out[end].map(lookup["lat"])
        out[f"{end}_lon"] = out[end].map(lookup["lon"])
    daily = mi_reference.groupby("date")["market_index"].mean()
    missing_dates = set(out["date"]) - set(daily.dropna().index)
    if missing_dates:
        raise KeyError(f"no market_index reference for dates: {sorted(missing_dates)[:5]}")
    out["market_index"] = out["date"].map(daily)
    out["quote_signal"] = np.nan
    return out
