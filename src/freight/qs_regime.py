"""Label-free quote_signal regime features.

EDA showed that, within a month, quote_signal is strongly rank-correlated with distance,
with a sign that flips between months (and ~0 correlation in Aug / Nov / Dec). This module
estimates that relationship per date from FEATURES ONLY (never the target) and builds a
regime-aware feature: rho_date * (quote_signal - within-date linear fit on log distance).
When a date has no detectable regime (rho ~ 0) or quote_signal is missing, the feature ~ 0.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def per_date_spearman(df: pd.DataFrame) -> pd.Series:
    """Spearman(quote_signal, log distance) for each date, from feature rows only."""
    d = pd.DataFrame({
        "date": df["date"].values,
        "rq": df.groupby("date")["quote_signal"].rank().values,
        "rd": df.groupby("date")["distance"].rank().values,
    })
    def _corr(g: pd.DataFrame) -> float:
        if len(g) < 10 or g["rq"].isna().all():
            return 0.0
        c = g["rq"].corr(g["rd"])
        return 0.0 if pd.isna(c) else float(c)
    return d.groupby("date")[["rq", "rd"]].apply(_corr)


def regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return per-row columns: qs_rho (date regime strength/sign) and qs_aligned."""
    # Only distance, date and quote_signal are read; the target column is never touched.
    ld = np.log(df["distance"].astype(float))
    qs = df["quote_signal"].astype(float)
    tmp = pd.DataFrame({"date": df["date"].values, "ld": ld.values, "qs": qs.values}, index=df.index)
    g = tmp.groupby("date")
    mean_ld, mean_qs = g["ld"].transform("mean"), g["qs"].transform("mean")
    cov = ((tmp["ld"] - mean_ld) * (tmp["qs"] - mean_qs)).groupby(tmp["date"]).transform("mean")
    var = ((tmp["ld"] - mean_ld) ** 2).groupby(tmp["date"]).transform("mean")
    slope = (cov / var.replace(0, np.nan)).fillna(0.0)
    resid = tmp["qs"] - (mean_qs + slope * (tmp["ld"] - mean_ld))
    rho = tmp["date"].map(per_date_spearman(df)).astype(float).fillna(0.0)
    aligned = (rho * resid).fillna(0.0)
    return pd.DataFrame({"qs_rho": rho.values, "qs_aligned": aligned.values}, index=df.index)
