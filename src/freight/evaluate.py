"""Time-ordered folds, cold-start simulation, December-mode scoring and metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .models import Pipeline
from .preprocess import december_mode


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    ape = np.abs(p - y) / y
    return {
        "n": int(len(y)),
        "mae": float(np.mean(np.abs(p - y))),
        "rmse": float(np.sqrt(np.mean((p - y) ** 2))),
        "mape": float(np.mean(ape)),
        "median_ape": float(np.median(ape)),
    }


def split(df: pd.DataFrame, fold: C.Fold) -> tuple[pd.DataFrame, pd.DataFrame]:
    tr = df[(df["date"] >= fold.train_start) & (df["date"] < fold.test_start)]
    te = df[(df["date"] >= fold.test_start) & (df["date"] <= fold.test_end)]
    assert tr["date"].max() < te["date"].min(), f"{fold.name}: training rows overlap test period"
    assert len(tr) and len(te)
    return tr.copy(), te.copy()


def features_only(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[C.TARGET])


def cold_start_cities(train: pd.DataFrame, k: int, seed: int) -> list[str]:
    cities = sorted(set(train["pickup"]) | set(train["delivery"]))
    return sorted(np.random.default_rng(seed).choice(cities, k, replace=False).tolist())


def touches(df: pd.DataFrame, cities: list[str]) -> pd.Series:
    return df["pickup"].isin(cities) | df["delivery"].isin(cities)


def evaluate_pipeline(cfg: dict, data: pd.DataFrame, fold: C.Fold, lookup: pd.DataFrame,
                      cold_seeds: list[int] | None = None, cold_k: int = C.COLD_START_K,
                      extra_cold_k: int | None = None) -> dict:
    """Fit on the fold's training window and score the test window in both modes."""
    tr, te = split(data, fold)
    y = te[C.TARGET].values
    feats = features_only(te)
    pipe = Pipeline(cfg).fit(tr)
    p_full = pipe.predict(feats)
    p_dec = pipe.predict(december_mode(feats, lookup, mi_reference=feats))

    res = {
        "fold": fold.name,
        "n_estimators": pipe.n_estimators_,
        "n_filtered": getattr(pipe, "n_filtered_", 0) if cfg["model"] != "blend" else getattr(pipe.parts_[0], "n_filtered_", 0),
        "full": metrics(y, p_full),
        "december_mode": metrics(y, p_dec),
        "by_month": {}, "by_equipment": {},
    }
    month = te["date"].dt.month.values
    for m in sorted(set(month)):
        mk = month == m
        res["by_month"][int(m)] = {"full": metrics(y[mk], p_full[mk]), "december_mode": metrics(y[mk], p_dec[mk])}
    for e in C.EQUIPMENT:
        mk = (te["equipment"] == e).values
        res["by_equipment"][e] = metrics(y[mk], p_full[mk])

    def _cold(k: int, seeds: list[int]) -> dict:
        runs = []
        for s in seeds:
            cities = cold_start_cities(tr, k, s)
            tr_cs = tr[~touches(tr, cities)]
            assert not touches(tr_cs, cities).any()
            p = Pipeline(cfg).fit(tr_cs, n_estimators=pipe.n_estimators_).predict(feats)
            unseen = touches(te, cities).values
            runs.append({
                "seed": s, "cities": cities, "exposure": float(unseen.mean()),
                "train_rows_removed": int(len(tr) - len(tr_cs)),
                "all": metrics(y, p), "unseen": metrics(y[unseen], p[unseen]),
                "seen": metrics(y[~unseen], p[~unseen]),
            })
        agg = {part: {k2: float(np.mean([r[part][k2] for r in runs])) for k2 in ["mae", "rmse", "mape", "median_ape"]}
               for part in ["all", "unseen", "seen"]}
        agg["exposure"] = float(np.mean([r["exposure"] for r in runs]))
        return {"k": k, "runs": runs, "mean": agg}

    if cold_seeds:
        res["cold_start"] = _cold(cold_k, cold_seeds)
        if extra_cold_k:
            res["cold_start_stress"] = _cold(extra_cold_k, cold_seeds)
    res["_pred_full"] = p_full
    return res
