"""Candidate models behind a single `Pipeline(cfg).fit(train).predict(features)` interface.

Every pipeline owns its fold-fitted Preprocessor, so nothing is learned outside `fit`.
Predictions are always returned in dollars.
"""
from __future__ import annotations

import copy

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor, LinearRegression
from sklearn.preprocessing import StandardScaler

from . import config as C
from .features import linear_matrix, tree_matrix
from .preprocess import Preprocessor

DEFAULT_FEATURES = {"dow": True, "coords": True, "qs_raw": False, "qs_aligned": False, "at_cap": False}
DEFAULT_LGBM = {
    "objective": "l2", "alpha": 0.9, "learning_rate": 0.03, "num_leaves": 31,
    "min_child_samples": 20, "feature_fraction": 0.9, "bagging_fraction": 0.8,
    "bagging_freq": 1, "lambda_l2": 0.0, "n_estimators": "early_stop", "max_rounds": 5000,
}


def default_config(model: str = "lgbm") -> dict:
    return {
        "model": model,
        "weight_negative": "abs",
        "features": dict(DEFAULT_FEATURES),
        "filter_threshold": None,        # |log residual| vs robust linear fit; None = no filtering
        "back_transform": "exp",         # "exp" (median-type) or "smear" (mean-type)
        "linear": {"huber": False},
        "lgbm": dict(DEFAULT_LGBM),
        "blend": None,                   # {"w_lgbm": float, "linear": cfg} for model == "blend"
    }


class LaneMedian:
    """B0: lane+equipment median rate-per-mile x distance with hierarchical fallbacks."""

    def fit(self, df: pd.DataFrame, y: np.ndarray) -> "LaneMedian":
        d = df.assign(rpm=y / df["distance"].values)
        self.bins_ = np.quantile(d["distance"], np.linspace(0, 1, 11)[1:-1])
        d["band"] = np.digitize(d["distance"], self.bins_)
        self.levels_ = []
        for keys in (["pickup", "delivery", "equipment"], ["pickup", "delivery"], ["equipment", "band"], ["band"]):
            g = d.groupby(keys)["rpm"].agg(["median", "size"])
            self.levels_.append((keys, g.loc[g["size"] >= 3, "median"]))
        self.global_ = float(d["rpm"].median())
        return self

    def predict_rpm(self, df: pd.DataFrame) -> np.ndarray:
        d = df.assign(band=np.digitize(df["distance"], self.bins_))
        out = pd.Series(np.nan, index=d.index)
        for keys, med in self.levels_:
            idx = pd.MultiIndex.from_frame(d[keys]) if len(keys) > 1 else d[keys[0]]
            vals = pd.Series(med.reindex(idx).values, index=d.index)
            out = out.fillna(vals)
        return out.fillna(self.global_).values


class Pipeline:
    def __init__(self, cfg: dict):
        self.cfg = copy.deepcopy(cfg)

    # ---- helpers -------------------------------------------------------------------------
    def _robust_filter_mask(self, tr: pd.DataFrame, ylog: np.ndarray) -> np.ndarray:
        thr = self.cfg["filter_threshold"]
        if thr is None:
            return np.ones(len(tr), dtype=bool)
        X = linear_matrix(tr, {})
        sc = StandardScaler().fit(X)
        hub = HuberRegressor(epsilon=1.35, alpha=0.0, max_iter=1000).fit(sc.transform(X), ylog)
        resid = ylog - hub.predict(sc.transform(X))
        return np.abs(resid) <= thr

    def _fit_linear(self, tr: pd.DataFrame, ylog: np.ndarray) -> None:
        X = linear_matrix(tr, self.cfg["features"])
        self.lin_cols_ = list(X.columns)
        self.scaler_ = StandardScaler().fit(X)
        Xs = self.scaler_.transform(X)
        if self.cfg["linear"]["huber"]:
            self.lin_ = HuberRegressor(epsilon=1.35, alpha=0.0, max_iter=1000).fit(Xs, ylog)
        else:
            self.lin_ = LinearRegression().fit(Xs, ylog)

    def _lgbm_params(self) -> dict:
        p = self.cfg["lgbm"]
        params = {
            "objective": {"l2": "regression", "l1": "regression_l1", "huber": "huber"}[p["objective"]],
            "learning_rate": p["learning_rate"], "num_leaves": p["num_leaves"],
            "min_child_samples": p["min_child_samples"], "feature_fraction": p["feature_fraction"],
            "bagging_fraction": p["bagging_fraction"], "bagging_freq": p["bagging_freq"],
            "lambda_l2": p["lambda_l2"], "seed": C.SEED, "deterministic": True,
            "force_row_wise": True, "num_threads": C.NUM_THREADS, "verbose": -1,
        }
        if p["objective"] == "huber":
            params["alpha"] = p["alpha"]
        return params

    def _fit_lgbm(self, tr: pd.DataFrame, ylog: np.ndarray, n_estimators=None) -> None:
        X = tree_matrix(tr, self.cfg["features"])
        self.tree_cols_ = list(X.columns)
        params = self._lgbm_params()
        n = n_estimators if n_estimators is not None else self.cfg["lgbm"]["n_estimators"]
        if n == "early_stop":
            # Inner time split: the last month of THIS training window is the early-stopping set.
            last_month = tr["date"].dt.to_period("M").max()
            inner = (tr["date"].dt.to_period("M") < last_month).values
            dtr = lgb.Dataset(X[inner], ylog[inner], categorical_feature=["equipment"], free_raw_data=False)
            dva = lgb.Dataset(X[~inner], ylog[~inner], categorical_feature=["equipment"], reference=dtr)
            booster = lgb.train(params, dtr, num_boost_round=self.cfg["lgbm"]["max_rounds"], valid_sets=[dva],
                                callbacks=[lgb.early_stopping(200, verbose=False)])
            n = int(booster.best_iteration)
        self.n_estimators_ = int(n)
        self.booster_ = lgb.train(params, lgb.Dataset(X, ylog, categorical_feature=["equipment"]),
                                  num_boost_round=self.n_estimators_)

    def _log_pred(self, df: pd.DataFrame) -> np.ndarray:
        m = self.cfg["model"]
        if m == "linear":
            X = linear_matrix(df, self.cfg["features"])[self.lin_cols_]
            return self.lin_.predict(self.scaler_.transform(X))
        if m == "lgbm":
            return self.booster_.predict(tree_matrix(df, self.cfg["features"])[self.tree_cols_])
        raise ValueError(m)

    # ---- public API ----------------------------------------------------------------------
    def fit(self, train: pd.DataFrame, n_estimators=None) -> "Pipeline":
        m = self.cfg["model"]
        if m == "blend":
            b = self.cfg["blend"]
            self.parts_ = [Pipeline(b["lgbm"]).fit(train, n_estimators), Pipeline(b["linear"]).fit(train)]
            self.n_estimators_ = self.parts_[0].n_estimators_
            return self
        self.prep_ = Preprocessor(self.cfg["weight_negative"]).fit(train)
        tr = self.prep_.transform(train)
        y = train[C.TARGET].values.astype(float)
        if m == "lane":
            self.lane_ = LaneMedian().fit(tr, y)
            self.n_estimators_ = None
            return self
        ylog = np.log(y)
        keep = self._robust_filter_mask(tr, ylog)
        self.n_filtered_ = int((~keep).sum())
        tr, ylog = tr[keep], ylog[keep]
        if m == "linear":
            self._fit_linear(tr, ylog)
            self.n_estimators_ = None
        elif m == "lgbm":
            self._fit_lgbm(tr, ylog, n_estimators)
        else:
            raise ValueError(m)
        resid = ylog - self._log_pred(tr)
        self.smear_ = float(np.mean(np.exp(resid))) if self.cfg["back_transform"] == "smear" else 1.0
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        assert C.TARGET not in features.columns, "predict() must receive feature columns only"
        m = self.cfg["model"]
        if m == "blend":
            w = self.cfg["blend"]["w_lgbm"]
            p1, p2 = (p.predict(features) for p in self.parts_)
            return np.exp(w * np.log(p1) + (1 - w) * np.log(p2))
        df = self.prep_.transform(features)
        if m == "lane":
            return self.lane_.predict_rpm(df) * df["distance"].values
        return np.exp(self._log_pred(df)) * self.smear_
