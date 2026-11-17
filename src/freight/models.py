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
from sklearn.model_selection import GroupKFold
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
        "trend": False,                  # fold-fitted linear time trend removed before fitting, added back after
        "half_life_days": None,          # recency weighting: weight = 0.5 ** (age_days / half_life)
    }


TREND_ORIGIN = pd.Timestamp("2025-01-01")


def months_since_origin(df: pd.DataFrame) -> np.ndarray:
    return ((df["date"] - TREND_ORIGIN).dt.days / 30.4).values


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
        """Flag rows whose OUT-OF-FOLD residual from a robust linear fit exceeds the threshold.

        5 folds grouped by date (a date's loads share market_index). The robust fit is trimmed OLS:
        fit, drop |resid| > threshold, refit. Only the rows passed in are used.
        """
        thr = self.cfg["filter_threshold"]
        if thr is None:
            return np.ones(len(tr), dtype=bool)
        return np.abs(self._oof_residuals(tr, ylog, thr)) <= thr

    @staticmethod
    def _oof_residuals(tr: pd.DataFrame, ylog: np.ndarray, thr: float) -> np.ndarray:
        X = linear_matrix(tr, {}).assign(const=1.0).values
        resid = np.zeros(len(tr))
        for fit_idx, oof_idx in GroupKFold(n_splits=5).split(X, ylog, groups=tr["date"].values):
            beta = np.linalg.lstsq(X[fit_idx], ylog[fit_idx], rcond=None)[0]
            ok = np.abs(ylog[fit_idx] - X[fit_idx] @ beta) <= thr
            beta = np.linalg.lstsq(X[fit_idx][ok], ylog[fit_idx][ok], rcond=None)[0]
            resid[oof_idx] = ylog[oof_idx] - X[oof_idx] @ beta
        return resid

    def _prepare(self, train: pd.DataFrame):
        """Fit preprocessing, outlier filter, trend and recency weights on exactly `train`."""
        prep = Preprocessor(self.cfg["weight_negative"]).fit(train)
        tr = prep.transform(train)
        ylog = np.log(train[C.TARGET].values.astype(float))
        keep = self._robust_filter_mask(tr, ylog)
        tr, ylog = tr[keep], ylog[keep]
        # Optional: a robust linear time trend removed before fitting, added back in predict().
        trend = 0.0
        if self.cfg.get("trend"):
            X = linear_matrix(tr, {}).assign(months=months_since_origin(tr))
            sc = StandardScaler().fit(X)
            hub = HuberRegressor(epsilon=1.35, alpha=0.0, max_iter=1000).fit(sc.transform(X), ylog)
            j = list(X.columns).index("months")
            trend = float(hub.coef_[j] / sc.scale_[j])
            ylog = ylog - trend * months_since_origin(tr)
        sw = None
        if self.cfg.get("half_life_days"):
            age = (tr["date"].max() - tr["date"]).dt.days.values
            sw = 0.5 ** (age / self.cfg["half_life_days"])
        return prep, tr, ylog, sw, trend, int((~keep).sum())

    def _early_stop_rounds(self, train: pd.DataFrame) -> int:
        """Choose the tree count on the last month of `train`, without that month leaking.

        Raw rows are split FIRST. Preprocessing, filtering and trend are fitted on the inner
        months only; every early-stopping row is kept (none filtered).
        """
        month = train["date"].dt.to_period("M")
        inner_raw, es_raw = train[month < month.max()], train[month == month.max()]
        prep, tr, ylog, sw, trend, _ = self._prepare(inner_raw)
        X = tree_matrix(tr, self.cfg["features"])
        es = prep.transform(es_raw.drop(columns=[C.TARGET]))
        X_es = tree_matrix(es, self.cfg["features"])[list(X.columns)]
        y_es = np.log(es_raw[C.TARGET].values.astype(float)) - trend * months_since_origin(es)
        self.es_rows_ = int(len(es_raw))
        dtr = lgb.Dataset(X, ylog, weight=sw, categorical_feature=["equipment"], free_raw_data=False)
        dva = lgb.Dataset(X_es, y_es, categorical_feature=["equipment"], reference=dtr)
        booster = lgb.train(self._lgbm_params(), dtr, num_boost_round=self.cfg["lgbm"]["max_rounds"],
                            valid_sets=[dva], callbacks=[lgb.early_stopping(200, verbose=False)])
        return int(booster.best_iteration)

    def _fit_linear(self, tr: pd.DataFrame, ylog: np.ndarray, sw=None) -> None:
        X = linear_matrix(tr, self.cfg["features"])
        self.lin_cols_ = list(X.columns)
        self.scaler_ = StandardScaler().fit(X)
        Xs = self.scaler_.transform(X)
        if self.cfg["linear"]["huber"]:
            self.lin_ = HuberRegressor(epsilon=1.35, alpha=0.0, max_iter=1000).fit(Xs, ylog, sample_weight=sw)
        else:
            self.lin_ = LinearRegression().fit(Xs, ylog, sample_weight=sw)

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

    def _fit_lgbm(self, tr: pd.DataFrame, ylog: np.ndarray, n: int, sw=None) -> None:
        X = tree_matrix(tr, self.cfg["features"])
        self.tree_cols_ = list(X.columns)
        self.n_estimators_ = int(n)
        self.booster_ = lgb.train(self._lgbm_params(),
                                  lgb.Dataset(X, ylog, weight=sw, categorical_feature=["equipment"]),
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
        if m == "lane":
            self.prep_ = Preprocessor(self.cfg["weight_negative"]).fit(train)
            self.lane_ = LaneMedian().fit(self.prep_.transform(train), train[C.TARGET].values.astype(float))
            self.n_estimators_ = None
            return self
        n = None
        if m == "lgbm":
            n = n_estimators if n_estimators is not None else self.cfg["lgbm"]["n_estimators"]
            if n == "early_stop":
                n = self._early_stop_rounds(train)
        self.prep_, tr, ylog, sw, self.trend_, self.n_filtered_ = self._prepare(train)
        if m == "linear":
            self._fit_linear(tr, ylog, sw)
            self.n_estimators_ = None
        elif m == "lgbm":
            self._fit_lgbm(tr, ylog, n, sw)
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
        return np.exp(self._log_pred(df) + self.trend_ * months_since_origin(df)) * self.smear_
