import numpy as np
import pandas as pd
import pytest

from freight import config as C
from freight.data import city_lookup, load_train, load_validation
from freight.evaluate import cold_start_cities, features_only, split, touches
from freight.preprocess import Preprocessor, december_mode


@pytest.fixture(scope="module")
def train():
    return load_train()


@pytest.fixture(scope="module")
def validation():
    return load_validation()


def test_december_mode_rejects_target(train, validation):
    lookup = city_lookup(features_only(train), validation)
    with pytest.raises(AssertionError):
        december_mode(train.head(50), lookup, mi_reference=features_only(train))
    with pytest.raises(AssertionError):
        december_mode(features_only(train.head(50)), lookup, mi_reference=train)


def test_december_mode_rebuilds_only_from_features(train, validation):
    lookup = city_lookup(features_only(train), validation)
    feats = features_only(train[train["date"] >= "2025-10-01"])
    rebuilt = december_mode(feats, lookup, mi_reference=feats)
    # coordinates rebuilt from the lookup equal the supplied ones (one point per city)
    assert np.allclose(rebuilt["pickup_lat"], feats["pickup_lat"])
    assert rebuilt["quote_signal"].isna().all()
    daily = feats.groupby("date")["market_index"].mean()
    assert np.allclose(rebuilt["market_index"], feats["date"].map(daily))


def test_unknown_city_raises(train, validation):
    lookup = city_lookup(features_only(train), validation).drop(index="Chicago")
    with pytest.raises(KeyError):
        december_mode(features_only(validation.head(200)), lookup, mi_reference=validation)


def test_time_split_has_no_future_rows(train):
    for fold in C.DEV_FOLDS + [C.FINAL_FOLD]:
        tr, te = split(train, fold)
        assert tr["date"].max() < pd.Timestamp(fold.test_start) <= te["date"].min()


def test_cold_start_removes_both_endpoints(train):
    tr, _ = split(train, C.DEV_FOLDS[0])
    cities = cold_start_cities(tr, C.COLD_START_K, 0)
    assert not touches(tr[~touches(tr, cities)], cities).any()


def test_preprocessor_uses_training_statistics_only(train):
    tr, te = split(train, C.DEV_FOLDS[0])
    prep = Preprocessor("abs").fit(tr)
    med_before = dict(prep.weight_median_)
    prep.transform(te)
    assert prep.weight_median_ == med_before
    out = prep.transform(te)
    assert out["weight_clean"].notna().all() and (out["weight_clean"] > 0).all()
    assert out["mi"].notna().all()


# ---- early stopping / outlier filter hygiene (regression tests for code-review finding) ----
from freight.models import Pipeline, default_config  # noqa: E402


def _small_cfg():
    cfg = default_config("lgbm")
    cfg["filter_threshold"] = 0.5
    cfg["lgbm"].update({"max_rounds": 30, "objective": "huber", "alpha": 0.1})
    return cfg


def test_early_stopping_month_is_unseen_and_unfiltered(train):
    window = train[train["date"] < "2025-04-01"]
    pipe = Pipeline(_small_cfg())
    seen = []
    original = pipe._prepare
    pipe._prepare = lambda df: (seen.append(df["date"].max()), original(df))[1]
    pipe.fit(window)
    last_month_rows = int((window["date"] >= "2025-03-01").sum())
    assert pipe.es_rows_ == last_month_rows          # every early-stopping row kept
    assert seen[0] < pd.Timestamp("2025-03-01")       # inner prep/filter never saw the ES month
    assert seen[1] == window["date"].max()            # final refit uses the full window


def test_filter_uses_out_of_fold_residuals(train):
    window = train[train["date"] < "2025-03-01"].copy()
    tr = Preprocessor("abs").fit(window).transform(window)
    ylog = np.log(window[C.TARGET].values)
    r1 = Pipeline._oof_residuals(tr, ylog, 0.5)
    i, delta = 123, 0.3
    bumped = ylog.copy()
    bumped[i] += delta
    r2 = Pipeline._oof_residuals(tr, bumped, 0.5)
    # Out-of-fold: the row's own fitted value ignores its label, so its residual moves by exactly delta
    # (an in-sample fit would move it by delta * (1 - leverage)).
    assert abs((r2[i] - r1[i]) - delta) < 1e-10
    # Rows on the same date share the fold, so their fits never see the changed label.
    same = (tr["date"] == tr["date"].iloc[i]).values
    same[i] = False
    assert np.allclose(r1[same], r2[same], atol=1e-12)
