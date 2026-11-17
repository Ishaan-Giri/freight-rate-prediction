"""Post-evaluation diagnosis of the Sep-Oct under-prediction (writes reports/drift_diagnosis.json).

Diagnostic only: nothing here feeds model selection.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from freight import config as C  # noqa: E402
from freight.data import load_train  # noqa: E402
from freight.evaluate import features_only, split  # noqa: E402
from freight.models import Pipeline  # noqa: E402


def main() -> None:
    cfg = yaml.safe_load(C.FINAL_CONFIG.read_text())["pipeline"]
    tr = load_train()
    out: dict = {"bias_by_test_month": []}
    for fold in C.DEV_FOLDS + [C.FINAL_FOLD]:
        a, b = split(tr, fold)
        p = Pipeline(cfg).fit(a).predict(features_only(b))
        lr = np.log(b[C.TARGET].values / p)
        for m in sorted(b["date"].dt.month.unique()):
            mk = (b["date"].dt.month == m).values & (np.abs(lr) < 0.4)
            out["bias_by_test_month"].append({"fold": fold.name, "month": int(m),
                                              "months_ahead": int(m - pd.Timestamp(fold.test_start).month + 1),
                                              "median_log_actual_over_pred": float(np.median(lr[mk]))})
    # month fixed effects controlling for distance, equipment, weight and market_index
    d = tr.copy()
    d["w"] = d["weight"].abs().fillna(d["weight"].abs().median())
    d["mi"] = d["market_index"].fillna(d.groupby("date")["market_index"].transform("mean"))
    X = pd.get_dummies(d["equipment"], drop_first=True).astype(float)
    X["ld"] = np.log(d["distance"]); X["ld2"] = X["ld"] ** 2
    X["lmi"] = np.log(d["mi"]); X["w"] = d["w"] / 1e4; X["c"] = 1.0
    X = X.join(pd.get_dummies(d["date"].dt.month, prefix="m", drop_first=True).astype(float))
    y = np.log(d[C.TARGET])
    beta = np.linalg.lstsq(X.values, y.values, rcond=None)[0]
    keep = np.abs(y - X.values @ beta) < 0.4
    beta = np.linalg.lstsq(X[keep].values, y[keep].values, rcond=None)[0]
    fe = {1: 0.0, **{int(c.split("_")[1]): float(b) for c, b in zip(X.columns, beta) if c.startswith("m_")}}
    out["month_fixed_effects_log"] = fe
    out["log_market_index_coef_with_month_fe"] = float(beta[list(X.columns).index("lmi")])
    # daily market_index elasticity by period: daily mean residual of a model WITHOUT market_index
    # regressed on log daily mean market_index
    Xn = X.drop(columns=["lmi"] + [c for c in X.columns if c.startswith("m_")])
    bn = np.linalg.lstsq(Xn[keep].values, y[keep].values, rcond=None)[0]
    d["r0"] = y - Xn.values @ bn
    day = d[keep].groupby("date").agg(r=("r0", "mean"), mi=("mi", "mean"))
    periods = {"Jan-Mar": (1, 3), "Apr-Jun": (4, 6), "Jul-Aug": (7, 8), "Sep-Oct": (9, 10)}
    out["daily_mi_elasticity_by_period"] = {}
    for name, (a, b) in periods.items():
        g = day[(day.index.month >= a) & (day.index.month <= b)]
        out["daily_mi_elasticity_by_period"][name] = float(np.polyfit(np.log(g["mi"]), g["r"], 1)[0])
    (C.REPORTS / "drift_diagnosis.json").write_text(json.dumps(out, indent=1))
    print(out["daily_mi_elasticity_by_period"])
    for r in out["bias_by_test_month"]:
        print(r)
    print({k: round(v, 4) for k, v in fe.items()})


if __name__ == "__main__":
    main()
