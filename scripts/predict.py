"""Production fit of the FROZEN pipeline on all labelled data (Jan-Oct), then:

1. validation_predictions.csv  (12,000 rows: load_id,predicted_rate) - full supplied features
2. data/december_chart_inputs.csv  - predicted_rate filled; missing columns rebuilt with the
   December-mode procedure (city coordinate lookup, per-date validation market_index, no quote_signal)
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
from freight.data import (check_december_raw, city_lookup, load_december, load_template,  # noqa: E402
                          load_train, load_validation)
from freight.evaluate import features_only  # noqa: E402
from freight.models import Pipeline  # noqa: E402
from freight.preprocess import december_mode  # noqa: E402


def main() -> None:
    frozen = yaml.safe_load(C.FINAL_CONFIG.read_text())
    train, val, dec, tmpl = load_train(), load_validation(), load_december(), load_template()
    check_december_raw(dec)
    pipe = Pipeline(frozen["pipeline"]).fit(train)

    # ---- 1. validation predictions (full supplied features)
    pred = pipe.predict(val)
    assert np.isfinite(pred).all() and (pred > 0).all()
    out = tmpl[["load_id"]].merge(pd.DataFrame({"load_id": val["load_id"], "predicted_rate": pred}),
                                  on="load_id", how="left", validate="one_to_one")
    assert len(out) == 12_000 and out["predicted_rate"].notna().all()
    assert out["load_id"].tolist() == tmpl["load_id"].tolist()
    out["predicted_rate"] = out["predicted_rate"].round(2)
    out.to_csv(C.PREDICTIONS_FILE, index=False)

    # ---- 2. December chart (December-mode reconstruction from feature data only)
    lookup = city_lookup(features_only(train), val)
    dec_feats = december_mode(dec.drop(columns=["predicted_rate"]), lookup, mi_reference=val)
    dec_pred = pipe.predict(dec_feats)
    assert np.isfinite(dec_pred).all() and (dec_pred > 0).all()
    filled = dec.copy()
    filled["predicted_rate"] = np.round(dec_pred, 2)
    filled["date"] = filled["date"].dt.strftime("%Y-%m-%d")
    filled.to_csv(C.DECEMBER_FILE, index=False)

    # ---- diagnostics for the report: what moves the December line?
    flat = dec_feats.copy()
    flat["market_index"] = dec_feats["market_index"].mean()
    p_flat_mi = pipe.predict(flat)
    sep_oct = train[train["date"] >= "2025-09-01"][C.TARGET]
    hist = train[(train.pickup == "Lexington") & (train.delivery == "Fort Wayne") & (train.equipment == "Dry Van")]
    summary = {
        "n_estimators": pipe.n_estimators_,
        "validation_pred_quantiles": out["predicted_rate"].quantile([0, .01, .25, .5, .75, .99, 1]).round(2).to_dict(),
        "sep_oct_actual_quantiles": sep_oct.quantile([0, .01, .25, .5, .75, .99, 1]).round(2).to_dict(),
        "validation_pred_mean": float(out["predicted_rate"].mean()),
        "sep_oct_actual_mean": float(sep_oct.mean()),
        "december": [
            {"date": d.strftime("%Y-%m-%d"), "dow": d.day_name()[:3], "market_index_used": float(mi),
             "predicted_rate": float(p), "predicted_rate_mi_held_constant": float(pf)}
            for d, mi, p, pf in zip(dec_feats["date"], dec_feats["market_index"], dec_pred, p_flat_mi)
        ],
        "december_range": [float(dec_pred.min()), float(dec_pred.max())],
        "december_range_mi_constant": [float(p_flat_mi.min()), float(p_flat_mi.max())],
        "corr_pred_vs_market_index": float(np.corrcoef(dec_pred, dec_feats["market_index"])[0, 1]),
        "historical_lex_fw_dry_van": hist[["date", "distance", "weight", "market_index", C.TARGET]]
            .assign(date=lambda x: x["date"].dt.strftime("%Y-%m-%d")).to_dict("records"),
    }
    (C.REPORTS / "predictions_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(f"Wrote {C.PREDICTIONS_FILE.name}: {len(out)} rows, mean {summary['validation_pred_mean']:.2f} "
          f"(Sep-Oct actual mean {summary['sep_oct_actual_mean']:.2f})")
    print(f"December: {summary['december_range'][0]:.2f} - {summary['december_range'][1]:.2f}; "
          f"with MI held constant: {summary['december_range_mi_constant'][0]:.2f} - "
          f"{summary['december_range_mi_constant'][1]:.2f}; corr(pred, MI) {summary['corr_pred_vs_market_index']:.3f}")


if __name__ == "__main__":
    main()
