"""Development round 2 (dev folds D1/D2 ONLY, same pre-declared rules as develop.py).

Why a second round: the first frozen pipeline under-predicted Sep-Oct by ~4-5% (original run).
Diagnosis showed an upward rate-level drift (~+0.65%/month beyond market_index) that tree models
cannot extrapolate; the same drift shows as a growing bias in the dev folds. Round 1 never offered
a trend candidate, so this round adds fold-fitted detrending and recency weighting and re-selects
on D1/D2 only. Because Sep-Oct prompted it, Sep-Oct is NOT independent evidence for this round.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import develop as D  # noqa: E402  (reuses run(), decide(), RESULTS and the same rules)
from freight import config as C  # noqa: E402


def set_trend(cfg: dict, on: bool) -> dict:
    c = copy.deepcopy(cfg)
    for part in [c["blend"]["lgbm"], c["blend"]["linear"]]:
        part["trend"] = on
    return c


def main() -> None:
    v1 = yaml.safe_load((C.CONFIGS / "final_v1.yaml").read_text())
    base = copy.deepcopy(v1["pipeline"])
    base["blend"]["lgbm"]["lgbm"]["n_estimators"] = "early_stop"  # re-derive per fold, as in round 1
    log: list[dict] = []

    print("== Round 2: fold-fitted trend")
    D.run("v1_blend_w0.75", base)
    cur = "v1_blend_w0.75"
    D.run("v1+trend", set_trend(base, True))
    if D.decide(cur, "v1+trend", False, log):
        cur = "v1+trend"
    # components, for the report
    for part in ["lgbm", "linear"]:
        for on in [False, True]:
            cfg = copy.deepcopy(base["blend"][part]); cfg["trend"] = on
            D.run(f"{part}_only{'+trend' if on else ''}", cfg)
    # recency-weighted training: tracks the recent level without extrapolating a slope
    print("== Round 2: recency weighting")
    best_hl = None
    for hl in [60, 120]:
        name = f"{cur}+halflife{hl}"
        cfg = copy.deepcopy(D.RESULTS[cur]["config"])
        for part in [cfg["blend"]["lgbm"], cfg["blend"]["linear"]]:
            part["half_life_days"] = hl
        D.run(name, cfg)
        if best_hl is None or D.RESULTS[name]["summary"]["primary_mae"] < D.RESULTS[best_hl]["summary"]["primary_mae"]:
            best_hl = name
    if D.decide(cur, best_hl, False, log):
        cur = best_hl
    # re-check blend weight under the chosen setting
    best = cur
    for w in [0.5, 1.0]:
        cfg = copy.deepcopy(D.RESULTS[cur]["config"]); cfg["blend"]["w_lgbm"] = w
        name = f"{cur}_w{w}"
        D.run(name, cfg)
        if D.RESULTS[name]["summary"]["primary_mae"] < D.RESULTS[best]["summary"]["primary_mae"]:
            best = name
    if best != cur and D.decide(cur, best, False, log):
        cur = best

    # bias diagnostic: median log(actual/pred) per test month for v1 vs selected
    final_cfg = copy.deepcopy(D.RESULTS[cur]["config"])
    n_list = [n for n in D.RESULTS[cur]["summary"]["n_estimators"] if n]
    final_cfg["blend"]["lgbm"]["lgbm"]["n_estimators"] = int(round(np.mean(n_list)))
    final = {"version": 2, "selected_candidate": cur,
             "why_round_2": __doc__.strip().splitlines()[2:8],
             "selection_rule": v1["selection_rule"], "pipeline": final_cfg,
             "reference_baselines": v1["reference_baselines"], "dev_summary": D.RESULTS[cur]["summary"]}
    (C.REPORTS / "dev_results_v2.json").write_text(json.dumps({"results": D.RESULTS, "decisions": log,
                                                              "selected": cur}, indent=1, default=str))
    C.FINAL_CONFIG.write_text(yaml.safe_dump(final, sort_keys=False))
    print("\nSELECTED (v2):", cur)


if __name__ == "__main__":
    main()
