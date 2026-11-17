"""Model development on the development folds ONLY (D1: Jan-Apr -> May-Jun, D2: Jan-Jun -> Jul-Aug).

Sep-Oct is never touched here. Selection follows rules declared before running:

  PRIMARY  : all-row dollar MAE, full-feature mode, mean over D1 and D2.
  ADOPT    : a more complex option is adopted only if it lowers PRIMARY by > 0.3% and does not
             worsen any secondary check by > 2%: December-mode MAE, August MAE (the quote_signal
             regime analogue of Nov-Dec), cold-start unseen-city MAE, worst-fold MAE.
  SIMPLIFY : removing a feature / option is adopted if PRIMARY worsens by <= 0.3% and no secondary
             check worsens by > 2% (simpler wins ties).

Writes reports/dev_results.json, reports/dev_summary.md and configs/final.yaml (frozen after review).
"""
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from freight import config as C  # noqa: E402
from freight.data import city_lookup, load_train, load_validation  # noqa: E402
from freight.evaluate import evaluate_pipeline, features_only  # noqa: E402
from freight.models import default_config  # noqa: E402

ADOPT_GAIN = 0.003
SECONDARY_TOL = 0.02

train = load_train()
lookup = city_lookup(features_only(train), load_validation())
RESULTS: dict[str, dict] = {}


def summarize(folds: list[dict]) -> dict:
    s = {
        "primary_mae": float(np.mean([f["full"]["mae"] for f in folds])),
        "december_mode_mae": float(np.mean([f["december_mode"]["mae"] for f in folds])),
        "worst_fold_mae": float(max(f["full"]["mae"] for f in folds)),
        "aug_mae": float(folds[1]["by_month"][8]["full"]["mae"]),
        "unseen_mae": float(np.mean([f["cold_start"]["mean"]["unseen"]["mae"] for f in folds])),
        "seen_mae": float(np.mean([f["cold_start"]["mean"]["seen"]["mae"] for f in folds])),
        "exposure": float(np.mean([f["cold_start"]["mean"]["exposure"] for f in folds])),
        "rmse": float(np.mean([f["full"]["rmse"] for f in folds])),
        "mape": float(np.mean([f["full"]["mape"] for f in folds])),
        "median_ape": float(np.mean([f["full"]["median_ape"] for f in folds])),
        "n_estimators": [f["n_estimators"] for f in folds],
        "n_filtered": [f["n_filtered"] for f in folds],
    }
    for fd in folds:
        s[f"{fd['fold']}_mae"] = fd["full"]["mae"]
    return s


def run(name: str, cfg: dict) -> dict:
    if name in RESULTS:
        return RESULTS[name]["summary"]
    t = time.time()
    folds = [evaluate_pipeline(cfg, train, f, lookup, cold_seeds=C.COLD_START_SEEDS_DEV) for f in C.DEV_FOLDS]
    for f in folds:
        f.pop("_pred_full")
    s = summarize(folds)
    RESULTS[name] = {"config": copy.deepcopy(cfg), "summary": s, "folds": folds}
    print(f"{name:<44} primary {s['primary_mae']:8.2f}  dec {s['december_mode_mae']:8.2f}  "
          f"aug {s['aug_mae']:8.2f}  unseen {s['unseen_mae']:8.2f}  mape {s['mape']:.4f}  "
          f"n_est {s['n_estimators']}  ({time.time() - t:.0f}s)", flush=True)
    return s


def secondary_ok(new: dict, base: dict) -> bool:
    keys = ["december_mode_mae", "aug_mae", "unseen_mae", "worst_fold_mae"]
    return all(new[k] <= base[k] * (1 + SECONDARY_TOL) for k in keys)


def decide(base_name: str, cand_name: str, simplify: bool, log: list) -> bool:
    b, c = RESULTS[base_name]["summary"], RESULTS[cand_name]["summary"]
    rel = (c["primary_mae"] - b["primary_mae"]) / b["primary_mae"]
    if simplify:
        ok = rel <= ADOPT_GAIN and secondary_ok(c, b)
    else:
        ok = rel < -ADOPT_GAIN and secondary_ok(c, b)
    log.append({"base": base_name, "candidate": cand_name, "type": "simplify" if simplify else "add",
                "rel_change_primary": rel, "secondary_ok": secondary_ok(c, b), "adopted": bool(ok)})
    print(f"   -> {'ADOPT' if ok else 'reject'} {cand_name} vs {base_name}: primary {rel:+.2%}, "
          f"secondary_ok={secondary_ok(c, b)}", flush=True)
    return ok


def with_(cfg: dict, **changes) -> dict:
    c = copy.deepcopy(cfg)
    for path, v in changes.items():
        d = c
        keys = path.split("__")
        for k in keys[:-1]:
            d = d[k]
        d[keys[-1]] = v
    return c


def main() -> None:
    log: list[dict] = []

    print("== Stage 1: baselines")
    run("B0_lane_median", default_config("lane"))
    lin = default_config("linear")
    run("B1_loglinear_ols", lin)
    run("B1_loglinear_huber", with_(lin, linear__huber=True))
    for thr in [0.5, 0.3]:
        run(f"B1_loglinear_ols_filter{thr}", with_(lin, filter_threshold=thr))
    lin_best = min([k for k in RESULTS if k.startswith("B1")], key=lambda k: RESULTS[k]["summary"]["primary_mae"])
    print("   best linear:", lin_best)

    print("== Stage 2: LightGBM objective x training-data variant")
    base = default_config("lgbm")
    for obj in ["l2", "l1", "huber"]:
        for thr in [None, 0.5, 0.3]:
            cfg = with_(base, lgbm__objective=obj, lgbm__alpha=0.1, filter_threshold=thr)
            run(f"M1_{obj}_filter{thr}", cfg)
    cur = min([k for k in RESULTS if k.startswith("M1_")], key=lambda k: RESULTS[k]["summary"]["primary_mae"])
    print("   best LightGBM variant:", cur)

    print("== Stage 3: feature ablations (quote_signal, DOW, coordinates, weight handling)")
    steps = [
        ("qs_raw", dict(features__qs_raw=True), False),
        ("qs_aligned", dict(features__qs_aligned=True), False),
        ("no_dow", dict(features__dow=False), True),
        ("no_coords", dict(features__coords=False), True),
        ("weight_neg_nan", dict(weight_negative="nan"), False),
        ("at_cap", dict(features__at_cap=True), False),
    ]
    for tag, ch, simplify in steps:
        cand = f"{cur}+{tag}"
        run(cand, with_(RESULTS[cur]["config"], **ch))
        if decide(cur, cand, simplify, log):
            cur = cand

    print("== Stage 4: hyper-parameters (small grid)")
    best = cur
    for nl in [15, 63, 127]:
        for mcs in [20, 100]:
            if nl == RESULTS[cur]["config"]["lgbm"]["num_leaves"] and mcs == RESULTS[cur]["config"]["lgbm"]["min_child_samples"]:
                continue
            cand = f"{cur}+nl{nl}_mcs{mcs}"
            run(cand, with_(RESULTS[cur]["config"], lgbm__num_leaves=nl, lgbm__min_child_samples=mcs))
            if RESULTS[cand]["summary"]["primary_mae"] < RESULTS[best]["summary"]["primary_mae"]:
                best = cand
    if best != cur and decide(cur, best, False, log):
        cur = best
    cand = f"{cur}+ff0.7"
    run(cand, with_(RESULTS[cur]["config"], lgbm__feature_fraction=0.7))
    if decide(cur, cand, False, log):
        cur = cand

    print("== Stage 5: back-transform and blending")
    cand = f"{cur}+smear"
    run(cand, with_(RESULTS[cur]["config"], back_transform="smear"))
    if decide(cur, cand, False, log):
        cur = cand
    lin_cfg = copy.deepcopy(RESULTS[lin_best]["config"])
    lin_cfg["features"].update({k: RESULTS[cur]["config"]["features"][k] for k in ["qs_raw", "qs_aligned", "at_cap"]})
    lin_cfg["weight_negative"] = RESULTS[cur]["config"]["weight_negative"]
    best_blend = None
    for w in [0.25, 0.5, 0.75]:
        name = f"blend_w{w}"
        cfg = default_config("blend")
        cfg["blend"] = {"w_lgbm": w, "lgbm": RESULTS[cur]["config"], "linear": lin_cfg}
        run(name, cfg)
        if best_blend is None or RESULTS[name]["summary"]["primary_mae"] < RESULTS[best_blend]["summary"]["primary_mae"]:
            best_blend = name
    if decide(cur, best_blend, False, log):
        cur = best_blend

    # Fixed tree count for final evaluation / production: mean of the dev folds' early-stopped counts.
    final_cfg = copy.deepcopy(RESULTS[cur]["config"])
    target = final_cfg["blend"]["lgbm"] if final_cfg["model"] == "blend" else final_cfg
    if target["model"] == "lgbm":
        n_list = [n for n in RESULTS[cur]["summary"]["n_estimators"] if n]
        target["lgbm"]["n_estimators"] = int(round(np.mean(n_list)))
    final = {"selected_candidate": cur, "selection_rule": __doc__.strip().splitlines()[2:9], "pipeline": final_cfg,
             "reference_baselines": ["B0_lane_median", lin_best], "dev_summary": RESULTS[cur]["summary"]}

    C.REPORTS.mkdir(exist_ok=True)
    C.CONFIGS.mkdir(exist_ok=True)
    (C.REPORTS / "dev_results.json").write_text(json.dumps({"results": RESULTS, "decisions": log, "selected": cur},
                                                           indent=1, default=str))
    C.FINAL_CONFIG.write_text(yaml.safe_dump(final, sort_keys=False))
    lines = ["| candidate | primary MAE | Dec-mode MAE | Aug MAE | unseen MAE | worst fold | MAPE | RMSE |",
             "|---|---|---|---|---|---|---|---|"]
    for k, v in RESULTS.items():
        s = v["summary"]
        lines.append(f"| {k} | {s['primary_mae']:.2f} | {s['december_mode_mae']:.2f} | {s['aug_mae']:.2f} | "
                     f"{s['unseen_mae']:.2f} | {s['worst_fold_mae']:.2f} | {s['mape']:.4f} | {s['rmse']:.1f} |")
    (C.REPORTS / "dev_summary.md").write_text("\n".join(lines) + f"\n\nSelected: **{cur}**\n")
    print("\nSELECTED:", cur)


if __name__ == "__main__":
    main()
