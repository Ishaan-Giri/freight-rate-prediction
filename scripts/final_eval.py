"""One-shot evaluation of the FROZEN pipeline on the held-out period (train Jan-Aug -> test Sep-Oct).

Sep-Oct was excluded from all fitting, tuning and selection, but its labels were examined during
EDA, so it is reported as a "held-out period examined during EDA", not untouched data.
Refuses to run unless configs/final.yaml is committed and unmodified.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from freight import config as C  # noqa: E402
from freight.data import city_lookup, load_train, load_validation  # noqa: E402
from freight.evaluate import evaluate_pipeline, features_only, metrics, split  # noqa: E402
from freight.features import linear_matrix  # noqa: E402
from freight.preprocess import Preprocessor  # noqa: E402


def assert_frozen() -> str:
    rel = str(C.FINAL_CONFIG.relative_to(C.ROOT))
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", rel], cwd=C.ROOT, capture_output=True)
    dirty = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel], cwd=C.ROOT)
    if tracked.returncode != 0 or dirty.returncode != 0:
        raise SystemExit(f"Refusing to run: {rel} must be committed and unmodified before final evaluation.")
    return subprocess.run(["git", "log", "-1", "--format=%H", "--", rel], cwd=C.ROOT,
                          capture_output=True, text=True).stdout.strip()


def clean_subset_mask(tr, te) -> np.ndarray:
    """SUPPLEMENTARY ONLY: test rows whose label is within exp(0.5) of a robust linear fit trained on
    the training fold. Uses test labels to define the subset, so it never drives any decision."""
    from sklearn.linear_model import HuberRegressor
    from sklearn.preprocessing import StandardScaler
    prep = Preprocessor("abs").fit(tr)
    a, b = prep.transform(tr), prep.transform(features_only(te))
    Xa, Xb = linear_matrix(a, {}), linear_matrix(b, {})
    sc = StandardScaler().fit(Xa)
    hub = HuberRegressor(epsilon=1.35, alpha=0.0, max_iter=1000).fit(sc.transform(Xa), np.log(tr[C.TARGET]))
    return np.abs(np.log(te[C.TARGET].values) - hub.predict(sc.transform(Xb))) <= 0.5


def main() -> None:
    commit = assert_frozen()
    frozen = yaml.safe_load(C.FINAL_CONFIG.read_text())
    dev = json.loads((C.REPORTS / "dev_results.json").read_text())
    references = {name: dev["results"][name]["config"] for name in frozen["reference_baselines"]}

    train = load_train()
    lookup = city_lookup(features_only(train), load_validation())
    tr, te = split(train, C.FINAL_FOLD)
    clean = clean_subset_mask(tr, te)
    y = te[C.TARGET].values

    out = {"frozen_config_commit": commit, "fold": vars(C.FINAL_FOLD),
           "status": "held-out period examined during EDA (excluded from fitting/tuning/selection)"}
    res = evaluate_pipeline(frozen["pipeline"], train, C.FINAL_FOLD, lookup, cold_seeds=C.COLD_START_SEEDS_FINAL,
                            extra_cold_k=C.COLD_START_STRESS_K)
    p = res.pop("_pred_full")
    res["clean_subset_supplementary"] = {"share_of_rows": float(clean.mean()), **metrics(y[clean], p[clean])}
    res["large_residual_rows_supplementary"] = metrics(y[~clean], p[~clean])
    out["selected"] = {"name": frozen["selected_candidate"], **res}
    out["references"] = {}
    for name, cfg in references.items():
        r = evaluate_pipeline(cfg, train, C.FINAL_FOLD, lookup)
        pr = r.pop("_pred_full")
        out["references"][name] = {"full": r["full"], "december_mode": r["december_mode"],
                                   "clean_subset_supplementary": metrics(y[clean], pr[clean])}
    (C.REPORTS / "final_eval.json").write_text(json.dumps(out, indent=1, default=str))

    s = out["selected"]
    print(f"Selected {s['name']} on Sep-Oct (all rows): MAE {s['full']['mae']:.2f}  RMSE {s['full']['rmse']:.1f}  "
          f"MAPE {s['full']['mape']:.4f}  median APE {s['full']['median_ape']:.4f}")
    print(f"December mode: MAE {s['december_mode']['mae']:.2f}")
    for m, v in s["by_month"].items():
        print(f"  month {m}: full MAE {v['full']['mae']:.2f}  Dec-mode MAE {v['december_mode']['mae']:.2f}")
    cs, st = s["cold_start"]["mean"], s["cold_start_stress"]["mean"]
    print(f"Cold start k={C.COLD_START_K}: exposure {cs['exposure']:.3f}  unseen MAE {cs['unseen']['mae']:.2f}  "
          f"seen MAE {cs['seen']['mae']:.2f}")
    print(f"Stress  k={C.COLD_START_STRESS_K}: exposure {st['exposure']:.3f}  unseen MAE {st['unseen']['mae']:.2f}  "
          f"seen MAE {st['seen']['mae']:.2f}")
    print(f"Clean subset (supplementary, {s['clean_subset_supplementary']['share_of_rows']:.3f} of rows): "
          f"MAE {s['clean_subset_supplementary']['mae']:.2f}")
    for name, r in out["references"].items():
        print(f"Reference {name}: MAE {r['full']['mae']:.2f}  Dec-mode {r['december_mode']['mae']:.2f}")


if __name__ == "__main__":
    main()
