"""Exploratory analysis: data-quality counts, label-free regime diagnostics and report figures.

Writes reports/eda.json and figures under reports/figures/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from freight import config as C  # noqa: E402
from freight.data import check_raw, load_train, load_validation  # noqa: E402

INK, ACCENT, GREY, WARN = "#1F2A30", "#064A56", "#9DAFB3", "#C2410C"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": GREY, "axes.titleweight": "bold", "axes.titlesize": 10})


def haversine(a, b, c, d):
    a, b, c, d = map(np.radians, [a, b, c, d])
    return 3958.8 * 2 * np.arcsin(np.sqrt(np.sin((c - a) / 2) ** 2 + np.cos(a) * np.cos(c) * np.sin((d - b) / 2) ** 2))


def main() -> None:
    C.FIGURES.mkdir(parents=True, exist_ok=True)
    tr, va = load_train(), load_validation()
    out: dict = {"train": check_raw(tr, "train", True), "validation": check_raw(va, "validation", False)}
    tr["rpm"] = tr[C.TARGET] / tr["distance"]
    tr["month"] = tr["date"].dt.month
    al = pd.concat([tr.assign(src="train"), va.assign(src="validation")], ignore_index=True)
    al["month"] = al["date"].dt.month

    # --- target / label quality: residuals vs a simple robust log-linear fit (diagnostic only)
    d = tr.copy()
    d["w"] = d["weight"].abs().fillna(d["weight"].abs().median())
    d["mi"] = d["market_index"].fillna(d.groupby("date")["market_index"].transform("mean"))
    X = pd.get_dummies(d["equipment"], drop_first=True).astype(float)
    X["ld"], X["lmi"], X["w"], X["c"] = np.log(d["distance"]), np.log(d["mi"]), d["w"] / 1e4, 1.0
    yl = np.log(d[C.TARGET])
    beta = np.linalg.lstsq(X.values, yl.values, rcond=None)[0]
    r = yl - X.values @ beta
    out["target"] = {
        "rate_quantiles": d[C.TARGET].quantile([0, .01, .5, .99, 1]).round(2).to_dict(),
        "loglinear_r2": float(1 - r.var() / yl.var()),
        "large_residual_rows_abs_gt_0.5": int((r.abs() > 0.5).sum()),
        "large_residual_share": float((r.abs() > 0.5).mean()),
        "large_residual_by_month": r.abs().gt(0.5).groupby(d["month"]).sum().astype(int).to_dict(),
        "ratio_bins_of_large_residuals": np.exp(r[r.abs() > 0.5]).round(2).value_counts(
            bins=[0, .3, .7, 1.5, 3, 7, 100]).sort_index().astype(int).rename(lambda i: str(i)).to_dict(),
        "clean_residual_std": float(r[r.abs() < 0.5].std()),
    }

    # --- market_index structure
    g = al.groupby("date")["market_index"]
    out["market_index"] = {
        "within_day_std": float(g.std().mean()), "across_day_std_of_daily_mean": float(g.mean().std()),
        "monthly_mean": {str(k): float(v) for k, v in al.groupby(al["date"].dt.to_period("M"))["market_index"].mean().items()},
        "daily_mean_by_weekday": al.groupby(al["date"].dt.day_name())["market_index"].mean().round(4).to_dict(),
    }

    # --- quote_signal regimes (label-free: vs distance; label-based only for train months)
    qs_dist = al.groupby("month")[["quote_signal", "distance"]].apply(
        lambda x: x["quote_signal"].corr(x["distance"], method="spearman"))
    qs_rpm = tr.groupby("month")[["quote_signal", "rpm"]].apply(
        lambda x: x["quote_signal"].corr(x["rpm"], method="spearman"))
    out["quote_signal"] = {"spearman_vs_distance_by_month": qs_dist.round(3).to_dict(),
                           "spearman_vs_rate_per_mile_by_month_train": qs_rpm.round(3).to_dict()}

    # --- geography / cold start
    al["ratio"] = al["distance"] / haversine(al.pickup_lat, al.pickup_lon, al.delivery_lat, al.delivery_lon)
    lane = al.groupby(["pickup", "delivery"])["ratio"].median().reset_index()
    per_city = pd.concat([lane.rename(columns={"pickup": "c"})[["c", "ratio"]],
                          lane.rename(columns={"delivery": "c"})[["c", "ratio"]]]).groupby("c")["ratio"].median()
    new = sorted((set(va.pickup) | set(va.delivery)) - (set(tr.pickup) | set(tr.delivery)))
    out["geography"] = {
        "lane_distance_to_haversine_ratio_quantiles": lane["ratio"].quantile([.01, .5, .99]).round(3).to_dict(),
        "per_city_median_ratio_min_max": [float(per_city.min()), float(per_city.max())],
        "new_orleans_median_ratio": float(per_city["New Orleans"]),
        "most_extreme_lanes": lane.nlargest(4, "ratio").round(2).to_dict("records"),
        "validation_only_cities": new,
        "validation_share_touching_new_cities": float((va.pickup.isin(new) | va.delivery.isin(new)).mean()),
    }
    lf = tr[(tr.pickup == "Lexington") & (tr.delivery == "Fort Wayne") & (tr.equipment == "Dry Van")]
    out["lexington_fort_wayne_dry_van"] = {"n": int(len(lf)), "min": float(lf[C.TARGET].min()),
                                           "median": float(lf[C.TARGET].median()), "max": float(lf[C.TARGET].max())}
    (C.REPORTS / "eda.json").write_text(json.dumps(out, indent=1, default=str))

    # ---------------- figures ----------------
    # 1. quote_signal regimes
    fig, ax = plt.subplots(figsize=(7.2, 2.8), dpi=160)
    months = list(range(1, 13))
    vals = [qs_dist.get(m, np.nan) for m in months]
    cols = [ACCENT if m <= 10 else WARN for m in months]
    ax.bar(months, vals, color=cols, label="vs distance (no prices needed)")
    ax.plot(list(qs_rpm.index), qs_rpm.values, "o", color=INK, ms=4, label="vs price per mile (training months)")
    ax.axhline(0, color=GREY, lw=0.8)
    ax.set_xticks(months, ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_ylabel("correlation (-1 to +1)")
    ax.set_title("quote_signal flips direction by month; no pattern in Aug, Nov, Dec", loc="left")
    ax.set_ylim(-1.6, 1.15)
    ax.legend(frameon=False, loc="lower center", ncol=2, fontsize=7.5)
    ax.text(11.5, 0.75, "months we\npredict", ha="center", fontsize=7.5, color=WARN)
    fig.tight_layout(); fig.savefig(C.FIGURES / "quote_signal_regimes.png"); plt.close(fig)

    # 2. market index daily mean
    daily = al.groupby(["date", "src"])["market_index"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(7.2, 2.6), dpi=160)
    for src, c in [("train", ACCENT), ("validation", WARN)]:
        s = daily[daily.src == src]
        ax.plot(s["date"], s["market_index"], color=c, lw=1,
                label={"train": "training months", "validation": "months we predict"}[src])
    ax.set_ylabel("market_index (daily average)")
    ax.set_title("market_index: one value per day, with a weekly up-and-down cycle", loc="left")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(C.FIGURES / "market_index_daily.png"); plt.close(fig)

    # 3. rate per mile vs distance
    fig, ax = plt.subplots(figsize=(7.2, 2.8), dpi=160)
    for e, c in zip(C.EQUIPMENT, [ACCENT, WARN, "#6B7280"]):
        s = tr[tr.equipment == e].sample(2500, random_state=0)
        ax.scatter(s["distance"], s["rpm"], s=3, alpha=0.35, color=c, label=e)
    ax.set_ylim(0, 6); ax.set_xlabel("distance (miles)"); ax.set_ylabel("price per mile ($)")
    ax.set_title("Price per mile falls with distance; equipment type shifts the level", loc="left")
    ax.legend(frameon=False, fontsize=8, markerscale=4)
    fig.tight_layout(); fig.savefig(C.FIGURES / "rpm_vs_distance.png"); plt.close(fig)

    # 4. split design
    fig, ax = plt.subplots(figsize=(7.2, 2.0), dpi=160)
    rows = [("Practice split 1", "2025-01-01", "2025-05-01", "2025-06-30"),
            ("Practice split 2", "2025-01-01", "2025-07-01", "2025-08-31"),
            ("Final check", "2025-01-01", "2025-09-01", "2025-10-31"),
            ("Production (predict Nov-Dec)", "2025-01-01", "2025-11-01", "2025-12-31")]
    for i, (lab, a, b, e) in enumerate(rows[::-1]):
        a, b, e = map(pd.Timestamp, (a, b, e))
        ax.barh(i, (b - a).days, left=a, color=ACCENT, height=0.55)
        ax.barh(i, (e - b).days + 1, left=b, color=WARN, height=0.55)
        ax.text(a - pd.Timedelta(days=3), i, lab, ha="right", va="center", fontsize=8)
    ax.set_yticks([]); ax.spines["left"].set_visible(False)
    ax.set_title("How the data is split (teal = train, orange = test or predict)", loc="left")
    ax.set_xlim(pd.Timestamp("2024-09-15"), pd.Timestamp("2026-01-05"))
    fig.tight_layout(); fig.savefig(C.FIGURES / "split_design.png"); plt.close(fig)
    print(json.dumps(out, indent=1, default=str)[:3000])


if __name__ == "__main__":
    main()
