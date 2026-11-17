"""Build reports/freight_rate_report.docx. Every number is read from the JSON outputs."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.shared import Inches, Pt, RGBColor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from freight import config as C  # noqa: E402

R = C.REPORTS


def load(name: str) -> dict:
    return json.loads((R / name).read_text())


eda, dev, dev2 = load("eda.json"), load("dev_results.json"), load("dev_results_v2.json")
fin, drift, pred = load("final_eval.json"), load("drift_diagnosis.json"), load("predictions_summary.json")
hist = load("history/evaluation_history.json")
prefix = load("history/prefix_final_eval.json")
DEV = dev["results"]
DEV2 = dev2["results"]
SEL = fin["selected"]


def money(x: float) -> str:
    return f"${x:,.2f}"


def pct(x: float, d: int = 1) -> str:
    return f"{x * 100:.{d}f}%"


doc = Document()
st = doc.styles["Normal"]
st.font.name = "Calibri"
st.font.size = Pt(10.5)
for s in ["Heading 1", "Heading 2"]:
    doc.styles[s].font.color.rgb = RGBColor(0x06, 0x4A, 0x56)


def para(text: str, bold_prefix: str | None = None, style: str | None = None):
    p = doc.add_paragraph(style=style)
    if bold_prefix:
        p.add_run(bold_prefix).bold = True
    p.add_run(text)
    p.paragraph_format.space_after = Pt(5)
    return p


def bullet(text: str, bold_prefix: str | None = None):
    return para(text, bold_prefix, style="List Bullet")


def table(header: list[str], rows: list[list], widths: list[float] | None = None):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Light Grid Accent 1"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(header):
        t.rows[0].cells[i].text = h
    for r in rows:
        cells = t.add_row().cells
        for i, v in enumerate(r):
            cells[i].text = str(v)
    for row in t.rows:
        for i, c in enumerate(row.cells):
            for p in c.paragraphs:
                for run in p.runs:
                    run.font.size = Pt(9)
            if widths:
                c.width = Inches(widths[i])
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


def figure(path: Path, caption: str, width: float = 6.3):
    doc.add_picture(str(path), width=Inches(width))
    cap = doc.add_paragraph(caption)
    cap.runs[0].italic = True
    cap.runs[0].font.size = Pt(9)


# ------------------------------------------------------------------------------------------------
tr, va, tg, mi, qs, geo = eda["train"], eda["validation"], eda["target"], eda["market_index"], eda["quote_signal"], eda["geography"]
dsum = DEV[dev["selected"]]["summary"]
cs, stress = SEL["cold_start"]["mean"], SEL["cold_start_stress"]["mean"]
refs = fin["references"]
lane, lin = refs["B0_lane_median"], refs["B1_loglinear_ols_filter0.5"]

doc.add_heading("Freight Rate Prediction: Validation Approach and Results", 0)
para("Spotter ML Engineer assessment. Code, data and full reproduction instructions are in the GitHub repository; "
     "every figure and number in this report is generated from the repository's JSON outputs.")

doc.add_heading("1. Summary", 1)
bullet(f" The development data covers {tr['date_min']} to {tr['date_max']}; the 12,000 loads to predict cover "
       f"{va['date_min']} to {va['date_max']}. All validation is therefore time-ordered (expanding window), never a random split.",
       "Out-of-time problem.")
bullet(" Model choices were made on two development folds only (D1: Jan-Apr -> May-Jun; D2: Jan-Jun -> Jul-Aug), using a "
       "selection metric fixed before any results were seen: all-row dollar MAE. September-October was held out from model "
       "fitting and selection; Section 3 states exactly how and when it was looked at.", "Pre-declared selection.")
bullet(f" A 0.75/0.25 geometric blend of a LightGBM model (Huber loss on log rate, large-residual rows filtered from "
       f"training) and a log-linear regression. Dev-fold MAE {money(dsum['primary_mae'])} vs {money(DEV['B0_lane_median']['summary']['primary_mae'])} "
       f"for a lane-median baseline.", "Selected model.")
bullet(f" On held-out Sep-Oct the frozen model scored MAE {money(SEL['full']['mae'])} (median APE {pct(SEL['full']['median_ape'])}). "
       f"This is materially worse than development and roughly equal to the lane-median reference ({money(lane['full']['mae'])}). "
       f"Diagnosis: an upward rate-level drift not explained by market_index, which the model under-predicts by "
       f"~{pct(drift['bias_by_test_month'][5]['median_log_actual_over_pred'], 0)}-{pct(drift['bias_by_test_month'][4]['median_log_actual_over_pred'], 0)} in Sep-Oct. This is reported as-is (Section 6).",
       "Honest held-out result.")
bullet(" quote_signal flips sign month to month and has no detectable regime in Aug/Nov/Dec. Using it would have "
       "improved the pooled dev score while badly degrading the regime-matched month and the December chart, so it is excluded.",
       "Key data finding.")

doc.add_heading("2. Data findings and quality issues", 1)
table(["Issue", "Evidence", "Handling"], [
    ["Negative weights", f"{tr['weight_negative']} train / {va['weight_negative']} validation rows",
     "Invalid; true value unknown. abs() vs set-to-missing compared on dev folds (abs() kept: missing was "
     f"{pct(DEV['M1_huber_filter0.5+no_dow+weight_neg_nan']['summary']['primary_mae'] / DEV['M1_huber_filter0.5+no_dow']['summary']['primary_mae'] - 1, 2)} worse). Invalid-weight flag retained."],
    ["Missing weights", f"{tr['weight_missing']} / {va['weight_missing']}", "Equipment median from the training fold + missing flag."],
    ["Weights at exactly 47,500 lb", f"{tr['weight_at_47500']} / {va['weight_at_47500']}",
     "Left unchanged (reaching a cap does not prove an error); an at-cap indicator did not help on dev folds."],
    ["Missing market_index", f"{tr['market_index_missing']} / {va['market_index_missing']}",
     f"market_index is a date-level signal (within-day std {mi['within_day_std']:.3f} vs across-day {mi['across_day_std_of_daily_mean']:.3f}); "
     "imputed with the same date's mean of feature values in the same file (label-free)."],
    ["Large label residuals", f"{tg['large_residual_rows_abs_gt_0.5']} rows ({pct(tg['large_residual_share'])}) more than "
     "e^0.5 from a log-linear fit; two clusters (~0.2-0.7x and ~1.5-7x), evenly spread across months",
     "Not assumed wrong. Unfiltered, robust-loss and fold-filtered training compared; all evaluation rows kept in headline metrics."],
    ["Coordinates", f"One fixed point per city; distance/great-circle ratio median {geo['lane_distance_to_haversine_ratio_quantiles']['0.5']:.2f} for every city "
     f"(New Orleans {geo['new_orleans_median_ratio']:.3f}); only a few short pairs look odd",
     "No repair (no evidence a city is misplaced). distance is the trusted geometry; coordinates are location descriptors."],
    ["Cities only in validation", f"{len(geo['validation_only_cities'])} cities ({', '.join(geo['validation_only_cities'])}), "
     f"{pct(geo['validation_share_touching_new_cities'])} of validation loads", "No city-ID features; cold-start explicitly simulated (Section 3)."],
    ["quote_signal regimes", "Within-month Spearman with distance ~ +/-0.77, sign flipping by month; ~0 in Aug, Nov, Dec",
     "Investigated in Section 5; excluded."],
], widths=[1.3, 2.6, 2.6])
figure(C.FIGURES / "rpm_vs_distance.png", "Figure 1. Rate per mile vs distance by equipment (train sample).")
figure(C.FIGURES / "market_index_daily.png", "Figure 2. Daily mean market_index: a date-level signal with a weekly cycle, including the validation period.")

doc.add_heading("3. Validation and split approach", 1)
para("The task is a forecast: train on Jan-Oct 2025, predict Nov-Dec 2025. A random K-fold split would let the model see "
     "loads from the same days and weeks it is tested on (market_index is shared by every load on a date), overstating "
     "accuracy and hiding drift. Every split is therefore time-ordered and expanding-window, with an assertion that no "
     "training row is on or after the test start date.")
figure(C.FIGURES / "split_design.png", "Figure 3. Development folds, held-out evaluation and the production fit.")
bullet(" D1 (Jan-Apr -> May-Jun) and D2 (Jan-Jun -> Jul-Aug) are the only data used for tuning, feature selection, outlier "
       "rules, early stopping and blending. Each fold mimics the real two-month horizon. Early stopping uses the last month "
       "of each fold's own training window: raw rows are split first, preprocessing and the outlier filter are fitted on the "
       "earlier months only, and every early-stopping row is kept. The frozen tree count is the mean of the dev folds' "
       "early-stopped counts.",
       "Development folds.")
bullet(" Primary: all-row dollar MAE in full-feature mode, mean of D1 and D2 (predictions back-transformed from log space). "
       "A more complex option is adopted only if it lowers the primary metric by more than 0.3% without worsening any "
       "secondary check by more than 2% (December-mode MAE, August MAE, unseen-city MAE, worst fold). Simpler wins ties.",
       "Pre-declared selection rule.")
bullet(f" Removing cities from both ends of training. Validation-only cities are low-volume, so 4 held-out cities match the "
       f"real exposure: {pct(dsum['exposure'])} of dev test loads (validation: {pct(geo['validation_share_touching_new_cities'])}). "
       f"An 8-city stress test is also reported. Seen and unseen loads are scored separately.", "Cold-start simulation.")
bullet(" The December chart file lacks coordinates, market_index and quote_signal. In every fold, test rows are also scored "
       "after dropping those columns and rebuilding them exactly as for December: coordinates from a city lookup built from "
       "supplied feature rows (unknown city = error), market_index as the per-date mean of the period's feature values, "
       "quote_signal absent. The held-out target is never passed to this reconstruction (unit-tested).", "December-mode evaluation.")
bullet(" Imputation statistics, outlier filtering, trend estimates and early stopping are all fitted inside each training "
       "fold through a single fit/transform pipeline. The outlier filter drops training rows whose out-of-fold residual "
       "(5 date-grouped folds, trimmed least squares on log rate) exceeds e^0.5; evaluation rows are never filtered.", "Fold hygiene.")
bullet(" Sep-Oct labels were looked at during EDA, and Sep-Oct was never used to fit or select a model. The evaluation "
       "script refuses to run unless the frozen pipeline is committed. Spotter's Nov-Dec labels are the only truly external test.",
       "Status of Sep-Oct.")
para("The sequence of events, recorded in reports/history/:", "Evaluation history. ")
bullet(f" Round-1 selection on the dev folds, frozen and committed, then evaluated on Sep-Oct: MAE "
       f"{money(prefix['selected']['full']['mae'])}.", "1.")
bullet(" After seeing that result (a systematic under-prediction), exploratory experiments were run on the dev folds only "
       "(linear trend, recency weighting). Neither met the adoption rule, so the pipeline was unchanged. From this point on, "
       "Sep-Oct is not independent evidence.", "2.")
bullet(" A code review found that early stopping was contaminated: preprocessing and the outlier filter had been fitted on "
       "the whole training window, including the early-stopping month, and early-stopping rows were filtered. The filter "
       "also used in-sample residuals. This was fixed, and round-1 selection, round 2, the freeze and the Sep-Oct evaluation "
       "were all re-run with unchanged rules. The numbers in this report are from the corrected run.", "3.")

doc.add_heading("4. Development results (dev folds only)", 1)
rows = []
for name, label in [
    ("B0_lane_median", "B0 lane-median rate/mile x distance"),
    ("B1_loglinear_ols", "B1 log-linear OLS"),
    ("B1_loglinear_ols_filter0.5", "B1 log-linear OLS, filtered"),
    ("M1_l2_filterNone", "LightGBM L2, unfiltered"),
    ("M1_huber_filter0.5", "LightGBM Huber, filtered"),
    ("M1_huber_filter0.5+qs_raw", "  + raw quote_signal (rejected)"),
    ("M1_huber_filter0.5+no_dow", "  - day-of-week (adopted)"),
    ("M1_huber_filter0.5+no_dow+no_coords", "  - coordinates (rejected)"),
    ("M1_huber_filter0.5+no_dow+nl15_mcs100", "  tuned: 15 leaves, min_child 100"),
    ("blend_w0.75", "Selected: 0.75 LightGBM + 0.25 log-linear"),
]:
    s = DEV[name]["summary"]
    rows.append([label, money(s["primary_mae"]), money(s["december_mode_mae"]), money(s["aug_mae"]),
                 money(s["unseen_mae"]), pct(s["mape"], 2)])
table(["Candidate", "Primary MAE", "Dec-mode MAE", "Aug MAE", "Unseen-city MAE", "MAPE"], rows,
      widths=[2.4, 0.9, 0.9, 0.8, 1.0, 0.7])
para("Robust losses matter: with ~1.4% of labels far from the bulk, L2 on log rate is pulled around by those rows "
     "(unfiltered L2 is the worst LightGBM variant). Coordinates carry real lane information beyond distance (removing them costs "
     f"{pct(DEV['M1_huber_filter0.5+no_dow+no_coords']['summary']['primary_mae'] / DEV['M1_huber_filter0.5+no_dow']['summary']['primary_mae'] - 1)}). "
     "Day-of-week was dropped: market_index already carries the weekly cycle. The linear component makes the blend more robust "
     "for unseen cities.")

doc.add_heading("5. quote_signal investigation", 1)
figure(C.FIGURES / "quote_signal_regimes.png",
       "Figure 4. Monthly Spearman correlation of quote_signal with distance (bars; computable without labels, including Nov-Dec) and with rate per mile (dots; train only).")
para(f"Within a month, quote_signal is almost a monotone function of rate per mile (|Spearman| ~ 0.96), but the sign flips "
     f"between months, and in August, November and December it has no relationship with distance at all. "
     f"Because the regime is visible from features alone, August is the right historical analogue for Nov-Dec. "
     f"Three variants were compared month by month on the dev folds:")
base, raw, ali = (DEV[k]["folds"] for k in ["M1_huber_filter0.5", "M1_huber_filter0.5+qs_raw", "M1_huber_filter0.5+qs_aligned"])
rows = []
for fi, months in [(0, ["5", "6"]), (1, ["7", "8"])]:
    for m in months:
        rows.append([{"5": "May", "6": "Jun", "7": "Jul", "8": "Aug (Nov-Dec analogue)"}[m]] +
                    [money(f[fi]["by_month"][m]["full"]["mae"]) for f in (base, raw, ali)] +
                    [money(raw[fi]["by_month"][m]["december_mode"]["mae"])])
table(["Test month", "No quote_signal", "Raw quote_signal", "Regime-aware", "Raw, but signal absent (Dec mode)"], rows,
      widths=[1.6, 1.1, 1.1, 1.1, 1.5])
para("Raw quote_signal helps in months where its regime matches training, but is clearly worse in August, and a model that "
     "depends on it collapses when the signal is absent, as it is in the December chart file. A label-free regime-aware "
     "variant did not help either. quote_signal is excluded from the final model.")

doc.add_heading("6. Held-out evaluation (Sep-Oct)", 1)
table(["Metric (all rows)", "Selected blend", "Lane-median ref.", "Log-linear ref."], [
    ["MAE", money(SEL["full"]["mae"]), money(lane["full"]["mae"]), money(lin["full"]["mae"])],
    ["MAE, December mode", money(SEL["december_mode"]["mae"]), money(lane["december_mode"]["mae"]), money(lin["december_mode"]["mae"])],
    ["RMSE", f"${SEL['full']['rmse']:,.0f}", f"${lane['full']['rmse']:,.0f}", f"${lin['full']['rmse']:,.0f}"],
    ["MAPE", pct(SEL["full"]["mape"], 2), pct(lane["full"]["mape"], 2), pct(lin["full"]["mape"], 2)],
    ["Median APE", pct(SEL["full"]["median_ape"], 2), pct(lane["full"]["median_ape"], 2), pct(lin["full"]["median_ape"], 2)],
    ["MAE, clean-label subset (supplementary)", money(SEL["clean_subset_supplementary"]["mae"]),
     money(lane["clean_subset_supplementary"]["mae"]), money(lin["clean_subset_supplementary"]["mae"])],
], widths=[2.4, 1.3, 1.3, 1.3])
table(["Slice", "MAE"], [
    ["September", money(SEL["by_month"]["9"]["full"]["mae"])],
    ["October", money(SEL["by_month"]["10"]["full"]["mae"])],
    *[[e, money(SEL["by_equipment"][e]["mae"])] for e in C.EQUIPMENT],
    [f"Cold start, 4 cities: seen loads (exposure {pct(cs['exposure'])})", money(cs["seen"]["mae"])],
    ["Cold start, 4 cities: unseen-city loads", money(cs["unseen"]["mae"])],
    [f"Stress, 8 cities: unseen-city loads (exposure {pct(stress['exposure'])})", money(stress["unseen"]["mae"])],
], widths=[4.0, 1.3])
para("The result is materially worse than development and roughly equal to the lane-median reference. "
     "The large gap between all-row and clean-subset MAE shows how much of the dollar error comes from the "
     f"{pct(1 - SEL['clean_subset_supplementary']['share_of_rows'])} of rows with extreme labels. On typical loads the lane "
     f"median is better (median APE {pct(lane['full']['median_ape'])} vs {pct(SEL['full']['median_ape'])}). It has no "
     "market_index term, so it is not pulled down by the low Sep-Oct index, while actual rates stayed high.", "Assessment. ")
bias = {(b["fold"], b["month"]): b["median_log_actual_over_pred"] for b in drift["bias_by_test_month"]}
table(["Months ahead", "D1", "D2", "Held-out"], [
    ["1", pct(bias[("D1", 5)]), pct(bias[("D2", 7)]), pct(bias[("FINAL", 9)])],
    ["2", pct(bias[("D1", 6)]), pct(bias[("D2", 8)]), pct(bias[("FINAL", 10)])],
], widths=[1.4, 1.2, 1.2, 1.2])
fe = drift["month_fixed_effects_log"]
para(f"Median log(actual/predicted) by horizon (table above): the model systematically under-predicts, and the gap grows with horizon "
     f"and over the year. Month fixed effects controlling for distance, equipment, weight and market_index rise from 0 in "
     f"January to {pct(fe['9'])} in September and {pct(fe['10'])} in October: a rate-level drift that market_index does not "
     "explain, with bumps in March, June and September. Within a single year this cannot be separated from a weakening "
     f"market_index effect (daily elasticity estimate {drift['daily_mi_elasticity_by_period']['Jan-Mar']:.2f} in Jan-Mar, "
     f"{drift['daily_mi_elasticity_by_period']['Jul-Aug']:.2f} in Jul-Aug, {drift['daily_mi_elasticity_by_period']['Sep-Oct']:.2f} in Sep-Oct). Either reading "
     "implies the same failure: the model expects low-index days to be cheaper than they now are.", "Diagnosis. ")
t2 = DEV2
para(f"Prompted by the under-prediction (so Sep-Oct is not independent evidence for this step), a second development "
     f"round was run on D1/D2 only, under the same rules. A fold-fitted linear trend was rejected "
     f"(dev MAE {money(t2['v1+trend']['summary']['primary_mae'])} vs {money(t2['v1_blend_w0.75']['summary']['primary_mae'])}): "
     f"slopes estimated from 4-6 months are unstable and over-extrapolate. Recency-weighted training (60-day half-life) "
     f"gave {money(t2['v1_blend_w0.75+halflife60']['summary']['primary_mae'])}, below the adoption threshold. "
     "The frozen pipeline was therefore left unchanged. Making a correction justified only by the held-out period "
     "would turn it into a tuning set.", "Response. ")
para("Implication for Nov-Dec: the production model is refit on Jan-Oct, so it sees the higher recent level, but it "
     f"still has no way to extrapolate further drift. Its average predicted rate on the validation loads is {money(pred['validation_pred_mean'])} "
     f"vs a Sep-Oct actual mean of {money(pred['sep_oct_actual_mean'])}. Spotter's Nov-Dec labels may well show a "
     "similar under-prediction.", "Risk. ")

doc.add_heading("7. December prediction chart", 1)
figure(C.ROOT / "scorer_results" / "candidate_december.png",
       "Figure 5. December 2025 predictions for Lexington -> Fort Wayne, 360 mi, Dry Van, 32,000 lb (generated by the provided score.py).")
d = pred["december"]
para(f"The December inputs fix everything except the date. The missing columns are rebuilt as described in Section 3: "
     f"coordinates from the city lookup, market_index from the validation file's per-date mean, and no quote_signal. "
     f"Rebuilding them this way costs about nothing on held-out data (MAE {money(SEL['december_mode']['mae'])} vs {money(SEL['full']['mae'])}).")
para(f"Predictions range from {money(pred['december_range'][0])} to {money(pred['december_range'][1])}. The day-to-day movement comes "
     f"mostly from market_index (correlation {pred['corr_pred_vs_market_index']:.2f}). With market_index held at its December "
     f"average, the line is nearly flat ({money(pred['december_range_mi_constant'][0])}-{money(pred['december_range_mi_constant'][1])}); "
     f"the small remaining variation comes from the day-of-week terms in the log-linear component. "
     f"The weekly peaks are midweek days with high market_index (e.g. {d[24]['date']}, a {d[24]['dow']}, index {d[24]['market_index_used']:.3f}). "
     f"For context, the {eda['lexington_fort_wayne_dry_van']['n']} historical Dry Van loads on this lane ranged "
     f"{money(eda['lexington_fort_wayne_dry_van']['min'])}-{money(eda['lexington_fort_wayne_dry_van']['max'])} "
     f"(median {money(eda['lexington_fort_wayne_dry_van']['median'])}) with varying weights, distances and dates.")
para("The model has never seen a December, so it cannot represent holiday or year-end effects. The weekly shape mostly "
     "follows the daily market_index rather than a learned calendar effect. Given the drift in Section 6, the overall December level is more "
     "likely to be too low than too high.")

doc.add_heading("8. Limitations and next steps", 1)
for t in [
    "Level drift: the main weakness. With more history (or an external rate index), a dampened or seasonal level component could be validated over several cycles instead of one year.",
    "Quarter-end bumps (Mar, Jun, Sep) appear in the residuals, but three occurrences are too few to model responsibly; December, as a quarter- and year-end month, may carry one.",
    "Extreme labels (~1.4% of rows) account for a large share of dollar MAE and most of the RMSE. Large residuals alone do not show whether they are recording errors or genuinely unusual loads that better features could explain.",
    "quote_signal is excluded because of unstable regimes. If its generating process were documented, a regime-aware use might recover the gains seen in matching months.",
    "Cold-start loads rely on coordinates and distance; unseen-city MAE is higher than seen-city MAE.",
]:
    bullet(t)

doc.add_heading("Appendix: reproduction", 1)
for t in ["python -m venv .venv && .venv/bin/pip install -r requirements.txt",
          ".venv/bin/python scripts/eda.py",
          ".venv/bin/python scripts/develop.py && .venv/bin/python scripts/develop_v2.py",
          ".venv/bin/python scripts/final_eval.py   # requires configs/final.yaml committed",
          ".venv/bin/python scripts/predict.py",
          ".venv/bin/python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv",
          ".venv/bin/python scripts/build_report.py"]:
    p = doc.add_paragraph(t)
    p.runs[0].font.name = "Consolas"
    p.runs[0].font.size = Pt(8.5)

out = R / "freight_rate_report.docx"
doc.save(out)
print("wrote", out)
