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
for sec in doc.sections:
    sec.left_margin = sec.right_margin = Inches(0.75)
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
    t.autofit = False
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
    if widths:
        for i, col in enumerate(t.columns):
            col.width = Inches(widths[i])
    if len(t.rows) <= 8:  # keep small tables on one page
        for row in t.rows[:-1]:
            for c in row.cells:
                for p in c.paragraphs:
                    p.paragraph_format.keep_with_next = True
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
REPO = "https://github.com/Ishaan-Giri/freight-rate-prediction"


def rel(a: str, b: str) -> float:
    return DEV[a]["summary"]["primary_mae"] / DEV[b]["summary"]["primary_mae"] - 1


doc.add_heading("Freight Rate Prediction: Validation Approach and Results", 0)
para(f"Spotter ML Engineer assessment. Code, data and run instructions: {REPO}. "
     "Every number and chart in this report is produced by the code in that repository.")

doc.add_heading("How to read this report", 1)
table(["Term", "Meaning"], [
    ["Average error (MAE)", "How far the predicted price is from the real price, in dollars, averaged over all loads. Lower is better. This is the main score."],
    ["% error (MAPE / median APE)", "The same error as a percentage of the real price. Median APE is the error of a typical load."],
    ["Practice splits", "Two tests used to choose the model: train on Jan-Apr and test on May-Jun (practice split 1), "
                        "and train on Jan-Jun and test on Jul-Aug (practice split 2)."],
    ["Final check", "Train on Jan-Aug, test on Sep-Oct. Never used to choose the model."],
    ["New-city error", "Error on loads touching cities the model never saw in training (we hide 4 cities to test this)."],
    ["December-file test", "Error when the columns missing from the December chart file are removed and rebuilt "
                           "the same way we do for December."],
    ["August error", "August behaves like Nov-Dec for quote_signal, so it is our best stand-in for the real target months."],
], widths=[1.6, 4.9])

doc.add_heading("1. Summary", 1)
bullet(f" We learn from loads dated {tr['date_min']} to {tr['date_max']} and predict 12,000 loads dated "
       f"{va['date_min']} to {va['date_max']}. Because we predict the future, every test trains on earlier months and "
       "tests on later months. We never use a random split.", "It is a forecasting problem.")
bullet(" The model was chosen on two practice splits, using a scoring rule written down before seeing any results. "
       "Sep-Oct was kept aside as a final check; Section 3 explains exactly how it was used.", "Fair model choice.")
bullet(f" A mix of 75% LightGBM (a decision-tree model) and 25% linear model. Average error on the practice splits: "
       f"{money(dsum['primary_mae'])} per load, against {money(DEV['B0_lane_median']['summary']['primary_mae'])} "
       "for a simple lane-average baseline.", "Chosen model.")
bullet(f" On Sep-Oct the average error was {money(SEL['full']['mae'])} (typical load off by {pct(SEL['full']['median_ape'])}). "
       f"That is worse than in practice and about the same as the simple lane average ({money(lane['full']['mae'])}). "
       f"The reason: prices rose over the year more than market_index explains, so the model predicts about "
       f"{pct(drift['bias_by_test_month'][5]['median_log_actual_over_pred'], 0)}-"
       f"{pct(drift['bias_by_test_month'][4]['median_log_actual_over_pred'], 0)} too low in Sep-Oct (Section 6).",
       "Honest final result.")
bullet(" quote_signal looks very predictive, but its direction flips from month to month, and in August, November and "
       "December it shows no pattern. Using it would have helped some months and badly hurt the ones that matter, "
       "so it is left out.", "Most important data finding.")

doc.add_heading("2. What we found in the data, and how we fixed problems", 1)
table(["Problem", "What we saw", "What we did"], [
    ["Negative weights", f"{tr['weight_negative']} training rows, {va['weight_negative']} prediction rows. A weight cannot be negative.",
     "Tried two fixes: turn them positive, or treat them as missing. Results were almost identical "
     f"({pct(rel('M1_huber_filter0.5+no_dow+weight_neg_nan', 'M1_huber_filter0.5+no_dow'), 2)} apart), so we turn them "
     "positive and add a flag."],
    ["Missing weights", f"{tr['weight_missing']} training, {va['weight_missing']} prediction rows",
     "Filled with the typical weight for that equipment type, plus a flag."],
    ["Weights of exactly 47,500 lb", f"{tr['weight_at_47500']} training, {va['weight_at_47500']} prediction rows",
     "Left as they are. Hitting a maximum does not prove an error, and a flag for it did not help."],
    ["Missing market_index", f"{tr['market_index_missing']} training, {va['market_index_missing']} prediction rows",
     "market_index is one number per day (loads on the same day differ very little), so we fill it with that day's average."],
    ["Extreme prices", f"{tg['large_residual_rows_abs_gt_0.5']} rows ({pct(tg['large_residual_share'])}) are 2 to 7 times "
     "too high or too low compared with similar loads, spread evenly across months.",
     "Not assumed to be mistakes. They always count in our error scores. For training, leaving them out worked best."],
    ["City coordinates", "Each city has one fixed location. Checked against the distance column: every city is consistent.",
     "No change. The distance column is used for distance; coordinates describe location."],
    ["Cities never seen in training", f"{len(geo['validation_only_cities'])} cities ({', '.join(geo['validation_only_cities'])}) "
     f"appear only in the prediction file: {pct(geo['validation_share_touching_new_cities'])} of its loads.",
     "No city-name features (they cannot help a new city). We test new-city performance directly (Section 3)."],
    ["quote_signal behaves strangely", "Its direction flips from month to month; no pattern in Aug, Nov, Dec.",
     "Investigated in Section 5 and left out."],
], widths=[1.3, 2.6, 2.6])
figure(C.FIGURES / "rpm_vs_distance.png",
       "Figure 1. Price per mile falls as distance grows; Reefer and Flatbed cost more than Dry Van.")
figure(C.FIGURES / "market_index_daily.png",
       "Figure 2. market_index is one value per day with a weekly up-and-down cycle (orange = the months we predict).")

doc.add_heading("3. How we split the data and tested the model", 1)
para("We must predict Nov-Dec from Jan-Oct. A random split would mix days: the model would learn from loads on the "
     "same days it is tested on (they share market_index) and look better than it really is. So every test trains on "
     "earlier months and tests on the following two months, the same situation as the real task. The code checks "
     "that no training row comes from the test period.")
figure(C.FIGURES / "split_design.png",
       "Figure 3. Teal = training months, orange = test months. The two practice splits choose the model; "
       "the final check tests it; production trains on everything and predicts Nov-Dec.")
bullet(" Practice split 1 trains on Jan-Apr and tests on May-Jun; practice split 2 trains on Jan-Jun and tests on Jul-Aug. "
       "Every decision (features, settings, outlier handling, mixing) was made on these two splits only.", "Practice splits.")
bullet(" Before looking at results we fixed the score (average dollar error) and the rule: a change is accepted only if it "
       "lowers the error by more than 0.3% and does not make August, new-city or December-file error more than 2% worse. "
       "If two options tie, the simpler one wins.", "A rule written down first.")
bullet(f" We hide 4 cities from training and measure the error on loads touching them. 4 cities give "
       f"{pct(dsum['exposure'])} of test loads, close to the {pct(geo['validation_share_touching_new_cities'])} of new-city loads "
       "in the real prediction file. An 8-city version is reported as a harder test.", "New-city test.")
bullet(" The December chart file has no coordinates, market_index or quote_signal. In every test we also remove those columns "
       "and rebuild them exactly as for December: coordinates from a city list, market_index as that day's average from the "
       "same file, quote_signal left out. The real prices are never used for this (checked by an automated test).",
       "December-file test.")
bullet(" All cleaning steps (filling missing values, removing extreme prices from training, choosing the number of trees) "
       "learn only from the training months of each split. Test rows are never cleaned or removed.", "No leakage.")
bullet(" Sep-Oct prices were looked at while exploring the data, but Sep-Oct was never used to train or choose a model. "
       "The model settings were saved in git before testing on Sep-Oct, and the test script refuses to run otherwise. "
       "Spotter's Nov-Dec prices are the only truly unseen test.", "About the final check.")
para("What happened, in order (records in reports/history/):", "History. ")
bullet(f" Model chosen on the practice splits, saved, then tested on Sep-Oct: average error {money(prefix['selected']['full']['mae'])}.", "1.")
bullet(" That result showed the model predicts too low. We tried two fixes on the practice splits only (a trend line, and "
       "giving recent months more weight). Neither passed the rule, so the model did not change. After this, Sep-Oct is no "
       "longer a fully independent test.", "2.")
bullet(" A code review found a leak: when choosing the number of trees, the month used for that choice had also been used "
       "in cleaning, and some of its rows were removed. We fixed it, added tests, and re-ran everything with the same rules. "
       "The same model was chosen. All numbers in this report are from the corrected run.", "3.")

doc.add_heading("4. Which model we chose and why", 1)
para("All numbers below are average dollar errors per load on the practice splits (lower is better). Each row is one "
     "thing we tried; indented rows change one thing in the row above.")
rows = []
for name, label, what, result in [
    ("B0_lane_median", "Lane average", "Average price per mile on that lane, times distance", "Baseline"),
    ("B1_loglinear_ols", "Linear model", "A straight-line formula on distance, equipment, weight, market_index", "Better than baseline"),
    ("B1_loglinear_ols_filter0.5", "Linear model, extreme prices removed", "Same, but trained without the extreme prices", "Used in final mix"),
    ("M1_l2_filterNone", "LightGBM, standard", "Decision-tree model trained on all rows", "Hurt by extreme prices"),
    ("M1_huber_filter0.5", "LightGBM, robust", "Uses a loss that ignores extreme prices (Huber) and trains without them", "Much better"),
    ("M1_huber_filter0.5+qs_raw", "   + quote_signal", "Adds quote_signal as an input", "Rejected: August and December-file error much worse"),
    ("M1_huber_filter0.5+no_dow", "   - day of week", "Removes day of week (market_index already has the weekly cycle)", "Accepted"),
    ("M1_huber_filter0.5+no_dow+no_coords", "   - coordinates", "Removes city locations", "Rejected: much worse"),
    ("M1_huber_filter0.5+no_dow+nl15_mcs100", "   tuned", "Smaller trees, each needing more loads per branch", "Accepted"),
    ("blend_w0.75", "Final: 75% LightGBM + 25% linear", "Average of the two models' predictions", "CHOSEN"),
]:
    s = DEV[name]["summary"]
    rows.append([label, what, money(s["primary_mae"]), money(s["aug_mae"]), money(s["unseen_mae"]),
                 money(s["december_mode_mae"]), result])
table(["Model tried", "What it is", "Average error", "August error", "New-city error", "December-file test", "Result"], rows,
      widths=[1.25, 1.85, 0.7, 0.7, 0.7, 0.85, 0.95])
bullet(" About 1.4% of prices are extreme. A standard model chases them; a robust one does not. That change alone "
       f"lowered the error from {money(DEV['M1_l2_filterNone']['summary']['primary_mae'])} to "
       f"{money(DEV['M1_huber_filter0.5']['summary']['primary_mae'])}.", "Why a robust LightGBM.")
bullet(f" Removing city locations made the error {pct(rel('M1_huber_filter0.5+no_dow+no_coords', 'M1_huber_filter0.5+no_dow'))} "
       "worse, so they carry real information beyond distance. Day of week added nothing that market_index did not already have.",
       "Why these inputs.")
bullet(f" Adding the linear model lowered the average error and the new-city error "
       f"({money(DEV['M1_huber_filter0.5+no_dow+nl15_mcs100']['summary']['unseen_mae'])} to {money(dsum['unseen_mae'])}): "
       "the simple formula is steadier on cities the trees have never seen.", "Why mix in a linear model.")
bullet(f" It lowered the overall error a little, but August error rose from "
       f"{money(DEV['M1_huber_filter0.5']['summary']['aug_mae'])} to {money(DEV['M1_huber_filter0.5+qs_raw']['summary']['aug_mae'])} "
       f"and the December-file error from {money(DEV['M1_huber_filter0.5']['summary']['december_mode_mae'])} to "
       f"{money(DEV['M1_huber_filter0.5+qs_raw']['summary']['december_mode_mae'])}. See Section 5.", "Why no quote_signal.")

doc.add_heading("5. Why quote_signal was left out", 1)
figure(C.FIGURES / "quote_signal_regimes.png",
       "Figure 4. Bars: how quote_signal moves with distance each month (can be measured without prices, so Nov-Dec "
       "is included). Dots: how it moves with price per mile (training months only). The direction keeps flipping.")
para("Within one month, quote_signal tracks the price almost perfectly, but the direction flips between months, and in "
     "August, November and December there is no pattern at all. Because this can be seen without prices, we know Nov-Dec "
     "behave like August. Error by test month with and without it:")
base, raw, ali = (DEV[k]["folds"] for k in ["M1_huber_filter0.5", "M1_huber_filter0.5+qs_raw", "M1_huber_filter0.5+qs_aligned"])
rows = []
for fi, months in [(0, ["5", "6"]), (1, ["7", "8"])]:
    for m in months:
        rows.append([{"5": "May", "6": "June", "7": "July", "8": "August (like Nov-Dec)"}[m]] +
                    [money(f[fi]["by_month"][m]["full"]["mae"]) for f in (base, raw, ali)] +
                    [money(raw[fi]["by_month"][m]["december_mode"]["mae"])])
table(["Test month", "Without quote_signal", "With quote_signal", "With a month-aware version",
       "With quote_signal, but it is missing (as in December)"], rows, widths=[1.5, 1.1, 1.1, 1.2, 1.6])
para("It helps in months that look like the training months, but it is clearly worse in August, and a model that relies "
     "on it falls apart when it is missing, as it is in the December file. A smarter month-aware version did not help "
     "either. So quote_signal is not used.")

doc.add_heading("6. Final check on Sep-Oct", 1)
table(["Score (all loads)", "Our model", "Lane average", "Linear model"], [
    ["Average error (MAE)", money(SEL["full"]["mae"]), money(lane["full"]["mae"]), money(lin["full"]["mae"])],
    ["Average error, December-file test", money(SEL["december_mode"]["mae"]), money(lane["december_mode"]["mae"]), money(lin["december_mode"]["mae"])],
    ["Root mean squared error (punishes big misses)", f"${SEL['full']['rmse']:,.0f}", f"${lane['full']['rmse']:,.0f}", f"${lin['full']['rmse']:,.0f}"],
    ["Average % error", pct(SEL["full"]["mape"], 2), pct(lane["full"]["mape"], 2), pct(lin["full"]["mape"], 2)],
    ["Typical load % error (median)", pct(SEL["full"]["median_ape"], 2), pct(lane["full"]["median_ape"], 2), pct(lin["full"]["median_ape"], 2)],
    ["Average error without the extreme prices (for reference only)", money(SEL["clean_subset_supplementary"]["mae"]),
     money(lane["clean_subset_supplementary"]["mae"]), money(lin["clean_subset_supplementary"]["mae"])],
], widths=[2.6, 1.2, 1.2, 1.2])
table(["Group of loads", "Average error"], [
    ["September", money(SEL["by_month"]["9"]["full"]["mae"])],
    ["October", money(SEL["by_month"]["10"]["full"]["mae"])],
    *[[e, money(SEL["by_equipment"][e]["mae"])] for e in C.EQUIPMENT],
    [f"New-city test, 4 hidden cities: loads on known cities ({pct(cs['exposure'])} of loads were affected)", money(cs["seen"]["mae"])],
    ["New-city test, 4 hidden cities: loads touching a hidden city", money(cs["unseen"]["mae"])],
    [f"Harder test, 8 hidden cities: loads touching a hidden city ({pct(stress['exposure'])} of loads)", money(stress["unseen"]["mae"])],
], widths=[4.6, 1.3])
para("The error is clearly higher than in practice and about equal to the simple lane average. The extreme prices "
     f"(in Sep-Oct, {pct(1 - SEL['clean_subset_supplementary']['share_of_rows'])} of loads) cause about a third of it: without them the "
     f"average error is {money(SEL['clean_subset_supplementary']['mae'])}. On a typical load the lane average is slightly "
     f"better ({pct(lane['full']['median_ape'])} vs {pct(SEL['full']['median_ape'])}).", "What it means. ")
bias = {(b["fold"], b["month"]): b["median_log_actual_over_pred"] for b in drift["bias_by_test_month"]}
table(["Months after training ends", "Practice split 1", "Practice split 2", "Final check"], [
    ["1st month", pct(bias[("D1", 5)]), pct(bias[("D2", 7)]), pct(bias[("FINAL", 9)])],
    ["2nd month", pct(bias[("D1", 6)]), pct(bias[("D2", 8)]), pct(bias[("FINAL", 10)])],
], widths=[2.0, 1.4, 1.4, 1.4])
fe = drift["month_fixed_effects_log"]
para("The table shows how much the real price is above our prediction for a typical load. The model always predicts "
     "too low, and the gap grows the further ahead and the later in the year we predict. After accounting for distance, "
     f"equipment, weight and market_index, prices still rise from January to about {pct(fe['9'])} higher in September and "
     f"{pct(fe['10'])} in October. market_index does not explain this rise, and tree models cannot project a rise forward. "
     f"(Another way to read the same data: market_index matters less over time, with its effect falling from "
     f"{drift['daily_mi_elasticity_by_period']['Jan-Mar']:.2f} in Jan-Mar to "
     f"{drift['daily_mi_elasticity_by_period']['Sep-Oct']:.2f} in Sep-Oct. One year of data cannot separate the two.)",
     "Why. ")
t2 = DEV2
para(f"Prompted by this result, we tried two fixes on the practice splits only, with the same rule. A trend line made "
     f"things much worse ({money(t2['v1+trend']['summary']['primary_mae'])} vs "
     f"{money(t2['v1_blend_w0.75']['summary']['primary_mae'])}): a slope learned from 4-6 months is unreliable. Giving "
     f"recent months more weight gave {money(t2['v1_blend_w0.75+halflife60']['summary']['primary_mae'])}, too small a gain "
     "to pass the rule. So the model was not changed. Fixing it using the Sep-Oct results would turn the final check into "
     "tuning.", "What we did. ")
para(f"The production model is trained on Jan-Oct, so it sees the higher recent prices, but it still cannot project "
     f"further rises. Its average prediction for Nov-Dec is {money(pred['validation_pred_mean'])}, against a real Sep-Oct "
     f"average of {money(pred['sep_oct_actual_mean'])}. Nov-Dec predictions are likely to be somewhat too low.", "Risk. ")

doc.add_heading("7. December prediction chart", 1)
figure(C.ROOT / "scorer_results" / "candidate_december.png",
       "Figure 5. December 2025 predictions for Lexington to Fort Wayne, 360 miles, Dry Van, 32,000 lb "
       "(chart produced by Spotter's score.py).")
d = pred["december"]
para("Only the date changes. The December file has no coordinates, market_index or quote_signal, so we fill them in as "
     "described in Section 3: coordinates from the city list, market_index from that day's average in the prediction "
     f"file, and no quote_signal. On Sep-Oct this way of filling in costs almost nothing ({money(SEL['december_mode']['mae'])} "
     f"vs {money(SEL['full']['mae'])}).")
para(f"Predictions range from {money(pred['december_range'][0])} to {money(pred['december_range'][1])}. The ups and downs "
     f"mostly follow the daily market_index (correlation {pred['corr_pred_vs_market_index']:.2f}): with market_index held "
     f"fixed, the line is almost flat ({money(pred['december_range_mi_constant'][0])}-"
     f"{money(pred['december_range_mi_constant'][1])}), and the small leftover movement comes from day-of-week terms in the "
     f"linear model. The weekly peaks are midweek days with a high market_index (for example {d[17]['date']}, a "
     f"{d[17]['dow']}). For comparison, the {eda['lexington_fort_wayne_dry_van']['n']} past Dry Van loads on this lane cost "
     f"{money(eda['lexington_fort_wayne_dry_van']['min'])}-{money(eda['lexington_fort_wayne_dry_van']['max'])} "
     f"(median {money(eda['lexington_fort_wayne_dry_van']['median'])}), with different weights, distances and dates.")
para("The model has never seen a December, so it cannot know about Christmas or year-end effects. Given the upward "
     "price drift in Section 6, the December level is more likely too low than too high.")

doc.add_heading("8. Limitations and next steps", 1)
for t in [
    "Rising prices are the main weakness. With more than one year of data (or an outside price index), a trend or seasonal "
    "effect could be tested over several cycles.",
    "March, June and September run a little above trend (quarter ends). Three cases are too few to model; December may do the same.",
    "Extreme prices (about 1.4% of loads) cause a large share of the dollar error. The data cannot tell us whether they are "
    "recording mistakes or genuinely unusual loads.",
    "quote_signal is left out because its behaviour keeps changing. If we knew how it is produced, it might become useful.",
    "Loads touching new cities rely on location and distance only, and their error is higher than for known cities.",
    "Next idea to test: a lane-level adjustment. The model leaves a small, consistent error on each lane (about 1%), "
    "which a carefully built lane feature might capture.",
]:
    bullet(t)

doc.add_heading("Appendix: how to reproduce", 1)
para("Python 3.12+. From the repository folder:")
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
