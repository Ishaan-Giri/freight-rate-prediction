# Freight Rate Prediction (Spotter ML Engineer assessment)

This repo predicts `posted_rate` for 12,000 Nov–Dec 2025 loads from labeled Jan–Oct 2025 data. It also fills the fixed December chart for Lexington → Fort Wayne.

**Deliverables**
- [`validation_predictions.csv`](validation_predictions.csv): `load_id,predicted_rate`, 12,000 rows
- [`reports/freight_rate_report.docx`](reports/freight_rate_report.docx): validation and split approach, results, December chart
- [`scorer_results/candidate_december.png`](scorer_results/candidate_december.png): produced by the provided `score.py`

## Results at a glance
| | All-row MAE | Notes |
|---|---|---|
| Dev folds (D1: Jan–Apr→May–Jun, D2: Jan–Jun→Jul–Aug) | $108.85 | Used for all model selection |
| Held-out Sep–Oct (frozen pipeline, run once) | $163.76 | Median APE 4.4%; about equal to a lane-median reference ($162.43) |

The held-out result is worse than development. The cause is an upward rate-level drift (~4–5% by Sep–Oct) that `market_index` doesn't explain. Section 6 of the report covers the diagnosis, the dev-fold-only attempts to fix it (a linear trend and recency weighting, neither adopted under the pre-declared rules), and the resulting risk for Nov–Dec.

## Approach (short version)
- **Time-ordered, expanding-window validation.** The target period is the future, and `market_index` is shared by every load on a date, so random K-fold would leak.
- **Pre-declared selection rule.** The primary metric is all-row dollar MAE on the dev folds. Secondary checks: December-mode MAE, August MAE, unseen-city MAE, worst fold. Sep–Oct stays out of all fitting and tuning. `configs/final.yaml` is committed before `final_eval.py` will run.
- **Cold-start simulation.** 8 cities appear only in validation (12.1% of loads). Holding out 4 cities from both ends of training reproduces that exposure.
- **December mode.** The chart file lacks coordinates, `market_index` and `quote_signal`. Every fold is also scored with those columns rebuilt the same way as for December, from features only.
- **`quote_signal` is excluded.** Its relationship to rate flips sign month to month, and it shows no regime in Aug/Nov/Dec. It improves the pooled dev score but degrades the regime-matched month and breaks December-mode predictions.
- **Selected model.** A geometric blend: 0.75 × LightGBM (Huber loss on log rate, large-residual rows filtered from training, 15 leaves) plus 0.25 × log-linear regression. All preprocessing is fitted inside each training fold.

## Setup
Python 3.10+.
```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Reproduce
```bash
.venv/bin/python -m pytest -q                      # leakage / split / preprocessing tests
.venv/bin/python scripts/eda.py                    # data-quality stats + figures -> reports/
.venv/bin/python scripts/develop.py                # dev-fold model selection (~15 min) -> configs/final.yaml
.venv/bin/python scripts/develop_v2.py             # dev round 2 (trend / recency), same rules
.venv/bin/python scripts/final_eval.py             # one-shot Sep-Oct evaluation (needs committed config)
.venv/bin/python scripts/predict.py                # production fit -> validation_predictions.csv + December file
.venv/bin/python score.py --predictions validation_predictions.csv \
    --december-predictions data/december_chart_inputs.csv
.venv/bin/python scripts/diagnose_drift.py         # post-evaluation drift diagnosis
.venv/bin/python scripts/build_report.py           # DOCX report from the JSON outputs
```
To regenerate only the submission files from the committed frozen config, run `predict.py` and then `score.py`, which takes about a minute. Outputs are deterministic: fixed seeds, `deterministic=True`, fixed thread count.

## Layout
```
data/                     supplied CSVs (december_chart_inputs.csv has predicted_rate filled)
src/freight/              config, data checks, fold-fitted preprocessing, features,
                          quote_signal regime features, models, evaluation (folds, cold start, December mode)
scripts/                  eda, develop, develop_v2, final_eval, predict, diagnose_drift, build_report
configs/final.yaml        frozen pipeline (final_v1.yaml = round-1 freeze; identical pipeline)
reports/                  JSON results, logs, figures, DOCX report
tests/                    unit tests (no target in reconstruction, time ordering, cold-start, fold-fitted stats)
score.py                  provided scorer (unchanged)
docs/assignment/          provided README (instructions PDF not redistributed)
```
