"""Paths, constants and the fixed evaluation design."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
CONFIGS = ROOT / "configs"
MODELS = ROOT / "models"

TRAIN_FILE = DATA / "train_test.csv"
VALIDATION_FILE = DATA / "validation.csv"
TEMPLATE_FILE = DATA / "validation_predictions_template.csv"
DECEMBER_FILE = DATA / "december_chart_inputs.csv"
PREDICTIONS_FILE = ROOT / "validation_predictions.csv"
FINAL_CONFIG = CONFIGS / "final.yaml"

TARGET = "posted_rate"
SEED = 42
NUM_THREADS = 8
WEIGHT_CAP = 47_500.0
EQUIPMENT = ["Dry Van", "Flatbed", "Reefer"]

# Columns absent from the December chart inputs; rebuilt by the December-mode procedure.
DECEMBER_MISSING = [
    "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon", "market_index", "quote_signal",
]


@dataclass(frozen=True)
class Fold:
    name: str
    train_start: str  # inclusive
    test_start: str   # inclusive; training data is strictly before this date
    test_end: str     # inclusive


# Development folds: the ONLY data used for tuning / selection.
DEV_FOLDS = [
    Fold("D1", "2025-01-01", "2025-05-01", "2025-06-30"),
    Fold("D2", "2025-01-01", "2025-07-01", "2025-08-31"),
]
# Held-out evaluation period (examined during EDA, excluded from all fitting/tuning).
FINAL_FOLD = Fold("FINAL", "2025-01-01", "2025-09-01", "2025-10-31")

# Cold-start simulation: 4 held-out cities ~= 12% load exposure, matching validation (12.1%).
COLD_START_K = 4
COLD_START_STRESS_K = 8
COLD_START_SEEDS_DEV = [0, 1, 2]
COLD_START_SEEDS_FINAL = [0, 1, 2, 3, 4]
