"""Project-wide constants: class definitions, colours, canonical paths."""

from __future__ import annotations

from pathlib import Path

# Repository root (this file lives at <root>/src/dentalscan/constants.py)
ROOT = Path(__file__).resolve().parents[2]

CONFIG_DIR = ROOT / "configs"
RESULTS_DIR = ROOT / "results"
FIGURE_DIR = RESULTS_DIR / "figures"
RUNS_DIR = ROOT / "runs"
REPORT_DIR = ROOT / "report"

# Class order MUST match the integer ids in the YOLO label files.
CLASS_NAMES: list[str] = [
    "Caries",
    "Infection",
    "Impacted",
    "BDC/BDR",
    "Fractured",
    "Healthy",
]
NUM_CLASSES = len(CLASS_NAMES)

CLASS_ID = {name: i for i, name in enumerate(CLASS_NAMES)}

# Short descriptions used in the model card and the report.
CLASS_DESCRIPTIONS: dict[str, str] = {
    "Caries": "Carious lesion (tooth decay) visible as radiolucency in enamel/dentine.",
    "Infection": "Periapical or periodontal infection; radiolucency at the root apex.",
    "Impacted": "Tooth that has failed to erupt into normal occlusal position.",
    "BDC/BDR": "Bone defect, coronal or root region.",
    "Fractured": "Crown or root fracture.",
    "Healthy": "Normal tooth region with no visible pathology.",
}

# Okabe-Ito categorical palette. Verified with the six-check validator: passes
# the lightness band, chroma floor and normal-vision separation on all pairs;
# the worst colour-vision-deficient pair sits at dE 7.6, which is legal only
# with a secondary encoding, so every figure in this repo also carries hatch
# patterns (bars) or distinct markers (lines) and direct labels.
CLASS_COLORS_HEX: dict[str, str] = {
    "Caries": "#0072B2",
    "Infection": "#E69F00",
    "Impacted": "#009E73",
    "BDC/BDR": "#D55E00",
    "Fractured": "#CC79A7",
    "Healthy": "#56B4E9",
}

# Secondary (non-colour) encoding, required by the palette's CVD margin.
CLASS_HATCH: dict[str, str] = {
    "Caries": "",
    "Infection": "///",
    "Impacted": "...",
    "BDC/BDR": "\\\\",
    "Fractured": "xxx",
    "Healthy": "---",
}

# Model families compared head to head.
MODEL_COLORS_HEX: dict[str, str] = {
    "YOLOv8n": "#0072B2",
    "YOLOv10s": "#E69F00",
    "YOLOv12m": "#009E73",
}
MODEL_MARKERS: dict[str, str] = {
    "YOLOv8n": "o",
    "YOLOv10s": "s",
    "YOLOv12m": "^",
}

# Detection thresholds used consistently across evaluation and the app.
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.7

# Random seeds used for the multi-seed protocol.
DEFAULT_SEEDS = (0, 1, 2, 3, 4)
