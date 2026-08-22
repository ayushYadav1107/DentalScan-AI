"""DentalScan AI - dental condition detection on panoramic radiographs.

Two screens. Before a study is loaded, the claim and the drop target sit side by
side over the per-class recall the model was actually measured at - the caveat
leads rather than trails. Once a study is loaded the page is a viewer: the
radiograph and the two controls that govern it on the left, an evidence rail on
the right. The radiograph is the brightest thing on screen throughout, the
chrome is recessive, and every number the interface shows is one the harness in
this repository can regenerate.

Three properties worth knowing about before reading the code:

* **Inference runs once per image, at conf=0.01.** Changing the confidence
  threshold or toggling a class re-filters that cached result rather than
  re-running the network, so the controls respond instantly and the displayed
  numbers always come from a single forward pass.
* **Class identity is never carried by colour alone.** Every box has a text
  label, every findings row repeats the class name, and ``Healthy`` - a
  background label, not a finding, and 58% of all boxes in the dataset - is
  drawn as a recessive grey outline so it cannot dominate the pathology.
* **Measured recall travels with every detection.** A model that finds half of
  all infections should not present an infection the same way it presents an
  impacted tooth, and this interface does not.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    import cv2
except ImportError:  # some hosted environments ship a broken OpenCV wheel
    os.system("pip uninstall -y opencv-python opencv-python-headless")
    os.system("pip install opencv-python-headless")
    import cv2

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from ultralytics import YOLO

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR / "src"))

try:
    from dentalscan.explain import EigenCAM, overlay_cam, pointing_energy
    EXPLAIN_AVAILABLE = True
    EXPLAIN_ERROR = ""
except Exception as exc:  # the viewer must still run without the research extras
    EXPLAIN_AVAILABLE = False
    EXPLAIN_ERROR = str(exc)

st.set_page_config(
    page_title="DentalScan · OPG viewer",
    page_icon="◧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------- #
# Classes
# --------------------------------------------------------------------------- #

CLASS_NAMES: dict[int, str] = {
    0: "Caries",
    1: "Infection",
    2: "Impacted",
    3: "BDC/BDR",
    4: "Fractured",
    5: "Healthy",
}
HEALTHY_ID = 5

CLASS_DESCRIPTIONS: dict[int, str] = {
    0: "Carious lesion - radiolucency in enamel or dentine",
    1: "Periapical or periodontal infection",
    2: "Tooth that failed to erupt into normal occlusion",
    3: "Bone defect, coronal or root region",
    4: "Crown or root fracture",
    5: "Normal tooth region - a background label, not a diagnosis",
}

# Five pathology hues, chosen by search inside the OKLCH lightness band for a
# dark surface and verified against the six-check palette validator: lightness
# band, chroma floor, all-pairs colour-vision-deficient separation (worst pair
# dE 9.6), all-pairs normal-vision separation (worst pair 17.1) and contrast
# against the surface all pass. Healthy is deliberately *not* one of them - it
# is a non-finding and is drawn in recessive grey.
CLASS_COLORS_HEX: dict[int, str] = {
    0: "#E74F00",   # Caries
    1: "#EB429D",   # Infection
    2: "#805EF9",   # Impacted
    3: "#009AD8",   # BDC/BDR
    4: "#078D64",   # Fractured
    5: "#7A828C",   # Healthy - neutral, recessive
}


def hex_to_bgr(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    r, g, b = (int(value[i:i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


CLASS_COLORS_BGR = {k: hex_to_bgr(v) for k, v in CLASS_COLORS_HEX.items()}

# Recall on the 23-image validation split at confidence 0.25, with the number of
# ground-truth instances behind each figure. Shown next to every detection so the
# interface never presents a finding as more trustworthy than its evidence.
# Regenerate with `python evaluate.py per-class`.
CLASS_RELIABILITY: dict[int, dict] = {
    0: {"recall": 0.79, "support": 14},
    1: {"recall": 0.50, "support": 4},
    2: {"recall": 1.00, "support": 18},
    3: {"recall": 1.00, "support": 3},
    4: {"recall": 0.89, "support": 9},
    5: {"recall": 0.64, "support": 67},
}
LOW_SUPPORT = 10


def reliability_grade(class_id: int) -> tuple[str, str, str]:
    """(grade, short label, tooltip) for how far a class's recall can be trusted."""
    info = CLASS_RELIABILITY.get(class_id)
    if not info:
        return ("unknown", "no data", "This class was not measured.")
    recall, support = info["recall"], info["support"]
    if support < LOW_SUPPORT:
        return ("unmeasured", f"n={support}",
                f"Only {support} validation instances. The recall of {recall:.2f} "
                f"is arithmetic, not evidence - treat this class as unmeasured.")
    if recall < 0.70:
        return ("weak", f"recall {recall:.2f}",
                f"Measured recall {recall:.2f} on {support} instances. Roughly "
                f"{(1 - recall) * 100:.0f}% of true cases are missed entirely.")
    return ("ok", f"recall {recall:.2f}",
            f"Measured recall {recall:.2f} on {support} instances.")


FINETUNED_PATH = APP_DIR / "runs" / "detect" / "Yolo_10s_train" / "weights" / "best.pt"
FALLBACK_MODEL = "yolov10s.pt"
MAX_UPLOAD_MB = 25
CACHE_CONF = 0.01          # inference floor; the UI threshold filters this cache
PATHOLOGY_IDS = [i for i in CLASS_NAMES if i != HEALTHY_ID]

# Marks. Inline SVG rather than an icon font: nothing to fetch, no flash of a
# missing glyph, and the stroke inherits currentColor.
REPO_URL = "https://github.com/ayushYadav1107/DentalScan-AI"

TOOTH_SVG = (
    '<svg width="21" height="21" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
    'stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
    '<path d="M12 4.2c-1.7-1.3-3.7-1.6-5.2-.7C5 4.5 4.3 6.7 4.7 9c.3 1.8.7 2.9 1 4.6.3 1.9.4 3.6.8 5 .3 1 .8 1.7 1.5 1.7.9 0 1.2-1 1.5-2.5.3-1.5.5-2.7 1.5-2.7s1.2 1.2 1.5 2.7c.3 1.5.6 2.5 1.5 2.5.7 0 1.2-.7 1.5-1.7.4-1.4.5-3.1.8-5 .3-1.7.7-2.8 1-4.6.4-2.3-.3-4.5-2.1-5.5-1.5-.9-3.5-.6-5.2.7Z"/></svg>')

CHECK_SVG = (
    '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#4FBF8B" '
    'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
    '<circle cx="12" cy="12" r="9"/><path d="m8.5 12 2.4 2.4 4.6-4.8"/></svg>')



# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

def file_digest(path: Path, length: int = 12) -> str:
    """Short SHA-256 of a file, so an exported record names exact weights."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:length]


@st.cache_resource(show_spinner=False)
def load_model() -> dict:
    """Load the fine-tuned detector, falling back to the base COCO model.

    The fallback exists so the interface still renders and can explain itself
    when the weights are missing - but it predicts COCO classes, so the result
    is nonsense for dental use and the UI says so in the strongest terms it has.
    """
    if FINETUNED_PATH.exists():
        try:
            return {
                "model": YOLO(str(FINETUNED_PATH)),
                "path": str(FINETUNED_PATH),
                "name": FINETUNED_PATH.name,
                "sha": file_digest(FINETUNED_PATH),
                "finetuned": True,
                "error": "",
            }
        except Exception as exc:
            return {"model": None, "path": str(FINETUNED_PATH), "name": FINETUNED_PATH.name,
                    "sha": "", "finetuned": False, "error": str(exc)}
    try:
        return {"model": YOLO(FALLBACK_MODEL), "path": FALLBACK_MODEL,
                "name": FALLBACK_MODEL, "sha": "", "finetuned": False, "error": ""}
    except Exception as exc:
        return {"model": None, "path": FALLBACK_MODEL, "name": FALLBACK_MODEL,
                "sha": "", "finetuned": False, "error": str(exc)}


# --------------------------------------------------------------------------- #
# Inference and caching
# --------------------------------------------------------------------------- #

def run_inference(model, image_bgr: np.ndarray, iou: float, imgsz: int) -> tuple[list[dict], float]:
    """One forward pass at a low confidence floor.

    Everything the interface does afterwards - thresholding, class filtering,
    the findings table, the export - reads this list. The model is never re-run
    for a slider move, which is what makes the controls feel immediate and what
    guarantees the exported record matches what is on screen.
    """
    started = time.perf_counter()
    result = model.predict(image_bgr, conf=CACHE_CONF, iou=iou,
                           imgsz=imgsz, verbose=False)[0]
    elapsed_ms = (time.perf_counter() - started) * 1000

    detections: list[dict] = []
    if result.boxes is not None and len(result.boxes):
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        for box, score, class_id in zip(boxes, scores, classes):
            if int(class_id) not in CLASS_NAMES:
                continue  # base COCO fallback emits classes we cannot name
            x1, y1, x2, y2 = (int(round(v)) for v in box)
            detections.append({
                "class_id": int(class_id),
                "class": CLASS_NAMES[int(class_id)],
                "confidence": float(score),
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "width": x2 - x1, "height": y2 - y1,
                "area_px": (x2 - x1) * (y2 - y1),
            })
    detections.sort(key=lambda d: -d["confidence"])
    return detections, elapsed_ms


def visible_detections(detections: list[dict], conf: float,
                       enabled: set[int]) -> list[dict]:
    """Filter the cache to what the current controls ask for, and number it.

    The index assigned here is the identity a finding keeps everywhere: on the
    box label, in the findings table, in the CSV and in the JSON record.
    """
    kept = [d for d in detections
            if d["confidence"] >= conf and d["class_id"] in enabled]
    for i, detection in enumerate(kept, start=1):
        detection["index"] = i
    return kept


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #

def draw_overlay(
    image_bgr: np.ndarray,
    detections: list[dict],
    show_labels: bool = True,
    box_opacity: float = 1.0,
    fill: bool = True,
) -> np.ndarray:
    """Render detections onto a copy of the radiograph.

    Design rules, in order of importance:

    * ``Healthy`` is a background label. It is drawn one pixel thick, in grey,
      unlabelled, so that pathology reads first. On this dataset it is 58% of
      all boxes; drawn like everything else it would bury the findings.
    * Every pathology box carries its index and class name as text, so colour is
      a redundant encoding rather than the only one.
    * Boxes get a faint interior wash. A 30x20 pixel carious lesion on a
      1935-pixel panoramic is nearly invisible as an outline alone.
    * The label chip flips inside the box when it would otherwise leave frame.
    """
    canvas = image_bgr.copy()
    height, width = canvas.shape[:2]
    scale = max(width / 1400.0, 0.7)
    thickness = max(int(round(2 * scale)), 2)
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.46 * scale
    overlay = canvas.copy()
    placed: list[tuple[int, int, int, int]] = []   # chip rects, for collision avoidance

    # Non-findings first, so pathology always draws on top of them.
    ordered = sorted(detections, key=lambda d: (d["class_id"] != HEALTHY_ID,
                                                d["confidence"]))

    for detection in ordered:
        class_id = detection["class_id"]
        colour = CLASS_COLORS_BGR[class_id]
        x1, y1, x2, y2 = detection["x1"], detection["y1"], detection["x2"], detection["y2"]

        if class_id == HEALTHY_ID:
            cv2.rectangle(overlay, (x1, y1), (x2, y2), colour,
                          max(thickness - 1, 1), cv2.LINE_AA)
            continue

        if fill:
            wash = overlay.copy()
            cv2.rectangle(wash, (x1, y1), (x2, y2), colour, -1)
            cv2.addWeighted(wash, 0.10, overlay, 0.90, 0, overlay)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), colour, thickness, cv2.LINE_AA)

        if not show_labels:
            continue

        # OpenCV's Hershey fonts are ASCII-only; anything else renders as '??'.
        label = f"{detection.get('index', '')}  {detection['class']}  {detection['confidence']:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, 1)
        pad_x, pad_y = int(7 * scale), int(5 * scale)
        chip_w, chip_h = text_w + 2 * pad_x, text_h + baseline + 2 * pad_y
        chip_x = min(max(x1, 0), max(width - chip_w, 0))

        # Adjacent teeth put boxes within a few pixels of each other, and chips
        # drawn naively then overwrite each other's text. Try stacked positions
        # above and below the box until one is clear; only overlap as a last
        # resort, and draw a leader line whenever the chip has moved off the box.
        candidates = [y1 - chip_h - int(round(k * chip_h * 1.05)) for k in range(3)]
        candidates += [y2 + int(round(k * chip_h * 1.05)) for k in range(3)]
        candidates.append(y1)
        chip_y = None
        for candidate in candidates:
            if candidate < 0 or candidate + chip_h > height:
                continue
            rect = (chip_x, candidate, chip_x + chip_w, candidate + chip_h)
            if not any(rect[0] < p[2] and p[0] < rect[2]
                       and rect[1] < p[3] and p[1] < rect[3] for p in placed):
                chip_y = candidate
                break
        if chip_y is None:
            chip_y = max(0, min(y1 - chip_h, height - chip_h))
        placed.append((chip_x, chip_y, chip_x + chip_w, chip_y + chip_h))

        anchor_y = y1 if chip_y < y1 else y2
        if abs((chip_y + chip_h if chip_y < y1 else chip_y) - anchor_y) > 3:
            cv2.line(overlay, (chip_x + chip_w // 2,
                               chip_y + chip_h if chip_y < y1 else chip_y),
                     (min(max(chip_x + chip_w // 2, x1), x2), anchor_y),
                     colour, max(thickness - 1, 1), cv2.LINE_AA)

        cv2.rectangle(overlay, (chip_x, chip_y), (chip_x + chip_w, chip_y + chip_h),
                      colour, -1)
        cv2.putText(overlay, label, (chip_x + pad_x, chip_y + chip_h - pad_y - baseline // 2),
                    font, font_scale, (12, 12, 12), 1, cv2.LINE_AA)

    if box_opacity >= 0.999:
        return overlay
    cv2.addWeighted(overlay, box_opacity, canvas, 1 - box_opacity, 0, canvas)
    return canvas


# --------------------------------------------------------------------------- #
# Encoding helpers
# --------------------------------------------------------------------------- #

def encode_jpeg(image_bgr: np.ndarray, max_width: int = 1600, quality: int = 88) -> str:
    """BGR array -> base64 JPEG, downscaled for display.

    The viewport does not need full resolution - it needs to load fast and pan
    smoothly. Full resolution is what the PNG export is for.
    """
    if image_bgr.shape[1] > max_width:
        scale = max_width / image_bgr.shape[1]
        image_bgr = cv2.resize(
            image_bgr, (max_width, int(round(image_bgr.shape[0] * scale))),
            interpolation=cv2.INTER_AREA)
    ok, buffer = cv2.imencode(".jpg", image_bgr,
                              [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return base64.b64encode(buffer.tobytes()).decode("ascii")


def encode_png(image_bgr: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(".png", image_bgr)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return buffer.tobytes()



# --------------------------------------------------------------------------- #
# Viewport
# --------------------------------------------------------------------------- #
#
# Boxes are drawn as SVG *over* the radiograph rather than burned into it. That
# buys three things a rasterised overlay cannot:
#
#   * they stay razor sharp at 12x zoom, where a baked-in 2px stroke turns to
#     mush and OpenCV's bitmap font is unreadable;
#   * strokes and badges counter-scale, so a box outline is 2 screen pixels
#     whether you are looking at the whole jaw or one molar;
#   * every box is a live element - it can animate in, respond to hover, and
#     carry a tooltip, none of which a flat JPEG can do.
#
# The rasterised version in draw_overlay() is still what the PNG export uses,
# because an exported file has to carry its annotations inside the pixels.

VIEWPORT_TEMPLATE = """
<style>
  *, *::before, *::after { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; background: transparent;
               font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }

  #shell { position: relative; height: __HEIGHT__px; border-radius: 20px;
           overflow: hidden; background:
             radial-gradient(120% 90% at 50% 0%, #1B1E2E 0%, #12141F 60%, #0F1018 100%);
           border: 1px solid rgba(255,255,255,.08);
           box-shadow: inset 0 1px 0 rgba(255,255,255,.05),
                       0 24px 60px -30px rgba(0,0,0,.9); }
  #shell::after { content: ''; position: absolute; inset: 0; pointer-events: none;
                  border-radius: 20px;
                  background: radial-gradient(80% 60% at 50% 120%,
                              rgba(145,132,217,.13), transparent 70%); }

  #stage { position: absolute; inset: 0; overflow: hidden; cursor: grab;
           opacity: 0; animation: rise .6s cubic-bezier(.22,1,.36,1) .05s forwards; }
  #stage.dragging { cursor: grabbing; }
  @keyframes rise { from { opacity: 0; transform: translateY(10px) scale(.99); }
                    to   { opacity: 1; transform: none; } }

  /* Centring is done by the transform alone - flex-centring the pane as well
     would offset every pan by the flex origin and make the image drift. */
  #pane { position: absolute; top: 0; left: 0; transform-origin: 0 0;
          will-change: transform; line-height: 0; }
  #pane img { display: block; max-width: none; user-select: none; -webkit-user-drag: none; }
  #cam { position: absolute; top: 0; left: 0; transition: opacity .35s ease; }
  #svg { position: absolute; top: 0; left: 0; overflow: visible; }

  .bx { fill: transparent; stroke-width: 2.25; vector-effect: non-scaling-stroke;
        transition: stroke-width .18s ease, filter .18s ease;
        stroke-dasharray: var(--len); stroke-dashoffset: var(--len);
        animation: draw .75s cubic-bezier(.22,1,.36,1) forwards; }
  @keyframes draw { to { stroke-dashoffset: 0; } }
  .wash { stroke: none; opacity: 0; animation: wash .5s ease forwards; }
  @keyframes wash { to { opacity: .12; } }
  .hit { fill: transparent; stroke: none; cursor: pointer; }
  g.det.on .bx { stroke-width: 4; filter: drop-shadow(0 0 7px currentColor); }
  g.det.on .wash { opacity: .26; }
  g.det.dim { opacity: .28; transition: opacity .2s ease; }

  /* Badges and chips live outside the transformed pane and are positioned each
     frame in screen space, so they never scale with the image. */
  #marks { position: absolute; inset: 0; pointer-events: none; }
  .badge { position: absolute; transform: translate(-50%,-50%) scale(.4);
           width: 26px; height: 26px; border-radius: 50%;
           display: grid; place-items: center;
           font-size: 13px; font-weight: 700; font-variant-numeric: tabular-nums;
           background: rgba(15,16,24,.94); border: 2px solid currentColor;
           box-shadow: 0 3px 10px rgba(0,0,0,.7);
           pointer-events: auto; cursor: pointer; opacity: 0;
           animation: pop .45s cubic-bezier(.34,1.56,.64,1) forwards;
           transition: transform .18s cubic-bezier(.34,1.56,.64,1),
                       background .18s ease, color .18s ease; }
  .badge:hover, .badge.on { background: currentColor; }
  .badge:hover span, .badge.on span { color: #0F1018; }
  @keyframes pop { to { opacity: 1; transform: translate(-50%,-50%) scale(1); } }
  .badge:hover, .badge.on { transform: translate(-50%,-50%) scale(1.25); }
  .chip { position: absolute; transform: translate(0,-50%);
          display: flex; align-items: center; gap: 7px; white-space: nowrap;
          padding: 5px 11px; border-radius: 9px; font-size: 13px; font-weight: 600;
          color: #0F1018; box-shadow: 0 3px 12px rgba(0,0,0,.6);
          opacity: 0; animation: fadein .5s ease forwards; }
  .chip .c { font-weight: 500; opacity: .78; font-variant-numeric: tabular-nums; }
  @keyframes fadein { to { opacity: 1; } }

  #tip { position: absolute; z-index: 9; pointer-events: none; opacity: 0;
         transform: translate(-50%, -100%) translateY(-14px);
         transition: opacity .16s ease; min-width: 168px;
         padding: 11px 13px; border-radius: 12px; font-size: 13px; line-height: 1.5;
         background: rgba(35,37,50,.94); backdrop-filter: blur(16px) saturate(1.5);
         border: 1px solid rgba(255,255,255,.12);
         box-shadow: 0 18px 44px -14px rgba(0,0,0,.9); color: #F3F5FE; }
  #tip.on { opacity: 1; }
  #tip .t { display: flex; align-items: center; gap: 8px; font-weight: 650;
            font-size: 14px; margin-bottom: 5px; }
  #tip .sw { width: 10px; height: 10px; border-radius: 3px; }
  #tip .m { color: #9397AB; font-size: 12.5px; }
  #tip .m b { color: #F3F5FE; font-weight: 600; font-variant-numeric: tabular-nums; }

  #bar { position: absolute; left: 50%; bottom: 18px; transform: translateX(-50%) translateY(6px);
         display: flex; align-items: center; gap: 5px; padding: 8px 10px;
         border-radius: 15px; background: rgba(22,24,38,.86);
         border: 1px solid rgba(255,255,255,.1);
         backdrop-filter: blur(18px) saturate(1.6);
         box-shadow: 0 16px 40px -14px rgba(0,0,0,.9);
         opacity: 0; transition: opacity .25s ease, transform .25s cubic-bezier(.22,1,.36,1); }
  #shell:hover #bar, #bar:focus-within { opacity: 1; transform: translateX(-50%) translateY(0); }

  .btn { appearance: none; border: 0; background: transparent; color: #B2B6CA;
         font-size: 13.5px; font-weight: 500; padding: 7px 12px; border-radius: 10px;
         cursor: pointer; line-height: 1.4; font-variant-numeric: tabular-nums;
         transition: background .16s ease, color .16s ease, transform .16s ease; }
  .btn:hover { background: rgba(255,255,255,.1); color: #FFF; transform: translateY(-1px); }
  .btn:active { transform: translateY(0) scale(.96); }
  .btn:focus-visible { outline: 2px solid #9184D9; outline-offset: 2px; }
  .btn.icon { min-width: 32px; text-align: center; font-size: 16px; padding: 5px 9px; }
  #zoomval { font-size: 13px; color: #9397AB; min-width: 50px; text-align: center;
             font-variant-numeric: tabular-nums; }
  .sep { width: 1px; height: 18px; background: rgba(255,255,255,.11); margin: 0 5px; }
  .lbl { font-size: 12px; color: #75798C; letter-spacing: .3px; padding-left: 5px; }
  input[type=range] { -webkit-appearance: none; appearance: none; height: 4px; width: 96px;
                      border-radius: 3px; background: rgba(255,255,255,.18); cursor: pointer; }
  input[type=range]::-webkit-slider-thumb { -webkit-appearance: none; width: 14px; height: 14px;
      border-radius: 50%; background: #F3F5FE; box-shadow: 0 1px 4px rgba(0,0,0,.7); }
  input[type=range]::-moz-range-thumb { width: 14px; height: 14px; border: 0;
      border-radius: 50%; background: #F3F5FE; }

  #hint { position: absolute; top: 16px; left: 50%; transform: translateX(-50%);
          font-size: 12.5px; color: #9397AB; padding: 7px 15px; border-radius: 100px;
          background: rgba(22,24,38,.86); border: 1px solid rgba(255,255,255,.08);
          backdrop-filter: blur(12px); pointer-events: none;
          animation: hint 5s ease forwards; }
  @keyframes hint { 0% { opacity: 0; transform: translateX(-50%) translateY(-6px); }
                    14% { opacity: 1; transform: translateX(-50%) translateY(0); }
                    72% { opacity: 1; } 100% { opacity: 0; } }

  #empty { position: absolute; inset: 0; display: grid; place-items: center;
           font-size: 14px; color: #75798C; pointer-events: none; }

  @media (prefers-reduced-motion: reduce) {
    * { animation-duration: .01ms !important; transition-duration: .01ms !important; }
    #bar { opacity: 1; }
  }
</style>

<div id="shell">
  <div id="stage" tabindex="0" aria-label="Radiograph viewport">
    <div id="pane">
      <img id="base" src="data:image/jpeg;base64,__BASE__" alt="Radiograph" draggable="false">
      __CAMTAG__
      <svg id="svg" width="__IW__" height="__IH__" viewBox="0 0 __IW__ __IH__"></svg>
    </div>
    <div id="marks"></div>
    <div id="tip"></div>
  </div>
  <div id="hint">scroll to zoom · drag to pan · click a marker to inspect</div>
  <div id="bar">
    <button class="btn icon" id="out" title="Zoom out (−)" aria-label="Zoom out">&minus;</button>
    <span id="zoomval">100%</span>
    <button class="btn icon" id="in" title="Zoom in (+)" aria-label="Zoom in">&plus;</button>
    <div class="sep"></div>
    <button class="btn" id="fit" title="Fit to frame (0)">Fit</button>
    <button class="btn" id="one" title="Actual pixels (1)">1:1</button>
    <div class="sep"></div>
    <span class="lbl">Overlay</span>
    <input id="mix" type="range" min="0" max="100" value="100" aria-label="Overlay opacity">
    <button class="btn" id="hold" title="Hold to see the radiograph underneath">Compare</button>
  </div>
</div>

<script>
(function () {
  const DETS = __DETS__, SHOW_CHIPS = __CHIPS__, IS_CAM = __ISCAM__;
  const NS = 'http://www.w3.org/2000/svg';
  const stage = document.getElementById('stage'), pane = document.getElementById('pane');
  const base = document.getElementById('base'), svg = document.getElementById('svg');
  const marks = document.getElementById('marks'), tip = document.getElementById('tip');
  const zoomval = document.getElementById('zoomval'), mix = document.getElementById('mix');
  const cam = document.getElementById('cam');

  let scale = 1, tx = 0, ty = 0, active = null;
  const MIN = 0.08, MAX = 14;
  const groups = [], badges = [], chips = [];

  // ── build the overlay ───────────────────────────────────────────────────
  DETS.forEach(function (d, i) {
    const w = d.x2 - d.x1, h = d.y2 - d.y1;
    const g = document.createElementNS(NS, 'g');
    g.setAttribute('class', 'det');
    g.style.color = d.color;

    const wash = document.createElementNS(NS, 'rect');
    wash.setAttribute('class', 'wash');
    wash.setAttribute('fill', d.color);
    const bx = document.createElementNS(NS, 'rect');
    bx.setAttribute('class', 'bx');
    bx.setAttribute('stroke', d.color);
    bx.style.setProperty('--len', 2 * (w + h));
    bx.style.animationDelay = (0.05 + i * 0.045) + 's';
    wash.style.animationDelay = (0.28 + i * 0.045) + 's';
    const hit = document.createElementNS(NS, 'rect');
    hit.setAttribute('class', 'hit');

    [wash, bx, hit].forEach(function (r) {
      r.setAttribute('x', d.x1); r.setAttribute('y', d.y1);
      r.setAttribute('width', w); r.setAttribute('height', h);
      r.setAttribute('rx', Math.min(7, w / 5, h / 5));
      g.appendChild(r);
    });
    svg.appendChild(g);
    groups.push(g);

    const badge = document.createElement('div');
    badge.className = 'badge';
    badge.innerHTML = '<span>' + d.index + '</span>';
    badge.style.color = d.color;
    badge.style.animationDelay = (0.3 + i * 0.045) + 's';
    marks.appendChild(badge);
    badges.push(badge);

    let chip = null;
    if (SHOW_CHIPS) {
      chip = document.createElement('div');
      chip.className = 'chip';
      chip.style.background = d.color;
      chip.innerHTML = '<span>' + d.name + '</span><span class="c">' +
                       d.conf.toFixed(2) + '</span>';
      chip.style.animationDelay = (0.34 + i * 0.045) + 's';
      marks.appendChild(chip);
    }
    chips.push(chip);

    function enter() {
      active = i;
      groups.forEach(function (o, j) { o.classList.toggle('dim', j !== i); });
      g.classList.add('on'); badge.classList.add('on');
      tip.innerHTML =
        '<div class="t"><span class="sw" style="background:' + d.color + '"></span>' +
        d.name + '</div>' +
        '<div class="m">confidence <b>' + d.conf.toFixed(2) + '</b> · ' +
        (d.x2 - d.x1) + '×' + (d.y2 - d.y1) + ' px</div>' +
        '<div class="m">' + d.note + '</div>';
      tip.classList.add('on');
      place();
    }
    function leave() {
      active = null;
      groups.forEach(function (o) { o.classList.remove('dim', 'on'); });
      badge.classList.remove('on');
      tip.classList.remove('on');
    }
    [hit, badge].forEach(function (el) {
      el.addEventListener('pointerenter', enter);
      el.addEventListener('pointerleave', leave);
    });
    // Click a marker to fly to it - the fastest way to inspect a small lesion.
    badge.addEventListener('click', function (e) { e.stopPropagation(); focus(d); });
    hit.addEventListener('click', function (e) { e.stopPropagation(); focus(d); });
  });

  if (!DETS.length && !IS_CAM) {
    const e = document.createElement('div');
    e.id = 'empty'; e.textContent = 'No findings at this threshold';
    stage.appendChild(e);
  }

  // ── transform ───────────────────────────────────────────────────────────
  function place() {
    pane.style.transform = 'translate(' + tx + 'px,' + ty + 'px) scale(' + scale + ')';
    zoomval.textContent = Math.round(scale * 100) + '%';
    DETS.forEach(function (d, i) {
      const bx = tx + d.x1 * scale, by = ty + d.y1 * scale;
      badges[i].style.left = bx + 'px';
      badges[i].style.top = by + 'px';
      if (chips[i]) {
        chips[i].style.left = (bx + 15) + 'px';
        chips[i].style.top = (by - 2) + 'px';
      }
    });
    if (active !== null) {
      const d = DETS[active];
      tip.style.left = (tx + (d.x1 + d.x2) / 2 * scale) + 'px';
      tip.style.top = (ty + d.y1 * scale) + 'px';
    }
  }
  function fit(animate) {
    const sw = stage.clientWidth, sh = stage.clientHeight;
    const iw = base.naturalWidth, ih = base.naturalHeight;
    if (!iw || !ih) return;
    go(Math.min(sw / iw, sh / ih) * 0.94,
       null, null, animate);
    tx = (sw - iw * scale) / 2; ty = (sh - ih * scale) / 2;
    place();
  }
  function go(next, cx, cy, animate) {
    next = Math.min(MAX, Math.max(MIN, next));
    if (cx !== null && cx !== undefined) {
      const k = next / scale;
      tx = cx - (cx - tx) * k; ty = cy - (cy - ty) * k;
    }
    scale = next;
    pane.style.transition = animate ? 'transform .42s cubic-bezier(.22,1,.36,1)' : 'none';
    if (animate) setTimeout(function () { pane.style.transition = 'none'; }, 460);
    place();
  }
  function centre(f) { go(scale * f, stage.clientWidth / 2, stage.clientHeight / 2, true); }
  function focus(d) {
    const sw = stage.clientWidth, sh = stage.clientHeight;
    const pad = 5.5;
    scale = Math.min(MAX, Math.min(sw / (d.x2 - d.x1), sh / (d.y2 - d.y1)) / pad * 2.4);
    tx = sw / 2 - (d.x1 + d.x2) / 2 * scale;
    ty = sh / 2 - (d.y1 + d.y2) / 2 * scale;
    pane.style.transition = 'transform .55s cubic-bezier(.22,1,.36,1)';
    place();
    setTimeout(function () { pane.style.transition = 'none'; }, 580);
  }

  stage.addEventListener('wheel', function (e) {
    e.preventDefault();
    const r = stage.getBoundingClientRect();
    go(scale * (e.deltaY < 0 ? 1.13 : 1 / 1.13), e.clientX - r.left, e.clientY - r.top, false);
  }, { passive: false });

  let dragging = false, lx = 0, ly = 0;
  stage.addEventListener('pointerdown', function (e) {
    if (e.target.closest('.badge')) return;
    dragging = true; lx = e.clientX; ly = e.clientY;
    stage.classList.add('dragging'); stage.setPointerCapture(e.pointerId);
  });
  stage.addEventListener('pointermove', function (e) {
    if (!dragging) return;
    tx += e.clientX - lx; ty += e.clientY - ly; lx = e.clientX; ly = e.clientY; place();
  });
  ['pointerup', 'pointercancel'].forEach(function (ev) {
    stage.addEventListener(ev, function () {
      dragging = false; stage.classList.remove('dragging');
    });
  });
  stage.addEventListener('dblclick', function () { fit(true); });
  stage.addEventListener('keydown', function (e) {
    if (e.key === '+' || e.key === '=') { centre(1.25); e.preventDefault(); }
    if (e.key === '-' || e.key === '_') { centre(1 / 1.25); e.preventDefault(); }
    if (e.key === '0') { fit(true); e.preventDefault(); }
    if (e.key === '1') { centre(1 / scale); e.preventDefault(); }
  });

  document.getElementById('in').onclick = function () { centre(1.25); };
  document.getElementById('out').onclick = function () { centre(1 / 1.25); };
  document.getElementById('fit').onclick = function () { fit(true); };
  document.getElementById('one').onclick = function () { centre(1 / scale); };

  function setOverlay(v) {
    svg.style.opacity = v; marks.style.opacity = v;
    if (cam) cam.style.opacity = v;
  }
  mix.addEventListener('input', function () { setOverlay(mix.value / 100); });

  // Press and hold to see the untouched radiograph - the single most useful
  // gesture when judging whether a box sits on a real feature.
  const hold = document.getElementById('hold');
  hold.addEventListener('pointerdown', function (e) { e.preventDefault(); setOverlay(0); });
  ['pointerup', 'pointerleave', 'pointercancel'].forEach(function (ev) {
    hold.addEventListener(ev, function () { setOverlay(mix.value / 100); });
  });

  if (base.complete) { fit(false); } else { base.addEventListener('load', function () { fit(false); }); }
  window.addEventListener('resize', function () { fit(false); });
})();
</script>
"""


def render_viewport(image_bgr: np.ndarray, detections: list[dict], height: int,
                    show_chips: bool = False, cam_bgr: np.ndarray | None = None) -> None:
    """Interactive viewport: pan, zoom, hover to inspect, click to fly to a finding."""
    display_w = min(image_bgr.shape[1], 1600)
    ratio = display_w / image_bgr.shape[1]

    payload = [{
        "index": d.get("index", i + 1),
        "name": d["class"],
        "conf": round(float(d["confidence"]), 4),
        "color": CLASS_COLORS_HEX[d["class_id"]],
        "note": reliability_grade(d["class_id"])[2],
        "x1": int(round(d["x1"] * ratio)), "y1": int(round(d["y1"] * ratio)),
        "x2": int(round(d["x2"] * ratio)), "y2": int(round(d["y2"] * ratio)),
    } for i, d in enumerate(detections)]

    cam_tag = ""
    if cam_bgr is not None:
        cam_tag = (f'<img id="cam" src="data:image/jpeg;base64,'
                   f'{encode_jpeg(cam_bgr, max_width=display_w)}" alt="Saliency" '
                   f'draggable="false">')

    html = (VIEWPORT_TEMPLATE
            .replace("__HEIGHT__", str(height))
            .replace("__BASE__", encode_jpeg(image_bgr, max_width=display_w))
            .replace("__CAMTAG__", cam_tag)
            .replace("__IW__", str(int(round(image_bgr.shape[1] * ratio))))
            .replace("__IH__", str(int(round(image_bgr.shape[0] * ratio))))
            .replace("__DETS__", json.dumps(payload))
            .replace("__CHIPS__", "true" if show_chips else "false")
            .replace("__ISCAM__", "true" if cam_bgr is not None else "false"))
    components.html(html, height=height + 10, scrolling=False)


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #

DISCLAIMER = (
    "Research artefact. Not a medical device, not validated for clinical use, "
    "confidence values are not calibrated probabilities."
)


def findings_csv(detections: list[dict]) -> bytes:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["index", "class", "confidence", "x1", "y1", "x2", "y2",
                     "width_px", "height_px", "area_px",
                     "class_validation_recall", "class_validation_support"])
    for detection in detections:
        info = CLASS_RELIABILITY.get(detection["class_id"], {})
        writer.writerow([
            detection.get("index", ""), detection["class"],
            f"{detection['confidence']:.4f}",
            detection["x1"], detection["y1"], detection["x2"], detection["y2"],
            detection["width"], detection["height"], detection["area_px"],
            info.get("recall", ""), info.get("support", ""),
        ])
    return buffer.getvalue().encode("utf-8")


def findings_json(detections: list[dict], meta: dict) -> bytes:
    """A record that can be audited later.

    It names the exact weights by hash, the exact image by hash, every threshold
    in force, the inference time, and the measured reliability of each class
    that appears - so a finding pulled out of this file six months from now can
    still be traced to the model and settings that produced it.
    """
    payload = {
        "schema": "dentalscan.findings/1",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "disclaimer": DISCLAIMER,
        "image": {
            "filename": meta.get("image_name"),
            "sha256_12": meta.get("image_sha"),
            "width": meta.get("image_width"),
            "height": meta.get("image_height"),
        },
        "model": {
            "weights": meta.get("model_path"),
            "sha256_12": meta.get("model_sha"),
            "fine_tuned": meta.get("finetuned"),
            "architecture": "YOLOv10s",
        },
        "settings": {
            "confidence_threshold": meta.get("conf"),
            "nms_iou": meta.get("iou"),
            "inference_size": meta.get("imgsz"),
            "cache_confidence_floor": CACHE_CONF,
            "classes_enabled": sorted(meta.get("enabled", [])),
        },
        "timing": {"inference_ms": round(meta.get("inference_ms", 0.0), 2)},
        "summary": {
            "n_findings": len([d for d in detections if d["class_id"] != HEALTHY_ID]),
            "n_boxes": len(detections),
            "classes_present": sorted({d["class"] for d in detections}),
        },
        "class_reliability": {
            CLASS_NAMES[i]: {**CLASS_RELIABILITY[i],
                             "split": "validation (23 radiographs)",
                             "reportable": CLASS_RELIABILITY[i]["support"] >= LOW_SUPPORT}
            for i in CLASS_NAMES
        },
        "detections": [
            {k: v for k, v in d.items() if k != "class_id"} | {"class_id": d["class_id"]}
            for d in detections
        ],
    }
    return json.dumps(payload, indent=2).encode("utf-8")


# --------------------------------------------------------------------------- #
# Chrome
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Design system
# --------------------------------------------------------------------------- #
#
# Nocturne: an indigo ground, a blurple accent, Inter at heading weight 500, and
# elevation carried by a hairline rather than a glow. The five pathology hues
# are unchanged - they are validated in constants.py and the design was drawn
# around them.
#
# Everything below is CSS only. Streamlit strips <script> from st.markdown, so
# the page's motion is built from keyframes, transitions and staggered delays -
# which also means it degrades to a static, readable page if anything fails, and
# collapses to nothing under prefers-reduced-motion.

BASE_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;450;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
  /* Ground and surfaces. */
  --bg:        #161826;
  --bg-deep:   #0F1018;   /* under the radiograph, so the image is the brightest thing */
  --surface:   #232532;
  --surface-2: #1C1E2B;

  /* Ink. The design system's neutral ramp, but secondary text is pulled one
     step brighter than the source: neutral-600 on --surface measures 3.4:1,
     which fails AA at body size. neutral-500 measures 5.1:1. neutral-600 is
     kept only for incidental metadata, set large or set in mono. */
  --ink:   #F3F5FE;
  --ink-2: #B2B6CA;
  --ink-3: #9397AB;
  --ink-4: #75798C;
  --ink-5: #595D6C;

  --line:   color-mix(in srgb, var(--ink) 9%,  transparent);
  --line-2: color-mix(in srgb, var(--ink) 20%, transparent);

  /* Accent: the blurple, with the ramp steps the design leans on. */
  --accent:       #9184D9;
  --accent-300:   #D2CEFD;
  --accent-400:   #B5ABFC;
  --accent-700:   #5D5294;
  --accent-900:   #2B2741;
  --accent-2-400: #B5AFE8;

  /* Section ground - saturated deep indigo, band-scale fills only. */
  --section:       #262A60;
  --section-ghost: #4C5397;

  /* Status. */
  --ok:    #4FBF8B;  --ok-ink:    #7FD7AC;
  --warn:  #D4A03A;  --warn-ink:  #E0B968;
  --alert: #E0685F;  --alert-ink: #F0968E;

  --r:    20px;
  --r-lg: 16px;
  --r-md: 14px;
  --r-s:  9px;
  --shadow:    0 2px 6px rgba(0,0,0,.35), 0 20px 50px -26px rgba(0,0,0,.85);
  --shadow-lg: 0 4px 12px rgba(0,0,0,.4),  0 36px 80px -34px rgba(0,0,0,.95);
  --ease: cubic-bezier(.22,1,.36,1);
}

html, body, .stApp, [class*="css"] {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  font-size: 16px;
  -webkit-font-smoothing: antialiased; text-rendering: optimizeLegibility;
}

/* One soft bloom off the top right rather than a pair of drifting blobs: the
   ground should be still, because the thing that moves is the radiograph. */
.stApp {
  color: var(--ink);
  background:
    radial-gradient(1200px 620px at 78% -8%,
                    color-mix(in srgb, var(--section-ghost) 42%, transparent) 0%,
                    rgba(35,39,70,0) 62%),
    var(--bg);
  background-attachment: fixed;
}

.block-container { max-width: 1400px !important; padding: 0 32px 110px !important;
                   position: relative; z-index: 1; }
/* st.markdown always claims a layout slot, so the stylesheet injected below
   would otherwise open the page with one empty row-gap and leave the sticky nav
   hanging 16px clear of the top. */
[data-testid="stElementContainer"]:has(> [data-testid="stMarkdown"] style:only-child) {
  display: none; }
[data-testid="stHeader"] { background: transparent; height: 0; }
[data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"],
footer, #MainMenu, [data-testid="stSidebar"] { display: none !important; }

::-webkit-scrollbar { width: 11px; height: 11px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #33364A; border-radius: 7px; border: 3px solid var(--bg); }
::-webkit-scrollbar-thumb:hover { background: #454962; }
*:focus-visible { outline: 2px solid var(--accent) !important; outline-offset: 2px; border-radius: 6px; }
::selection { background: color-mix(in srgb, var(--accent) 30%, transparent); }

/* -- entry motion ---------------------------------------------------------
   One keyframe, applied with a stagger. Streamlit re-runs the script on every
   interaction, so these must be short: long enough to feel deliberate on load,
   short enough that nudging a slider never feels like waiting. */
@keyframes rise    { from { opacity: 0; transform: translateY(22px); } to { opacity: 1; transform: none; } }
@keyframes riseSm  { from { opacity: 0; transform: translateY(8px);  } to { opacity: 1; transform: none; } }
@keyframes fade    { from { opacity: 0; } to { opacity: 1; } }
@keyframes grow    { from { transform: scaleX(0); } to { transform: scaleX(1); } }
@keyframes shimmer { to { background-position: 220% 0; } }
@keyframes pulse   { 0%,100% { opacity: .4; transform: scale(1); }
                     50%     { opacity: .95; transform: scale(1.35); } }
@keyframes spin    { to { transform: rotate(360deg); } }
@keyframes glow    { 0%,100% { opacity: .5; } 50% { opacity: .9; } }
@keyframes sweep   { 0% { transform: translateX(-30%); opacity: 0; } 12% { opacity: 1; }
                     88% { opacity: 1; } 100% { transform: translateX(130%); opacity: 0; } }

.r  { opacity: 0; animation: rise .7s var(--ease) forwards; }
.r1 { animation-delay: .04s; } .r2 { animation-delay: .10s; } .r3 { animation-delay: .16s; }
.r4 { animation-delay: .22s; } .r5 { animation-delay: .28s; } .r6 { animation-delay: .34s; }

/* -- nav ------------------------------------------------------------------
   Sticky, and bled out to the container's padding so the hairline runs the
   full width of the page rather than the width of the text column. */
.nav { position: sticky; top: 0; z-index: 40;
       display: flex; align-items: center; gap: 16px; flex-wrap: wrap;
       margin: 0 -32px 4px; padding: 15px 32px;
       background: rgba(22,24,38,.74); backdrop-filter: blur(18px) saturate(1.4);
       border-bottom: 1px solid var(--line); }
.nav-logo { width: 38px; height: 38px; border-radius: 11px; flex: none; display: grid;
            place-items: center; color: var(--accent-400);
            background: linear-gradient(155deg, var(--accent-900), var(--surface));
            border: 1px solid color-mix(in srgb, var(--accent) 45%, transparent);
            box-shadow: 0 0 22px -6px color-mix(in srgb, var(--accent) 55%, transparent);
            transition: transform .4s var(--ease); }
.nav:hover .nav-logo { transform: rotate(-6deg) scale(1.06); }
.nav-brand { font-size: 17px; font-weight: 600; letter-spacing: -.3px; line-height: 1.15; }
.nav-sub { font-size: 12.5px; color: var(--ink-3); line-height: 1.3; margin-top: 1px; }
.nav-links { display: flex; align-items: center; gap: 26px; font-size: 14px; }
.nav-links a { color: var(--ink-3); font-weight: 400; transition: color .18s ease; }
.nav-links a:hover { color: var(--ink); text-decoration: none; }
.nav-rule { width: 1px; height: 24px; background: var(--line-2); margin: 0 4px; }
.sp { flex: 1 1 auto; }

.status { display: inline-flex; align-items: center; gap: 9px; padding: 8px 15px;
          border-radius: 100px; font-size: 13.5px; white-space: nowrap;
          background: color-mix(in srgb, var(--ok) 8%, transparent);
          border: 1px solid color-mix(in srgb, var(--ok) 28%, transparent);
          color: var(--ok-ink); }
.status.bad { background: color-mix(in srgb, var(--alert) 9%, transparent);
              border-color: color-mix(in srgb, var(--alert) 30%, transparent);
              color: var(--alert-ink); }
.status b { font-weight: 600; }
.dot { position: relative; width: 8px; height: 8px; flex: none; }
.dot::before, .dot::after { content: ''; position: absolute; border-radius: 50%;
                            background: var(--ok); }
.dot::before { inset: 0; animation: pulse 2.4s ease-in-out infinite; }
.dot::after  { inset: 1.5px; }
.status.bad .dot::before, .status.bad .dot::after { background: var(--alert); }

/* -- hero ------------------------------------------------------------------ */
.hero { padding: 62px 0 18px; }
.eyebrow { display: inline-flex; align-items: center; gap: 10px; padding: 7px 14px 7px 10px;
           border-radius: 100px; font-size: 13px; color: var(--accent-400);
           background: color-mix(in srgb, var(--accent-900) 70%, transparent);
           border: 1px solid color-mix(in srgb, var(--accent) 30%, transparent);
           margin-bottom: 26px; }
.eyebrow i { width: 5px; height: 5px; border-radius: 50%; background: var(--accent); }
.hero h1 { font-size: clamp(40px, 4.6vw, 62px); font-weight: 500; line-height: 1.05;
           letter-spacing: -2.4px; margin: 0 0 22px; color: var(--ink); text-wrap: pretty; }
.hero p { font-size: 19px; line-height: 1.6; color: var(--ink-2); max-width: 520px;
          margin: 0 0 4px; text-wrap: pretty; }
.hero .fine { font-size: 14px; color: var(--ink-3); line-height: 1.7; max-width: 500px;
              margin-top: 26px; }

/* The frame around the drop target. The design put a worked example here; the
   real app has nothing to show until a study is uploaded, so the frame *is*
   the upload and the scan sweep runs over an empty stage. */
.st-key-dropframe { position: relative; border-radius: var(--r); overflow: hidden;
  background: var(--bg-deep); border: 1px solid var(--line);
  box-shadow: 0 40px 90px -40px rgba(0,0,0,.9),
              0 0 0 1px color-mix(in srgb, var(--accent) 7%, transparent);
  animation: rise .8s .12s var(--ease) both; }
.st-key-dropframe::after {
  content: ''; position: absolute; inset: 0 0 52px; pointer-events: none;
  mix-blend-mode: screen;
  background: linear-gradient(100deg,
    transparent 42%, color-mix(in srgb, var(--accent) 16%, transparent) 50%, transparent 58%);
  animation: sweep 4.6s cubic-bezier(.5,0,.5,1) infinite; }
.frame-foot { display: flex; align-items: center; gap: 16px; flex-wrap: wrap;
              padding: 15px 20px; border-top: 1px solid var(--line);
              background: rgba(22,24,38,.7);
              font-family: 'IBM Plex Mono', monospace; font-size: 12.5px; color: var(--ink-3); }
.frame-foot .ready { display: inline-flex; align-items: center; gap: 7px; margin-left: auto;
                     font-family: 'Inter', sans-serif; font-size: 13px; color: var(--ink-2); }

/* -- evidence band --------------------------------------------------------- */
.band { position: relative; overflow: hidden; border-radius: 24px; margin-top: 34px;
        background: linear-gradient(150deg, var(--section) 0%, var(--section) 55%, var(--bg) 100%);
        border: 1px solid color-mix(in srgb, var(--section-ghost) 50%, transparent); }
.band::before { content: ''; position: absolute; top: -140px; right: -80px;
                width: 420px; height: 420px; border-radius: 50%; pointer-events: none;
                background: radial-gradient(circle,
                  color-mix(in srgb, var(--accent) 30%, transparent), transparent 66%);
                animation: glow 6s ease-in-out infinite; }
.band-in { position: relative; padding: 42px 42px 38px; }
.band h2 { font-size: 30px; font-weight: 500; letter-spacing: -.9px; margin: 0; color: var(--ink); }
.band-head { display: flex; flex-wrap: wrap; align-items: baseline; gap: 16px; margin-bottom: 10px; }
.band-head span { font-size: 14.5px; color: var(--accent-2-400); }
.band p { font-size: 15.5px; line-height: 1.65; color: var(--ink-2); max-width: 760px;
          margin: 0 0 32px; }
/* Six classes, so the grid steps 6 -> 3 -> 2 rather than reflowing freely:
   auto-fit leaves Healthy stranded alone on a second row at most widths. */
.band-grid { display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 20px; }
.cls { padding: 20px 20px 18px; border-radius: var(--r-md); background: rgba(15,16,24,.42);
       border: 1px solid var(--line);
       transition: transform .25s var(--ease), border-color .25s ease, background .25s ease; }
.cls:hover { transform: translateY(-5px); border-color: var(--line-2); background: rgba(15,16,24,.62); }
.cls-bar { height: 4px; border-radius: 3px; margin-bottom: 16px; background: var(--c);
           box-shadow: 0 0 14px -2px var(--c); transform-origin: left;
           animation: grow .9s var(--ease) both; }
.cls-n { font-size: 15px; font-weight: 500; color: var(--ink); }
.cls-v { display: flex; align-items: baseline; gap: 6px; margin-top: 12px; }
.cls-v b { font-size: 38px; font-weight: 500; letter-spacing: -1.4px; line-height: 1;
           font-variant-numeric: tabular-nums; }
.cls-v span { font-size: 14px; color: var(--ink-3); }
.cls-note { margin-top: 14px; font-size: 13.5px; line-height: 1.5; color: var(--ink-3); }
.cls.low .cls-v b { color: var(--ink-4); }
.cls.low .cls-note { color: var(--warn-ink); }

/* -- notes ----------------------------------------------------------------- */
.notes { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
         gap: 24px; padding: 56px 0 20px; }
.note { padding: 30px 28px; border-radius: var(--r-lg); background: var(--surface);
        border: 1px solid var(--line);
        transition: transform .25s var(--ease), border-color .25s ease; }
.note:hover { transform: translateY(-6px);
              border-color: color-mix(in srgb, var(--accent) 42%, transparent); }
.note-n { width: 42px; height: 42px; border-radius: 12px; display: grid; place-items: center;
          margin-bottom: 20px; font-family: 'IBM Plex Mono', monospace; font-size: 15px;
          color: var(--accent-400);
          background: color-mix(in srgb, var(--accent) 12%, transparent);
          border: 1px solid color-mix(in srgb, var(--accent) 28%, transparent); }
.note-t { font-size: 18px; font-weight: 500; letter-spacing: -.3px; margin-bottom: 10px; }
.note-d { font-size: 15px; line-height: 1.68; color: var(--ink-3); }

/* -- study head ------------------------------------------------------------ */
.studybar { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
.study-name { font-size: 22px; font-weight: 500; letter-spacing: -.5px; color: var(--ink); }
.study-dim { padding: 4px 10px; border-radius: 6px; font-family: 'IBM Plex Mono', monospace;
             font-size: 12.5px; color: var(--ink-3);
             background: color-mix(in srgb, var(--ink) 6%, transparent);
             border: 1px solid var(--line); }
.study-meta { font-size: 14px; color: var(--ink-3); margin-top: 6px; }
.study-meta b { color: var(--ink); font-weight: 500; }

/* -- threshold card -------------------------------------------------------- */
.st-key-thresh { padding: 22px 26px 6px; border-radius: var(--r-lg);
                 background: var(--surface); border: 1px solid var(--line); }
.thresh-head { display: flex; flex-wrap: wrap; align-items: baseline; gap: 12px; }
.thresh-head .t { font-size: 16px; font-weight: 500; color: var(--ink); }
.thresh-head .v { font-family: 'IBM Plex Mono', monospace; font-size: 26px; font-weight: 500;
                  letter-spacing: -.6px; color: var(--accent-400);
                  font-variant-numeric: tabular-nums; }
.thresh-head .n { font-size: 14.5px; color: var(--ink-3); margin-left: auto; text-align: right; }
.thresh-ends { display: flex; justify-content: space-between; margin-top: -6px;
               font-family: 'IBM Plex Mono', monospace; font-size: 12.5px; color: var(--ink-5); }

/* -- stat tiles ------------------------------------------------------------ */
.tiles { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
.tile { padding: 20px 20px 18px; border-radius: var(--r-md); background: var(--surface);
        border: 1px solid var(--line); opacity: 0; animation: riseSm .5s var(--ease) forwards;
        transition: border-color .3s ease; }
.tile:hover { border-color: var(--line-2); }
.tile-v { font-size: 34px; font-weight: 500; line-height: 1; letter-spacing: -1.3px;
          font-variant-numeric: tabular-nums; color: var(--ink); }
.tile-v.muted { color: var(--ink-5); }
.tile-v.warn  { color: var(--warn-ink); }
.tile-v small { font-size: 15px; font-weight: 400; letter-spacing: 0; color: var(--ink-4);
                margin-left: 3px; }
.tile-k { font-size: 13.5px; line-height: 1.35; color: var(--ink-3); margin-top: 10px; }

/* -- section heading ------------------------------------------------------- */
.sec { display: flex; align-items: baseline; gap: 10px; flex-wrap: wrap; margin: 6px 0 0; }
.sec h2 { font-size: 17px; font-weight: 500; letter-spacing: -.2px; margin: 0; color: var(--ink); }
.sec span { font-size: 14px; color: var(--ink-3); }

/* -- findings -------------------------------------------------------------- */
.flist { display: flex; flex-direction: column; gap: 12px; }
.find { padding: 17px 20px; border-radius: var(--r-md); background: var(--surface);
        border: 1px solid var(--line); border-left: 3px solid var(--c);
        opacity: 0; animation: riseSm .45s var(--ease) forwards;
        transition: transform .22s var(--ease), background .22s ease, border-color .22s ease; }
.find:hover { transform: translateX(4px); background: var(--accent-900);
              border-color: color-mix(in srgb, var(--accent) 34%, transparent);
              border-left-color: var(--c); }
.find-top { display: flex; align-items: center; gap: 12px; }
.find-i { font-family: 'IBM Plex Mono', monospace; font-size: 13px; color: var(--ink-4);
          width: 16px; flex: none; }
.find-n { font-size: 18px; font-weight: 500; letter-spacing: -.2px; color: var(--ink); }
.find-c { margin-left: auto; font-size: 22px; font-weight: 500; letter-spacing: -.6px;
          color: var(--c); font-variant-numeric: tabular-nums; }
.find-bar { height: 5px; border-radius: 3px; margin: 14px 0; overflow: hidden;
            background: color-mix(in srgb, var(--ink) 7%, transparent); }
.find-bar i { display: block; height: 100%; border-radius: 3px; background: var(--c);
              transform-origin: left; animation: grow .85s var(--ease) forwards;
              box-shadow: 0 0 12px -2px var(--c); }
.find-foot { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
.find-box { font-family: 'IBM Plex Mono', monospace; font-size: 12px; color: var(--ink-4);
            margin-left: auto; }
.tag { display: inline-flex; align-items: center; gap: 8px; padding: 5px 11px 5px 9px;
       border-radius: 100px; font-size: 13px; white-space: nowrap;
       border: 1px solid transparent; font-variant-numeric: tabular-nums; }
.tag i { width: 6px; height: 6px; border-radius: 50%; background: currentColor; flex: none; }
.tag.ok         { color: var(--ok-ink);    background: color-mix(in srgb, var(--ok) 9%, transparent);
                  border-color: color-mix(in srgb, var(--ok) 28%, transparent); }
.tag.weak       { color: var(--warn-ink);  background: color-mix(in srgb, var(--warn) 10%, transparent);
                  border-color: color-mix(in srgb, var(--warn) 30%, transparent); }
.tag.unmeasured { color: var(--alert-ink); background: color-mix(in srgb, var(--alert) 10%, transparent);
                  border-color: color-mix(in srgb, var(--alert) 30%, transparent); }
.tag.unknown    { color: var(--ink-3); background: color-mix(in srgb, var(--ink) 6%, transparent); }

.railnote { padding: 18px 20px; border-radius: var(--r-md); font-size: 13.5px; line-height: 1.68;
            color: var(--ink-2);
            background: color-mix(in srgb, var(--accent) 6%, transparent);
            border: 1px solid color-mix(in srgb, var(--accent) 20%, transparent); }
.railnote b { color: var(--ink); font-weight: 500; }
.railnote b.w { color: var(--warn-ink); }

/* -- skeleton -------------------------------------------------------------- */
.skel { border-radius: var(--r); border: 1px solid var(--line); overflow: hidden;
        background: linear-gradient(100deg, var(--surface) 30%, #33364A 50%, var(--surface) 70%);
        background-size: 220% 100%; animation: shimmer 1.4s linear infinite; }
.skel-note { display: flex; align-items: center; gap: 13px; font-size: 15px;
             color: var(--ink-2); margin-top: 18px; }
.spinner { width: 17px; height: 17px; border-radius: 50%; flex: none;
           border: 2px solid color-mix(in srgb, var(--accent) 25%, transparent);
           border-top-color: var(--accent); animation: spin .8s linear infinite; }

/* -- notices --------------------------------------------------------------- */
.notice { border-radius: var(--r-md); padding: 17px 20px; margin-bottom: 20px;
          font-size: 15px; line-height: 1.62; border: 1px solid; animation: fade .4s ease; }
.notice.alert { background: color-mix(in srgb, var(--alert) 8%, transparent);
                border-color: color-mix(in srgb, var(--alert) 30%, transparent); color: var(--alert-ink); }
.notice.warn  { background: color-mix(in srgb, var(--warn) 8%, transparent);
                border-color: color-mix(in srgb, var(--warn) 28%, transparent); color: var(--warn-ink); }
.notice.quiet { background: color-mix(in srgb, var(--ink) 3%, transparent);
                border: 1px dashed var(--line-2); color: var(--ink-3); }
.notice b { color: var(--ink); font-weight: 600; }
.notice code, .study-meta code { font-family: 'IBM Plex Mono', monospace; font-size: 13.5px;
                                 padding: 2px 7px; border-radius: 6px;
                                 background: color-mix(in srgb, var(--ink) 7%, transparent); }

.foot { font-size: 13.5px; color: var(--ink-4); line-height: 1.72; margin-top: 18px; }
.foot b { color: var(--ink-2); font-weight: 500; }

/* -- Streamlit widgets ----------------------------------------------------- */
[data-testid="stFileUploaderDropzone"] {
  background: transparent; border: 0; border-radius: 0; padding: 54px 30px;
  min-height: 250px; align-items: center; justify-content: center;
  transition: background .3s ease; }
[data-testid="stFileUploaderDropzone"]:hover {
  background: color-mix(in srgb, var(--accent) 6%, transparent); }
[data-testid="stFileUploaderDropzone"] button {
  background: transparent !important; color: var(--accent-300) !important;
  border: 1px solid var(--accent) !important; font-weight: 500 !important;
  font-size: 15px !important; border-radius: 10px !important; padding: 12px 22px !important;
  transition: background .2s ease, transform .2s var(--ease), box-shadow .2s ease !important; }
[data-testid="stFileUploaderDropzone"] button:hover {
  background: color-mix(in srgb, var(--accent) 14%, transparent) !important;
  transform: translateY(-2px);
  box-shadow: 0 10px 28px -12px color-mix(in srgb, var(--accent) 80%, transparent) !important; }
[data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploaderDropzone"] span { color: var(--ink-3) !important; font-size: 14.5px !important; }
[data-testid="stFileUploaderFile"] { background: color-mix(in srgb, var(--ink) 5%, transparent);
                                     border-radius: 10px; font-size: 14.5px; }

/* Buttons are outlined, not filled - Nocturne carries emphasis with an edge. */
.stButton > button, .stDownloadButton > button, [data-testid="stPopover"] button {
  background: transparent; color: var(--ink-2); border: 1px solid var(--line-2);
  border-radius: var(--r-s); font-size: 14.5px; font-weight: 450; padding: 10px 18px;
  transition: background .2s ease, border-color .2s ease, color .2s ease, transform .2s var(--ease); }
.stButton > button:hover, .stDownloadButton > button:hover, [data-testid="stPopover"] button:hover {
  background: color-mix(in srgb, var(--ink) 4%, transparent);
  border-color: color-mix(in srgb, var(--ink) 34%, transparent); color: var(--ink); }
.stButton > button:active, .stDownloadButton > button:active { transform: scale(.97); }
.stDownloadButton > button { color: var(--accent-300); border-color: var(--accent-700); }
.stDownloadButton > button:hover { background: color-mix(in srgb, var(--accent) 12%, transparent) !important;
                                   border-color: var(--accent) !important; color: var(--accent-300) !important; }
[data-testid="stPopoverBody"] { background: rgba(28,30,43,.96); border: 1px solid var(--line-2);
                                border-radius: var(--r-lg); box-shadow: var(--shadow-lg);
                                backdrop-filter: blur(28px) saturate(1.5); padding: 10px 6px; }

/* Pills. Streamlit 1.40 reported selection through aria-pressed; 1.42 and
   after swap the button's data-testid between stBaseButton-pills and
   -pillsActive. Both are matched, so the filter reads correctly across the
   supported range. Each pill wears its own class's hue through --c, assigned by
   index below, so the palette in this file stays the single source of truth. */
div[data-baseweb="button-group"] { gap: 10px; flex-wrap: wrap; }
[data-testid="stBaseButton-pills"], [data-testid="stBaseButton-pillsActive"],
button[aria-pressed] {
  background: transparent !important; border: 1px solid var(--line-2) !important;
  color: var(--ink-3) !important; border-radius: 100px !important; font-size: 14.5px !important;
  font-weight: 450 !important; padding: 9px 18px !important;
  transition: transform .2s var(--ease), background .2s ease,
              border-color .2s ease, color .2s ease !important; }
[data-testid="stBaseButton-pills"]:hover, [data-testid="stBaseButton-pillsActive"]:hover,
button[aria-pressed]:hover { transform: translateY(-2px); color: var(--ink) !important; }
[data-testid="stBaseButton-pillsActive"], button[aria-pressed="true"] {
  background: color-mix(in srgb, var(--c, var(--accent)) 17%, transparent) !important;
  border-color: color-mix(in srgb, var(--c, var(--accent)) 55%, transparent) !important;
  color: var(--ink) !important; font-weight: 500 !important; }

.stSlider [data-baseweb="slider"] div[role="slider"] {
  background: var(--ink) !important; border: 3px solid var(--bg) !important;
  height: 22px !important; width: 22px !important;
  box-shadow: 0 3px 12px rgba(0,0,0,.65) !important; }
.stSlider [data-baseweb="slider"] [data-testid="stSliderTickBar"] { display: none; }
/* The threshold card prints its own value, at a size worth reading across the
   room; the thumb label beneath it would only say the same thing again. */
.st-key-thresh [data-testid="stSliderThumbValue"] { visibility: hidden; }
[data-testid="stWidgetLabel"] p, .stSlider label, .stCheckbox label, .stToggle label {
  font-size: 14.5px !important; color: var(--ink-3) !important; font-weight: 450 !important; }
[data-testid="stCheckbox"] label span, [data-testid="stToggle"] label span { font-size: 14.5px; }

[data-testid="stExpander"] { border: 1px solid var(--line); border-radius: var(--r-md);
  background: var(--surface); transition: border-color .3s ease; }
[data-testid="stExpander"]:hover { border-color: var(--line-2); }
[data-testid="stExpander"] summary { font-size: 15px; color: var(--ink-2); padding: 8px 4px;
                                     font-weight: 450; }
[data-testid="stExpander"] summary:hover { color: var(--ink); }
[data-testid="stExpanderDetails"] p, [data-testid="stExpanderDetails"] li,
[data-testid="stExpanderDetails"] td, [data-testid="stExpanderDetails"] th {
  font-size: 15px; line-height: 1.78; color: var(--ink-2); }
[data-testid="stExpanderDetails"] strong { color: var(--ink); }
[data-testid="stExpanderDetails"] table { border-collapse: collapse; }
[data-testid="stExpanderDetails"] th { color: var(--ink-3); font-weight: 500; font-size: 13.5px; }

[data-testid="stCaptionContainer"] p { font-size: 14px; color: var(--ink-3); line-height: 1.7; }
a { color: var(--accent-400); text-decoration: none; font-weight: 450; }
a:hover { color: var(--accent-300); text-decoration: underline; }
hr { border-color: var(--line); }

/* -- responsive ------------------------------------------------------------ */
@media (max-width: 1240px) {
  .band-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); }
}
@media (max-width: 1100px) {
  .band-in { padding: 32px 26px 30px; }
  .band h2 { font-size: 25px; }
}
@media (max-width: 640px) {
  .band-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
@media (max-width: 860px) {
  .block-container { padding: 0 16px 80px !important; }
  .nav { margin: 0 -16px 4px; padding: 13px 16px; gap: 12px; }
  .nav-links, .nav-rule { display: none; }
  .hero { padding: 34px 0 8px; }
  .hero h1 { letter-spacing: -1.4px; }
  .hero p { font-size: 17px; }
  .notes { padding: 40px 0 12px; gap: 16px; }
  .tiles { gap: 10px; }
  .find-box { margin-left: 0; flex-basis: 100%; }
}
@media (max-width: 520px) {
  /* The strapline is the first thing to go: without it the brand block is
     narrow enough that the status chip stays on the same row, which halves the
     height of a header that is sticky. */
  .nav-sub { display: none; }
  .status { font-size: 12.5px; padding: 7px 12px; }
  .nav-brand { font-size: 16px; }
  .band-grid { gap: 12px; }
}

@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: .01ms !important; animation-iteration-count: 1 !important;
    transition-duration: .01ms !important; }
  .r, .tile, .find { opacity: 1 !important; }
}
"""

# Per-class pill hues, generated from the palette so it stays the one source of
# truth. Streamlit has wrapped pill buttons differently across versions, so both
# the wrapped and the direct-child form are emitted.
PILL_CSS = "\n".join(
    f'div[data-baseweb="button-group"] > *:nth-child({position + 1}) '
    f'{{ --c: {CLASS_COLORS_HEX[class_id]}; }}'
    for position, class_id in enumerate(CLASS_NAMES)
)

st.markdown(f"<style>{BASE_CSS}\n{PILL_CSS}</style>", unsafe_allow_html=True)

bundle = load_model()
model = bundle["model"]

# The status chip names the architecture and whether the weights are the
# fine-tuned ones. The checkpoint hash used to sit here in full; it is
# provenance, not a headline, so it moved to the tooltip and the JSON export.
if model is None:
    status_class, status_html = "status bad", "<b>Model unavailable</b>"
elif bundle["finetuned"]:
    status_class, status_html = "status", "Fine-tuned · <b>YOLOv10s</b>"
else:
    status_class, status_html = "status bad", "<b>Base COCO weights</b>"

st.markdown(
    f"""
    <header class="nav">
      <div class="nav-logo">{TOOTH_SVG}</div>
      <div>
        <div class="nav-brand">DentalScan</div>
        <div class="nav-sub">panoramic radiograph analysis</div>
      </div>
      <div class="sp"></div>
      <nav class="nav-links">
        <a href="{REPO_URL}/blob/main/MODEL_CARD.md" target="_blank" rel="noopener">Model card</a>
        <a href="{REPO_URL}/blob/main/report/main.pdf" target="_blank" rel="noopener">Technical report</a>
        <a href="{REPO_URL}" target="_blank" rel="noopener">Repository</a>
      </nav>
      <div class="nav-rule"></div>
      <span class="{status_class}" title="checkpoint {bundle['sha'] or bundle['path']}">
        <span class="dot"></span>{status_html}</span>
    </header>
    """,
    unsafe_allow_html=True,
)

if model is None:
    st.markdown(
        f'<div class="notice alert"><b>The detector could not be loaded.</b><br>'
        f'{bundle["error"]}<br>Expected weights at <code>{FINETUNED_PATH}</code>. '
        f'Train them with <code>python model.py train --name baseline</code>, or copy '
        f'<code>best.pt</code> into that path.</div>', unsafe_allow_html=True)
    st.stop()

if not bundle["finetuned"]:
    st.markdown(
        f'<div class="notice alert"><b>Fine-tuned weights are missing, so this is the '
        f'base COCO model.</b> It predicts everyday objects, not dental conditions — '
        f'every label below will be wrong. Place the trained checkpoint at '
        f'<code>{FINETUNED_PATH}</code>.</div>', unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #

state = st.session_state
for key, default in [("image", None), ("image_name", ""), ("image_sha", ""),
                     ("cache_key", None), ("detections", []), ("inference_ms", 0.0),
                     ("cam_key", None), ("cam", None), ("error", ""),
                     ("conf_thresh", 0.25)]:
    state.setdefault(key, default)


def decode_upload(file) -> tuple[np.ndarray | None, str]:
    """Bytes -> BGR array, with the failure modes named rather than thrown."""
    raw = file.getvalue()
    if len(raw) > MAX_UPLOAD_MB * 1024 * 1024:
        return None, f"That file is larger than {MAX_UPLOAD_MB} MB."
    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
    except Exception as exc:
        return None, f"That file could not be read as an image ({exc})."
    if min(image.size) < 64:
        return None, f"Image is only {image.size[0]}×{image.size[1]} px — too small to analyse."
    return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR), ""


def ingest(upload) -> bool:
    """Adopt a new upload. Returns True when state changed, which the caller
    turns into a rerun.

    Deliberately does nothing when the uploader reports None: the widget is torn
    down when the page switches from the empty state to the loaded one, and
    reading that transient None as "the user cleared the study" would wipe the
    image on the very rerun meant to display it. Closing is explicit instead.
    """
    if upload is None:
        return False
    sha = hashlib.sha256(upload.getvalue()).hexdigest()[:12]
    if sha == state["image_sha"]:
        return False
    image, problem = decode_upload(upload)
    state.update({"image": image, "image_name": upload.name, "image_sha": sha,
                  "error": problem, "detections": [], "cache_key": None,
                  "cam": None, "cam_key": None})
    return True


def clear_study() -> None:
    state.update({"image": None, "image_name": "", "image_sha": "", "detections": [],
                  "cache_key": None, "cam": None, "cam_key": None, "error": ""})


# --------------------------------------------------------------------------- #
# Empty state - the upload *is* the page
# --------------------------------------------------------------------------- #
#
# Two columns: the claim on the left, the thing that acts on it on the right.
# The design drew a worked example into that right-hand frame; this app has no
# study to show until one is given to it, so the frame is the drop target and
# the scan sweep runs over an empty stage.

def evidence_band() -> str:
    """The per-class recall band. Built from CLASS_RELIABILITY, so it cannot
    drift from the numbers the findings list badges every detection with."""
    cards = []
    for class_id, name in CLASS_NAMES.items():
        recall = CLASS_RELIABILITY[class_id]["recall"]
        support = CLASS_RELIABILITY[class_id]["support"]
        low = support < LOW_SUPPORT
        note = (f"only {support} validation instances — treat as unmeasured" if low
                else f"{support} validation instances")
        cards.append(
            f'<div class="cls{" low" if low else ""}" '
            f'style="--c:{CLASS_COLORS_HEX[class_id]};">'
            f'<div class="cls-bar" style="width:{28 + recall * 62:.0f}%;'
            f'animation-delay:{0.1 + class_id * 0.07:.2f}s;"></div>'
            f'<div class="cls-n">{name}</div>'
            f'<div class="cls-v"><b>{recall:.2f}</b><span>recall</span></div>'
            f'<div class="cls-note">{note}</div></div>')
    return (
        '<section class="band r r5"><div class="band-in">'
        '<div class="band-head"><h2>What each class is actually worth</h2>'
        '<span>measured recall on the 23-image validation split, at confidence 0.25</span>'
        '</div>'
        '<p>The headline mAP@0.5 of 0.923 is carried by the three rarest classes. Over '
        'the three with at least ten validation instances it is 0.894 — the defensible '
        'number, and the one this interface shows next to every finding.</p>'
        f'<div class="band-grid">{"".join(cards)}</div></div></section>')


if state["image"] is None:
    hero_col, drop_col = st.columns([1, 1.15], gap="large", vertical_alignment="center")

    with hero_col:
        st.markdown(
            f"""
            <div class="hero">
              <div class="eyebrow r r2"><i></i>Six conditions · 231 annotated radiographs</div>
              <h1 class="r r3">Read a panoramic<br>radiograph in<br>one pass.</h1>
              <p class="r r4">Upload an OPG and get colour-coded findings with confidence
                 scores — each one carrying the measured recall of its class, so nothing
                 on screen claims more certainty than the evidence behind it.</p>
              <div class="fine r r5">JPG · PNG · TIFF up to {MAX_UPLOAD_MB} MB. Nothing is
                 stored — the study is discarded when you close it.</div>
            </div>
            """, unsafe_allow_html=True)

    with drop_col:
        with st.container(key="dropframe"):
            upload = st.file_uploader("Radiograph",
                                      type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
                                      label_visibility="collapsed", key="uploader")
            st.markdown(
                f'<div class="frame-foot">awaiting study · one forward pass at '
                f'confidence {CACHE_CONF:g}<span class="ready">{CHECK_SVG}'
                f'model loaded</span></div>', unsafe_allow_html=True)

    if ingest(upload) and state["image"] is not None:
        st.rerun()
    if state["error"]:
        st.markdown(f'<div class="notice alert" style="margin-top:18px;">'
                    f'{state["error"]}</div>', unsafe_allow_html=True)

    st.markdown(evidence_band(), unsafe_allow_html=True)

    st.markdown(
        """
        <div class="notes">
          <div class="note r r4">
            <div class="note-n">01</div>
            <div class="note-t">One pass, instant controls</div>
            <div class="note-d">The model runs once at a low confidence floor. The
              threshold and the condition filters re-read that cached result, so they
              respond immediately and every number on screen comes from a single pass.</div>
          </div>
          <div class="note r r5">
            <div class="note-n">02</div>
            <div class="note-t">Zoom that matters</div>
            <div class="note-d">A 30-pixel carious lesion on a 1935-pixel panoramic is
              unreadable at fit-to-width. Scroll to zoom at the cursor, drag to pan, hold
              Compare to blank the overlay and check a box sits on a real feature.</div>
          </div>
          <div class="note r r6">
            <div class="note-n">03</div>
            <div class="note-t">Healthy is grey, and off</div>
            <div class="note-d">Healthy is a background label rather than a finding, and
              58% of all boxes in the training data. Drawn like pathology it buries the
              pathology, so it is recessive and disabled by default.</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f'<div class="foot" style="opacity:.75;">{DISCLAIMER}</div>',
                unsafe_allow_html=True)
    st.stop()


# --------------------------------------------------------------------------- #
# Loaded study
# --------------------------------------------------------------------------- #

image_bgr = state["image"]
img_h, img_w = image_bgr.shape[:2]

name_col, tog_col, chg_col, set_col, exp_col, close_col = st.columns(
    [4.6, 1.4, 1.2, 1.2, 1.1, 0.55], vertical_alignment="center")

meta_slot = name_col.empty()

with tog_col:
    saliency_on = st.toggle("Saliency", value=False, key="saliency_on",
                            help="Show where the model looked, instead of what it found.")

with chg_col:
    with st.popover("Change", use_container_width=True):
        upload = st.file_uploader("Replace study",
                                  type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
                                  label_visibility="collapsed", key="uploader_loaded")
    if ingest(upload):
        st.rerun()

with set_col:
    with st.popover("Settings", use_container_width=True):
        st.caption("Display")
        show_chips = st.checkbox("Always show labels", value=False,
                                 help="Off by default: numbered markers keep the "
                                      "radiograph readable, and hovering one shows "
                                      "the full label.")
        fill_boxes = st.checkbox("Interior wash", value=True,
                                 help="Applies to the exported PNG.")
        box_opacity = st.slider("Overlay strength (export)", 0.2, 1.0, 1.0, 0.05)
        st.divider()
        st.caption("Inference — changing these re-runs the model")
        iou_thresh = st.slider("NMS IoU", 0.10, 0.90, 0.70, 0.05)
        imgsz = st.select_slider("Inference size", options=[512, 640, 800, 960], value=640)

export_slot = exp_col.empty()

with close_col:
    if st.button("✕", help="Close this study", use_container_width=True):
        clear_study()
        st.rerun()

# --------------------------------------------------------------------------- #
# Inference, with a skeleton in place of the viewport while it runs
# --------------------------------------------------------------------------- #
#
# The viewport now sits in a column roughly 58% of the page rather than the full
# width, so the height that keeps a panoramic legible is smaller than it was.

vp_height = int(np.clip(round(880 / max(img_w / img_h, 0.35)), 360, 580))
skeleton = st.empty()

if not state["error"]:
    key = (state["image_sha"], round(iou_thresh, 3), int(imgsz),
           bundle["sha"] or bundle["path"])
    if state["cache_key"] != key:
        skeleton.markdown(
            f'<div class="skel" style="height:{vp_height}px;"></div>'
            f'<div class="skel-note"><span class="spinner"></span>'
            f'Reading radiograph at {imgsz}px — one forward pass at confidence '
            f'{CACHE_CONF:g}…</div>', unsafe_allow_html=True)
        try:
            detections, elapsed = run_inference(model, image_bgr, iou_thresh, imgsz)
            state.update({"detections": detections, "inference_ms": elapsed,
                          "cache_key": key, "error": ""})
        except Exception as exc:
            state.update({"detections": [], "cache_key": None,
                          "error": f"Inference failed: {exc}"})
        skeleton.empty()

meta_slot.markdown(
    f'<div class="studybar r r1"><span class="study-name">{state["image_name"]}</span>'
    f'<span class="study-dim">{img_w} × {img_h}</span></div>'
    f'<div class="study-meta">analysed in <b>{state["inference_ms"]:.0f} ms</b> · one '
    f'forward pass, cached at confidence {CACHE_CONF:g}</div>',
    unsafe_allow_html=True)

if state["error"]:
    st.markdown(f'<div class="notice alert" style="margin-top:16px;">'
                f'<b>Could not analyse this study.</b><br>{state["error"]}</div>',
                unsafe_allow_html=True)
    st.stop()

raw_detections: list[dict] = state["detections"]

# --------------------------------------------------------------------------- #
# Study layout: the radiograph and its controls on the left, the evidence rail
# on the right.
# --------------------------------------------------------------------------- #
#
# Source order inside the left column follows the dependency chain - the
# threshold governs the pill counts, which govern what the viewport draws - so
# only the viewport needs a slot to be filled out of order.

st.markdown('<div style="height:18px;"></div>', unsafe_allow_html=True)
scan_col, rail_col = st.columns([1.52, 1], gap="large")

with scan_col:
    viewport_slot = st.container()

    with st.container(key="thresh"):
        thresh_head = st.empty()
        conf_thresh = st.slider("Confidence", 0.05, 0.95,
                                value=float(state["conf_thresh"]), step=0.01,
                                key="conf_slider", label_visibility="collapsed")
        st.markdown('<div class="thresh-ends"><span>0.05</span><span>0.95</span></div>',
                    unsafe_allow_html=True)
    state["conf_thresh"] = conf_thresh

    available = {
        class_id: sum(1 for d in raw_detections
                      if d["class_id"] == class_id and d["confidence"] >= conf_thresh)
        for class_id in CLASS_NAMES
    }

    # Options are class ids, not label strings: the visible label carries a count
    # that moves with the threshold, and pills keyed on the label would reset
    # their selection every time that count changed.
    picked = st.pills(
        "Conditions", list(CLASS_NAMES), selection_mode="multi",
        format_func=lambda i: f"{CLASS_NAMES[i]}  {available[i]}",
        default=[i for i in CLASS_NAMES if i != HEALTHY_ID],
        label_visibility="collapsed", key="class_pills")

enabled = set(picked or [])
shown = visible_detections(raw_detections, conf_thresh, enabled)
findings = [d for d in shown if d["class_id"] != HEALTHY_ID]
total_pathology = sum(1 for d in raw_detections if d["class_id"] != HEALTHY_ID)

thresh_head.markdown(
    f'<div class="thresh-head"><span class="t">Confidence threshold</span>'
    f'<span class="v">{conf_thresh:.2f}</span>'
    f'<span class="n">{len(findings)} of {total_pathology} findings shown · '
    f'filters the cache, no re-run</span></div>', unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# Saliency
# --------------------------------------------------------------------------- #

cam_stats: tuple[float, float, float] | None = None
cam_image: np.ndarray | None = None

if saliency_on and not EXPLAIN_AVAILABLE:
    st.markdown('<div class="notice warn"><b>Saliency needs the research package.</b> '
                'Install it with <code>pip install -r requirements-research.txt</code>. '
                'Showing detections instead.</div>', unsafe_allow_html=True)
    saliency_on = False

if saliency_on:
    cam_key = (state["image_sha"], int(imgsz), bundle["sha"] or bundle["path"])
    if state["cam"] is None or state["cam_key"] != cam_key:
        with st.spinner("Computing saliency…"):
            try:
                state.update({"cam": EigenCAM(model)(image_bgr, imgsz=int(imgsz)),
                              "cam_key": cam_key})
            except Exception as exc:
                state.update({"cam": None, "cam_key": None})
                st.markdown(f'<div class="notice alert">Saliency failed: {exc}</div>',
                            unsafe_allow_html=True)
    if state["cam"] is not None:
        cam = state["cam"]
        boxes = (np.array([[d["x1"], d["y1"], d["x2"], d["y2"]] for d in findings],
                          dtype=float) if findings else np.zeros((0, 4)))
        energy, chance = pointing_energy(cam, boxes)
        cam_stats = (energy, chance,
                     (energy / chance) if chance and chance > 0 else float("nan"))
        cam_image = overlay_cam(image_bgr, cam)

with viewport_slot:
    render_viewport(image_bgr, shown, height=vp_height,
                    show_chips=show_chips, cam_bgr=cam_image)

# --------------------------------------------------------------------------- #
# Rail: what was found, and how far each class can be trusted
# --------------------------------------------------------------------------- #

mean_conf = (sum(d["confidence"] for d in findings) / len(findings)) if findings else 0.0
class_count = len({d["class_id"] for d in findings})
weak = [d for d in findings
        if reliability_grade(d["class_id"])[0] in ("weak", "unmeasured")]


def tile(value: str, label: str, *, tone: str = "", delay: float = 0.0) -> str:
    return (f'<div class="tile" style="animation-delay:{delay:.2f}s;">'
            f'<div class="tile-v {tone}">{value}</div>'
            f'<div class="tile-k">{label}</div></div>')


with rail_col:
    st.markdown(
        '<div class="tiles">'
        + tile(str(len(findings)), "Findings",
               tone="" if findings else "muted", delay=0.04)
        + tile(str(class_count), "Conditions",
               tone="" if class_count else "muted", delay=0.09)
        + tile(f"{mean_conf:.2f}", "Mean confidence",
               tone="" if findings else "muted", delay=0.14)
        + tile(str(len(weak)), "On weakly-measured classes",
               tone="warn" if weak else "muted", delay=0.19)
        + '</div>', unsafe_allow_html=True)

    st.markdown(
        f'<div class="sec"><h2>Findings</h2><span>'
        + (f'above {conf_thresh:.2f} confidence · hover a marker on the scan to isolate one'
           if findings else f'none above {conf_thresh:.2f}')
        + '</span></div>', unsafe_allow_html=True)

    if not findings:
        st.markdown(
            f'<div class="notice quiet">Nothing above {conf_thresh:.2f} confidence in the '
            f'selected conditions. Lower the threshold or select more — the model was run '
            f'down to {CACHE_CONF:g}, so nothing needs re-analysing.</div>',
            unsafe_allow_html=True)
    else:
        rows = []
        for position, detection in enumerate(findings):
            class_id = detection["class_id"]
            colour = CLASS_COLORS_HEX[class_id]
            grade, tag_text, tooltip = reliability_grade(class_id)
            icon = {"ok": "", "weak": "↓ ", "unmeasured": "⚠ ", "unknown": ""}[grade]
            pct = int(round(detection["confidence"] * 100))
            delay = 0.04 + position * 0.05
            rows.append(
                f'<div class="find" style="--c:{colour};animation-delay:{delay:.2f}s;">'
                f'<div class="find-top">'
                f'<span class="find-i">{detection["index"]}</span>'
                f'<span class="find-n">{detection["class"]}</span>'
                f'<span class="find-c">{detection["confidence"]:.2f}</span></div>'
                f'<div class="find-bar"><i style="width:{pct}%;'
                f'animation-delay:{delay + 0.12:.2f}s;"></i></div>'
                f'<div class="find-foot">'
                f'<span class="tag {grade}" title="{tooltip}"><i></i>{icon}{tag_text}</span>'
                f'<span class="find-box">{detection["width"]}×{detection["height"]} px '
                f'at {detection["x1"]},{detection["y1"]}</span></div></div>')
        st.markdown('<div class="flist">' + "".join(rows) + '</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="railnote" style="margin-top:16px;">The pill on each finding is that '
        '<b>class\'s</b> measured recall on the validation split, not a property of the '
        'individual detection. <b class="w">⚠ unmeasured</b> means fewer than ten '
        'validation instances behind the number; <b class="w">↓</b> marks measured recall '
        'below 0.70.</div>', unsafe_allow_html=True)

if cam_stats is not None:
    energy, chance, lift = cam_stats
    if findings:
        verdict = ("concentrated on the detections" if lift == lift and lift >= 1.5
                   else "close to what a uniform heatmap would score — read it with caution")
        st.caption(
            f"Eigen-CAM over the last feature map before the detection head. "
            f"**{energy * 100:.0f}%** of the saliency falls inside the predicted boxes, "
            f"against **{chance * 100:.0f}%** by chance — a lift of **{lift:.1f}×**. "
            f"Saliency is {verdict}. A saliency map describes the model, never a patient.")
    else:
        st.caption("No conditions are selected, so there are no boxes to score the map against.")

# --------------------------------------------------------------------------- #
# Export (rendered into the slot reserved in the study bar)
# --------------------------------------------------------------------------- #

stem = Path(state["image_name"]).stem[:48] or "study"
meta = {"image_name": state["image_name"], "image_sha": state["image_sha"],
        "image_width": int(img_w), "image_height": int(img_h),
        "model_path": bundle["path"], "model_sha": bundle["sha"],
        "finetuned": bundle["finetuned"], "conf": conf_thresh, "iou": iou_thresh,
        "imgsz": int(imgsz), "enabled": enabled, "inference_ms": state["inference_ms"]}
try:
    png_bytes = encode_png(draw_overlay(image_bgr, shown, show_labels=True,
                                        box_opacity=box_opacity, fill=fill_boxes))
except Exception:
    png_bytes = b""

with export_slot.container():
    with st.popover("Export", use_container_width=True):
        st.download_button("Annotated PNG", data=png_bytes,
                           file_name=f"{stem}_annotated.png", mime="image/png",
                           disabled=not png_bytes, use_container_width=True)
        st.download_button("Findings CSV", data=findings_csv(shown),
                           file_name=f"{stem}_findings.csv", mime="text/csv",
                           disabled=not shown, use_container_width=True)
        st.download_button("Record JSON", data=findings_json(shown, meta),
                           file_name=f"{stem}_record.json", mime="application/json",
                           use_container_width=True)
        st.caption("The JSON record names the weights and the image by hash and "
                   "carries every threshold in force.")

# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #

st.markdown('<div style="height:44px;"></div>', unsafe_allow_html=True)

with st.expander("Where these numbers come from"):
    st.markdown(
        """
Everything on this page is reproducible from the harness in this repository.

**Per-class reliability.** Measured on a 23-image validation split at confidence
0.25 — `python evaluate.py per-class`. Aggregate mAP@0.5 is 0.923 over all six
classes but 0.894 over the three with at least ten validation instances. The
second number is the defensible one.

| Condition | Recall | Instances | 95% interval |
|---|---|---|---|
| Impacted | 1.00 | 18 | [0.82, 1.00] |
| Caries | 0.79 | 14 | [0.52, 0.92] |
| Healthy | 0.64 | 67 | [0.52, 0.75] |
| Fractured | 0.89 | 9 ⚠ | [0.56, 0.98] |
| Infection | 0.50 | 4 ⚠ | [0.15, 0.85] |
| BDC/BDR | 1.00 | 3 ⚠ | [0.44, 1.00] |

**Why `Healthy` is off by default.** It is a background label, not a diagnosis —
a `Healthy` box means "this region looked unremarkable to a detector". It is 58%
of all boxes in the training data and would bury the pathology.

**What the model actually gets wrong.** Inter-class confusion is roughly 1% of
predictions: given a proposal, the model names it correctly. Almost all residual
error is ground truth never proposed at all, concentrated on `Infection` (recall
0.50) and `Healthy` (0.64). Of false positives raised on background, 82% are
labelled `Healthy`.

**Confidence is not calibrated.** A displayed 0.80 does not mean 80% of such
detections are correct.

Full analysis: `report/main.pdf` · Scope and limits: `MODEL_CARD.md`
        """.strip())

st.markdown(f'<div class="foot" style="margin-top:26px;opacity:.75;">{DISCLAIMER}</div>',
            unsafe_allow_html=True)
