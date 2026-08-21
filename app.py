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

VIEWPORT_TEMPLATE = """
<style>
  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; background: transparent;
               font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
  #shell { position: relative; height: __HEIGHT__px; background: #08090B;
           border: 1px solid rgba(255,255,255,.07); border-radius: 16px;
           overflow: hidden; }
  /* Centring is done by the transform alone. Flex-centring the pane as well
     would offset every pan by the flex origin and make the image drift. */
  #stage { position: absolute; inset: 0; overflow: hidden; cursor: grab; }
  #stage.dragging { cursor: grabbing; }
  #pane { position: absolute; top: 0; left: 0; transform-origin: 0 0;
          will-change: transform; line-height: 0; }
  #pane img { display: block; max-width: none; user-select: none;
              -webkit-user-drag: none; }
  #top { position: absolute; top: 0; left: 0; }

  #bar { position: absolute; left: 50%; bottom: 16px; transform: translateX(-50%);
         display: flex; align-items: center; gap: 6px;
         padding: 7px 9px; border-radius: 13px;
         background: rgba(16,18,22,.82); border: 1px solid rgba(255,255,255,.09);
         backdrop-filter: blur(14px) saturate(1.3);
         box-shadow: 0 8px 30px -10px rgba(0,0,0,.8);
         opacity: 0; transition: opacity .22s ease; }
  #shell:hover #bar, #bar:focus-within { opacity: 1; }

  .btn { appearance: none; border: 0; background: transparent;
         color: #AEB5BE; font-size: 12.5px; font-weight: 450;
         padding: 6px 11px; border-radius: 8px; cursor: pointer; line-height: 1.4;
         font-variant-numeric: tabular-nums; }
  .btn:hover { background: rgba(255,255,255,.08); color: #FFF; }
  .btn:active { background: rgba(255,255,255,.14); }
  .btn:focus-visible { outline: 2px solid #5B9DFF; outline-offset: 1px; }
  .btn.icon { min-width: 30px; text-align: center; font-size: 15px; padding: 4px 8px; }
  #zoomval { font-size: 12px; color: #79818B; min-width: 44px; text-align: center;
             font-variant-numeric: tabular-nums; }
  .sep { width: 1px; height: 16px; background: rgba(255,255,255,.1); margin: 0 4px; }
  .lbl { font-size: 11px; color: #6C747E; letter-spacing: .3px; padding-left: 4px; }
  input[type=range] { -webkit-appearance: none; appearance: none; height: 3px;
                      background: rgba(255,255,255,.16); border-radius: 3px; width: 92px; }
  input[type=range]::-webkit-slider-thumb { -webkit-appearance: none; width: 12px;
      height: 12px; border-radius: 50%; background: #F2F4F6; cursor: pointer;
      box-shadow: 0 1px 3px rgba(0,0,0,.6); }
  input[type=range]::-moz-range-thumb { width: 12px; height: 12px; border-radius: 50%;
      background: #F2F4F6; border: 0; cursor: pointer; }

  #hint { position: absolute; top: 14px; left: 50%; transform: translateX(-50%);
          font-size: 11.5px; color: #79818B; background: rgba(16,18,22,.78);
          padding: 6px 13px; border-radius: 100px; border: 1px solid rgba(255,255,255,.07);
          pointer-events: none; backdrop-filter: blur(10px);
          transition: opacity .5s ease; }
</style>

<div id="shell">
  <div id="stage" tabindex="0" aria-label="Radiograph viewport">
    <div id="pane">
      <img id="base" src="data:image/jpeg;base64,__BASE__" alt="Radiograph" draggable="false">
      <img id="top"  src="data:image/jpeg;base64,__TOP__"  alt="Detection overlay" draggable="false">
    </div>
  </div>
  <div id="hint">scroll to zoom · drag to pan · double-click to fit</div>
  <div id="bar">
    <button class="btn icon" id="out"  title="Zoom out (-)" aria-label="Zoom out">&minus;</button>
    <span id="zoomval">100%</span>
    <button class="btn icon" id="in"   title="Zoom in (+)" aria-label="Zoom in">&plus;</button>
    <div class="sep"></div>
    <button class="btn" id="fit"  title="Fit to viewport (0)">Fit</button>
    <button class="btn" id="one"  title="Actual pixels (1)">1:1</button>
    <div class="sep"></div>
    <span class="lbl">Overlay</span>
    <input id="mix" type="range" min="0" max="100" value="100" aria-label="Overlay opacity">
    <button class="btn" id="hold" title="Hold to see the radiograph underneath">Compare</button>
  </div>
</div>

<script>
(function () {
  const stage = document.getElementById('stage');
  const pane  = document.getElementById('pane');
  const base  = document.getElementById('base');
  const top   = document.getElementById('top');
  const zoomval = document.getElementById('zoomval');
  const mix   = document.getElementById('mix');

  let scale = 1, tx = 0, ty = 0;
  const MIN = 0.1, MAX = 12;

  function apply() {
    pane.style.transform = 'translate(' + tx + 'px,' + ty + 'px) scale(' + scale + ')';
    zoomval.textContent = Math.round(scale * 100) + '%';
  }
  function fit() {
    const sw = stage.clientWidth, sh = stage.clientHeight;
    const iw = base.naturalWidth, ih = base.naturalHeight;
    if (!iw || !ih) return;
    scale = Math.min(sw / iw, sh / ih) * 0.94;
    tx = (sw - iw * scale) / 2;
    ty = (sh - ih * scale) / 2;
    apply();
  }
  function zoomAt(cx, cy, factor) {
    const next = Math.min(MAX, Math.max(MIN, scale * factor));
    const k = next / scale;
    tx = cx - (cx - tx) * k;
    ty = cy - (cy - ty) * k;
    scale = next;
    apply();
  }
  function centreZoom(f) { zoomAt(stage.clientWidth / 2, stage.clientHeight / 2, f); }

  stage.addEventListener('wheel', function (e) {
    e.preventDefault();
    const r = stage.getBoundingClientRect();
    zoomAt(e.clientX - r.left, e.clientY - r.top, e.deltaY < 0 ? 1.12 : 1 / 1.12);
  }, { passive: false });

  let dragging = false, lastX = 0, lastY = 0;
  stage.addEventListener('pointerdown', function (e) {
    dragging = true; lastX = e.clientX; lastY = e.clientY;
    stage.classList.add('dragging'); stage.setPointerCapture(e.pointerId);
  });
  stage.addEventListener('pointermove', function (e) {
    if (!dragging) return;
    tx += e.clientX - lastX; ty += e.clientY - lastY;
    lastX = e.clientX; lastY = e.clientY; apply();
  });
  ['pointerup', 'pointercancel'].forEach(function (ev) {
    stage.addEventListener(ev, function () {
      dragging = false; stage.classList.remove('dragging');
    });
  });
  stage.addEventListener('dblclick', fit);

  stage.addEventListener('keydown', function (e) {
    if (e.key === '+' || e.key === '=') { centreZoom(1.2); e.preventDefault(); }
    if (e.key === '-' || e.key === '_') { centreZoom(1 / 1.2); e.preventDefault(); }
    if (e.key === '0') { fit(); e.preventDefault(); }
    if (e.key === '1') { centreZoom(1 / scale); e.preventDefault(); }
  });

  document.getElementById('in').onclick  = function () { centreZoom(1.2); };
  document.getElementById('out').onclick = function () { centreZoom(1 / 1.2); };
  document.getElementById('fit').onclick = fit;
  document.getElementById('one').onclick = function () { centreZoom(1 / scale); };

  mix.addEventListener('input', function () { top.style.opacity = mix.value / 100; });

  // Press and hold to look at the untouched radiograph - the single most useful
  // gesture when judging whether a box sits on a real feature.
  const hold = document.getElementById('hold');
  hold.addEventListener('pointerdown', function (e) { e.preventDefault(); top.style.opacity = 0; });
  ['pointerup', 'pointerleave', 'pointercancel'].forEach(function (ev) {
    hold.addEventListener(ev, function () { top.style.opacity = mix.value / 100; });
  });

  // The affordance hint earns its place for a few seconds, then gets out of the way.
  const hint = document.getElementById('hint');
  setTimeout(function () { hint.style.opacity = '0'; }, 3600);

  if (base.complete) { fit(); } else { base.addEventListener('load', fit); }
  window.addEventListener('resize', fit);
})();
</script>
"""


def render_viewport(base_bgr: np.ndarray, overlay_bgr: np.ndarray,
                    height: int = 560) -> None:
    """Zoom/pan viewport with an overlay crossfade and a hold-to-compare button.

    Streamlit has no native image viewport, and a 1935-pixel panoramic scaled to
    fit a browser column makes a 30-pixel carious lesion unreadable. This is a
    self-contained component: two stacked images, CSS transforms for zoom and
    pan, no external libraries and no round-trip to the server.
    """
    html = (VIEWPORT_TEMPLATE
            .replace("__HEIGHT__", str(height))
            .replace("__BASE__", encode_jpeg(base_bgr))
            .replace("__TOP__", encode_jpeg(overlay_bgr)))
    components.html(html, height=height + 8, scrolling=False)


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

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;450;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
  --bg:      #0A0B0D;
  --card:    #101216;
  --card-2:  #15181D;
  --card-3:  #1C2026;
  --line:    rgba(255,255,255,.065);
  --line-2:  rgba(255,255,255,.11);
  --ink:     #F2F4F6;
  --ink-2:   #99A1AB;
  --ink-3:   #646C76;
  --accent:  #5B9DFF;
  --ok:      #3BAE7B;
  --warn:    #D4A03A;
  --alert:   #E0685F;
  --r:       16px;
  --r-s:     10px;
  --shadow:  0 1px 2px rgba(0,0,0,.4), 0 8px 28px -12px rgba(0,0,0,.6);
}

html, body, .stApp, [class*="css"] {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  -webkit-font-smoothing: antialiased; text-rendering: optimizeLegibility;
}
.stApp { background: var(--bg); color: var(--ink); }

/* A single centred column. No rail, no panels-inside-panels: the radiograph is
   the page, and everything else is a quiet line of controls around it. */
.block-container {
  max-width: 1180px !important;
  padding: 34px 28px 96px !important;
}
[data-testid="stHeader"] { background: transparent; height: 0; }
[data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"],
footer, #MainMenu { display: none !important; }
[data-testid="stSidebar"] { display: none; }

::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #262B31; border-radius: 6px;
                            border: 3px solid var(--bg); }
::-webkit-scrollbar-thumb:hover { background: #343A42; }
*:focus-visible { outline: 2px solid var(--accent) !important; outline-offset: 3px; }

/* ── Masthead ────────────────────────────────────────────────────────── */
.mast { display: flex; align-items: center; gap: 14px; margin-bottom: 30px; }
.mast-logo {
  width: 34px; height: 34px; border-radius: 10px; flex: none;
  background: linear-gradient(150deg, #22262C, #14171B);
  border: 1px solid var(--line-2);
  display: grid; place-items: center;
  font-size: 15px; color: var(--accent);
}
.mast-name { font-size: 17px; font-weight: 600; letter-spacing: -0.35px; line-height: 1.2; }
.mast-sub  { font-size: 12.5px; color: var(--ink-3); margin-top: 1px; }
.mast-sp { flex: 1 1 auto; }
.status {
  display: inline-flex; align-items: center; gap: 8px; padding: 6px 13px;
  border-radius: 100px; background: var(--card); border: 1px solid var(--line);
  font-size: 12.5px; color: var(--ink-2); white-space: nowrap;
}
.status code { font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: var(--ink-3); }
.dot { width: 6px; height: 6px; border-radius: 50%; flex: none; }
.dot.ok    { background: var(--ok);    box-shadow: 0 0 8px rgba(59,174,123,.7); }
.dot.alert { background: var(--alert); box-shadow: 0 0 8px rgba(224,104,95,.7); }

/* ── Hero copy (empty state) ─────────────────────────────────────────── */
.hero { text-align: center; padding: 26px 0 30px; }
.hero h1 {
  font-size: 40px; font-weight: 600; letter-spacing: -1.3px; line-height: 1.12;
  margin: 0 0 14px; color: var(--ink);
}
.hero p { font-size: 15.5px; line-height: 1.65; color: var(--ink-2);
          max-width: 560px; margin: 0 auto; }
.hero-notes { display: flex; justify-content: center; gap: 40px; flex-wrap: wrap;
              margin-top: 42px; }
.note { max-width: 210px; text-align: left; }
.note-t { font-size: 13px; font-weight: 550; color: var(--ink); margin-bottom: 5px; }
.note-d { font-size: 12.5px; line-height: 1.6; color: var(--ink-3); }

/* ── Study bar ───────────────────────────────────────────────────────── */
.studybar { display: flex; align-items: center; gap: 12px; margin-bottom: 4px; }
.study-name { font-size: 14.5px; font-weight: 550; color: var(--ink); }
.study-meta { font-size: 12.5px; color: var(--ink-3); }
.study-meta code { font-family: 'IBM Plex Mono', monospace; font-size: 11px; }

/* ── Stats ───────────────────────────────────────────────────────────── */
.stats { display: flex; gap: 54px; flex-wrap: wrap; margin: 30px 2px 8px; }
.stat-v { font-size: 34px; font-weight: 550; letter-spacing: -1.1px; line-height: 1;
          color: var(--ink); font-variant-numeric: tabular-nums; }
.stat-v.muted { color: var(--ink-3); }
.stat-v small { font-size: 15px; font-weight: 450; color: var(--ink-3);
                letter-spacing: 0; margin-left: 2px; }
.stat-k { font-size: 12px; color: var(--ink-3); margin-top: 9px; letter-spacing: .1px; }

/* ── Findings ────────────────────────────────────────────────────────── */
.sec-h { font-size: 13px; font-weight: 600; color: var(--ink-2); letter-spacing: .2px;
         margin: 40px 2px 14px; display: flex; align-items: baseline; gap: 10px; }
.sec-h span { font-size: 12.5px; font-weight: 400; color: var(--ink-3); }

.flist { border: 1px solid var(--line); border-radius: var(--r);
         background: var(--card); overflow: hidden; }
.frow { display: flex; align-items: center; gap: 16px; padding: 15px 20px;
        border-bottom: 1px solid var(--line); }
.frow:last-child { border-bottom: none; }
.frow:hover { background: var(--card-2); }
.f-i { font-family: 'IBM Plex Mono', monospace; font-size: 12px; color: var(--ink-3);
       width: 20px; flex: none; }
.f-dot { width: 9px; height: 9px; border-radius: 3px; flex: none; }
.f-name { font-size: 14.5px; font-weight: 500; min-width: 96px; }
.f-conf { font-size: 14px; font-variant-numeric: tabular-nums; color: var(--ink-2);
          width: 46px; text-align: right; flex: none; }
.f-bar { height: 3px; border-radius: 2px; background: var(--card-3);
         flex: 1 1 auto; max-width: 220px; min-width: 60px; overflow: hidden; }
.f-bar i { display: block; height: 100%; border-radius: 2px; }
.f-box { font-family: 'IBM Plex Mono', monospace; font-size: 11.5px; color: var(--ink-3);
         white-space: nowrap; margin-left: auto; }
.tag { display: inline-flex; align-items: center; gap: 5px; padding: 3px 10px;
       border-radius: 100px; font-size: 11.5px; white-space: nowrap; flex: none;
       font-variant-numeric: tabular-nums; }
.tag.ok         { background: rgba(59,174,123,.12); color: #6ECAA0; }
.tag.weak       { background: rgba(212,160,58,.14); color: #E0B968; }
.tag.unmeasured { background: rgba(224,104,95,.14); color: #F0968E; }
.tag.unknown    { background: var(--card-3);        color: var(--ink-3); }

/* ── Notices ─────────────────────────────────────────────────────────── */
.notice { border-radius: var(--r-s); padding: 14px 17px; margin-bottom: 18px;
          font-size: 13.5px; line-height: 1.6; border: 1px solid; }
.notice.alert { background: rgba(224,104,95,.07); border-color: rgba(224,104,95,.3);
                color: #F0968E; }
.notice.warn  { background: rgba(212,160,58,.07); border-color: rgba(212,160,58,.28);
                color: #E0B968; }
.notice.quiet { background: var(--card); border-color: var(--line); color: var(--ink-2); }
.notice b { color: var(--ink); font-weight: 600; }
.notice code { font-family: 'IBM Plex Mono', monospace; font-size: 12px; }

.foot { font-size: 12.5px; color: var(--ink-3); line-height: 1.7; margin-top: 14px; }
.foot b { color: var(--ink-2); font-weight: 500; }

/* ── Streamlit widgets ───────────────────────────────────────────────── */
[data-testid="stFileUploaderDropzone"] {
  background: var(--card); border: 1px dashed var(--line-2); border-radius: var(--r);
  padding: 30px 26px; transition: border-color .15s ease, background .15s ease;
}
[data-testid="stFileUploaderDropzone"]:hover {
  border-color: rgba(91,157,255,.55); background: var(--card-2);
}
[data-testid="stFileUploaderDropzone"] button {
  background: var(--ink) !important; color: #0A0B0D !important; border: none !important;
  font-weight: 550 !important; border-radius: 8px !important;
}
[data-testid="stFileUploaderDropzone"] small { color: var(--ink-3); }
[data-testid="stFileUploaderFile"] { background: var(--card-2); border-radius: 8px; }

.stButton > button, .stDownloadButton > button, [data-testid="stPopover"] button {
  background: var(--card); color: var(--ink-2); border: 1px solid var(--line);
  border-radius: 9px; font-size: 13px; font-weight: 500; padding: 8px 15px;
}
.stButton > button:hover, .stDownloadButton > button:hover,
[data-testid="stPopover"] button:hover {
  background: var(--card-2); border-color: var(--line-2); color: var(--ink);
}
.stButton > button[kind="primary"] {
  background: var(--ink); border-color: var(--ink); color: #0A0B0D; font-weight: 600;
}
.stButton > button[kind="primary"]:hover { background: #FFF; border-color: #FFF; }
[data-testid="stPopoverBody"] {
  background: var(--card); border: 1px solid var(--line-2); border-radius: 14px;
  box-shadow: var(--shadow); padding: 6px 4px;
}

/* Pills report their state through aria-pressed, not aria-checked. */
button[aria-pressed] {
  background: transparent !important; border: 1px solid var(--line) !important;
  color: var(--ink-3) !important; border-radius: 100px !important;
  font-size: 12.5px !important; font-weight: 450 !important;
  padding: 6px 15px !important;
}
button[aria-pressed]:hover { border-color: var(--line-2) !important;
                             color: var(--ink-2) !important; }
button[aria-pressed="true"] {
  background: var(--ink) !important; border-color: var(--ink) !important;
  color: #0A0B0D !important; font-weight: 550 !important;
}
button[aria-pressed="true"]:hover { background: #FFF !important; color: #0A0B0D !important; }

.stSlider [data-baseweb="slider"] div[role="slider"] {
  background: var(--ink) !important; border: 2px solid var(--bg) !important;
  box-shadow: 0 1px 4px rgba(0,0,0,.5) !important;
}
[data-testid="stWidgetLabel"] p, .stSlider label, .stCheckbox label, .stToggle label {
  font-size: 12.5px !important; color: var(--ink-2) !important; font-weight: 450 !important;
}
[data-testid="stExpander"] {
  border: 1px solid var(--line); border-radius: var(--r-s); background: var(--card);
}
[data-testid="stExpander"] summary { font-size: 13px; color: var(--ink-2); padding: 4px 2px; }
[data-testid="stExpander"] summary:hover { color: var(--ink); }
[data-testid="stExpanderDetails"] p, [data-testid="stExpanderDetails"] li {
  font-size: 13.5px; line-height: 1.75; color: var(--ink-2);
}
[data-testid="stExpanderDetails"] strong { color: var(--ink); }
[data-testid="stCaptionContainer"] p { font-size: 12.5px; color: var(--ink-3);
                                       line-height: 1.65; }
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
hr { border-color: var(--line); }
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

bundle = load_model()
model = bundle["model"]

if model is None:
    status_class, status_text = "alert", "model unavailable"
elif bundle["finetuned"]:
    status_class, status_text = "ok", "fine-tuned"
else:
    status_class, status_text = "alert", "base COCO weights"

st.markdown(
    f"""
    <div class="mast">
      <div class="mast-logo">◧</div>
      <div>
        <div class="mast-name">DentalScan</div>
        <div class="mast-sub">condition detection on panoramic radiographs</div>
      </div>
      <div class="mast-sp"></div>
      <span class="status"><span class="dot {status_class}"></span>{status_text}
        {f'<code>{bundle["sha"]}</code>' if bundle["sha"] else ''}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

if model is None:
    st.markdown(
        f'<div class="notice alert"><b>The detector could not be loaded.</b><br>'
        f'{bundle["error"]}<br>Expected weights at <code>{FINETUNED_PATH}</code>. '
        f'Train them with <code>python model.py train --name baseline</code>, or copy '
        f'<code>best.pt</code> into that path.</div>',
        unsafe_allow_html=True)
    st.stop()

if not bundle["finetuned"]:
    st.markdown(
        f'<div class="notice alert"><b>Fine-tuned weights are missing, so this is the '
        f'base COCO model.</b> It predicts everyday objects, not dental conditions — '
        f'every label below will be wrong. Place the trained checkpoint at '
        f'<code>{FINETUNED_PATH}</code>.</div>',
        unsafe_allow_html=True)

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
    """Adopt a new upload, or clear state when the uploader is emptied.

    Returns True when state changed, which the caller turns into a rerun: the
    widget callback that delivered the file has already consumed this run, so
    without an explicit rerun the page would render one step behind the upload.
    """
    # Deliberately does nothing when the uploader reports None. The widget is
    # torn down when the page switches from the empty state to the loaded one,
    # and treating that transient None as "the user cleared the study" wipes the
    # image on the very rerun that was supposed to display it. Closing a study
    # is an explicit action instead - see clear_study().
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
    """Explicitly close the current study."""
    state.update({"image": None, "image_name": "", "image_sha": "",
                  "detections": [], "cache_key": None, "cam": None,
                  "cam_key": None, "error": ""})


has_image = state["image"] is not None

# --------------------------------------------------------------------------- #
# Empty state - the upload *is* the page
# --------------------------------------------------------------------------- #

if not has_image:
    st.markdown(
        """
        <div class="hero">
          <h1>Read a panoramic radiograph.</h1>
          <p>Six dental conditions, detected and scored — with the measured
             reliability of each class shown next to every finding, so nothing
             claims more confidence than the evidence behind it.</p>
        </div>
        """, unsafe_allow_html=True)

    pad_l, centre, pad_r = st.columns([1, 3, 1])
    with centre:
        upload = st.file_uploader(
            "Radiograph", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
            label_visibility="collapsed", key="uploader")
    if ingest(upload) and state["image"] is not None:
        st.rerun()
    if state["error"]:
        with centre:
            st.markdown(f'<div class="notice alert" style="margin-top:16px;">'
                        f'{state["error"]}</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="hero-notes">
          <div class="note">
            <div class="note-t">One pass, instant controls</div>
            <div class="note-d">The model runs once. Thresholds and class filters
              re-read that result rather than re-running it.</div>
          </div>
          <div class="note">
            <div class="note-t">Zoom that matters</div>
            <div class="note-d">A 30-pixel lesion on a 1935-pixel panoramic is
              invisible at fit-to-width. Scroll to zoom, drag to pan.</div>
          </div>
          <div class="note">
            <div class="note-t">Honest numbers</div>
            <div class="note-d">Classes measured on fewer than ten instances are
              marked as unmeasured rather than quietly averaged in.</div>
          </div>
        </div>
        """, unsafe_allow_html=True)
    st.stop()

# --------------------------------------------------------------------------- #
# Loaded study
# --------------------------------------------------------------------------- #

image_bgr = state["image"]
img_h, img_w = image_bgr.shape[:2]

name_col, tog_col, chg_col, set_col, exp_col, close_col = st.columns(
    [4.4, 1.45, 1.3, 1.2, 1.1, 0.55], vertical_alignment="center")

with name_col:
    st.markdown(
        f'<div class="studybar"><span class="study-name">{state["image_name"]}</span>'
        f'<span class="study-meta">{img_w} × {img_h} px · '
        f'<code>{state["image_sha"]}</code></span></div>',
        unsafe_allow_html=True)

with tog_col:
    saliency_on = st.toggle("Saliency", value=False, key="saliency_on",
                            help="Show where the model looked instead of what it found.")

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
        show_labels = st.checkbox("Box labels", value=True)
        fill_boxes = st.checkbox("Interior wash", value=True,
                                 help="A faint fill makes small boxes findable when "
                                      "the whole radiograph is in view.")
        box_opacity = st.slider("Overlay strength", 0.2, 1.0, 1.0, 0.05)
        st.divider()
        st.caption("Inference — changing these re-runs the model")
        iou_thresh = st.slider("NMS IoU", 0.10, 0.90, 0.70, 0.05)
        imgsz = st.select_slider("Inference size", options=[512, 640, 800, 960], value=640)

export_slot = exp_col.empty()

with close_col:
    if st.button("✕", help="Close this study", use_container_width=True):
        clear_study()
        st.rerun()

# Inference is keyed on everything that can change its output. Confidence is
# deliberately not in the key: it filters the cache instead of re-running.
if not state["error"]:
    key = (state["image_sha"], round(iou_thresh, 3), int(imgsz),
           bundle["sha"] or bundle["path"])
    if state["cache_key"] != key:
        with st.spinner("Reading radiograph…"):
            try:
                detections, elapsed = run_inference(model, image_bgr, iou_thresh, imgsz)
                state.update({"detections": detections, "inference_ms": elapsed,
                              "cache_key": key, "error": ""})
            except Exception as exc:
                state.update({"detections": [], "cache_key": None,
                              "error": f"Inference failed: {exc}"})

if state["error"]:
    st.markdown(f'<div class="notice alert" style="margin-top:14px;">'
                f'<b>Could not analyse this study.</b><br>{state["error"]}</div>',
                unsafe_allow_html=True)
    st.stop()

raw_detections: list[dict] = state["detections"]

# --------------------------------------------------------------------------- #
# Layout slots
# --------------------------------------------------------------------------- #
# The confidence threshold is read before the class pills are drawn, because the
# count on each pill depends on it. Slots let the pills and the viewport appear
# above the slider while being *built* after it, so nothing on screen is ever a
# step behind the control that governs it.

pills_slot = st.container()
viewport_slot = st.container()

slider_col, readout_col = st.columns([5, 2], vertical_alignment="center")
with slider_col:
    conf_thresh = st.slider("Confidence", 0.05, 0.95, value=float(state["conf_thresh"]),
                            step=0.01, key="conf_slider", label_visibility="collapsed")
state["conf_thresh"] = conf_thresh

available = {
    class_id: sum(1 for d in raw_detections
                  if d["class_id"] == class_id and d["confidence"] >= conf_thresh)
    for class_id in CLASS_NAMES
}

with pills_slot:
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

with readout_col:
    st.markdown(
        f'<div class="study-meta" style="text-align:right;">confidence &ge; '
        f'<b style="color:var(--ink);">{conf_thresh:.2f}</b> &middot; '
        f'{len(findings)} shown of {total_pathology} above {CACHE_CONF:g}</div>',
        unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# Viewport
# --------------------------------------------------------------------------- #

overlay_bgr = draw_overlay(image_bgr, shown, show_labels=show_labels,
                           box_opacity=box_opacity, fill=fill_boxes)

# Size the frame to the study rather than to a fixed number, so a panoramic does
# not sit in a tall letterbox and a periapical is not crushed into a strip.
vp_height = int(np.clip(round(1100 / max(img_w / img_h, 0.35)), 360, 620))

cam_stats: tuple[float, float, float] | None = None

if saliency_on and not EXPLAIN_AVAILABLE:
    st.markdown(
        '<div class="notice warn"><b>Saliency needs the research package.</b> '
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
        with viewport_slot:
            render_viewport(image_bgr, overlay_cam(image_bgr, cam), height=vp_height)
    else:
        saliency_on = False
        with viewport_slot:
            render_viewport(image_bgr, overlay_bgr, height=vp_height)
else:
    with viewport_slot:
        render_viewport(image_bgr, overlay_bgr, height=vp_height)

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
# Summary and findings
# --------------------------------------------------------------------------- #

mean_conf = (sum(d["confidence"] for d in findings) / len(findings)) if findings else 0.0
class_count = len({d["class_id"] for d in findings})
weak = [d for d in findings
        if reliability_grade(d["class_id"])[0] in ("weak", "unmeasured")]

st.markdown(
    f"""
    <div class="stats">
      <div>
        <div class="stat-v{'' if findings else ' muted'}">{len(findings)}</div>
        <div class="stat-k">Findings</div>
      </div>
      <div>
        <div class="stat-v{'' if class_count else ' muted'}">{class_count}</div>
        <div class="stat-k">Conditions</div>
      </div>
      <div>
        <div class="stat-v{'' if findings else ' muted'}">{mean_conf:.2f}</div>
        <div class="stat-k">Mean confidence</div>
      </div>
      <div>
        <div class="stat-v{'' if weak else ' muted'}">{len(weak)}</div>
        <div class="stat-k">On weakly-measured classes</div>
      </div>
      <div>
        <div class="stat-v muted">{state['inference_ms']:.0f}<small>ms</small></div>
        <div class="stat-k">Inference</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown(
    f'<div class="sec-h">Findings<span>{len(findings)} above {conf_thresh:.2f} '
    f'confidence</span></div>', unsafe_allow_html=True)

if not findings:
    st.markdown(
        f'<div class="notice quiet">Nothing above {conf_thresh:.2f} confidence in the '
        f'selected conditions. Lower the threshold or select more — the model was run '
        f'down to {CACHE_CONF:g}, so nothing needs re-analysing.</div>',
        unsafe_allow_html=True)
else:
    rows = []
    for detection in findings:
        class_id = detection["class_id"]
        colour = CLASS_COLORS_HEX[class_id]
        grade, tag_text, tooltip = reliability_grade(class_id)
        icon = {"ok": "", "weak": "↓ ", "unmeasured": "⚠ ", "unknown": ""}[grade]
        pct = int(round(detection["confidence"] * 100))
        rows.append(
            f'<div class="frow">'
            f'<span class="f-i">{detection["index"]}</span>'
            f'<span class="f-dot" style="background:{colour};"></span>'
            f'<span class="f-name">{detection["class"]}</span>'
            f'<span class="f-conf">{detection["confidence"]:.2f}</span>'
            f'<span class="f-bar"><i style="width:{pct}%;background:{colour};"></i></span>'
            f'<span class="tag {grade}" title="{tooltip}">{icon}{tag_text}</span>'
            f'<span class="f-box">{detection["x1"]},{detection["y1"]} → '
            f'{detection["x2"]},{detection["y2"]} · {detection["width"]}×{detection["height"]}</span>'
            f'</div>')
    st.markdown('<div class="flist">' + "".join(rows) + '</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="foot">The pill on each row is that <b>class\'s</b> measured recall '
        'on the validation split — not a property of the individual finding. '
        '<b>⚠</b> marks a class with fewer than ten validation instances, which makes it '
        'effectively unmeasured; <b>↓</b> marks measured recall below 0.70.</div>',
        unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# Export (rendered into the slot reserved in the study bar)
# --------------------------------------------------------------------------- #

stem = Path(state["image_name"]).stem[:48] or "study"
meta = {
    "image_name": state["image_name"], "image_sha": state["image_sha"],
    "image_width": int(img_w), "image_height": int(img_h),
    "model_path": bundle["path"], "model_sha": bundle["sha"],
    "finetuned": bundle["finetuned"], "conf": conf_thresh, "iou": iou_thresh,
    "imgsz": int(imgsz), "enabled": enabled, "inference_ms": state["inference_ms"],
}
try:
    png_bytes = encode_png(overlay_bgr)
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

st.markdown('<div style="height:30px;"></div>', unsafe_allow_html=True)

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

st.markdown(
    f'<div class="foot" style="margin-top:26px;opacity:.75;">{DISCLAIMER}</div>',
    unsafe_allow_html=True)
