"""Run a detector once and cache its output for repeated offline analysis.

Every downstream analysis in this repo - per-class metrics, bootstrap intervals,
error decomposition, threshold sweeps, model-vs-model comparison - reads from
this cache rather than re-running the network. One inference pass, many
analyses, and every number in the report traceable to a single stored artefact.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .constants import CLASS_NAMES, DEFAULT_IOU
from .data import find_label_path, list_images, read_label_file
from .metrics import ImageRecord


def _yolo_to_xyxy(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    """Normalised ``(xc, yc, w, h)`` -> absolute ``(x1, y1, x2, y2)``."""
    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float64)
    xc, yc, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    return np.stack(
        [
            (xc - w / 2) * width,
            (yc - h / 2) * height,
            (xc + w / 2) * width,
            (yc + h / 2) * height,
        ],
        axis=1,
    )


def build_prediction_cache(
    model_path: str | Path,
    image_dirs: Iterable[str | Path],
    conf: float = 0.001,
    iou: float = DEFAULT_IOU,
    imgsz: int = 640,
    device: str | None = None,
    batch: int = 8,
    max_det: int = 300,
) -> list[ImageRecord]:
    """Predict on every image under ``image_dirs`` and pair with ground truth.

    ``conf`` defaults to 0.001, not the app's 0.25: average precision needs the
    full score range. Operating-point precision/recall is then computed by
    thresholding this cache, so a threshold sweep costs nothing.
    """
    from ultralytics import YOLO  # imported lazily - keeps analysis importable without torch

    images: list[Path] = []
    for directory in image_dirs:
        images.extend(list_images(Path(directory)))
    if not images:
        raise FileNotFoundError(f"No images found under {list(image_dirs)}")

    model = YOLO(str(model_path))
    records: list[ImageRecord] = []

    for start in range(0, len(images), batch):
        chunk = images[start:start + batch]
        results = model.predict(
            [str(p) for p in chunk],
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=device,
            max_det=max_det,
            verbose=False,
        )
        for image_path, result in zip(chunk, results):
            height, width = result.orig_shape

            if result.boxes is not None and len(result.boxes):
                pred_boxes = result.boxes.xyxy.cpu().numpy().astype(np.float64)
                pred_scores = result.boxes.conf.cpu().numpy().astype(np.float64)
                pred_classes = result.boxes.cls.cpu().numpy().astype(np.int64)
            else:
                pred_boxes = np.zeros((0, 4))
                pred_scores = np.zeros(0)
                pred_classes = np.zeros(0, dtype=np.int64)

            label_rows = read_label_file(find_label_path(image_path))
            if label_rows:
                gt_classes = np.array([r[0] for r in label_rows], dtype=np.int64)
                gt_boxes = _yolo_to_xyxy(
                    np.array([r[1:] for r in label_rows], dtype=np.float64), width, height
                )
            else:
                gt_classes = np.zeros(0, dtype=np.int64)
                gt_boxes = np.zeros((0, 4))

            records.append(
                ImageRecord(
                    image_id=str(image_path),
                    pred_boxes=pred_boxes,
                    pred_scores=pred_scores,
                    pred_classes=pred_classes,
                    gt_boxes=gt_boxes,
                    gt_classes=gt_classes,
                )
            )
    return records


def save_cache(records: Sequence[ImageRecord], path: str | Path) -> Path:
    """Write a prediction cache to a compressed ``.npz``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {}
    index: list[str] = []
    for i, record in enumerate(records):
        index.append(record.image_id)
        payload[f"pb_{i}"] = record.pred_boxes
        payload[f"ps_{i}"] = record.pred_scores
        payload[f"pc_{i}"] = record.pred_classes
        payload[f"gb_{i}"] = record.gt_boxes
        payload[f"gc_{i}"] = record.gt_classes
    payload["_index"] = np.array(index, dtype=object)
    np.savez_compressed(path, **payload, allow_pickle=True)
    return path


def load_cache(path: str | Path) -> list[ImageRecord]:
    """Read back a cache written by :func:`save_cache`."""
    data = np.load(Path(path), allow_pickle=True)
    index = list(data["_index"])
    return [
        ImageRecord(
            image_id=str(image_id),
            pred_boxes=data[f"pb_{i}"],
            pred_scores=data[f"ps_{i}"],
            pred_classes=data[f"pc_{i}"],
            gt_boxes=data[f"gb_{i}"],
            gt_classes=data[f"gc_{i}"],
        )
        for i, image_id in enumerate(index)
    ]


def cache_summary(records: Sequence[ImageRecord]) -> dict:
    """Small human-readable description of what a cache contains."""
    gt = np.concatenate([r.gt_classes for r in records]) if records else np.zeros(0)
    pred = np.concatenate([r.pred_classes for r in records]) if records else np.zeros(0)
    return {
        "n_images": len(records),
        "n_gt_boxes": int(gt.size),
        "n_pred_boxes": int(pred.size),
        "gt_per_class": {
            name: int((gt == i).sum()) for i, name in enumerate(CLASS_NAMES)
        },
        "pred_per_class": {
            name: int((pred == i).sum()) for i, name in enumerate(CLASS_NAMES)
        },
    }
