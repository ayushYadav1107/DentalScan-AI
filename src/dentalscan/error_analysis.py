"""Decompose detection errors into named, actionable categories.

A single mAP number tells you the model is wrong; it does not tell you *how*.
This module implements a TIDE-style decomposition (Bolya et al., ECCV 2020),
which sorts every false positive into one of five buckets and separately counts
ground truth the model never proposed at all:

============  ==========================================================
Category      Meaning
============  ==========================================================
classification  Right box, wrong label (IoU >= 0.5 with a GT of another class)
localization    Right label, sloppy box (0.1 <= IoU < 0.5, same class)
both            Wrong label *and* sloppy box
duplicate       Correct object already claimed by a higher-scoring detection
background      IoU < 0.1 with everything - a hallucination
missed          Ground truth with no detection above threshold
============  ==========================================================

The distinction matters for what you do next. Classification errors point at the
head and the label taxonomy; localization errors at the box regression and
anchor/stride choice; background errors at the negative distribution and the
confidence threshold; missed detections at recall, class imbalance and loss
weighting. On this dataset the decomposition is decisive - see the report.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .constants import CLASS_NAMES
from .metrics import ImageRecord, box_iou

BACKGROUND_IOU = 0.1
FOREGROUND_IOU = 0.5

ERROR_CATEGORIES = (
    "classification",
    "localization",
    "both",
    "duplicate",
    "background",
)


@dataclass
class ErrorBreakdown:
    """Error counts overall and per class."""

    conf_thr: float
    n_predictions: int = 0
    n_ground_truth: int = 0
    n_true_positive: int = 0
    n_missed: int = 0
    by_category: dict[str, int] = field(default_factory=dict)
    by_class: dict[str, dict[str, int]] = field(default_factory=dict)
    # Per-class counts of what a missed GT was confused with (or nothing).
    missed_by_class: dict[str, int] = field(default_factory=dict)

    def as_fractions(self) -> dict[str, float]:
        total = sum(self.by_category.values()) + self.n_missed
        if not total:
            return {k: 0.0 for k in (*ERROR_CATEGORIES, "missed")}
        out = {k: v / total for k, v in self.by_category.items()}
        out["missed"] = self.n_missed / total
        return out

    def dominant_failure(self) -> str:
        """The single category that accounts for most of the error mass."""
        fractions = self.as_fractions()
        return max(fractions, key=fractions.get) if fractions else ""


def analyse_errors(
    records: Sequence[ImageRecord],
    conf_thr: float = 0.25,
    class_names: Sequence[str] = tuple(CLASS_NAMES),
) -> ErrorBreakdown:
    """Categorise every false positive and every missed ground-truth box."""
    breakdown = ErrorBreakdown(conf_thr=conf_thr)
    breakdown.by_category = {c: 0 for c in ERROR_CATEGORIES}
    breakdown.by_class = {
        n: {c: 0 for c in (*ERROR_CATEGORIES, "true_positive", "missed")}
        for n in class_names
    }
    breakdown.missed_by_class = {n: 0 for n in class_names}

    for record in records:
        keep = record.pred_scores >= conf_thr
        boxes = record.pred_boxes[keep]
        scores = record.pred_scores[keep]
        classes = record.pred_classes[keep]

        order = np.argsort(-scores, kind="stable")
        boxes, classes = boxes[order], classes[order]

        breakdown.n_predictions += len(boxes)
        breakdown.n_ground_truth += len(record.gt_boxes)

        ious = box_iou(boxes, record.gt_boxes)
        claimed = np.zeros(len(record.gt_boxes), dtype=bool)
        matched_gt = np.zeros(len(record.gt_boxes), dtype=bool)

        for i in range(len(boxes)):
            pred_cls = int(classes[i])
            pred_name = class_names[pred_cls] if pred_cls < len(class_names) else str(pred_cls)
            row = ious[i] if ious.size else np.zeros(0)

            same_cls = np.where(record.gt_classes == pred_cls)[0]
            other_cls = np.where(record.gt_classes != pred_cls)[0]

            best_same = same_cls[np.argmax(row[same_cls])] if same_cls.size else -1
            best_other = other_cls[np.argmax(row[other_cls])] if other_cls.size else -1
            iou_same = row[best_same] if best_same >= 0 else 0.0
            iou_other = row[best_other] if best_other >= 0 else 0.0

            if iou_same >= FOREGROUND_IOU:
                if claimed[best_same]:
                    category = "duplicate"
                else:
                    claimed[best_same] = True
                    matched_gt[best_same] = True
                    breakdown.n_true_positive += 1
                    breakdown.by_class[pred_name]["true_positive"] += 1
                    continue
            elif iou_other >= FOREGROUND_IOU:
                category = "classification"
            elif iou_same >= BACKGROUND_IOU:
                category = "localization"
            elif iou_other >= BACKGROUND_IOU:
                category = "both"
            else:
                category = "background"

            breakdown.by_category[category] += 1
            breakdown.by_class[pred_name][category] += 1

        for j, hit in enumerate(matched_gt):
            if hit:
                continue
            gt_cls = int(record.gt_classes[j])
            name = class_names[gt_cls] if gt_cls < len(class_names) else str(gt_cls)
            breakdown.n_missed += 1
            breakdown.missed_by_class[name] += 1
            breakdown.by_class[name]["missed"] += 1

    return breakdown


def size_stratified_recall(
    records: Sequence[ImageRecord],
    conf_thr: float = 0.25,
    iou_thr: float = 0.5,
    class_names: Sequence[str] = tuple(CLASS_NAMES),
) -> dict[str, dict[str, float]]:
    """Recall split by ground-truth box area terciles.

    Detectors on radiographs usually fail on the smallest lesions first. If
    recall is flat across terciles the bottleneck is contrast or class support,
    not scale - which changes what you would fix.
    """
    areas: list[float] = []
    for record in records:
        wh = np.clip(record.gt_boxes[:, 2:] - record.gt_boxes[:, :2], 0, None)
        areas.extend((wh[:, 0] * wh[:, 1]).tolist())
    if not areas:
        return {}

    q33, q66 = np.percentile(areas, [33.3, 66.7])

    def bucket(area: float) -> str:
        return "small" if area < q33 else ("medium" if area < q66 else "large")

    hits: dict[tuple[str, str], int] = defaultdict(int)
    totals: dict[tuple[str, str], int] = defaultdict(int)

    for record in records:
        keep = record.pred_scores >= conf_thr
        boxes, classes = record.pred_boxes[keep], record.pred_classes[keep]
        ious = box_iou(boxes, record.gt_boxes)

        for j in range(len(record.gt_boxes)):
            gt_cls = int(record.gt_classes[j])
            name = class_names[gt_cls] if gt_cls < len(class_names) else str(gt_cls)
            wh = np.clip(record.gt_boxes[j, 2:] - record.gt_boxes[j, :2], 0, None)
            key = (name, bucket(float(wh[0] * wh[1])))
            totals[key] += 1
            if ious.size:
                same = np.where(classes == gt_cls)[0]
                if same.size and ious[same, j].max() >= iou_thr:
                    hits[key] += 1

    out: dict[str, dict[str, float]] = {}
    for name in (*class_names, "ALL"):
        row: dict[str, float] = {}
        for size in ("small", "medium", "large"):
            if name == "ALL":
                total = sum(v for (c, s), v in totals.items() if s == size)
                hit = sum(v for (c, s), v in hits.items() if s == size)
            else:
                total = totals.get((name, size), 0)
                hit = hits.get((name, size), 0)
            row[size] = hit / total if total else float("nan")
            row[f"{size}_n"] = total
        out[name] = row
    out["_thresholds"] = {"area_q33_px2": float(q33), "area_q66_px2": float(q66)}
    return out


def confidence_sweep(
    records: Sequence[ImageRecord],
    thresholds: Sequence[float] = tuple(np.round(np.arange(0.05, 0.91, 0.05), 2)),
    iou_thr: float = 0.5,
    class_names: Sequence[str] = tuple(CLASS_NAMES),
) -> list[dict]:
    """Per-class precision/recall/F1 as the confidence threshold varies.

    Screening and triage want different operating points: a triage tool that
    flags radiographs for review should sit at high recall, a tool that writes
    into a record should sit at high precision. A single default threshold
    silently picks one. This sweep makes the choice explicit and lets the report
    state a per-class F1-optimal threshold.
    """
    from .metrics import evaluate_detections

    rows: list[dict] = []
    for thr in thresholds:
        metrics = evaluate_detections(
            records, iou_thr=iou_thr, conf_thr=float(thr),
            iou_sweep=[iou_thr], class_names=class_names,
        )
        for name, m in metrics.items():
            rows.append({
                "conf": float(thr),
                "class": name,
                "support": m.support,
                "precision": m.precision,
                "recall": m.recall,
                "f1": m.f1,
                "tp": m.tp, "fp": m.fp, "fn": m.fn,
            })
    return rows


def best_thresholds(sweep_rows: Sequence[dict]) -> dict[str, dict[str, float]]:
    """F1-optimal confidence threshold per class, from a sweep."""
    best: dict[str, dict[str, float]] = {}
    by_class: dict[str, list[dict]] = defaultdict(list)
    for row in sweep_rows:
        by_class[row["class"]].append(row)
    for name, rows in by_class.items():
        valid = [r for r in rows if not np.isnan(r["f1"])]
        if not valid:
            continue
        top = max(valid, key=lambda r: r["f1"])
        best[name] = {
            "conf": top["conf"], "f1": top["f1"],
            "precision": top["precision"], "recall": top["recall"],
            "support": top["support"],
        }
    return best
