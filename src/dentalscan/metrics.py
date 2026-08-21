"""Detection metrics computed from cached predictions.

Why re-implement AP instead of reading it off ``model.val()``?

Because we need *uncertainty*. A bootstrap confidence interval requires
recomputing the metric on hundreds of resamples of the evaluation set, and
re-running inference each time is wasteful. Here inference runs once, its output
is cached per image, and every metric - including every bootstrap replicate - is
derived from that cache in NumPy.

The AP definition follows COCO: greedy matching of detections to ground truth in
descending confidence order at a fixed IoU threshold, then all-point
interpolated area under the precision-recall curve. Values agree with
Ultralytics to within numerical noise; ``tests/test_metrics.py`` pins the
behaviour on hand-checked cases.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .constants import CLASS_NAMES, NUM_CLASSES


@dataclass
class ImageRecord:
    """Cached detector output and ground truth for a single image."""

    image_id: str
    pred_boxes: np.ndarray      # (N, 4) xyxy in pixels
    pred_scores: np.ndarray     # (N,)
    pred_classes: np.ndarray    # (N,) int
    gt_boxes: np.ndarray        # (M, 4) xyxy in pixels
    gt_classes: np.ndarray      # (M,) int

    def __post_init__(self) -> None:
        self.pred_boxes = np.asarray(self.pred_boxes, dtype=np.float64).reshape(-1, 4)
        self.pred_scores = np.asarray(self.pred_scores, dtype=np.float64).reshape(-1)
        self.pred_classes = np.asarray(self.pred_classes, dtype=np.int64).reshape(-1)
        self.gt_boxes = np.asarray(self.gt_boxes, dtype=np.float64).reshape(-1, 4)
        self.gt_classes = np.asarray(self.gt_classes, dtype=np.int64).reshape(-1)


def box_iou(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two sets of xyxy boxes -> (len(a), len(b))."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float64)

    lt = np.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    rb = np.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    wh = np.clip(rb - lt, 0.0, None)
    inter = wh[..., 0] * wh[..., 1]

    area_a = np.prod(np.clip(boxes_a[:, 2:] - boxes_a[:, :2], 0.0, None), axis=1)
    area_b = np.prod(np.clip(boxes_b[:, 2:] - boxes_b[:, :2], 0.0, None), axis=1)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def _match_one_image(
    record: ImageRecord, cls: int, iou_thr: float
) -> tuple[np.ndarray, np.ndarray, int]:
    """Match one class in one image.

    Returns ``(scores, is_true_positive, n_ground_truth)``, detections sorted by
    descending score. Each ground-truth box may be claimed at most once; extra
    detections on an already-claimed box are duplicates and count as false
    positives, which is what makes duplicate suppression visible in the metric.
    """
    pred_mask = record.pred_classes == cls
    gt_mask = record.gt_classes == cls

    scores = record.pred_scores[pred_mask]
    boxes = record.pred_boxes[pred_mask]
    gts = record.gt_boxes[gt_mask]
    n_gt = int(gt_mask.sum())

    order = np.argsort(-scores, kind="stable")
    scores = scores[order]
    boxes = boxes[order]

    tp = np.zeros(len(scores), dtype=bool)
    if n_gt and len(boxes):
        ious = box_iou(boxes, gts)
        claimed = np.zeros(n_gt, dtype=bool)
        for i in range(len(boxes)):
            candidates = np.where(~claimed)[0]
            if candidates.size == 0:
                break
            best_local = candidates[np.argmax(ious[i, candidates])]
            if ious[i, best_local] >= iou_thr:
                claimed[best_local] = True
                tp[i] = True
    return scores, tp, n_gt


def average_precision(scores: np.ndarray, tp: np.ndarray, n_gt: int) -> float:
    """All-point interpolated AP (COCO convention). Undefined -> NaN."""
    if n_gt == 0:
        return float("nan")
    if len(scores) == 0:
        return 0.0

    order = np.argsort(-scores, kind="stable")
    tp_sorted = tp[order].astype(np.float64)
    fp_sorted = 1.0 - tp_sorted

    tp_cum = np.cumsum(tp_sorted)
    fp_cum = np.cumsum(fp_sorted)

    recall = tp_cum / n_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)

    # Monotonic envelope: precision at recall r is the max precision at recall >= r.
    mrec = np.concatenate(([0.0], recall, [recall[-1] if len(recall) else 0.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


@dataclass
class ClassMetrics:
    """Everything we are willing to claim about one class."""

    name: str
    support: int = 0                 # ground-truth instances
    n_predictions: int = 0
    precision: float = float("nan")  # at the operating confidence threshold
    recall: float = float("nan")
    f1: float = float("nan")
    ap50: float = float("nan")
    ap50_95: float = float("nan")
    tp: int = 0
    fp: int = 0
    fn: int = 0
    # Populated by bootstrap_metrics.
    ci: dict[str, tuple[float, float]] = field(default_factory=dict)

    @property
    def measurable(self) -> bool:
        """Support below 10 instances cannot support a per-class claim."""
        return self.support >= 10


def evaluate_detections(
    records: Sequence[ImageRecord],
    iou_thr: float = 0.5,
    conf_thr: float = 0.25,
    iou_sweep: Sequence[float] | None = None,
    class_names: Sequence[str] = tuple(CLASS_NAMES),
) -> dict[str, ClassMetrics]:
    """Per-class metrics over a set of cached images.

    ``precision``/``recall``/``f1`` are reported at ``conf_thr`` - the threshold
    the deployed app actually uses - because a PR curve summary such as AP says
    nothing about behaviour at the operating point a clinician would see.
    ``ap50``/``ap50_95`` use all detections regardless of ``conf_thr``.
    """
    if iou_sweep is None:
        iou_sweep = np.arange(0.5, 1.0, 0.05)

    out: dict[str, ClassMetrics] = {}

    for cls, name in enumerate(class_names):
        metrics = ClassMetrics(name=name)

        # --- AP over the full score range -----------------------------------
        all_scores: list[np.ndarray] = []
        all_tp: list[np.ndarray] = []
        n_gt_total = 0
        for record in records:
            scores, tp, n_gt = _match_one_image(record, cls, iou_thr)
            all_scores.append(scores)
            all_tp.append(tp)
            n_gt_total += n_gt

        scores_cat = np.concatenate(all_scores) if all_scores else np.zeros(0)
        tp_cat = np.concatenate(all_tp) if all_tp else np.zeros(0, dtype=bool)

        metrics.support = n_gt_total
        metrics.n_predictions = int(len(scores_cat))
        metrics.ap50 = average_precision(scores_cat, tp_cat, n_gt_total)

        aps: list[float] = []
        for thr in iou_sweep:
            s_list, t_list, g_total = [], [], 0
            for record in records:
                s, t, g = _match_one_image(record, cls, float(thr))
                s_list.append(s)
                t_list.append(t)
                g_total += g
            ap = average_precision(
                np.concatenate(s_list) if s_list else np.zeros(0),
                np.concatenate(t_list) if t_list else np.zeros(0, dtype=bool),
                g_total,
            )
            aps.append(ap)
        with warnings.catch_warnings():
            # An all-NaN sweep just means the class has no ground truth here;
            # NaN is the correct answer, not a condition worth warning about.
            warnings.simplefilter("ignore", RuntimeWarning)
            metrics.ap50_95 = float(np.nanmean(aps)) if aps else float("nan")

        # --- Operating point at conf_thr ------------------------------------
        keep = scores_cat >= conf_thr
        tp_at = int(tp_cat[keep].sum())
        fp_at = int((~tp_cat[keep]).sum())
        fn_at = max(n_gt_total - tp_at, 0)

        metrics.tp, metrics.fp, metrics.fn = tp_at, fp_at, fn_at
        denom_p = tp_at + fp_at
        denom_r = tp_at + fn_at
        metrics.precision = tp_at / denom_p if denom_p else float("nan")
        metrics.recall = tp_at / denom_r if denom_r else float("nan")
        if metrics.precision + metrics.recall > 0:
            metrics.f1 = (
                2 * metrics.precision * metrics.recall
                / (metrics.precision + metrics.recall)
            )

        out[name] = metrics

    return out


def macro_average(metrics: dict[str, ClassMetrics], field_name: str) -> float:
    """Mean of a field over classes that have any ground truth at all."""
    values = [
        getattr(m, field_name)
        for m in metrics.values()
        if m.support > 0 and not np.isnan(getattr(m, field_name))
    ]
    return float(np.mean(values)) if values else float("nan")


def bootstrap_metrics(
    records: Sequence[ImageRecord],
    n_boot: int = 1000,
    iou_thr: float = 0.5,
    conf_thr: float = 0.25,
    alpha: float = 0.05,
    seed: int = 0,
    class_names: Sequence[str] = tuple(CLASS_NAMES),
    fields: Sequence[str] = ("precision", "recall", "f1", "ap50"),
) -> dict[str, ClassMetrics]:
    """Percentile bootstrap over *images* for per-class confidence intervals.

    Resampling images (not boxes) is the right unit: boxes within a radiograph
    are correlated, so a box-level bootstrap would understate the interval.

    With 23 evaluation images the resulting intervals are wide. That is the
    finding, not a defect of the method - it is precisely why the cross-
    validation protocol behind ``dataset.py folds`` exists.
    """
    base = evaluate_detections(
        records, iou_thr=iou_thr, conf_thr=conf_thr, class_names=class_names
    )
    if not records:
        return base

    rng = np.random.default_rng(seed)
    n = len(records)
    samples: dict[str, dict[str, list[float]]] = {
        name: {f: [] for f in fields} for name in class_names
    }

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        resample = [records[i] for i in idx]
        # AP@0.5 only inside the bootstrap loop - the full IoU sweep would make
        # this 10x slower for no gain in the reported intervals.
        rep = evaluate_detections(
            resample,
            iou_thr=iou_thr,
            conf_thr=conf_thr,
            iou_sweep=[iou_thr],
            class_names=class_names,
        )
        for name, m in rep.items():
            for f in fields:
                value = getattr(m, f)
                if not np.isnan(value):
                    samples[name][f].append(value)

    lo_q, hi_q = 100 * alpha / 2, 100 * (1 - alpha / 2)
    for name, metrics in base.items():
        for f in fields:
            values = samples[name][f]
            if len(values) >= 20:
                metrics.ci[f] = (
                    float(np.percentile(values, lo_q)),
                    float(np.percentile(values, hi_q)),
                )
    return base


def paired_bootstrap_delta(
    records_a: Sequence[ImageRecord],
    records_b: Sequence[ImageRecord],
    metric: str = "ap50",
    n_boot: int = 1000,
    conf_thr: float = 0.25,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """Is model B better than model A, or is the gap noise?

    Both models are evaluated on the *same* resample of images, so the
    comparison is paired and the image-difficulty variance that dominates a
    small evaluation set cancels out. Returns, per class, the observed delta,
    a 95% interval on it, and the fraction of resamples where B beat A.
    """
    by_id_a = {r.image_id: r for r in records_a}
    by_id_b = {r.image_id: r for r in records_b}
    shared = sorted(set(by_id_a) & set(by_id_b))
    if not shared:
        raise ValueError("The two runs share no image ids - cannot pair them.")

    rng = np.random.default_rng(seed)
    deltas: dict[str, list[float]] = {name: [] for name in CLASS_NAMES}

    def metrics_for(ids: Sequence[str], table: dict[str, ImageRecord]) -> dict:
        return evaluate_detections(
            [table[i] for i in ids], conf_thr=conf_thr, iou_sweep=[0.5]
        )

    observed_a = metrics_for(shared, by_id_a)
    observed_b = metrics_for(shared, by_id_b)

    for _ in range(n_boot):
        idx = rng.integers(0, len(shared), size=len(shared))
        ids = [shared[i] for i in idx]
        ma = metrics_for(ids, by_id_a)
        mb = metrics_for(ids, by_id_b)
        for name in CLASS_NAMES:
            va, vb = getattr(ma[name], metric), getattr(mb[name], metric)
            if not (np.isnan(va) or np.isnan(vb)):
                deltas[name].append(vb - va)

    out: dict[str, dict[str, float]] = {}
    for name in CLASS_NAMES:
        values = np.asarray(deltas[name], dtype=np.float64)
        obs = getattr(observed_b[name], metric) - getattr(observed_a[name], metric)
        if values.size < 20:
            out[name] = {"delta": float(obs), "ci_low": float("nan"),
                         "ci_high": float("nan"), "p_b_better": float("nan"),
                         "n_boot_valid": int(values.size)}
            continue
        out[name] = {
            "delta": float(obs),
            "ci_low": float(np.percentile(values, 2.5)),
            "ci_high": float(np.percentile(values, 97.5)),
            "p_b_better": float((values > 0).mean()),
            "n_boot_valid": int(values.size),
        }
    return out
