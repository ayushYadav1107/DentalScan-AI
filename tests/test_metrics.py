"""Known-answer tests for the metric implementation.

The AP code in ``dentalscan.metrics`` is the foundation everything else in the
project stands on, so it is pinned two ways: against hand-computable cases, and
against Ultralytics' own ``ap_per_class`` on randomised inputs. If those two
disagree, the number in the report is wrong.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dentalscan.error_analysis import analyse_errors
from dentalscan.metrics import (
    ImageRecord, average_precision, box_iou, evaluate_detections
)


def make_record(image_id, preds, gts):
    """preds: list of (x1,y1,x2,y2,score,cls); gts: list of (x1,y1,x2,y2,cls)."""
    preds = np.asarray(preds, dtype=float).reshape(-1, 6)
    gts = np.asarray(gts, dtype=float).reshape(-1, 5)
    return ImageRecord(
        image_id=image_id,
        pred_boxes=preds[:, :4], pred_scores=preds[:, 4],
        pred_classes=preds[:, 5].astype(int),
        gt_boxes=gts[:, :4], gt_classes=gts[:, 4].astype(int),
    )


def test_box_iou_identical_boxes_is_one():
    a = np.array([[0.0, 0.0, 10.0, 10.0]])
    assert box_iou(a, a)[0, 0] == pytest.approx(1.0)


def test_box_iou_half_overlap():
    a = np.array([[0.0, 0.0, 10.0, 10.0]])
    b = np.array([[5.0, 0.0, 15.0, 10.0]])
    # intersection 50, union 150
    assert box_iou(a, b)[0, 0] == pytest.approx(50 / 150)


def test_box_iou_disjoint_is_zero():
    a = np.array([[0.0, 0.0, 10.0, 10.0]])
    b = np.array([[20.0, 20.0, 30.0, 30.0]])
    assert box_iou(a, b)[0, 0] == 0.0


def test_perfect_detector_scores_one():
    record = make_record(
        "a",
        preds=[[0, 0, 10, 10, 0.9, 0]],
        gts=[[0, 0, 10, 10, 0]],
    )
    metrics = evaluate_detections([record], conf_thr=0.5, iou_sweep=[0.5])
    caries = metrics["Caries"]
    assert caries.ap50 == pytest.approx(1.0)
    assert caries.precision == pytest.approx(1.0)
    assert caries.recall == pytest.approx(1.0)
    assert (caries.tp, caries.fp, caries.fn) == (1, 0, 0)


def test_one_hit_one_hallucination():
    record = make_record(
        "a",
        preds=[[0, 0, 10, 10, 0.9, 0], [100, 100, 110, 110, 0.8, 0]],
        gts=[[0, 0, 10, 10, 0]],
    )
    caries = evaluate_detections([record], conf_thr=0.5, iou_sweep=[0.5])["Caries"]
    assert caries.precision == pytest.approx(0.5)
    assert caries.recall == pytest.approx(1.0)
    assert (caries.tp, caries.fp) == (1, 1)


def test_duplicate_detection_is_a_false_positive():
    """Two boxes on one object: the second must not be credited."""
    record = make_record(
        "a",
        preds=[[0, 0, 10, 10, 0.9, 0], [1, 1, 11, 11, 0.8, 0]],
        gts=[[0, 0, 10, 10, 0]],
    )
    caries = evaluate_detections([record], conf_thr=0.5, iou_sweep=[0.5])["Caries"]
    assert caries.tp == 1
    assert caries.fp == 1

    breakdown = analyse_errors([record], conf_thr=0.5)
    assert breakdown.by_category["duplicate"] == 1


def test_wrong_class_is_a_classification_error_not_a_localization_one():
    record = make_record(
        "a",
        preds=[[0, 0, 10, 10, 0.9, 1]],   # predicts Infection
        gts=[[0, 0, 10, 10, 0]],          # ground truth Caries
    )
    breakdown = analyse_errors([record], conf_thr=0.5)
    assert breakdown.by_category["classification"] == 1
    assert breakdown.by_category["localization"] == 0
    assert breakdown.n_missed == 1


def test_sloppy_box_right_class_is_a_localization_error():
    record = make_record(
        "a",
        preds=[[0, 0, 10, 10, 0.9, 0]],
        gts=[[6, 0, 20, 10, 0]],          # IoU ~0.2: above 0.1, below 0.5
    )
    breakdown = analyse_errors([record], conf_thr=0.5)
    assert breakdown.by_category["localization"] == 1


def test_far_away_box_is_a_background_error():
    record = make_record(
        "a",
        preds=[[500, 500, 510, 510, 0.9, 0]],
        gts=[[0, 0, 10, 10, 0]],
    )
    breakdown = analyse_errors([record], conf_thr=0.5)
    assert breakdown.by_category["background"] == 1


def test_class_with_no_ground_truth_is_nan_not_zero():
    """The single most dangerous silent bug: a class with no instances must not
    contribute a 0.0 that drags the macro average down, nor a 1.0 that lifts it."""
    record = make_record("a", preds=[[0, 0, 10, 10, 0.9, 0]], gts=[[0, 0, 10, 10, 0]])
    metrics = evaluate_detections([record], conf_thr=0.5, iou_sweep=[0.5])
    assert np.isnan(metrics["Fractured"].ap50)
    assert metrics["Fractured"].support == 0


def test_confidence_threshold_changes_operating_point_not_ap():
    record = make_record(
        "a",
        preds=[[0, 0, 10, 10, 0.9, 0], [100, 100, 110, 110, 0.3, 0]],
        gts=[[0, 0, 10, 10, 0]],
    )
    high = evaluate_detections([record], conf_thr=0.5, iou_sweep=[0.5])["Caries"]
    low = evaluate_detections([record], conf_thr=0.1, iou_sweep=[0.5])["Caries"]
    assert high.precision == pytest.approx(1.0)
    assert low.precision == pytest.approx(0.5)
    assert high.ap50 == pytest.approx(low.ap50)  # AP ignores the threshold


def test_average_precision_step_curve():
    """Two GT, detections ranked hit-miss-hit.

    Precision at each true positive is 1/1 and 2/3, recall 0.5 and 1.0, so the
    all-point interpolated area is 0.5*1 + 0.5*(2/3).
    """
    scores = np.array([0.9, 0.8, 0.7])
    tp = np.array([True, False, True])
    assert average_precision(scores, tp, n_gt=2) == pytest.approx(0.5 + 0.5 * (2 / 3))


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_agrees_with_ultralytics_ap_per_class(seed):
    """Cross-check AP@0.5 against Ultralytics' own implementation."""
    ultra = pytest.importorskip("ultralytics.utils.metrics")

    rng = np.random.default_rng(seed)
    records, tp_flags, confs, pred_cls, target_cls = [], [], [], [], []

    for i in range(12):
        n_gt = int(rng.integers(1, 5))
        gts = []
        for _ in range(n_gt):
            x, y = rng.uniform(0, 400, size=2)
            w, h = rng.uniform(20, 80, size=2)
            gts.append([x, y, x + w, y + h, int(rng.integers(0, 3))])

        preds = []
        for g in gts:
            if rng.random() < 0.75:  # a detection near this GT
                jitter = rng.normal(0, 6, size=4)
                cls = g[4] if rng.random() < 0.85 else int(rng.integers(0, 3))
                preds.append([g[0] + jitter[0], g[1] + jitter[1],
                              g[2] + jitter[2], g[3] + jitter[3],
                              float(rng.uniform(0.1, 1.0)), cls])
        for _ in range(int(rng.integers(0, 3))):  # background hallucinations
            x, y = rng.uniform(0, 400, size=2)
            preds.append([x, y, x + 40, y + 40, float(rng.uniform(0.1, 1.0)),
                          int(rng.integers(0, 3))])

        records.append(make_record(f"img{i}", preds, gts))

    # Build the (tp, conf, pred_cls, target_cls) arrays Ultralytics expects,
    # using our own matching so that only the AP integration differs.
    from dentalscan.metrics import _match_one_image
    for cls in range(3):
        for record in records:
            scores, tp, _ = _match_one_image(record, cls, 0.5)
            tp_flags.extend(tp.tolist())
            confs.extend(scores.tolist())
            pred_cls.extend([cls] * len(scores))
        target_cls.extend([cls] * sum(int((r.gt_classes == cls).sum()) for r in records))

    if not tp_flags or not target_cls:
        pytest.skip("degenerate random draw")

    result = ultra.ap_per_class(
        np.asarray(tp_flags, dtype=bool).reshape(-1, 1),
        np.asarray(confs), np.asarray(pred_cls), np.asarray(target_cls),
        plot=False,
    )
    # ap_per_class returns (tp, fp, p, r, f1, ap, unique_classes, ...)
    ap_reference = {int(c): float(a[0]) for c, a in zip(result[6], result[5])}

    ours = evaluate_detections(records, iou_thr=0.5, iou_sweep=[0.5])
    from dentalscan.constants import CLASS_NAMES
    for cls, reference in ap_reference.items():
        mine = ours[CLASS_NAMES[cls]].ap50
        # Ultralytics integrates on a 101-point recall grid; we use the exact
        # all-point envelope. The two differ only by interpolation resolution.
        assert mine == pytest.approx(reference, abs=0.02), (
            f"class {CLASS_NAMES[cls]}: ours={mine:.4f} ultralytics={reference:.4f}"
        )
