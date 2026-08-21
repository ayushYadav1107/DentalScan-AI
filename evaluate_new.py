#!/usr/bin/env python3
"""Diagnostics beyond the headline metrics: why the model is wrong, and where it looks.

    python evaluate_new.py errors  --cache results/v10s/predictions.npz --out results/v10s
    python evaluate_new.py sweep   --cache results/v10s/predictions.npz
    python evaluate_new.py explain --weights <best.pt> --images <test/images> --faithfulness

What changed from the original script. That version ran a separate ResNet-18
classifier over YOLO's own crops and reported the agreement rate as "CNN
accuracy". Two problems made the number uninterpretable: agreement between two
models is not accuracy against ground truth, and the crops were selected by one
of the two models being compared, so the measurement was circular.

What that analysis was reaching for - "which classes does the model get wrong,
and is it right for the right reason?" - is answered here directly.
``errors`` sorts every false positive into classification / localization /
duplicate / background and counts ground truth the model never proposed.
``sweep`` shows how precision and recall trade off per class as the confidence
threshold moves. ``explain`` produces saliency maps and, crucially, scores
whether they actually describe the model rather than just looking plausible.

``errors`` and ``sweep`` read the prediction cache written by
``evaluate.py per-class``, so they cost no extra inference.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from dentalscan.constants import CLASS_NAMES, DEFAULT_CONF, RESULTS_DIR
from dentalscan.data import list_images
from dentalscan.error_analysis import (
    analyse_errors, best_thresholds, confidence_sweep, size_stratified_recall,
)
from dentalscan.explain import (
    CAM_METHODS, faithfulness, list_candidate_layers, overlay_cam, pointing_energy,
)
from dentalscan.predict import load_cache
from dentalscan.report import error_breakdown_markdown, markdown_table


# --------------------------------------------------------------------------- #
# errors
# --------------------------------------------------------------------------- #

def cmd_errors(args: argparse.Namespace) -> int:
    args.out.mkdir(parents=True, exist_ok=True)
    breakdowns, payload = {}, {"conf": args.conf, "models": {}}

    for cache_path in args.cache:
        label = cache_path.parent.name
        records = load_cache(cache_path)
        breakdown = analyse_errors(records, conf_thr=args.conf)
        breakdowns[label] = breakdown

        print(f"\n=== {label} ===")
        print(f"{breakdown.n_ground_truth} ground-truth boxes, "
              f"{breakdown.n_predictions} predictions above conf={args.conf}, "
              f"{breakdown.n_true_positive} true positives")
        print(error_breakdown_markdown(breakdown))
        print(f"\nDominant failure mode: {breakdown.dominant_failure()}")

        sizes = size_stratified_recall(records, conf_thr=args.conf)
        print("\nRecall by ground-truth box size tercile:")
        print(markdown_table(["Class", "Small", "Medium", "Large"], [
            [name,
             f"{row['small']:.2f} (n={int(row['small_n'])})",
             f"{row['medium']:.2f} (n={int(row['medium_n'])})",
             f"{row['large']:.2f} (n={int(row['large_n'])})"]
            for name, row in sizes.items() if not name.startswith("_")
        ]))

        sweep = confidence_sweep(records)
        best = best_thresholds(sweep)
        print("\nF1-optimal confidence threshold per class:")
        print(markdown_table(
            ["Class", "Best conf", "F1", "Precision", "Recall", "N"],
            [[n, f"{v['conf']:.2f}", f"{v['f1']:.3f}", f"{v['precision']:.3f}",
              f"{v['recall']:.3f}", v["support"]] for n, v in best.items()]))

        payload["models"][label] = {
            "breakdown": asdict(breakdown), "fractions": breakdown.as_fractions(),
            "dominant_failure": breakdown.dominant_failure(),
            "size_stratified_recall": sizes, "confidence_sweep": sweep,
            "best_thresholds": best,
        }

        if not args.no_figure:
            from dentalscan.viz import plot_confidence_sweep
            plot_confidence_sweep(sweep, args.out / f"{label}_confidence_sweep.png")

    (args.out / "error_analysis.json").write_text(
        json.dumps(payload, indent=2, default=float))
    if not args.no_figure and breakdowns:
        from dentalscan.viz import plot_error_breakdown
        print(f"\nWrote {plot_error_breakdown(breakdowns, args.out / 'error_breakdown.png')}")
    print(f"Wrote {args.out / 'error_analysis.json'}")
    return 0


# --------------------------------------------------------------------------- #
# sweep
# --------------------------------------------------------------------------- #

def cmd_sweep(args: argparse.Namespace) -> int:
    """Where should the confidence threshold sit?

    Screening and record-writing want different answers: a triage tool that
    flags radiographs for review should sit at high recall, a tool that writes
    into a record should sit at high precision. One global default silently
    picks one. This makes the choice explicit, per class.
    """
    records = load_cache(args.cache)
    rows = confidence_sweep(records)
    best = best_thresholds(rows)

    print(markdown_table(
        ["Class", "N", "Best conf (F1)", "F1", "Precision", "Recall"],
        [[n, v["support"], f"{v['conf']:.2f}", f"{v['f1']:.3f}",
          f"{v['precision']:.3f}", f"{v['recall']:.3f}"] for n, v in best.items()]))

    print(f"\nThe app currently uses a single threshold of {DEFAULT_CONF:.2f} for "
          "every class. Where the per-class optima above disagree with it, that "
          "is free performance left on the table.")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "threshold_sweep.json").write_text(
        json.dumps({"sweep": rows, "best": best}, indent=2, default=float))
    if not args.no_figure:
        from dentalscan.viz import plot_confidence_sweep
        print(f"Wrote {plot_confidence_sweep(rows, args.out / 'confidence_sweep.png')}")
    print(f"Wrote {args.out / 'threshold_sweep.json'}")
    return 0


# --------------------------------------------------------------------------- #
# explain
# --------------------------------------------------------------------------- #

def cmd_explain(args: argparse.Namespace) -> int:
    import cv2
    from ultralytics import YOLO

    model = YOLO(str(args.weights))

    if args.list_layers:
        for index, name in list_candidate_layers(model):
            print(f"{index:3d}  {name}")
        return 0

    cam_fn = CAM_METHODS[args.method](model, layer_index=args.layer)
    print(f"{args.method} hooked at module index {cam_fn.layer_index}")

    images: list[Path] = []
    for path in args.images:
        images.extend(list_images(path) if path.is_dir() else [path])
    images = images[: args.n]
    if not images:
        print("No images found.", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    panels: list[tuple[str, np.ndarray]] = []
    rows: list[dict] = []

    for image_path in images:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Could not read {image_path}", file=sys.stderr)
            continue

        result = model.predict(image, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        detection_overlay = result.plot()

        if result.boxes is not None and len(result.boxes):
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            scores = result.boxes.conf.cpu().numpy()
            top = int(classes[int(np.argmax(scores))])
        else:
            boxes, top = np.zeros((0, 4)), 0

        target = args.target_class if args.target_class is not None else top
        cam = (cam_fn(image, imgsz=args.imgsz) if args.method == "eigencam"
               else cam_fn(image, target_class=target, imgsz=args.imgsz))
        cam_overlay = overlay_cam(image, cam)

        stem = image_path.stem[:40]
        cv2.imwrite(str(args.out / f"{stem}_detections.jpg"), detection_overlay)
        cv2.imwrite(str(args.out / f"{stem}_{args.method}.jpg"), cam_overlay)
        panels.append((f"{stem}\n{len(boxes)} detections", detection_overlay))
        panels.append((f"{args.method} - {CLASS_NAMES[target]}", cam_overlay))

        row: dict = {"image": str(image_path), "target_class": CLASS_NAMES[target],
                     "n_detections": int(len(boxes))}
        if args.faithfulness:
            scores_obj = faithfulness(model, image, cam, boxes, target,
                                      steps=args.steps, imgsz=args.imgsz)
            row.update(asdict(scores_obj))
            print(f"{stem:42s} energy={scores_obj.pointing_energy:.3f} "
                  f"(chance {scores_obj.box_area_fraction:.3f}, "
                  f"lift {scores_obj.energy_lift:.2f}x)  "
                  f"deletion gain={scores_obj.deletion_gain:+.3f}")
        else:
            energy, chance = pointing_energy(cam, boxes)
            row.update({"pointing_energy": energy, "box_area_fraction": chance,
                        "energy_lift": energy / chance if chance else float("nan")})
            print(f"{stem:42s} energy={energy:.3f} (chance {chance:.3f})")
        rows.append(row)

    if rows:
        keys = [k for k in ("pointing_energy", "energy_lift", "deletion_auc",
                            "deletion_auc_random", "deletion_gain") if k in rows[0]]
        aggregate = {k: float(np.nanmean([r[k] for r in rows])) for k in keys}
        print("\nMean over images:")
        for key, value in aggregate.items():
            print(f"  {key:22s} {value:.4f}")
        print("\nA lift near 1x means the map is no better than uniform. "
              "A positive deletion gain means blanking high-saliency pixels hurts "
              "the prediction more than blanking random ones - which is what makes "
              "the map evidence rather than decoration.")
        (args.out / f"faithfulness_{args.method}.json").write_text(json.dumps(
            {"method": args.method, "layer": cam_fn.layer_index,
             "per_image": rows, "mean": aggregate}, indent=2, default=float))

    from dentalscan.viz import plot_cam_panel
    print(f"\nWrote {plot_cam_panel(panels[:12], args.out / f'panel_{args.method}.png', suptitle=f'Detections and {args.method} saliency')}")
    return 0


# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("errors", help="TIDE-style error decomposition")
    p.add_argument("--cache", type=Path, nargs="+", required=True,
                   help="One or more predictions.npz files (labelled by parent dir).")
    p.add_argument("--out", type=Path, default=RESULTS_DIR / "error_analysis")
    p.add_argument("--conf", type=float, default=DEFAULT_CONF)
    p.add_argument("--no-figure", action="store_true")
    p.set_defaults(func=cmd_errors)

    p = sub.add_parser("sweep", help="per-class F1-optimal confidence threshold")
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--out", type=Path, default=RESULTS_DIR / "error_analysis")
    p.add_argument("--no-figure", action="store_true")
    p.set_defaults(func=cmd_sweep)

    p = sub.add_parser("explain", help="saliency maps, scored for faithfulness")
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--images", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, default=RESULTS_DIR / "explainability")
    p.add_argument("--method", default="eigencam", choices=sorted(CAM_METHODS))
    p.add_argument("--layer", type=int, default=None,
                   help="Module index to hook; default is the last pre-head layer.")
    p.add_argument("--list-layers", action="store_true")
    p.add_argument("--n", type=int, default=6)
    p.add_argument("--conf", type=float, default=DEFAULT_CONF)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--target-class", type=int, default=None,
                   help="Class id for class-conditional CAM (gradcam only).")
    p.add_argument("--faithfulness", action="store_true",
                   help="Also compute deletion curves (~26 extra forward passes/image).")
    p.add_argument("--steps", type=int, default=10)
    p.set_defaults(func=cmd_explain)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
