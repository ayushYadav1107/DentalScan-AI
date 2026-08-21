#!/usr/bin/env python3
"""Per-class evaluation with uncertainty, and model-to-model comparison.

    python evaluate.py per-class --weights <best.pt> --images <test/images> --out results/v10s
    python evaluate.py compare   --runs runs/detect/* --labels YOLOv8n YOLOv10s YOLOv12m

What changed from the original script. That version printed four aggregate
numbers - precision, recall, mAP@0.5, mAP@0.5:0.95 - averaged over six classes
whose evaluation support ranges from 3 instances to 67. On this dataset that
average is not a measurement: the classes contributing most to it are the ones
with the least evidence behind them, and restricting the mean to the three
classes with adequate support reverses which architecture wins. See
``report/main.pdf``.

``per-class`` reports, for every class: support, precision, recall and F1 at the
deployed confidence threshold, AP@0.5 and AP@0.5:0.95, each with a bootstrap
confidence interval - and flags any class too rare to be reported at all. It
also prints two macro averages, over all classes and over measurable ones only,
because they are different numbers and only one of them is defensible.

Inference runs once and is cached to ``predictions.npz``; every metric,
including every bootstrap replicate, is derived from that cache. Deeper
diagnostics on the same cache live in ``evaluate_new.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from dentalscan.constants import (
    CLASS_NAMES, DEFAULT_CONF, DEFAULT_IOU, FIGURE_DIR, RESULTS_DIR,
)
from dentalscan.metrics import bootstrap_metrics, evaluate_detections, macro_average
from dentalscan.predict import (
    build_prediction_cache, cache_summary, load_cache, save_cache,
)
from dentalscan.report import (
    LOW_SUPPORT_NOTE, PER_CLASS_HEADERS, markdown_table, per_class_rows,
)


# --------------------------------------------------------------------------- #
# per-class
# --------------------------------------------------------------------------- #

def cmd_per_class(args: argparse.Namespace) -> int:
    args.out.mkdir(parents=True, exist_ok=True)
    cache_path = args.out / "predictions.npz"

    if args.reuse_cache and cache_path.exists():
        print(f"Loading cached predictions from {cache_path}")
        records = load_cache(cache_path)
    else:
        print(f"Running inference with {args.weights} ...")
        records = build_prediction_cache(
            args.weights, args.images, conf=0.001, iou=args.iou,
            imgsz=args.imgsz, device=args.device)
        save_cache(records, cache_path)
        print(f"Cached {len(records)} images to {cache_path}")

    summary = cache_summary(records)
    print(json.dumps(summary, indent=2))

    metrics = (
        bootstrap_metrics(records, n_boot=args.bootstrap, conf_thr=args.conf, seed=0)
        if args.bootstrap else
        evaluate_detections(records, conf_thr=args.conf)
    )

    table = markdown_table(
        PER_CLASS_HEADERS, per_class_rows(metrics, with_ci=bool(args.bootstrap)))
    print("\n" + table + "\n\n" + LOW_SUPPORT_NOTE)

    fields = ("precision", "recall", "f1", "ap50", "ap50_95")
    macro = {f: macro_average(metrics, f) for f in fields}
    print("\nMacro average over classes with any support:")
    for key, value in macro.items():
        print(f"  {key:10s} {value:.4f}")

    # A macro average that includes classes with three instances is not the same
    # number as one restricted to measurable classes. Report both, explicitly.
    measurable = {n: m for n, m in metrics.items() if m.measurable}
    if measurable and len(measurable) < len(metrics):
        macro_measurable = {f: macro_average(measurable, f) for f in fields}
        print(f"\nMacro average over the {len(measurable)} classes with >= 10 instances:")
        for key, value in macro_measurable.items():
            print(f"  {key:10s} {value:.4f}")
    else:
        macro_measurable = macro

    (args.out / "metrics.json").write_text(json.dumps({
        "weights": str(args.weights), "images": [str(p) for p in args.images],
        "conf": args.conf, "iou": args.iou, "imgsz": args.imgsz,
        "bootstrap": args.bootstrap, "cache_summary": summary,
        "per_class": {
            name: {**{k: v for k, v in asdict(m).items() if k != "ci"},
                   "ci": {k: list(v) for k, v in m.ci.items()}}
            for name, m in metrics.items()
        },
        "macro": macro, "macro_measurable_only": macro_measurable,
    }, indent=2))
    (args.out / "metrics.md").write_text(table + "\n\n" + LOW_SUPPORT_NOTE + "\n")
    print(f"\nWrote {args.out / 'metrics.json'} and metrics.md")

    if not args.no_figure:
        from dentalscan.viz import plot_per_class_with_ci
        print(f"Wrote {plot_per_class_with_ci(metrics, args.out / 'per_class_ci.png')}")
    return 0


# --------------------------------------------------------------------------- #
# compare
# --------------------------------------------------------------------------- #

def cmd_compare(args: argparse.Namespace) -> int:
    import pandas as pd

    from dentalscan.viz import plot_per_class_comparison, plot_training_curves

    labels = args.labels or [p.name for p in args.runs]
    if len(labels) != len(args.runs):
        print("--labels must match --runs in length", file=sys.stderr)
        return 1

    curves = {}
    for label, run_dir in zip(labels, args.runs):
        csv_path = Path(run_dir) / "results.csv"
        if not csv_path.exists():
            print(f"No results.csv in {run_dir}", file=sys.stderr)
            return 1
        frame = pd.read_csv(csv_path)
        frame.columns = [c.strip() for c in frame.columns]
        curves[label] = frame

    rows, summary = [], {}
    for label, frame in curves.items():
        best = frame.loc[frame["metrics/mAP50(B)"].idxmax()]
        final = frame.iloc[-1]
        summary[label] = {
            "epochs": int(final["epoch"]), "best_epoch": int(best["epoch"]),
            "best_mAP50": float(best["metrics/mAP50(B)"]),
            "best_mAP50_95": float(best["metrics/mAP50-95(B)"]),
            "final_mAP50": float(final["metrics/mAP50(B)"]),
            "train_minutes": round(float(final["time"]) / 60, 1),
        }
        e = summary[label]
        rows.append([label, e["epochs"], e["best_epoch"], f"{e['best_mAP50']:.3f}",
                     f"{e['best_mAP50_95']:.3f}", f"{e['final_mAP50']:.3f}",
                     f"{e['train_minutes']:.0f}"])

    print(markdown_table(
        ["Run", "Epochs", "Best epoch", "Best mAP@0.5", "Best mAP@0.5:0.95",
         "Final mAP@0.5", "Train (min)"], rows))
    print("\nNote: 'best' is the maximum over epochs on the validation split, which "
          "is itself a selection on that split - it is optimistically biased and is "
          "not a held-out estimate.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nWrote {plot_training_curves(curves, args.out_dir / 'training_curves.png')}")

    if args.per_class and args.per_class.exists():
        payload = json.loads(args.per_class.read_text())
        results = payload["ap50"]
        supports = payload.get("val_support", {})
        print(f"Wrote {plot_per_class_comparison(results, args.out_dir / 'per_class_ap50.png', metric_label='AP@0.5 (validation split)', supports=supports)}")
        print()
        print(markdown_table(["Model", *CLASS_NAMES, "mAP@0.5"], [
            [name, *[f"{pc.get(c, float('nan')):.3f}" for c in CLASS_NAMES],
             f"{sum(pc.values()) / len(pc):.3f}"]
            for name, pc in results.items()
        ]))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "existing_runs_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {RESULTS_DIR / 'existing_runs_summary.json'}")
    return 0


# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("per-class", help="per-class metrics with bootstrap intervals")
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--images", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, default=RESULTS_DIR / "evaluation")
    p.add_argument("--conf", type=float, default=DEFAULT_CONF,
                   help="Operating threshold for precision/recall/F1.")
    p.add_argument("--iou", type=float, default=DEFAULT_IOU, help="NMS IoU.")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default=None)
    p.add_argument("--bootstrap", type=int, default=1000,
                   help="Bootstrap replicates; 0 disables intervals.")
    p.add_argument("--reuse-cache", action="store_true",
                   help="Load an existing prediction cache instead of re-running inference.")
    p.add_argument("--no-figure", action="store_true")
    p.set_defaults(func=cmd_per_class)

    p = sub.add_parser("compare", help="head-to-head from runs already on disk")
    p.add_argument("--runs", type=Path, nargs="+", required=True)
    p.add_argument("--labels", nargs="+", default=None)
    p.add_argument("--per-class", type=Path,
                   default=RESULTS_DIR / "existing_runs_per_class.json")
    p.add_argument("--out-dir", type=Path, default=FIGURE_DIR)
    p.set_defaults(func=cmd_compare)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
