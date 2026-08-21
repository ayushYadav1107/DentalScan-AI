#!/usr/bin/env python3
"""Training: seeded runs, cross-validation, and the ablation grid.

    python model.py train  --name baseline --seeds 0 1 2 3 4
    python model.py train  --name baseline_cv --cv-dir configs/cv
    python model.py ablate --dry-run
    python model.py ablate --seeds 0 1 2
    python model.py aggregate --manifest results/ablation_manifest.json

What changed from the original one-shot script. That version hard-coded a single
recipe, a single seed and a single output directory, which made runs impossible
to compare after the fact and made a lucky seed indistinguishable from a real
improvement. Here, the same hyperparameters are the *baseline*, ablations are
expressed as explicit deltas from it, seeds are swept, and every run writes its
seed, resolved config, library versions and git commit next to its weights.

The original recipe is preserved verbatim in
``dentalscan.train.BASELINE_HYPERPARAMS`` and is what ``--name baseline`` trains.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from dentalscan.aggregate import aggregate, compare_configurations
from dentalscan.constants import CLASS_NAMES, CONFIG_DIR, RESULTS_DIR, ROOT, RUNS_DIR
from dentalscan.report import ablation_markdown, latex_table, markdown_table
from dentalscan.train import (
    AUGMENTATION_GROUPS, BASELINE_HYPERPARAMS, RunConfig, apply_ablation,
    train_once, weights_path,
)


def _parse_overrides(pairs: list[str]) -> dict[str, object]:
    """``--set lr0=0.005 cos_lr=true`` -> ``{'lr0': 0.005, 'cos_lr': True}``."""
    overrides: dict[str, object] = {}
    for item in pairs:
        key, _, raw = item.partition("=")
        try:
            overrides[key] = json.loads(raw)
        except json.JSONDecodeError:
            overrides[key] = raw
    return overrides


# --------------------------------------------------------------------------- #
# train
# --------------------------------------------------------------------------- #

def cmd_train(args: argparse.Namespace) -> int:
    overrides = _parse_overrides(args.set)
    for key, value in (("epochs", args.epochs), ("batch", args.batch), ("imgsz", args.imgsz)):
        if value is not None:
            overrides[key] = value
    hyperparams = apply_ablation(BASELINE_HYPERPARAMS, args.disable, overrides)

    jobs: list[tuple[str, Path, int]] = []
    if args.cv_dir:
        fold_yamls = sorted(Path(args.cv_dir).glob("fold*.yaml"))
        if not fold_yamls:
            print(f"No fold*.yaml under {args.cv_dir}; run `python dataset.py folds` first.",
                  file=sys.stderr)
            return 1
        jobs = [(f"{args.name}_{y.stem}", y, args.seeds[0]) for y in fold_yamls]
    else:
        jobs = [(f"{args.name}_seed{s}", Path(args.data), s) for s in args.seeds]

    run_dirs: list[str] = []
    for i, (run_name, data_path, seed) in enumerate(jobs, start=1):
        print(f"\n=== [{i}/{len(jobs)}] {run_name} "
              f"(data={data_path.name}, seed={seed}) ===")
        run_dir = train_once(RunConfig(
            name=run_name, model=args.model, data=str(data_path), seed=seed,
            device=args.device, project=str(args.project),
            hyperparams=dict(hyperparams),
            notes=f"disable={args.disable} overrides={overrides}",
        ))
        run_dirs.append(str(run_dir))
        print(f"--> {run_dir}")

    manifest = Path(args.project) / f"{args.name}_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({
        "name": args.name, "model": args.model, "runs": run_dirs,
        "disabled_groups": args.disable, "overrides": overrides,
        "hyperparams": hyperparams,
    }, indent=2))
    print(f"\nManifest: {manifest}")
    return 0


# --------------------------------------------------------------------------- #
# ablate
# --------------------------------------------------------------------------- #

def _evaluate_run(run_dir: Path, images: list[str], conf: float, bootstrap: int) -> None:
    """Shell out to ``evaluate.py`` so ablation rows and the headline numbers in
    the README are produced by exactly the same code path."""
    subprocess.run([
        sys.executable, str(ROOT / "evaluate.py"), "per-class",
        "--weights", str(weights_path(run_dir)),
        "--images", *images,
        "--out", str(run_dir / "eval"),
        "--conf", str(conf),
        "--bootstrap", str(bootstrap),
        "--no-figure",
    ], check=True)


def cmd_ablate(args: argparse.Namespace) -> int:
    spec = yaml.safe_load(args.config.read_text())
    seeds = args.seeds or spec.get("seeds", [0])
    base_model = spec.get("baseline", {}).get("model", "yolov10s.pt")

    eval_images = args.eval_images
    if eval_images is None:
        data_spec = yaml.safe_load(Path(args.data).read_text())
        val = data_spec.get("val")
        if val is None:
            print("No val split in the data config and no --eval-images given.",
                  file=sys.stderr)
            return 1
        eval_images = [str(val)]

    jobs: list[dict] = []
    if not args.only_models:
        jobs.append({"name": "baseline", "model": base_model, "disable": [],
                     "set": {}, "data": str(args.data)})
        for variant in spec.get("variants", []):
            jobs.append({
                "name": variant["name"],
                "model": variant.get("model", base_model),
                "disable": variant.get("disable", []),
                "set": variant.get("set", {}) or {},
                "data": variant.get("data", str(args.data)),
            })
    if not args.only_variants:
        for entry in spec.get("models", []):
            jobs.append({"name": f"arch_{entry['name']}", "model": entry["model"],
                         "disable": [], "set": {}, "data": str(args.data)})

    if args.only:
        jobs = [j for j in jobs if j["name"] in set(args.only)]

    # A variant may point at a data config that has not been generated yet
    # (e.g. data_single_copy.yaml). Skip it loudly rather than failing mid-grid
    # after several hours of training.
    runnable = []
    for job in jobs:
        if Path(job["data"]).exists():
            runnable.append(job)
        else:
            print(f"SKIP {job['name']}: missing data config {job['data']}. "
                  f"See configs/ablation.yaml for how to generate it.", file=sys.stderr)
    jobs = runnable

    total = len(jobs) * len(seeds)
    print(f"{len(jobs)} configurations x {len(seeds)} seeds = {total} training runs")
    for job in jobs:
        print(f"  - {job['name']:18s} model={job['model']:14s} "
              f"disable={job['disable']} set={job['set']}")
    if args.dry_run:
        print("\n--dry-run: nothing was trained.")
        return 0

    completed: list[dict] = []
    started = time.time()
    for job in jobs:
        for seed in seeds:
            run_name = f"{job['name']}_seed{seed}"
            print(f"\n=== {run_name} ({len(completed) + 1}/{total}) ===")
            overrides = dict(job["set"])
            if args.epochs is not None:
                overrides["epochs"] = args.epochs
            hyperparams = apply_ablation(BASELINE_HYPERPARAMS, job["disable"], overrides)
            try:
                run_dir = train_once(RunConfig(
                    name=run_name, model=job["model"], data=job["data"], seed=seed,
                    device=args.device, project=str(args.project),
                    hyperparams=hyperparams, notes=json.dumps(job),
                ))
                _evaluate_run(run_dir, eval_images, args.conf, args.bootstrap)
                completed.append({"variant": job["name"], "seed": seed,
                                  "run_dir": str(run_dir), "status": "ok"})
            except Exception as exc:  # keep the grid going; record the failure
                print(f"FAILED {run_name}: {exc}", file=sys.stderr)
                completed.append({"variant": job["name"], "seed": seed,
                                  "run_dir": "", "status": f"failed: {exc}"})

            manifest = RESULTS_DIR / "ablation_manifest.json"
            manifest.parent.mkdir(parents=True, exist_ok=True)
            manifest.write_text(json.dumps({
                "config": str(args.config), "seeds": seeds,
                "elapsed_hours": round((time.time() - started) / 3600, 2),
                "runs": completed,
            }, indent=2))

    print(f"\nDone in {(time.time() - started) / 3600:.1f} h. "
          f"Manifest: {RESULTS_DIR / 'ablation_manifest.json'}")
    return 0


# --------------------------------------------------------------------------- #
# aggregate
# --------------------------------------------------------------------------- #

def _load_metrics(path: Path) -> dict | None:
    path = Path(path)
    if path.is_dir():
        for candidate in (path / "metrics.json", path / "eval" / "metrics.json"):
            if candidate.exists():
                path = candidate
                break
        else:
            return None
    return json.loads(path.read_text()) if path.exists() else None


def _variant_and_seed(run_dir: str) -> tuple[str, int]:
    name = Path(run_dir).name
    match = re.match(r"^(?P<variant>.+?)_seed(?P<seed>\d+)$", name)
    return (match.group("variant"), int(match.group("seed"))) if match else (name, 0)


def cmd_aggregate(args: argparse.Namespace) -> int:
    entries: list[tuple[str, int, dict]] = []

    if args.manifest and args.manifest.exists():
        for run in json.loads(args.manifest.read_text()).get("runs", []):
            if run.get("status") != "ok":
                continue
            metrics = _load_metrics(Path(run["run_dir"]))
            if metrics:
                entries.append((run["variant"], run["seed"], metrics))

    for path in args.runs:
        metrics = _load_metrics(path)
        if metrics:
            base = Path(path).parent if Path(path).name == "eval" else Path(path)
            variant, seed = _variant_and_seed(str(base))
            entries.append((variant, seed, metrics))

    if not entries:
        print("No metrics.json files found. Run `python evaluate.py per-class` first.",
              file=sys.stderr)
        return 1

    by_variant: dict[str, list[dict]] = defaultdict(list)
    for variant, _seed, metrics in entries:
        by_variant[variant].append(metrics)
    print(f"Loaded {len(entries)} runs across {len(by_variant)} configurations\n")

    headline = {
        variant: [m[args.headline]["ap50"] for m in runs if args.headline in m]
        for variant, runs in by_variant.items()
    }

    rows = []
    for variant in sorted(headline):
        agg = aggregate(headline[variant])
        rows.append([variant, agg.n, agg.format(),
                     f"[{agg.ci_low:.3f}, {agg.ci_high:.3f}]" if agg.n > 1 else "--"])
    print("Headline mAP@0.5 (" + args.headline.replace("_", " ") + "):")
    print(markdown_table(["Configuration", "Runs", "mean ± std", "95% CI on the mean"], rows))

    comparisons = []
    if args.baseline in headline:
        for variant in sorted(headline):
            if variant != args.baseline:
                comparisons.append(compare_configurations(
                    headline[args.baseline], headline[variant], label=variant))
        print(f"\nAgainst '{args.baseline}':")
        print(ablation_markdown(comparisons))
        print("\n'within noise' means the seed-to-seed spread is larger than the "
              "effect; with three seeds only fairly large effects are detectable.")
    else:
        print(f"\n(No configuration named '{args.baseline}'; skipping delta table.)")

    per_class_rows = []
    for variant in sorted(by_variant):
        row = [variant]
        for name in CLASS_NAMES:
            values = [r["per_class"][name][args.metric] for r in by_variant[variant]
                      if name in r.get("per_class", {})]
            supports = [r["per_class"][name]["support"] for r in by_variant[variant]
                        if name in r.get("per_class", {})]
            row.append(aggregate(values).format()
                       + ("*" if supports and max(supports) < 10 else ""))
        per_class_rows.append(row)
    print(f"\nPer-class {args.metric}:")
    print(markdown_table(["Configuration", *CLASS_NAMES], per_class_rows))
    print("\n* fewer than 10 ground-truth instances - not a reliable estimate.")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "aggregate.json").write_text(json.dumps({
        "headline_field": args.headline,
        "headline": {v: aggregate(vals).__dict__ for v, vals in headline.items()},
        "comparisons": comparisons, "metric": args.metric, "n_runs": len(entries),
    }, indent=2, default=float))
    (args.out / "ablation.md").write_text(
        ablation_markdown(comparisons) if comparisons else "no comparisons\n")
    (args.out / "ablation.tex").write_text(latex_table(
        ["Configuration", "mAP@0.5", "$\\Delta$", "d", "p", "Verdict"],
        [[c["label"], f"{c['variant_mean']:.3f} $\\pm$ {c['variant_std']:.3f}",
          f"{c['delta']:+.3f}", f"{c['cohens_d']:.2f}", f"{c['p_value']:.3f}",
          c["verdict"]] for c in comparisons],
        caption="Ablation over the baseline recipe. Mean $\\pm$ sample standard "
                "deviation over seeds; $p$ from Welch's t-test.",
        label="tab:ablation"))
    print(f"\nWrote {args.out}/aggregate.json, ablation.md, ablation.tex")
    return 0


# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("train", help="train one configuration across seeds or CV folds")
    p.add_argument("--name", required=True)
    p.add_argument("--model", default="yolov10s.pt")
    p.add_argument("--data", type=Path, default=CONFIG_DIR / "data.yaml")
    p.add_argument("--cv-dir", type=Path, default=None,
                   help="Train one run per fold*.yaml in this directory.")
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--device", default="0")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--disable", nargs="*", default=[], choices=sorted(AUGMENTATION_GROUPS),
                   help="Augmentation groups to switch off.")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="Hyperparameter overrides, e.g. --set lr0=0.005")
    p.add_argument("--project", type=Path, default=RUNS_DIR / "experiments")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("ablate", help="run the grid in configs/ablation.yaml")
    p.add_argument("--config", type=Path, default=CONFIG_DIR / "ablation.yaml")
    p.add_argument("--data", type=Path, default=CONFIG_DIR / "data.yaml")
    p.add_argument("--eval-images", nargs="+", default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--device", default="0")
    p.add_argument("--epochs", type=int, default=None,
                   help="Override the schedule; use a short one for a first pass.")
    p.add_argument("--project", type=Path, default=RUNS_DIR / "ablation")
    p.add_argument("--only", nargs="*", default=None)
    p.add_argument("--only-models", action="store_true")
    p.add_argument("--only-variants", action="store_true")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_ablate)

    p = sub.add_parser("aggregate", help="mean ± std and significance across runs")
    p.add_argument("--runs", type=Path, nargs="*", default=[])
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument("--baseline", default="baseline")
    p.add_argument("--metric", default="ap50")
    p.add_argument("--headline", default="macro_measurable_only",
                   choices=["macro", "macro_measurable_only"])
    p.add_argument("--out", type=Path, default=RESULTS_DIR / "aggregate")
    p.set_defaults(func=cmd_aggregate)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
