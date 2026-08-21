"""Dataset auditing and split construction.

Two jobs live here.

1. **Audit.** Before trusting any metric we need to know how many instances of
   each class actually exist in each split. On this dataset several classes are
   represented by a handful of boxes in the validation split, which makes
   per-class AP on that split essentially unmeasurable. `audit_dataset` makes
   that explicit instead of letting a single aggregate mAP hide it.

2. **Split construction.** The public Roboflow export augments each source
   radiograph into three variants named ``<id>_jpg.rf.<hash>.jpg``. Any split
   that ignores that grouping risks putting augmented copies of the same
   radiograph on both sides of the train/test boundary. Every split built here
   is *grouped by source id* and *stratified by class presence*.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable, Sequence

from .constants import CLASS_NAMES, NUM_CLASSES

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# ``123_jpg.rf.deadbeef.jpg`` -> ``123``;  ``123.jpg`` -> ``123``
_SOURCE_ID_RE = re.compile(r"^(?P<sid>.+?)(?:_(?:jpg|jpeg|png)\.rf\.[0-9a-f]+)?$")


def source_id(path: str | Path) -> str:
    """Return the identifier of the *original* radiograph behind a file.

    Roboflow offline augmentation produces several files per source image. They
    must never be split across train and test, so grouping keys come from here.
    """
    stem = Path(path).stem
    match = _SOURCE_ID_RE.match(stem)
    return match.group("sid") if match else stem


def read_label_file(path: Path) -> list[tuple[int, float, float, float, float]]:
    """Parse a YOLO ``.txt`` label file into ``(cls, xc, yc, w, h)`` tuples."""
    boxes: list[tuple[int, float, float, float, float]] = []
    if not path.exists():
        return boxes
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cls = int(float(parts[0]))
            xc, yc, w, h = (float(v) for v in parts[1:5])
        except ValueError:
            continue
        boxes.append((cls, xc, yc, w, h))
    return boxes


def find_label_path(image_path: Path) -> Path:
    """Map ``.../images/foo.jpg`` to ``.../labels/foo.txt``."""
    parts = list(image_path.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


def list_images(directory: Path) -> list[Path]:
    return sorted(
        p for p in directory.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES
    )


@dataclass
class SplitStats:
    """Per-split summary of what a model can actually be measured on."""

    name: str
    n_images: int = 0
    n_source_images: int = 0
    n_boxes: int = 0
    boxes_per_class: dict[str, int] = field(default_factory=dict)
    images_per_class: dict[str, int] = field(default_factory=dict)
    # Median normalised box area per class - small objects are harder.
    median_rel_area: dict[str, float] = field(default_factory=dict)
    source_ids: list[str] = field(default_factory=list)

    @property
    def augmentation_factor(self) -> float:
        return self.n_images / self.n_source_images if self.n_source_images else 0.0

    def min_support(self) -> tuple[str, int]:
        """Class with the fewest boxes - the bottleneck for per-class claims."""
        if not self.boxes_per_class:
            return ("", 0)
        return min(self.boxes_per_class.items(), key=lambda kv: kv[1])


def scan_split(images_dir: Path, name: str) -> SplitStats:
    """Collect instance counts and geometry statistics for one split."""
    stats = SplitStats(name=name)
    areas: dict[int, list[float]] = defaultdict(list)
    box_counter: Counter[int] = Counter()
    image_counter: Counter[int] = Counter()
    sources: set[str] = set()

    for image_path in list_images(images_dir):
        stats.n_images += 1
        sources.add(source_id(image_path))
        boxes = read_label_file(find_label_path(image_path))
        present: set[int] = set()
        for cls, _xc, _yc, w, h in boxes:
            if not 0 <= cls < NUM_CLASSES:
                continue
            box_counter[cls] += 1
            areas[cls].append(w * h)
            present.add(cls)
            stats.n_boxes += 1
        for cls in present:
            image_counter[cls] += 1

    stats.n_source_images = len(sources)
    stats.source_ids = sorted(sources)
    stats.boxes_per_class = {n: box_counter.get(i, 0) for i, n in enumerate(CLASS_NAMES)}
    stats.images_per_class = {n: image_counter.get(i, 0) for i, n in enumerate(CLASS_NAMES)}
    stats.median_rel_area = {
        n: (sorted(areas[i])[len(areas[i]) // 2] if areas[i] else 0.0)
        for i, n in enumerate(CLASS_NAMES)
    }
    return stats


def check_leakage(splits: dict[str, SplitStats]) -> dict[str, int]:
    """Count source images shared between every pair of splits.

    A non-zero entry means augmented copies of the same radiograph appear on
    both sides of an evaluation boundary and every reported number is inflated.
    """
    overlaps: dict[str, int] = {}
    names = sorted(splits)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = set(splits[a].source_ids) & set(splits[b].source_ids)
            overlaps[f"{a}|{b}"] = len(shared)
    return overlaps


def audit_dataset(
    dataset_root: Path,
    split_dirs: dict[str, str] | None = None,
) -> dict:
    """Audit a YOLO-format dataset and return a JSON-serialisable summary.

    Parameters
    ----------
    dataset_root
        Directory containing the split folders.
    split_dirs
        Mapping of split name to relative path of its ``images`` folder.
    """
    split_dirs = split_dirs or {
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
    }

    splits: dict[str, SplitStats] = {}
    for name, rel in split_dirs.items():
        directory = dataset_root / rel
        if directory.exists():
            splits[name] = scan_split(directory, name)

    if not splits:
        raise FileNotFoundError(
            f"No split directories found under {dataset_root}. "
            f"Looked for: {list(split_dirs.values())}"
        )

    leakage = check_leakage(splits)

    # Imbalance ratio on the training split drives the rare-class failure mode.
    train = splits.get("train")
    imbalance = None
    if train and train.boxes_per_class:
        counts = [c for c in train.boxes_per_class.values() if c > 0]
        if counts:
            imbalance = max(counts) / min(counts)

    warnings: list[str] = []
    for name, stats in splits.items():
        if name == "train":
            continue
        for cls, count in stats.boxes_per_class.items():
            if count == 0:
                warnings.append(
                    f"[{name}] '{cls}' has 0 instances - AP for this class is undefined "
                    f"on this split and must not be reported."
                )
            elif count < 10:
                warnings.append(
                    f"[{name}] '{cls}' has only {count} instances - per-class AP here "
                    f"has a confidence interval wider than most effects you would "
                    f"want to detect. Use cross-validation instead."
                )
    for pair, n in leakage.items():
        if n:
            warnings.append(
                f"LEAKAGE: {n} source radiograph(s) appear in both {pair.replace('|', ' and ')}."
            )
    if imbalance and imbalance > 10:
        warnings.append(
            f"Training class imbalance is {imbalance:.0f}:1 "
            f"(most vs least frequent class)."
        )

    return {
        "dataset_root": str(dataset_root),
        "splits": {name: asdict(s) for name, s in splits.items()},
        "leakage": leakage,
        "train_imbalance_ratio": imbalance,
        "warnings": warnings,
    }


# --------------------------------------------------------------------------- #
# Cross-validation split construction
# --------------------------------------------------------------------------- #

def _class_signature(image_paths: Sequence[Path]) -> frozenset[int]:
    """Set of classes present across all augmented copies of one source image."""
    classes: set[int] = set()
    for path in image_paths:
        for cls, *_ in read_label_file(find_label_path(path)):
            if 0 <= cls < NUM_CLASSES:
                classes.add(cls)
    return frozenset(classes)


def build_cv_folds(
    image_dirs: Iterable[Path],
    n_folds: int = 5,
    seed: int = 0,
) -> list[list[Path]]:
    """Group-aware, rarity-aware k-fold assignment of images.

    Images are grouped by source radiograph so augmented copies never straddle a
    fold boundary. Groups are then assigned greedily in order of rarest class
    first, which keeps classes such as ``BDC/BDR`` (14 instances in the whole
    dataset) present in as many folds as possible.

    Returns
    -------
    list of length ``n_folds``; element *i* is the list of image paths in fold *i*.
    """
    groups: dict[str, list[Path]] = defaultdict(list)
    for directory in image_dirs:
        for image_path in list_images(Path(directory)):
            groups[source_id(image_path)].append(image_path)

    if not groups:
        raise FileNotFoundError(f"No images found under {list(image_dirs)}")

    # Global class frequency, so we can process rare classes first.
    global_counts: Counter[int] = Counter()
    signatures: dict[str, frozenset[int]] = {}
    for gid, paths in groups.items():
        sig = _class_signature(paths)
        signatures[gid] = sig
        for cls in sig:
            global_counts[cls] += 1

    def rarity(gid: str) -> int:
        sig = signatures[gid]
        return min((global_counts[c] for c in sig), default=10**9)

    rng = random.Random(seed)
    order = sorted(groups, key=lambda g: (rarity(g), rng.random()))

    folds: list[list[Path]] = [[] for _ in range(n_folds)]
    # Per-fold count of each class, used to pick the emptiest fold.
    fold_class_counts: list[Counter[int]] = [Counter() for _ in range(n_folds)]

    for gid in order:
        sig = signatures[gid]

        def cost(idx: int) -> tuple:
            # Prefer the fold that is most starved of this group's classes,
            # breaking ties by overall fold size to keep folds balanced.
            starvation = tuple(sorted(fold_class_counts[idx][c] for c in sig))
            return (starvation, len(folds[idx]))

        target = min(range(n_folds), key=cost)
        folds[target].extend(groups[gid])
        for cls in sig:
            fold_class_counts[target][cls] += 1

    return folds


def write_fold_configs(
    folds: list[list[Path]],
    out_dir: Path,
    class_names: Sequence[str] = tuple(CLASS_NAMES),
    eval_roots: Sequence[Path] | None = None,
) -> list[Path]:
    """Materialise each CV fold as an Ultralytics ``data.yaml`` + image lists.

    Ultralytics accepts a path to a ``.txt`` file listing images, so no image
    data is copied - the folds are just index files.

    ``eval_roots`` restricts *validation* to images under the given directories
    while still training on everything else in the other folds. Use it to
    validate on the 231 original radiographs while training on their offline
    augmented copies: scoring three augmented variants of one radiograph would
    otherwise triple-count that radiograph in the metric and quietly weight the
    evaluation towards whichever images the augmenter happened to duplicate.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    yaml_paths: list[Path] = []

    resolved_roots = [Path(r).resolve() for r in (eval_roots or [])]

    def eligible_for_val(path: Path) -> bool:
        if not resolved_roots:
            return True
        resolved = path.resolve()
        return any(root in resolved.parents for root in resolved_roots)

    for i in range(len(folds)):
        val_images = [p for p in folds[i] if eligible_for_val(p)]
        if not val_images:
            raise ValueError(
                f"Fold {i} has no images under eval_roots={resolved_roots}. "
                "Check the paths, or drop eval_roots to validate on everything."
            )
        train_images = [p for j, f in enumerate(folds) if j != i for p in f]

        train_txt = out_dir / f"fold{i}_train.txt"
        val_txt = out_dir / f"fold{i}_val.txt"
        train_txt.write_text("\n".join(str(p.resolve()) for p in train_images) + "\n")
        val_txt.write_text("\n".join(str(p.resolve()) for p in val_images) + "\n")

        names_block = "\n".join(f"  {i}: {n}" for i, n in enumerate(class_names))
        yaml_path = out_dir / f"fold{i}.yaml"
        yaml_path.write_text(
            "# Auto-generated by dentalscan.data.write_fold_configs - do not edit.\n"
            f"path: {out_dir.resolve()}\n"
            f"train: {train_txt.resolve()}\n"
            f"val: {val_txt.resolve()}\n"
            "names:\n"
            f"{names_block}\n"
        )
        yaml_paths.append(yaml_path)

    manifest = {
        "n_folds": len(folds),
        "fold_sizes": [len(f) for f in folds],
        "fold_source_counts": [len({source_id(p) for p in f}) for f in folds],
        "eval_roots": [str(r) for r in resolved_roots],
    }
    (out_dir / "folds_manifest.json").write_text(json.dumps(manifest, indent=2))
    return yaml_paths
