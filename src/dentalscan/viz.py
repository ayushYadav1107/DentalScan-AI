"""Figures for the report and the README.

Style rules applied throughout: one measure per axis (never a second y-scale),
recessive grid, thin marks, direct labels where they fit, a legend whenever more
than one series is drawn, and a hatch pattern or marker shape alongside every
colour so the figures survive colour-vision deficiency, greyscale printing and
the reviewer who prints the PDF single-colour.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .constants import (
    CLASS_COLORS_HEX,
    CLASS_HATCH,
    CLASS_NAMES,
    MODEL_COLORS_HEX,
    MODEL_MARKERS,
)

GRID_KW = dict(color="#D8D8D6", linewidth=0.6, alpha=0.9)
INK = "#1A1A1A"
MUTED = "#6B6B6B"


def apply_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.edgecolor": "#B9B9B6",
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "hatch.linewidth": 0.6,
    })


def _save(fig, out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    if out_path.suffix != ".pdf":
        fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #

def plot_class_support(
    counts_by_split: Mapping[str, Mapping[str, int]],
    out_path: Path,
    title: str = "Ground-truth instances per class",
) -> Path:
    """Horizontal bars of instance counts, log x-axis.

    Magnitude spanning two orders of magnitude is the whole point of this
    figure, so the axis is log and the exact counts are direct-labelled - a
    linear axis would render the rare classes as invisible slivers and hide the
    finding.
    """
    apply_style()
    splits = list(counts_by_split)
    fig, axes = plt.subplots(
        1, len(splits), figsize=(2.6 * len(splits) + 1.2, 2.9), sharey=True
    )
    axes = np.atleast_1d(axes)

    y = np.arange(len(CLASS_NAMES))
    for ax, split in zip(axes, splits):
        values = [counts_by_split[split].get(name, 0) for name in CLASS_NAMES]
        for i, (name, value) in enumerate(zip(CLASS_NAMES, values)):
            ax.barh(i, max(value, 0.6), height=0.62,
                    color=CLASS_COLORS_HEX[name], hatch=CLASS_HATCH[name],
                    edgecolor="white", linewidth=0.8)
            ax.text(max(value, 0.6) * 1.15, i, str(value), va="center",
                    fontsize=7.5, color=INK)
        ax.set_xscale("log")
        ax.set_xlim(0.6, max(max(values), 1) * 4)
        ax.set_title(f"{split}  (n={sum(values)} boxes)", color=INK)
        ax.xaxis.grid(True, **GRID_KW)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", labelbottom=False)
        ax.set_xlabel("instances (log scale)", fontsize=7.5)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(CLASS_NAMES)
    axes[0].invert_yaxis()
    fig.suptitle(title, fontsize=10.5, y=1.02, color=INK)
    return _save(fig, out_path)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #

def plot_training_curves(
    curves: Mapping[str, "object"],
    out_path: Path,
    metrics: Sequence[tuple[str, str]] = (
        ("metrics/mAP50(B)", "mAP@0.5"),
        ("metrics/mAP50-95(B)", "mAP@0.5:0.95"),
        ("val/cls_loss", "validation classification loss"),
    ),
) -> Path:
    """Small multiples of training progress, one panel per metric.

    Separate panels rather than a twin axis: mAP and loss have different units
    and opposite polarity, and overlaying them on two y-scales is the single
    most misleading thing a training-curve figure can do.
    """
    apply_style()
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.1 * len(metrics), 2.7))
    axes = np.atleast_1d(axes)

    for ax, (column, label) in zip(axes, metrics):
        for model_name, frame in curves.items():
            if column not in frame:
                continue
            epochs = frame["epoch"]
            values = frame[column]
            color = MODEL_COLORS_HEX.get(model_name, "#0072B2")
            marker = MODEL_MARKERS.get(model_name, "o")
            ax.plot(epochs, values, linewidth=1.4, color=color, label=model_name)
            # Sparse markers give the second (non-colour) encoding without
            # turning the line into a dotted mess.
            step = max(len(epochs) // 8, 1)
            ax.plot(epochs[::step], values[::step], linestyle="none",
                    marker=marker, markersize=4, color=color,
                    markeredgecolor="white", markeredgewidth=0.6)
        ax.set_xlabel("epoch")
        ax.set_title(label, color=INK)
        ax.yaxis.grid(True, **GRID_KW)
        ax.set_axisbelow(True)

    axes[0].legend(loc="lower right", ncol=1)
    return _save(fig, out_path)


# --------------------------------------------------------------------------- #
# Per-class results
# --------------------------------------------------------------------------- #

def plot_per_class_comparison(
    results: Mapping[str, Mapping[str, float]],
    out_path: Path,
    metric_label: str = "AP@0.5",
    supports: Mapping[str, int] | None = None,
    low_support: int = 10,
) -> Path:
    """Grouped bars: one group per class, one bar per model.

    Classes whose support is too small to measure are drawn with a hatched,
    outlined bar and marked in the tick label, so the eye cannot read a tall bar
    on three instances as a real result.
    """
    apply_style()
    models = list(results)
    x = np.arange(len(CLASS_NAMES))
    width = 0.8 / max(len(models), 1)

    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    for i, model_name in enumerate(models):
        offset = (i - (len(models) - 1) / 2) * width
        values = [results[model_name].get(name, np.nan) for name in CLASS_NAMES]
        color = MODEL_COLORS_HEX.get(model_name, "#0072B2")
        bars = ax.bar(x + offset, values, width * 0.88, label=model_name,
                      color=color, edgecolor="white", linewidth=0.9)
        for bar, name, value in zip(bars, CLASS_NAMES, values):
            unreliable = supports is not None and supports.get(name, 0) < low_support
            if unreliable:
                bar.set_hatch("///")
                bar.set_alpha(0.55)
            if not np.isnan(value):
                ax.text(bar.get_x() + bar.get_width() / 2, value + 0.015,
                        f"{value:.2f}", ha="center", fontsize=6.4, color=MUTED)

    labels = [
        f"{n}\n(n={supports[n]})" + ("*" if supports[n] < low_support else "")
        if supports else n
        for n in CLASS_NAMES
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(metric_label)
    ax.set_ylim(0, 1.08)
    ax.yaxis.grid(True, **GRID_KW)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", ncol=len(models), bbox_to_anchor=(0, 1.02))
    if supports:
        ax.text(0, -0.32, f"* fewer than {low_support} instances - not a reliable estimate",
                transform=ax.transAxes, fontsize=7, color=MUTED)
    return _save(fig, out_path)


def plot_per_class_with_ci(
    metrics: Mapping[str, object],
    out_path: Path,
    fields: Sequence[tuple[str, str]] = (("precision", "Precision"), ("recall", "Recall")),
) -> Path:
    """Dot plot with bootstrap intervals - the honest version of a bar chart."""
    apply_style()
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    y = np.arange(len(CLASS_NAMES))
    offsets = np.linspace(-0.16, 0.16, len(fields))
    markers = ["o", "s", "^", "D"]

    for (field, label), offset, marker in zip(fields, offsets, markers):
        for i, name in enumerate(CLASS_NAMES):
            m = metrics.get(name)
            if m is None:
                continue
            value = getattr(m, field, np.nan)
            ci = getattr(m, "ci", {}).get(field)
            color = CLASS_COLORS_HEX[name]
            if ci and not any(np.isnan(v) for v in ci):
                ax.plot([ci[0], ci[1]], [i + offset] * 2, color=color,
                        linewidth=1.6, alpha=0.5, solid_capstyle="round")
            ax.plot(value, i + offset, marker=marker, markersize=6, color=color,
                    markeredgecolor="white", markeredgewidth=0.8,
                    label=label if i == 0 else None)

    ax.set_yticks(y)
    ax.set_yticklabels(CLASS_NAMES)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("value (bars: 95% bootstrap interval over images)")
    ax.xaxis.grid(True, **GRID_KW)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", ncol=len(fields), bbox_to_anchor=(0, 1.02))
    return _save(fig, out_path)


# --------------------------------------------------------------------------- #
# Errors and explanations
# --------------------------------------------------------------------------- #

ERROR_COLORS = {
    "classification": "#0072B2",
    "localization": "#E69F00",
    "both": "#009E73",
    "duplicate": "#D55E00",
    "background": "#CC79A7",
    "missed": "#56B4E9",
}
ERROR_HATCH = {
    "classification": "", "localization": "///", "both": "...",
    "duplicate": "\\\\", "background": "xxx", "missed": "---",
}


def plot_error_breakdown(breakdowns: Mapping[str, object], out_path: Path) -> Path:
    """Stacked shares of each error category, one bar per model."""
    apply_style()
    models = list(breakdowns)
    categories = ["missed", "background", "classification",
                  "localization", "both", "duplicate"]

    fig, ax = plt.subplots(figsize=(6.6, 0.7 * len(models) + 1.8))
    left = np.zeros(len(models))
    for category in categories:
        values = np.array([
            breakdowns[m].as_fractions().get(category, 0.0) for m in models
        ])
        ax.barh(models, values, left=left, height=0.55,
                color=ERROR_COLORS[category], hatch=ERROR_HATCH[category],
                edgecolor="white", linewidth=1.2, label=category)
        for i, (value, base) in enumerate(zip(values, left)):
            if value > 0.06:
                ax.text(base + value / 2, i, f"{value * 100:.0f}%", ha="center",
                        va="center", fontsize=7, color="white", weight="bold")
        left += values

    ax.set_xlim(0, 1)
    ax.set_xlabel("share of all errors")
    ax.xaxis.grid(True, **GRID_KW)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=3)
    return _save(fig, out_path)


def plot_confidence_sweep(rows: Sequence[dict], out_path: Path,
                          metric: str = "f1") -> Path:
    """Per-class metric as a function of the confidence threshold."""
    apply_style()
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    by_class: dict[str, list[dict]] = {}
    for row in rows:
        by_class.setdefault(row["class"], []).append(row)

    for name, class_rows in by_class.items():
        class_rows = sorted(class_rows, key=lambda r: r["conf"])
        xs = [r["conf"] for r in class_rows]
        ys = [r[metric] for r in class_rows]
        color = CLASS_COLORS_HEX.get(name, "#0072B2")
        ax.plot(xs, ys, linewidth=1.5, color=color, label=name)
        best = max(
            (r for r in class_rows if not np.isnan(r[metric])),
            key=lambda r: r[metric], default=None,
        )
        if best:
            ax.plot(best["conf"], best[metric], marker="o", markersize=5,
                    color=color, markeredgecolor="white", markeredgewidth=0.8)

    ax.set_xlabel("confidence threshold")
    ax.set_ylabel(metric.upper())
    ax.set_ylim(0, 1.02)
    ax.yaxis.grid(True, **GRID_KW)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=3)
    ax.set_title("dots mark each class's best threshold", fontsize=8,
                 color=MUTED, loc="right")
    return _save(fig, out_path)


def plot_cam_panel(
    panels: Sequence[tuple[str, np.ndarray]],
    out_path: Path,
    suptitle: str = "",
) -> Path:
    """Grid of ``(caption, BGR image)`` panels for the explainability figure."""
    apply_style()
    import cv2

    n = len(panels)
    cols = min(n, 3)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.1 * cols, 2.6 * rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, (caption, image) in zip(axes, panels):
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax.set_title(caption, fontsize=8, color=INK)
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")
    if suptitle:
        fig.suptitle(suptitle, fontsize=10.5, y=1.0, color=INK)
    return _save(fig, out_path)
