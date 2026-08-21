"""Render results as Markdown and LaTeX tables.

The report, the README and the model card all quote the same numbers. Rendering
them from one source of truth means they cannot silently drift apart, and a
re-run regenerates every table without anyone retyping a decimal.
"""

from __future__ import annotations

import math
from typing import Iterable, Mapping, Sequence

from .constants import CLASS_NAMES

LOW_SUPPORT = 10  # below this many instances a per-class figure is not reportable


def _fmt(value: float | None, digits: int = 3, dash: str = "--") -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return dash
    return f"{value:.{digits}f}"


def _fmt_ci(metrics, field: str, digits: int = 3) -> str:
    value = getattr(metrics, field, None) if not isinstance(metrics, Mapping) else metrics.get(field)
    ci = (metrics.ci if not isinstance(metrics, Mapping) else metrics.get("ci", {})).get(field)
    base = _fmt(value, digits)
    if not ci or any(math.isnan(v) for v in ci):
        return base
    return f"{base} [{ci[0]:.{digits}f}, {ci[1]:.{digits}f}]"


def markdown_table(headers: Sequence[str], rows: Iterable[Sequence[str]]) -> str:
    rows = [list(map(str, r)) for r in rows]
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    out = ["| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |"]
    out.append("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    for row in rows:
        out.append("| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(row)) + " |")
    return "\n".join(out)


def latex_table(
    headers: Sequence[str],
    rows: Iterable[Sequence[str]],
    caption: str = "",
    label: str = "",
    align: str | None = None,
) -> str:
    align = align or "l" + "r" * (len(headers) - 1)
    body = "\n".join(
        " & ".join(str(c).replace("&", r"\&").replace("_", r"\_") for c in row) + r" \\"
        for row in rows
    )
    header = " & ".join(
        r"\textbf{" + str(h).replace("&", r"\&").replace("_", r"\_") + "}" for h in headers
    )
    return "\n".join([
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        rf"\begin{{tabular}}{{{align}}}",
        r"\toprule",
        header + r" \\",
        r"\midrule",
        body,
        r"\bottomrule",
        r"\end{tabular}",
        rf"\caption{{{caption}}}" if caption else "",
        rf"\label{{{label}}}" if label else "",
        r"\end{table}",
    ])


def per_class_rows(
    metrics: Mapping[str, object],
    with_ci: bool = True,
    class_names: Sequence[str] = tuple(CLASS_NAMES),
) -> list[list[str]]:
    """Rows for a per-class results table, flagging unmeasurable classes."""
    rows: list[list[str]] = []
    for name in class_names:
        m = metrics.get(name)
        if m is None:
            continue
        support = getattr(m, "support", 0)
        flag = " *" if support < LOW_SUPPORT else ""
        if with_ci:
            rows.append([
                name + flag, str(support),
                _fmt_ci(m, "precision"), _fmt_ci(m, "recall"),
                _fmt(getattr(m, "f1", float("nan"))),
                _fmt_ci(m, "ap50"), _fmt(getattr(m, "ap50_95", float("nan"))),
            ])
        else:
            rows.append([
                name + flag, str(support),
                _fmt(getattr(m, "precision", float("nan"))),
                _fmt(getattr(m, "recall", float("nan"))),
                _fmt(getattr(m, "f1", float("nan"))),
                _fmt(getattr(m, "ap50", float("nan"))),
                _fmt(getattr(m, "ap50_95", float("nan"))),
            ])
    return rows


PER_CLASS_HEADERS = ["Class", "N", "Precision", "Recall", "F1", "AP@0.5", "AP@0.5:0.95"]

LOW_SUPPORT_NOTE = (
    f"* fewer than {LOW_SUPPORT} ground-truth instances in this split; the "
    "per-class figure is reported for completeness but is not a reliable estimate."
)


def per_class_markdown(metrics: Mapping[str, object], with_ci: bool = True) -> str:
    table = markdown_table(PER_CLASS_HEADERS, per_class_rows(metrics, with_ci))
    return f"{table}\n\n{LOW_SUPPORT_NOTE}\n"


def aggregate_rows(
    per_class: Mapping[str, object],
    supports: Mapping[str, int] | None = None,
) -> list[list[str]]:
    """Rows for a mean ± std table produced by ``aggregate_per_class``."""
    rows = []
    for name, agg in per_class.items():
        support = supports.get(name, "") if supports else ""
        rows.append([name, str(support), str(agg.n), agg.format(),
                     f"[{_fmt(agg.ci_low)}, {_fmt(agg.ci_high)}]"])
    return rows


def ablation_markdown(comparisons: Sequence[Mapping[str, object]], metric: str = "mAP@0.5") -> str:
    headers = ["Configuration", f"{metric} (mean ± std)", "Δ vs baseline",
               "Cohen's d", "p", "Verdict"]
    rows = []
    for comparison in comparisons:
        rows.append([
            comparison["label"],
            f"{comparison['variant_mean']:.3f} ± {comparison['variant_std']:.3f}",
            f"{comparison['delta']:+.3f}",
            _fmt(comparison["cohens_d"], 2),
            _fmt(comparison["p_value"], 3),
            comparison["verdict"],
        ])
    return markdown_table(headers, rows)


def error_breakdown_markdown(breakdown) -> str:
    fractions = breakdown.as_fractions()
    headers = ["Error type", "Count", "Share of all errors"]
    counts = dict(breakdown.by_category)
    counts["missed"] = breakdown.n_missed
    rows = [
        [key, str(counts.get(key, 0)), f"{fractions.get(key, 0.0) * 100:.1f}%"]
        for key in sorted(counts, key=lambda k: -counts.get(k, 0))
    ]
    return markdown_table(headers, rows)
