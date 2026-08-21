"""Aggregate repeated runs into estimates with uncertainty.

A single training run gives a single number, and on a 23-image validation split
that number moves by several points between seeds. Everything reported in this
project is therefore an aggregate over repeats - either seeds (same split,
different initialisation and data order) or cross-validation folds (different
splits) - together with a spread and, where two configurations are compared, a
test that asks whether the gap survives that spread.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from .constants import CLASS_NAMES


@dataclass
class Aggregate:
    """Mean, spread and interval for one metric over repeated runs."""

    n: int
    mean: float
    std: float            # sample standard deviation (ddof=1)
    sem: float
    ci_low: float
    ci_high: float
    values: list[float] = field(default_factory=list)

    def format(self, digits: int = 3) -> str:
        if self.n == 0 or math.isnan(self.mean):
            return "n/a"
        if self.n == 1:
            return f"{self.mean:.{digits}f}"
        return f"{self.mean:.{digits}f} ± {self.std:.{digits}f}"


# Student-t 97.5th percentiles for small samples; avoids a scipy dependency
# in the aggregation path, which keeps the analysis importable anywhere.
_T_CRIT_95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
              7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 15: 2.131, 20: 2.086,
              30: 2.042, 60: 2.000}


def _t_crit(df: int) -> float:
    if df <= 0:
        return float("nan")
    keys = sorted(_T_CRIT_95)
    for k in keys:
        if df <= k:
            return _T_CRIT_95[k]
    return 1.96


def aggregate(values: Sequence[float]) -> Aggregate:
    """Mean ± sample std with a t-based 95% interval on the mean."""
    clean = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    n = len(clean)
    if n == 0:
        nan = float("nan")
        return Aggregate(0, nan, nan, nan, nan, nan, [])
    mean = float(np.mean(clean))
    if n == 1:
        return Aggregate(1, mean, 0.0, 0.0, mean, mean, clean)
    std = float(np.std(clean, ddof=1))
    sem = std / math.sqrt(n)
    half = _t_crit(n - 1) * sem
    return Aggregate(n, mean, std, sem, mean - half, mean + half, clean)


def aggregate_per_class(
    runs: Sequence[dict[str, dict[str, float]]],
    metric: str = "ap50",
    class_names: Sequence[str] = tuple(CLASS_NAMES),
) -> dict[str, Aggregate]:
    """Aggregate one metric across runs, per class.

    ``runs`` is a list of ``{class_name: {metric: value}}`` dictionaries, one
    per seed or fold.
    """
    out: dict[str, Aggregate] = {}
    for name in class_names:
        values = [
            run[name][metric]
            for run in runs
            if name in run and metric in run[name]
        ]
        out[name] = aggregate(values)
    return out


def welch_ttest(a: Sequence[float], b: Sequence[float]) -> dict[str, float]:
    """Welch's t-test - unequal variances, which is the realistic assumption.

    Reported alongside the effect size because with five seeds a p-value is a
    weak instrument: Cohen's d says how big the difference is, p says how
    confident we are it is not zero, and both belong in the table.
    """
    a = np.asarray([v for v in a if not math.isnan(v)], dtype=float)
    b = np.asarray([v for v in b if not math.isnan(v)], dtype=float)
    if len(a) < 2 or len(b) < 2:
        return {"t": float("nan"), "df": float("nan"), "p": float("nan"),
                "cohens_d": float("nan"), "mean_diff": float("nan")}

    va, vb = a.var(ddof=1), b.var(ddof=1)
    na, nb = len(a), len(b)
    se = math.sqrt(va / na + vb / nb)
    diff = float(b.mean() - a.mean())
    if se == 0:
        return {"t": float("inf") if diff else 0.0, "df": float(na + nb - 2),
                "p": 0.0 if diff else 1.0, "cohens_d": float("inf") if diff else 0.0,
                "mean_diff": diff}

    t = diff / se
    df = (va / na + vb / nb) ** 2 / (
        (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1)
    )
    try:
        from scipy import stats
        p = float(2 * stats.t.sf(abs(t), df))
    except Exception:
        # Normal approximation fallback if scipy is absent.
        p = float(math.erfc(abs(t) / math.sqrt(2)))

    pooled = math.sqrt(((na - 1) * va + (nb - 1) * vb) / max(na + nb - 2, 1))
    return {
        "t": float(t), "df": float(df), "p": p,
        "cohens_d": float(diff / pooled) if pooled > 0 else float("nan"),
        "mean_diff": diff,
    }


def compare_configurations(
    baseline: Sequence[float],
    variant: Sequence[float],
    label: str = "",
) -> dict[str, object]:
    """Full comparison record for an ablation row."""
    base_agg, var_agg = aggregate(baseline), aggregate(variant)
    test = welch_ttest(baseline, variant)
    delta = var_agg.mean - base_agg.mean

    if math.isnan(test["p"]):
        verdict = "insufficient repeats"
    elif test["p"] < 0.05 and abs(test["cohens_d"]) >= 0.8:
        verdict = "moved the needle"
    elif test["p"] < 0.05:
        verdict = "significant but small"
    else:
        verdict = "within noise"

    return {
        "label": label,
        "baseline_mean": base_agg.mean, "baseline_std": base_agg.std,
        "variant_mean": var_agg.mean, "variant_std": var_agg.std,
        "delta": delta, "p_value": test["p"], "cohens_d": test["cohens_d"],
        "n_baseline": base_agg.n, "n_variant": var_agg.n,
        "verdict": verdict,
    }


def load_run_metrics(paths: Sequence[str | Path]) -> list[dict]:
    """Load a set of ``metrics.json`` files written by ``evaluate.py per-class``."""
    runs: list[dict] = []
    for path in paths:
        path = Path(path)
        if path.is_dir():
            path = path / "metrics.json"
        if path.exists():
            runs.append(json.loads(path.read_text()))
    return runs
