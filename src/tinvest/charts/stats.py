"""Shared chart aggregations used by both matplotlib and JSON builders."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from tinvest.charts.spec import BoxStats, HistogramBin


def density_histogram(values: np.ndarray | pd.Series, *, bins: int = 60) -> list[HistogramBin]:
    """Density histogram bins matching matplotlib ``density=True``."""
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return []
    counts, edges = np.histogram(array, bins=bins, density=True)
    return [
        HistogramBin(x0=float(edges[i]), x1=float(edges[i + 1]), density=float(counts[i]))
        for i in range(len(counts))
    ]


def normal_pdf_curve(x_min: float = -4.0, x_max: float = 4.0, n: int = 400) -> list[dict[str, float]]:
    xs = np.linspace(x_min, x_max, n)
    return [{"x": float(x), "y": float(y)} for x, y in zip(xs, stats.norm.pdf(xs), strict=True)]


def boxplot_stats(values: np.ndarray | pd.Series, *, label: str, show_outliers: bool = True) -> BoxStats:
    """Five-number summary with matplotlib-style 1.5 IQR whiskers."""
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return BoxStats(label=label, min=0.0, q1=0.0, median=0.0, q3=0.0, max=0.0)
    q1 = float(np.percentile(array, 25))
    median = float(np.percentile(array, 50))
    q3 = float(np.percentile(array, 75))
    iqr = q3 - q1
    low_fence = q1 - 1.5 * iqr
    high_fence = q3 + 1.5 * iqr
    inside = array[(array >= low_fence) & (array <= high_fence)]
    if inside.size == 0:
        whisker_low = float(array.min())
        whisker_high = float(array.max())
        outliers: list[float] = []
    else:
        whisker_low = float(inside.min())
        whisker_high = float(inside.max())
        outliers = (
            [float(v) for v in array[(array < low_fence) | (array > high_fence)]]
            if show_outliers
            else []
        )
    return BoxStats(
        label=label,
        min=whisker_low,
        q1=q1,
        median=median,
        q3=q3,
        max=whisker_high,
        outliers=outliers,
    )


def qq_points(standardized: np.ndarray | pd.Series) -> tuple[list[float], list[float]]:
    """Theoretical vs sample quantiles for a normal QQ plot."""
    array = np.asarray(standardized, dtype=float)
    array = array[np.isfinite(array)]
    if array.size < 2:
        return [], []
    sample = np.sort(array)
    n = sample.size
    probs = (np.arange(1, n + 1) - 0.5) / n
    theoretical = stats.norm.ppf(probs)
    return [float(v) for v in theoretical], [float(v) for v in sample]


def decile_means(
    sample: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
) -> pd.DataFrame:
    """Mean of ``y_col`` by decile of ``x_col`` with 95% SE bands."""
    frame = sample[[x_col, y_col]].dropna().copy()
    if len(frame) < 10:
        return pd.DataFrame(columns=["decile", "mean", "low", "high"])
    frame["decile"] = pd.qcut(frame[x_col], 10, labels=False, duplicates="drop") + 1
    rows: list[dict[str, float]] = []
    for decile, chunk in frame.groupby("decile", sort=True):
        mean = float(chunk[y_col].mean())
        n = int(len(chunk))
        std = float(chunk[y_col].std(ddof=1)) if n > 1 else 0.0
        se = std / np.sqrt(n) if n > 0 else 0.0
        rows.append(
            {
                "decile": float(decile),
                "mean": mean,
                "low": mean - 1.96 * se,
                "high": mean + 1.96 * se,
            }
        )
    return pd.DataFrame(rows)


def standardize_returns(panel: pd.DataFrame) -> pd.Series:
    """Pool per-ticker standardized log returns."""
    parts: list[pd.Series] = []
    for _, chunk in panel.groupby("ticker"):
        returns = chunk["log_return"].dropna()
        if len(returns) < 2:
            continue
        std = returns.std(ddof=1)
        if std == 0 or np.isnan(std):
            continue
        parts.append((returns - returns.mean()) / std)
    if not parts:
        return pd.Series(dtype=float)
    return pd.concat(parts, ignore_index=True)
