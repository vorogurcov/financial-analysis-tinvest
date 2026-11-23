"""Typed chart payloads for the web UI."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

ChartKind = Literal[
    "line",
    "histogram",
    "scatter",
    "boxplot",
    "errorbar",
    "heatmap",
    "dual_line",
    "bar",
    "grouped_bar",
    "qq_grid",
]


@dataclass(frozen=True)
class ChartSeries:
    """One named series on a chart."""

    name: str
    values: list[float | None] = field(default_factory=list)
    axis: Literal["left", "right"] | None = None
    x: list[float | None] | None = None
    yerr_low: list[float | None] | None = None
    yerr_high: list[float | None] | None = None
    band_low: list[float | None] | None = None
    band_high: list[float | None] | None = None


@dataclass(frozen=True)
class HistogramBin:
    """One density histogram bin."""

    x0: float
    x1: float
    density: float


@dataclass(frozen=True)
class BoxStats:
    """Five-number summary plus optional outliers (matplotlib 1.5 IQR)."""

    label: str
    min: float
    q1: float
    median: float
    q3: float
    max: float
    outliers: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class ChartSpec:
    """One interactive chart rendered from series data, never from an image."""

    id: str
    title: str
    kind: ChartKind
    x: list[str | float] | None = None
    series: list[ChartSeries] = field(default_factory=list)
    x_label: str | None = None
    y_label: str | None = None
    y_label_right: str | None = None
    bins: list[HistogramBin] | None = None
    reference_line: list[dict[str, float]] | None = None
    boxes: list[BoxStats] | None = None
    categories: list[str] | None = None
    values: list[float | None] | None = None
    groups: list[dict[str, Any]] | None = None
    matrix: list[list[float | None]] | None = None
    x_labels: list[str] | None = None
    y_labels: list[str] | None = None
    panels: list[ChartSpec] | None = None
    horizontal: bool = False


def chart_to_dict(chart: ChartSpec) -> dict[str, Any]:
    """Serialize a chart for JSON responses."""
    return asdict(chart)


def charts_to_dicts(charts: list[ChartSpec]) -> list[dict[str, Any]]:
    return [chart_to_dict(chart) for chart in charts]
