"""Linear trend of daily closes with SMA(20)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from tinvest.analytics.base import Analysis
from tinvest.analytics.frames import TrendCandlesFrame
from tinvest.client.frames import CandleFrame


@dataclass(frozen=True)
class TrendConfig:
    """Display options for the linear-trend analysis."""

    ticker: str
    figure_path: Path | None = None

    def __post_init__(self) -> None:
        ticker = self.ticker.strip().upper()
        if not ticker:
            raise ValueError("ticker must be non-empty")
        object.__setattr__(self, "ticker", ticker)
        if self.figure_path is not None:
            object.__setattr__(self, "figure_path", Path(self.figure_path))


@dataclass(frozen=True)
class TrendSummary:
    """Slope of the fitted line and the period return of the close series."""

    slope: float
    intercept: float
    label: str
    period_return: float
    candle_count: int


class LinearTrend(Analysis[TrendCandlesFrame]):
    """Add return and SMA(20), fit a degree-1 trend on close, then plot."""

    def __init__(self, config: TrendConfig) -> None:
        super().__init__(config)
        self.config: TrendConfig = config
        self._summary: TrendSummary | None = None

    def prepare(self, candles: CandleFrame) -> TrendCandlesFrame:
        if candles.empty:
            raise ValueError("Cannot prepare a trend on an empty candle frame")
        working = pd.DataFrame(candles).copy()
        working["return"] = working["close"].pct_change()
        working["sma_20"] = working["close"].rolling(20).mean()
        working["trend"] = np.nan
        frame = TrendCandlesFrame.build(working)
        self._working = frame
        self._results = None
        self._summary = None
        return frame

    def analyze(self) -> TrendCandlesFrame:
        working = pd.DataFrame(self._require_working()).copy()
        x = np.arange(len(working), dtype=float)
        close = working["close"].to_numpy(dtype=float)
        slope, intercept = np.polyfit(x, close, 1)
        working["trend"] = intercept + slope * x

        if abs(slope) < 1e-4:
            label = "около нуля"
        elif slope > 0:
            label = "рост"
        else:
            label = "снижение"

        period_return = float(working["close"].iloc[-1] / working["close"].iloc[0] - 1)
        self._summary = TrendSummary(
            slope=float(slope),
            intercept=float(intercept),
            label=label,
            period_return=period_return,
            candle_count=len(working),
        )
        result = TrendCandlesFrame.build(working)
        self._working = result
        self._results = result
        return result

    def get_summary(self) -> TrendSummary:
        if self._summary is None:
            raise RuntimeError("Call analyze() before get_summary()")
        return self._summary

    def plot(self) -> Figure:
        frame = self.get_results()
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(frame["time"], frame["close"], label="Close", linewidth=1.5)
        ax.plot(frame["time"], frame["sma_20"], label="SMA(20)", linewidth=1.2)
        ax.plot(frame["time"], frame["trend"], label="Trend", linestyle="--", linewidth=1.2)
        ax.set_title(f"{self.config.ticker}: дневные свечи и тренд")
        ax.set_xlabel("Дата")
        ax.set_ylabel("Цена, руб.")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.autofmt_xdate()
        fig.tight_layout()
        self._save(fig)
        return fig
