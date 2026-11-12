"""Shared steps for an analysis that reads candles and returns a typed result."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, TypeVar

import pandas as pd
from matplotlib.figure import Figure

from tinvest.client.frames import CandleFrame
from tinvest.common.frame import TypedFrame

ResultT = TypeVar("ResultT")


class Analysis(ABC, Generic[ResultT]):
    """One notebook comparison: prepare a working frame, then analyze and plot it."""

    def __init__(self, config: object) -> None:
        self.config = config
        self._working: TypedFrame | None = None
        self._results: ResultT | None = None

    @abstractmethod
    def prepare(self, candles: CandleFrame) -> TypedFrame:
        """Build this analysis's working frame from candles."""

    @abstractmethod
    def analyze(self) -> ResultT:
        """Compute the typed result from the working frame."""

    def get_results(self) -> ResultT:
        if self._results is None:
            raise RuntimeError("Call analyze() before get_results()")
        return self._results

    @abstractmethod
    def plot(self) -> Figure:
        """Draw this analysis's single figure."""

    def to_csv(self, path: str | Path) -> None:
        results = self.get_results()
        if not isinstance(results, pd.DataFrame):
            raise TypeError("This analysis result is not a single frame")
        results.to_csv(path, index=False, encoding="utf-8-sig")

    def _save(self, fig: Figure) -> None:
        figure_path = getattr(self.config, "figure_path", None)
        if figure_path is None:
            return
        destination = Path(figure_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, dpi=160, bbox_inches="tight")

    def _require_working(self) -> TypedFrame:
        if self._working is None:
            raise RuntimeError("Call prepare() before analyze()")
        return self._working
