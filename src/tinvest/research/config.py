"""Configuration for the multi-instrument research pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path

from dotenv import load_dotenv

from tinvest.common.paths import default_output_dir, project_root


@dataclass(frozen=True)
class ResearchConfig:
    """Window and sample limits for the research universe and panel."""

    lookback_days: int = 1095
    min_bars: int = 252
    max_names: int = 80
    return_jump: float = 0.5
    refresh: bool = False
    output: Path = field(default_factory=default_output_dir)
    compare_ticker: str | None = None

    def __post_init__(self) -> None:
        if self.lookback_days < 1:
            raise ValueError("lookback_days must be at least 1")
        if self.min_bars < 2:
            raise ValueError("min_bars must be at least 2")
        if self.max_names < 1:
            raise ValueError("max_names must be at least 1")
        if self.return_jump <= 0:
            raise ValueError("return_jump must be positive")
        object.__setattr__(self, "output", Path(self.output))
        raw_ticker = self.compare_ticker.strip().upper() if isinstance(self.compare_ticker, str) else ""
        object.__setattr__(self, "compare_ticker", raw_ticker or None)


def load_research_settings(env_path: Path | None = None) -> ResearchConfig:
    """Read research overrides from ``.env`` at the project root."""
    root = project_root()
    path = env_path if env_path is not None else root / ".env"
    if path.is_file():
        load_dotenv(path)
    lookback = int(os.getenv("TINVEST_RESEARCH_LOOKBACK_DAYS", "1095") or "1095")
    max_names = int(os.getenv("TINVEST_RESEARCH_MAX_NAMES", "80") or "80")
    compare_ticker = os.getenv("TINVEST_RESEARCH_COMPARE_TICKER", "")
    return ResearchConfig(
        lookback_days=lookback,
        max_names=max_names,
        compare_ticker=compare_ticker,
    )
