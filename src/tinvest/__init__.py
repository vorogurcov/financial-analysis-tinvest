"""T-Invest market analytics: fetch candles, store them, then analyze."""

from tinvest.analytics import LinearTrend, TrendConfig, TrendSummary
from tinvest.client import CandleFrame, FetchConfig, fetch_candles
from tinvest.pipeline import PipelineConfig, collect, load_settings, run
from tinvest.storage import candles_dir, read_candles, write_candles

__all__ = [
    "CandleFrame",
    "FetchConfig",
    "LinearTrend",
    "PipelineConfig",
    "TrendConfig",
    "TrendSummary",
    "candles_dir",
    "collect",
    "fetch_candles",
    "load_settings",
    "read_candles",
    "run",
    "write_candles",
]
