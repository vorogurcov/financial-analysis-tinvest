"""Typed frames produced by candle analytics."""

from datetime import datetime

from tinvest.common.frame import TypedFrame


class TrendCandlesFrame(TypedFrame):
    """OHLCV candles enriched with return, SMA(20), and linear trend."""

    time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int


# ``return`` is a Python keyword, so the column is declared after the class body.
TrendCandlesFrame.__annotations__ = {
    "time": datetime,
    "open": float,
    "high": float,
    "low": float,
    "close": float,
    "volume": int,
    "return": float,
    "sma_20": float,
    "trend": float,
}
