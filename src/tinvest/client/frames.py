"""Typed frames produced by the market-data client."""

from datetime import datetime

from tinvest.common.frame import TypedFrame


class CandleFrame(TypedFrame):
    """Daily OHLCV candles for one instrument."""

    time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
