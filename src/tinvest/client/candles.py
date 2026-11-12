"""Fetch daily candles from the T-Invest API into a typed frame."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import pandas as pd
from t_tech.invest import CandleInterval, Client
from t_tech.invest.utils import now, quotation_to_decimal

from tinvest.client.frames import CandleFrame


@dataclass(frozen=True)
class FetchConfig:
    """Which instrument and window to download.

    The API token stays outside this dataclass and is passed to :func:`fetch_candles`.
    """

    ticker: str = "SBER"
    lookback_days: int = 365
    class_code: str = "TQBR"
    interval: CandleInterval = CandleInterval.CANDLE_INTERVAL_DAY

    def __post_init__(self) -> None:
        ticker = self.ticker.strip().upper()
        if not ticker:
            raise ValueError("ticker must be non-empty")
        object.__setattr__(self, "ticker", ticker)
        if self.lookback_days < 1:
            raise ValueError("lookback_days must be at least 1")
        if not self.class_code.strip():
            raise ValueError("class_code must be non-empty")


def fetch_candles(token: str, config: FetchConfig | None = None) -> CandleFrame:
    """Resolve the ticker on the configured board and download daily candles."""
    settings = config or FetchConfig()
    if not token.strip():
        raise ValueError("token must be non-empty")

    with Client(token) as client:
        found = client.instruments.find_instrument(query=settings.ticker)
        matches = [
            item
            for item in found.instruments
            if item.ticker == settings.ticker and item.class_code == settings.class_code
        ]
        if not matches:
            raise LookupError(
                f"Instrument {settings.ticker} on {settings.class_code} was not found."
            )

        instrument = matches[0]
        end = now()
        start = end - timedelta(days=settings.lookback_days)
        candles = list(
            client.get_all_candles(
                instrument_id=instrument.figi,
                from_=start,
                to=end,
                interval=settings.interval,
            )
        )

    frame = _candles_to_frame(candles)
    if frame.empty:
        raise RuntimeError("No candles were returned. Check the token and ticker.")
    return frame


def _candles_to_frame(candles: list) -> CandleFrame:
    rows = [
        {
            "time": pd.Timestamp(c.time),
            "open": float(quotation_to_decimal(c.open)),
            "high": float(quotation_to_decimal(c.high)),
            "low": float(quotation_to_decimal(c.low)),
            "close": float(quotation_to_decimal(c.close)),
            "volume": int(c.volume),
        }
        for c in candles
    ]
    if not rows:
        return CandleFrame.blank()
    frame = pd.DataFrame(rows).sort_values("time").reset_index(drop=True)
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    return CandleFrame.build(frame)
