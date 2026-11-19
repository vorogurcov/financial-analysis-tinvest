"""Long panel of included tickers and the equal-weight market series."""

from __future__ import annotations

from datetime import datetime
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd

from tinvest.common.frame import TypedFrame
from tinvest.common.paths import project_root
from tinvest.research.universe import (
    clean_ohlcv,
    moscow_today,
    processed_dir,
    read_universe,
)
from tinvest.storage import candles_dir, read_candles


class PanelFrame(TypedFrame):
    """Daily OHLCV panel with log returns for included tickers."""

    ticker: str
    time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    lot: int
    sector: str
    share_type: str
    log_return: float


class MarketFrame(TypedFrame):
    """Equal-weight average of non-empty ticker log returns by day."""

    time: datetime
    market_return: float
    n_names: int


def dropped_returns_path(root: Path | None = None) -> Path:
    return processed_dir(root) / "dropped_returns.csv"


def _ticker_panel_rows(
    candles: pd.DataFrame,
    *,
    ticker: str,
    lot: int,
    sector: str,
    share_type: str,
    return_jump: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build panel rows and dropped-return rows for one ticker."""
    working = candles.copy()
    if working.empty:
        empty_panel = pd.DataFrame(
            columns=[
                "ticker",
                "time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "lot",
                "sector",
                "share_type",
                "log_return",
            ]
        )
        empty_dropped = pd.DataFrame(
            columns=["ticker", "time", "close_prev", "close", "simple_return", "reason"]
        )
        return empty_panel, empty_dropped

    close = working["close"].astype(float)
    prev = close.shift(1)
    simple = close / prev - 1.0
    log_return = np.log(close / prev)

    jump_mask = simple.notna() & ((simple < -return_jump) | (simple > return_jump))
    dropped_rows: list[dict[str, object]] = []
    for idx in working.index[jump_mask]:
        dropped_rows.append(
            {
                "ticker": ticker,
                "time": working.at[idx, "time"],
                "close_prev": float(prev.at[idx]),
                "close": float(close.at[idx]),
                "simple_return": float(simple.at[idx]),
                "reason": "abs_simple_return_gt_0.5",
            }
        )
        log_return.at[idx] = np.nan

    panel = pd.DataFrame(
        {
            "ticker": ticker,
            "time": working["time"],
            "open": working["open"].astype(float),
            "high": working["high"].astype(float),
            "low": working["low"].astype(float),
            "close": close,
            "volume": working["volume"].astype("int64"),
            "lot": int(lot),
            "sector": sector,
            "share_type": share_type,
            "log_return": log_return.astype(float),
        }
    )
    dropped = pd.DataFrame(dropped_rows)
    if dropped.empty:
        dropped = pd.DataFrame(
            columns=["ticker", "time", "close_prev", "close", "simple_return", "reason"]
        )
    return panel, dropped


def load_panel(
    root: Path | str | None = None,
    *,
    return_jump: float = 0.5,
) -> PanelFrame:
    """Assemble the long panel for included universe tickers.

    The panel is kept in memory. Jump-filtered returns are written to
    ``data/processed/dropped_returns.csv`` (rewritten each call).
    """
    base = Path(root) if root is not None else project_root()
    universe = read_universe(base)
    included = universe.loc[universe["included"]].copy()
    today = moscow_today()

    panels: list[pd.DataFrame] = []
    dropped_parts: list[pd.DataFrame] = []

    for row in included.itertuples(index=False):
        candles = read_candles(candles_dir(base, row.ticker))
        cleaned = clean_ohlcv(pd.DataFrame(candles), today=today)
        panel_part, dropped_part = _ticker_panel_rows(
            cleaned,
            ticker=str(row.ticker),
            lot=int(row.lot),
            sector=str(row.sector),
            share_type=str(row.share_type),
            return_jump=return_jump,
        )
        if not panel_part.empty:
            panels.append(panel_part)
        if not dropped_part.empty:
            dropped_parts.append(dropped_part)

    drop_path = dropped_returns_path(base)
    drop_path.parent.mkdir(parents=True, exist_ok=True)
    if dropped_parts:
        dropped = pd.concat(dropped_parts, ignore_index=True)
        dropped["time"] = pd.to_datetime(dropped["time"], utc=True)
        dropped.to_csv(drop_path, index=False, encoding="utf-8-sig")
    else:
        empty = pd.DataFrame(
            columns=["ticker", "time", "close_prev", "close", "simple_return", "reason"]
        )
        empty.to_csv(drop_path, index=False, encoding="utf-8-sig")

    if not panels:
        return PanelFrame.blank()

    frame = pd.concat(panels, ignore_index=True)
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    frame["volume"] = frame["volume"].astype("int64")
    frame["lot"] = frame["lot"].astype("int64")
    for column in ("open", "high", "low", "close", "log_return"):
        frame[column] = frame[column].astype("float64")
    frame["ticker"] = frame["ticker"].astype(str)
    frame["sector"] = frame["sector"].astype(str)
    frame["share_type"] = frame["share_type"].astype(str)
    return PanelFrame.build(frame)


def equal_weight_market(panel: PanelFrame) -> MarketFrame:
    """Average non-empty log returns; keep days with at least 80% coverage."""
    if panel.empty:
        return MarketFrame.blank()

    working = pd.DataFrame(panel)
    n_tickers = working["ticker"].nunique()
    if n_tickers < 1:
        return MarketFrame.blank()
    min_names = ceil(0.8 * n_tickers)

    grouped = working.groupby("time", sort=True)
    rows: list[dict[str, object]] = []
    for time_value, chunk in grouped:
        valid = chunk["log_return"].dropna()
        n_names = int(len(valid))
        if n_names < min_names:
            continue
        rows.append(
            {
                "time": time_value,
                "market_return": float(valid.mean()),
                "n_names": n_names,
            }
        )

    if not rows:
        return MarketFrame.blank()

    frame = pd.DataFrame(rows)
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    frame["market_return"] = frame["market_return"].astype("float64")
    frame["n_names"] = frame["n_names"].astype("int64")
    return MarketFrame.build(frame)
