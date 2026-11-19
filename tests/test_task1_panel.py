"""Offline checks for the task-1 panel, market series, and distribution prep."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tinvest.analytics.distribution import DistributionConfig, ReturnDistribution
from tinvest.client.frames import CandleFrame
from tinvest.research.panel import PanelFrame, equal_weight_market, load_panel
from tinvest.research.universe import (
    CandidateEval,
    UniverseFrame,
    select_included,
    universe_path,
)
from tinvest.storage import write_candles


def _trading_days(n: int, start: datetime | None = None) -> pd.DatetimeIndex:
    start = start or datetime(2020, 1, 2, tzinfo=timezone.utc)
    # Business days keep a dense daily series without weekend gaps for synthetic tests.
    return pd.bdate_range(start=start, periods=n, tz="UTC")


def _synthetic_candles(
    n: int = 300,
    *,
    seed: int = 0,
    lot_volume: int = 100,
    jump_at: int | None = None,
    start_price: float = 100.0,
) -> CandleFrame:
    rng = np.random.default_rng(seed)
    times = _trading_days(n)
    noise = rng.normal(0.0, 0.01, size=n)
    log_prices = np.log(start_price) + np.cumsum(noise)
    close = np.exp(log_prices)
    if jump_at is not None:
        close = close.copy()
        close[jump_at:] = close[jump_at:] * 2.0
    open_ = close * (1.0 + rng.normal(0.0, 0.0005, size=n))
    high = np.maximum(open_, close) * (1.0 + 0.001)
    low = np.minimum(open_, close) * (1.0 - 0.001)
    frame = pd.DataFrame(
        {
            "time": times,
            "open": open_.astype(float),
            "high": high.astype(float),
            "low": low.astype(float),
            "close": close.astype(float),
            "volume": np.full(n, lot_volume, dtype=np.int64),
        }
    )
    return CandleFrame.build(frame)


def _write_universe(root: Path, rows: list[dict[str, object]]) -> UniverseFrame:
    frame = pd.DataFrame(rows)
    frame["ipo_date"] = pd.to_datetime(frame["ipo_date"], utc=True)
    frame["lot"] = frame["lot"].astype("int64")
    frame["issue_size"] = frame["issue_size"].astype("int64")
    frame["nominal"] = frame["nominal"].astype("float64")
    for flag in ("div_yield_flag", "liquidity_flag", "short_enabled_flag", "included"):
        frame[flag] = frame[flag].astype(bool)
    catalog = UniverseFrame.build(frame)
    path = universe_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_parquet(path, index=False)
    return catalog


def _base_card(ticker: str, *, lot: int, included: bool, exclude_reason: str = "") -> dict[str, object]:
    return {
        "ticker": ticker,
        "figi": f"FIGI_{ticker}",
        "name": ticker,
        "class_code": "TQBR",
        "currency": "rub",
        "country_of_risk": "RU",
        "lot": lot,
        "sector": "energy",
        "share_type": "common",
        "issue_size": 1_000_000,
        "nominal": 1.0,
        "div_yield_flag": True,
        "liquidity_flag": True,
        "short_enabled_flag": False,
        "ipo_date": datetime(2005, 1, 1, tzinfo=timezone.utc),
        "included": included,
        "exclude_reason": exclude_reason,
    }


def test_ann_vol_matches_std_formula(tmp_path: Path) -> None:
    tickers = [("AAA", 0), ("BBB", 1), ("CCC", 2)]
    cards = []
    for ticker, seed in tickers:
        candles = _synthetic_candles(300, seed=seed, lot_volume=100)
        write_candles(candles, tmp_path / "data" / "processed" / ticker)
        cards.append(_base_card(ticker, lot=10, included=True))
    _write_universe(tmp_path, cards)

    panel = load_panel(tmp_path)
    analysis = ReturnDistribution(
        DistributionConfig(
            figures_dir=tmp_path / "figures",
            universe_path=universe_path(tmp_path),
        )
    )
    metrics = analysis.prepare(panel)
    metrics_df = pd.DataFrame(metrics).set_index("ticker")

    panel_df = pd.DataFrame(panel)
    for ticker, _ in tickers:
        returns = panel_df.loc[panel_df["ticker"] == ticker, "log_return"].dropna()
        expected = float(returns.std(ddof=1) * np.sqrt(252.0))
        assert metrics_df.loc[ticker, "ann_vol"] == pytest.approx(expected, abs=1e-6)


def test_price_jump_sets_nan_and_logs_drop(tmp_path: Path) -> None:
    candles = _synthetic_candles(300, seed=3, jump_at=150)
    write_candles(candles, tmp_path / "data" / "processed" / "JMP")
    _write_universe(tmp_path, [_base_card("JMP", lot=1, included=True)])

    panel = load_panel(tmp_path)
    row = pd.DataFrame(panel).iloc[150]
    assert np.isnan(row["log_return"])

    returns = pd.DataFrame(panel)["log_return"]
    assert np.isfinite(returns.iloc[149])
    assert np.isfinite(returns.iloc[151])

    dropped = pd.read_csv(tmp_path / "data" / "processed" / "dropped_returns.csv")
    assert len(dropped) == 1
    assert dropped.iloc[0]["reason"] == "abs_simple_return_gt_0.5"
    assert dropped.iloc[0]["ticker"] == "JMP"


def test_max_names_keeps_top_turnover() -> None:
    evaluations = [
        CandidateEval("LOW", "f1", n_bars=300, median_turnover_rub=10.0),
        CandidateEval("MID", "f2", n_bars=300, median_turnover_rub=50.0),
        CandidateEval("HIGH", "f3", n_bars=300, median_turnover_rub=100.0),
    ]
    reasons = select_included(evaluations, min_bars=252, max_names=2)
    assert reasons["HIGH"] == ""
    assert reasons["MID"] == ""
    assert reasons["LOW"] == "over_cap"


def test_equal_weight_market_requires_coverage() -> None:
    times = _trading_days(3)
    rows = []
    # Day 0: all three present; day 1: only one of three (< 80%); day 2: all three.
    for ticker in ("A", "B", "C"):
        for index, time_value in enumerate(times):
            if index == 1 and ticker != "A":
                log_return = float("nan")
            else:
                log_return = 0.01
            rows.append(
                {
                    "ticker": ticker,
                    "time": time_value,
                    "open": 1.0,
                    "high": 1.0,
                    "low": 1.0,
                    "close": 1.0,
                    "volume": 1,
                    "lot": 1,
                    "sector": "energy",
                    "share_type": "common",
                    "log_return": log_return,
                }
            )
    frame = pd.DataFrame(rows)
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    frame["volume"] = frame["volume"].astype("int64")
    frame["lot"] = frame["lot"].astype("int64")
    for column in ("open", "high", "low", "close", "log_return"):
        frame[column] = frame[column].astype("float64")
    panel = PanelFrame.build(frame)
    market = equal_weight_market(panel)
    market_times = set(pd.to_datetime(market["time"], utc=True))
    assert times[0] in market_times
    assert times[1] not in market_times
    assert times[2] in market_times


def test_prepare_empty_panel_raises(tmp_path: Path) -> None:
    analysis = ReturnDistribution(
        DistributionConfig(
            figures_dir=tmp_path / "figures",
            universe_path=tmp_path / "data" / "processed" / "universe.parquet",
        )
    )
    with pytest.raises(ValueError, match="выборка пуста"):
        analysis.prepare(PanelFrame.blank())
