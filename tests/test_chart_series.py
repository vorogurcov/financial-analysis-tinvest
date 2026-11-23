"""Offline checks for JSON chart series builders."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from tinvest.analytics.distribution import DistributionConfig, ReturnDistribution
from tinvest.charts.spec import chart_to_dict
from tinvest.client.frames import CandleFrame
from tinvest.research.panel import load_panel
from tinvest.research.universe import UniverseFrame, universe_path
from tinvest.storage import write_candles


def _trading_days(n: int, start: datetime | None = None) -> pd.DatetimeIndex:
    start = start or datetime(2020, 1, 2, tzinfo=timezone.utc)
    return pd.bdate_range(start=start, periods=n, tz="UTC")


def _synthetic_candles(n: int = 80, *, seed: int = 0) -> CandleFrame:
    rng = np.random.default_rng(seed)
    times = _trading_days(n)
    noise = rng.normal(0.0, 0.01, size=n)
    close = np.exp(np.log(100.0) + np.cumsum(noise))
    frame = pd.DataFrame(
        {
            "time": times,
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close.astype(float),
            "volume": np.full(n, 100, dtype=np.int64),
        }
    )
    return CandleFrame.build(frame)


def _write_universe(root: Path, rows: list[dict[str, object]]) -> None:
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


def _card(ticker: str, *, lot: int = 10) -> dict[str, object]:
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
        "included": True,
        "exclude_reason": "",
    }


def test_distribution_chart_specs_on_synthetic_panel(tmp_path: Path) -> None:
    for index, ticker in enumerate(("AAA", "BBB", "CCC", "DDD", "EEE")):
        candles = _synthetic_candles(120, seed=index)
        write_candles(candles, tmp_path / "data" / "processed" / ticker)
    _write_universe(
        tmp_path,
        [_card(ticker) for ticker in ("AAA", "BBB", "CCC", "DDD", "EEE")],
    )

    panel = load_panel(tmp_path)
    analysis = ReturnDistribution(
        DistributionConfig(
            figures_dir=tmp_path / "figures",
            universe_path=universe_path(tmp_path),
        )
    )
    analysis.prepare(panel)
    analysis.analyze()
    charts = analysis.chart_specs()
    ids = [chart.id for chart in charts]
    assert ids == ["return_hist", "qq_grid", "vol_by_sector", "vol_vs_size"]

    hist = chart_to_dict(charts[0])
    assert hist["kind"] == "histogram"
    assert hist["bins"]
    assert hist["reference_line"]
    assert "image" not in hist
    assert all("density" in bin_ for bin_ in hist["bins"])

    qq = chart_to_dict(charts[1])
    assert qq["kind"] == "qq_grid"
    assert qq["panels"]
