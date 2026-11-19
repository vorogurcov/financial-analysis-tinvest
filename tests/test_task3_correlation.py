"""Offline checks for task-3 return correlations and vol regimes."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tinvest.analytics.correlation import (
    CorrelationConfig,
    ReturnCorrelation,
)
from tinvest.research.panel import MarketFrame, PanelFrame


def _trading_days(n: int, start: datetime | None = None) -> pd.DatetimeIndex:
    start = start or datetime(2020, 1, 2, tzinfo=timezone.utc)
    return pd.bdate_range(start=start, periods=n, tz="UTC")


def _panel_from_returns(
    returns: dict[str, np.ndarray],
    *,
    sectors: dict[str, str] | None = None,
    start: datetime | None = None,
) -> PanelFrame:
    n = len(next(iter(returns.values())))
    times = _trading_days(n, start=start)
    parts: list[pd.DataFrame] = []
    for ticker, log_return in returns.items():
        close = np.exp(np.nancumsum(np.nan_to_num(log_return, nan=0.0)) + np.log(100.0))
        open_ = close.copy()
        high = close * 1.001
        low = close * 0.999
        parts.append(
            pd.DataFrame(
                {
                    "ticker": ticker,
                    "time": times,
                    "open": open_.astype(float),
                    "high": high.astype(float),
                    "low": low.astype(float),
                    "close": close.astype(float),
                    "volume": np.full(n, 1000, dtype=np.int64),
                    "lot": np.full(n, 1, dtype=np.int64),
                    "sector": (sectors or {}).get(ticker, "energy"),
                    "share_type": "common",
                    "log_return": np.asarray(log_return, dtype=float),
                }
            )
        )
    frame = pd.concat(parts, ignore_index=True)
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    return PanelFrame.build(frame)


def _market_from_returns(
    times: pd.DatetimeIndex,
    market_return: np.ndarray,
) -> MarketFrame:
    frame = pd.DataFrame(
        {
            "time": times,
            "market_return": np.asarray(market_return, dtype=float),
            "n_names": np.full(len(times), 20, dtype=np.int64),
        }
    )
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    return MarketFrame.build(frame)


def _analysis(
    panel: PanelFrame,
    market: MarketFrame,
    tmp_path: Path,
    **overrides: object,
) -> ReturnCorrelation:
    figures = tmp_path / "figures"
    config = CorrelationConfig(
        heatmap_path=figures / "task3_corr_heatmap.png",
        rolling_path=figures / "task3_rolling_corr.png",
        regimes_path=figures / "task3_regime_bars.png",
        **overrides,  # type: ignore[arg-type]
    )
    analysis = ReturnCorrelation(config)
    analysis.prepare(panel, market)
    return analysis


def test_common_factor_block_higher_than_independent(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    n = 120
    factor = rng.normal(0.0, 0.02, size=n)
    common = {
        f"T{i:02d}": factor + rng.normal(0.0, 0.002, size=n) for i in range(5)
    }
    independent = {
        f"U{i:02d}": rng.normal(0.0, 0.02, size=n) for i in range(5)
    }
    panel_common = _panel_from_returns(common)
    panel_indep = _panel_from_returns(independent)
    times = _trading_days(n)
    market = _market_from_returns(times, rng.normal(0.0, 0.01, size=n))

    a_common = _analysis(panel_common, market, tmp_path / "c")
    a_indep = _analysis(panel_indep, market, tmp_path / "i")
    mean_common = a_common.analyze().pairs["pearson"].mean()
    mean_indep = a_indep.analyze().pairs["pearson"].mean()
    assert float(mean_common) > float(mean_indep)


def test_stress_day_threshold(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    n = 80
    returns = {f"T{i:02d}": rng.normal(0.0, 0.01, size=n) for i in range(5)}
    panel = _panel_from_returns(returns)
    times = _trading_days(n)
    market_return = rng.normal(0.0, 0.005, size=n)
    market_return[10] = -0.03
    market_return[11] = -0.01
    # Pad enough stress days so the row is published with numbers path available.
    market_return[20:40] = -0.025
    market = _market_from_returns(times, market_return)

    tables = _analysis(
        panel,
        market,
        tmp_path,
        min_stress_days=15,
        min_overlap=10,
    ).analyze()
    stress = pd.DataFrame(tables.regimes).loc[
        lambda df: df["regime"] == "stress"
    ].iloc[0]
    assert int(stress["n_days"]) == 21  # one -0.03 and twenty -0.025
    assert stress["note"] == ""

    market_return_few = rng.normal(0.0, 0.005, size=n)
    market_return_few[10] = -0.03
    market_return_few[11] = -0.01
    few = _market_from_returns(times, market_return_few)
    tables_few = _analysis(panel, few, tmp_path / "few", min_stress_days=15).analyze()
    stress_few = pd.DataFrame(tables_few.regimes).loc[
        lambda df: df["regime"] == "stress"
    ].iloc[0]
    assert int(stress_few["n_days"]) == 1
    assert stress_few["note"] == "too_few_stress_days"
    assert pd.isna(stress_few["mean_pair_corr"])


def test_pair_overlap_threshold(tmp_path: Path) -> None:
    n = 100
    a = np.full(n, np.nan)
    b = np.full(n, np.nan)
    c = np.full(n, np.nan)
    a[:60] = 0.01
    b[:60] = 0.012
    c[:59] = 0.011
    # Shared calendar so all three appear; C overlaps A/B on 59 days only.
    panel = _panel_from_returns({"AAA": a, "BBB": b, "CCC": c})
    times = _trading_days(n)
    market = _market_from_returns(times, np.zeros(n))
    tables = _analysis(panel, market, tmp_path, min_overlap=60).analyze()
    pairs = pd.DataFrame(tables.pairs)
    assert set(zip(pairs["ticker_a"], pairs["ticker_b"], strict=True)) == {("AAA", "BBB")}
    assert (pairs["ticker_a"] < pairs["ticker_b"]).all()
    assert (pairs["n_overlap"] >= 60).all()


def test_short_market_window_skips_market_rv(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    n = 80
    # Need enough names for rolling pair_corr; lower floors for the synthetic set.
    returns = {
        f"T{i:02d}": rng.normal(0.0, 0.01, size=n) for i in range(22)
    }
    panel = _panel_from_returns(returns)
    times = _trading_days(n)
    market_return = rng.normal(0.0, 0.01, size=n)
    # Drop the first market observation so the first 63-day window is short.
    market_return[0] = np.nan
    market = _market_from_returns(times, market_return)

    tables = _analysis(
        panel,
        market,
        tmp_path,
        window=63,
        min_names_in_window=20,
        min_returns_in_window=50,
        min_regression_rows=10,
    ).analyze()
    rolling = pd.DataFrame(tables.rolling)
    # Window ending at index 62 covers dates 0..62 and includes the NaN at 0.
    first_full = rolling.loc[rolling["time"] == times[62]].iloc[0]
    assert pd.isna(first_full["market_rv"])
    # Window ending at 79 covers 17..79 and excludes day 0.
    later = rolling.loc[rolling["time"] == times[79]].iloc[0]
    assert pd.notna(later["market_rv"])


def test_regression_positive_on_noiseless_link(tmp_path: Path) -> None:
    n = 150
    market_rv = np.linspace(0.1, 0.4, n)
    pair_corr = 0.2 + 1.5 * market_rv
    times = _trading_days(n)
    rolling = pd.DataFrame(
        {
            "time": times,
            "pair_corr": pair_corr,
            "market_rv": market_rv,
            "n_names": np.full(n, 25, dtype=np.int64),
        }
    )
    rolling["time"] = pd.to_datetime(rolling["time"], utc=True)

    # Bypass prepare: inject rolling into a minimal analysis path via private fit.
    from tinvest.analytics.correlation import (
        CorrelationRollingFrame,
        _fit_hac_regression,
    )

    frame = CorrelationRollingFrame.build(rolling)
    reg = pd.DataFrame(_fit_hac_regression(frame, min_rows=100))
    slope = reg.loc[reg["term"] == "market_rv"].iloc[0]
    assert float(slope["coef"]) > 0
    assert slope["cov"] == "HAC_63"
    assert slope["note"] == ""


def test_too_few_windows_skips_coefficients(tmp_path: Path) -> None:
    n = 50
    rolling = pd.DataFrame(
        {
            "time": _trading_days(n),
            "pair_corr": np.linspace(0.1, 0.5, n),
            "market_rv": np.linspace(0.1, 0.3, n),
            "n_names": np.full(n, 25, dtype=np.int64),
        }
    )
    rolling["time"] = pd.to_datetime(rolling["time"], utc=True)
    from tinvest.analytics.correlation import (
        CorrelationRollingFrame,
        _fit_hac_regression,
    )

    reg = pd.DataFrame(
        _fit_hac_regression(CorrelationRollingFrame.build(rolling), min_rows=100)
    )
    assert len(reg) == 1
    assert reg.iloc[0]["note"] == "too_few_windows"
    assert pd.isna(reg.iloc[0]["coef"])
    assert reg.iloc[0]["cov"] == ""


def test_empty_panel_raises(tmp_path: Path) -> None:
    analysis = ReturnCorrelation(
        CorrelationConfig(
            heatmap_path=tmp_path / "h.png",
            rolling_path=tmp_path / "r.png",
            regimes_path=tmp_path / "b.png",
        )
    )
    with pytest.raises(ValueError, match="выборка пуста"):
        analysis.prepare(PanelFrame.blank(), MarketFrame.blank())
