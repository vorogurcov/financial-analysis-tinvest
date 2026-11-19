"""Offline checks for task-5 elevated volatility forecast."""

from __future__ import annotations

from datetime import datetime, timezone
from math import floor
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from tinvest.analytics.vol_forecast import (
    VolForecastConfig,
    VolatilityForecast,
    _NUMERIC_FEATURES,
    _prepare_sample,
    _split_sample,
    beats_baseline,
    build_label_threshold,
    time_split_dates,
)
from tinvest.research.panel import MarketFrame, PanelFrame


def _trading_days(n: int, start: datetime | None = None) -> pd.DatetimeIndex:
    start = start or datetime(2020, 1, 2, tzinfo=timezone.utc)
    return pd.bdate_range(start=start, periods=n, tz="UTC")


def _panel_from_returns(
    returns: dict[str, np.ndarray],
    *,
    times: pd.DatetimeIndex | None = None,
    sectors: dict[str, str] | None = None,
    volume: int = 1000,
) -> PanelFrame:
    n = len(next(iter(returns.values())))
    index = times if times is not None else _trading_days(n)
    parts: list[pd.DataFrame] = []
    for ticker, log_return in returns.items():
        close = np.exp(np.nancumsum(np.nan_to_num(log_return, nan=0.0)) + np.log(100.0))
        high = close * 1.01
        low = close * 0.99
        parts.append(
            pd.DataFrame(
                {
                    "ticker": ticker,
                    "time": index,
                    "open": close.astype(float),
                    "high": high.astype(float),
                    "low": low.astype(float),
                    "close": close.astype(float),
                    "volume": np.full(n, volume, dtype=np.int64),
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


def _fixed_market(times: pd.DatetimeIndex, values: np.ndarray | None = None) -> MarketFrame:
    if values is None:
        values = np.full(len(times), 0.001, dtype=float)
    frame = pd.DataFrame(
        {
            "time": times,
            "market_return": np.asarray(values, dtype=float),
            "n_names": np.full(len(times), 40, dtype=np.int64),
        }
    )
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    return MarketFrame.build(frame)


def _block_returns(n: int, *, seed: int = 0) -> np.ndarray:
    """Alternate calm and wide blocks so both label classes appear."""
    rng = np.random.default_rng(seed)
    out = np.empty(n, dtype=float)
    out[0] = np.nan
    position = 1
    wide = False
    while position < n:
        length = min(10, n - position)
        scale = 0.04 if wide else 0.005
        out[position : position + length] = rng.normal(0.0, scale, size=length)
        position += length
        wide = not wide
    return out


def _gate_panel(
    n_tickers: int = 40,
    n_days: int = 520,
    *,
    seed: int = 0,
) -> tuple[PanelFrame, MarketFrame]:
    sectors = [f"sector_{index // 5}" for index in range(n_tickers)]
    returns = {
        f"T{index:02d}": _block_returns(n_days, seed=seed + index)
        for index in range(n_tickers)
    }
    sector_map = {f"T{index:02d}": sectors[index] for index in range(n_tickers)}
    panel = _panel_from_returns(returns, sectors=sector_map)
    times = _trading_days(n_days)
    market = _fixed_market(times)
    return panel, market


def test_horizon_no_leakage() -> None:
    n = 250
    returns = {"AAA": _block_returns(n, seed=1)}
    panel = _panel_from_returns(returns)
    market = _fixed_market(_trading_days(n))
    _labeled, sample = _prepare_sample(
        panel, market, horizon=5, quantile=0.8, min_past_labels=120
    )
    assert not sample.empty
    row = sample.iloc[len(sample) // 2]
    ticker = str(row["ticker"])
    time = pd.Timestamp(row["time"])
    feature_cols = list(_NUMERIC_FEATURES)

    before = sample.loc[(sample["ticker"] == ticker) & (sample["time"] == time)].iloc[0]

    panel_df = pd.DataFrame(panel).copy()
    mask = (panel_df["ticker"] == ticker) & (panel_df["time"] > time)
    shuffled = panel_df.loc[mask, "log_return"].to_numpy(dtype=float).copy()
    rng = np.random.default_rng(42)
    rng.shuffle(shuffled)
    panel_df.loc[mask, "log_return"] = shuffled
    # Rebuild OHLC consistently is unnecessary for feature leakage of return-based fields;
    # high/low/close stay fixed so parkinson/range/sma_gap stay put.
    panel_shuf = PanelFrame.build(panel_df)
    _labeled2, sample2 = _prepare_sample(
        panel_shuf, market, horizon=5, quantile=0.8, min_past_labels=120
    )
    after = sample2.loc[(sample2["ticker"] == ticker) & (sample2["time"] == time)].iloc[0]

    for column in feature_cols:
        assert after[column] == pytest.approx(before[column], rel=0, abs=1e-12)
    assert after["future_rv_5"] != pytest.approx(before["future_rv_5"], rel=0, abs=1e-12)


def test_label_threshold_excludes_overlapping_windows() -> None:
    horizon = 5
    values = pd.Series(np.linspace(0.1, 1.0, 200), dtype=float)
    labels, thresholds = build_label_threshold(
        values,
        horizon=horizon,
        quantile=0.8,
        min_past_labels=120,
    )
    for index in range(len(values)):
        if not np.isfinite(thresholds.iloc[index]):
            continue
        end = index - horizon
        past = values.iloc[:end].to_numpy(dtype=float)
        past = past[np.isfinite(past)]
        assert past.size >= 120
        expected = float(np.quantile(past, 0.8))
        assert thresholds.iloc[index] == pytest.approx(expected)
        # No window that starts at s with s + horizon >= index enters the past set.
        assert end == index - horizon


def test_time_split_fraction_and_order() -> None:
    dates = _trading_days(100)
    cut, train, test = time_split_dates(dates, train_fraction=0.8)
    d = len(dates)
    assert len(train) / d == pytest.approx(floor(0.8 * d) / d)
    assert train.max() < test.min()
    assert cut == test.min()


def test_scaler_fit_on_train_only() -> None:
    panel, market = _gate_panel(n_tickers=40, n_days=520, seed=3)
    _labeled, sample = _prepare_sample(
        panel, market, horizon=5, quantile=0.8, min_past_labels=120
    )
    train, test, _cut = _split_sample(sample, train_fraction=0.8)
    feature = "rv_21"
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train[[feature]].to_numpy(dtype=float))
    assert float(np.mean(train_scaled)) == pytest.approx(0.0, abs=1e-9)

    mean_before = float(scaler.mean_[0])
    scale_before = float(scaler.scale_[0])
    test_mut = test.copy()
    test_mut[feature] = 1_000.0
    _ = scaler.transform(test_mut[[feature]].to_numpy(dtype=float))
    assert float(scaler.mean_[0]) == pytest.approx(mean_before)
    assert float(scaler.scale_[0]) == pytest.approx(scale_before)


def test_gate_skips_training(tmp_path: Path) -> None:
    n_tickers = 10
    n_days = 300
    returns = {f"T{index:02d}": _block_returns(n_days, seed=index) for index in range(n_tickers)}
    panel = _panel_from_returns(returns)
    market = _fixed_market(_trading_days(n_days))
    figures = tmp_path / "figures"
    config = VolForecastConfig(
        output_dir=tmp_path,
        roc_path=figures / "task5_roc.png",
        importance_path=figures / "task5_importance.png",
        calibration_path=figures / "task5_calibration.png",
    )
    analysis = VolatilityForecast(config)
    analysis.prepare(panel, market)

    with patch(
        "tinvest.analytics.vol_forecast._fit_models",
        side_effect=AssertionError("fit must not run"),
    ) as mocked:
        tables = analysis.analyze()
        mocked.assert_not_called()

    assert (tmp_path / "task5_skipped.txt").is_file()
    assert (tmp_path / "task5_gate.csv").is_file()
    assert not (tmp_path / "task5_predictions.csv").is_file()
    assert tables.metrics.empty
    summary = analysis.get_summary()
    assert summary.gate_passed is False
    analysis.plot()


def test_beats_baseline_requires_margin() -> None:
    assert beats_baseline(0.55, 0.54, auc_floor=0.5, auc_margin=0.02) is False
    assert beats_baseline(0.57, 0.54, auc_floor=0.5, auc_margin=0.02) is True
    assert beats_baseline(0.49, 0.40, auc_floor=0.5, auc_margin=0.02) is False


def test_analyze_writes_metrics_when_gate_passes(tmp_path: Path) -> None:
    panel, market = _gate_panel(n_tickers=40, n_days=520, seed=7)
    # Market must cover >= 400 days independently of equal_weight filtering.
    assert len(market) >= 400
    figures = tmp_path / "figures"
    config = VolForecastConfig(
        output_dir=tmp_path,
        roc_path=figures / "task5_roc.png",
        importance_path=figures / "task5_importance.png",
        calibration_path=figures / "task5_calibration.png",
    )
    analysis = VolatilityForecast(config)
    analysis.prepare(panel, market)
    tables = analysis.analyze()
    summary = analysis.get_summary()
    assert summary.gate_passed is True
    metrics_path = tmp_path / "task5_metrics.csv"
    assert metrics_path.is_file()
    metrics = pd.read_csv(metrics_path)
    assert set(metrics["model"]) == {"constant", "rv_21", "logit", "hgb"}
    assert not tables.predictions.empty
    assert tables.predictions["time"].min() >= analysis._train["time"].max()  # type: ignore[index]
    probs = tables.predictions
    assert ((probs["p_logit"] > 0) & (probs["p_logit"] < 1)).all()
    assert ((probs["p_hgb"] > 0) & (probs["p_hgb"] < 1)).all()
    analysis.plot()
    assert (figures / "task5_roc.png").is_file()
    assert (figures / "task5_importance.png").is_file()
