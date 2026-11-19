"""Offline checks for task-2 volume features and panel regressions."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tinvest.analytics.volume import (
    VolumeConfig,
    VolumeImpact,
    add_volume_features,
)
from tinvest.research.panel import PanelFrame


def _trading_days(n: int, start: datetime | None = None) -> pd.DatetimeIndex:
    start = start or datetime(2020, 1, 2, tzinfo=timezone.utc)
    return pd.bdate_range(start=start, periods=n, tz="UTC")


def _synthetic_panel(
    tickers: list[str],
    n: int = 400,
    *,
    seed: int = 0,
    volume_scale: dict[str, float] | None = None,
    zero_volume_at: dict[str, int] | None = None,
) -> PanelFrame:
    rng = np.random.default_rng(seed)
    parts: list[pd.DataFrame] = []
    for index, ticker in enumerate(tickers):
        times = _trading_days(n)
        noise = rng.normal(0.0, 0.01, size=n)
        close = np.exp(np.log(100.0) + np.cumsum(noise))
        open_ = close * (1.0 + rng.normal(0.0, 0.0005, size=n))
        high = np.maximum(open_, close) * (1.0 + 0.001)
        low = np.minimum(open_, close) * (1.0 - 0.001)
        scale = 1.0 if volume_scale is None else float(volume_scale.get(ticker, 1.0))
        volume = np.maximum(
            1,
            np.round(1_000 * scale * (1.0 + 0.5 * np.sin(np.arange(n) / 7.0 + index))).astype(
                np.int64
            ),
        )
        if zero_volume_at is not None and ticker in zero_volume_at:
            volume[int(zero_volume_at[ticker])] = 0
        log_return = np.concatenate([[np.nan], np.diff(np.log(close))])
        parts.append(
            pd.DataFrame(
                {
                    "ticker": ticker,
                    "time": times,
                    "open": open_.astype(float),
                    "high": high.astype(float),
                    "low": low.astype(float),
                    "close": close.astype(float),
                    "volume": volume,
                    "lot": np.full(n, 10, dtype=np.int64),
                    "sector": "energy",
                    "share_type": "common",
                    "log_return": log_return.astype(float),
                }
            )
        )
    frame = pd.concat(parts, ignore_index=True)
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    return PanelFrame.build(frame)


def _abn_coef(tables, model: str = "A") -> tuple[float, float]:
    coef = pd.DataFrame(tables.coefficients)
    row = coef.loc[(coef["model"] == model) & (coef["term"] == "abn_volume")].iloc[0]
    return float(row["coef"]), float(row["p_value"])


def test_noiseless_model_a_coefficient(tmp_path: Path) -> None:
    panel = _synthetic_panel(["AAA", "BBB", "CCC"], n=250, seed=1)
    analysis = VolumeImpact(
        VolumeConfig(
            min_rows=50,
            figure_path=tmp_path / "figures" / "task2_abs_return_vs_volume.png",
            lead_vol_figure_path=tmp_path / "figures" / "task2_lead_vol.png",
            coefficients_figure_path=tmp_path / "figures" / "task2_coefficients.png",
        )
    )
    features = analysis.prepare(panel)
    working = pd.DataFrame(features).copy()
    working["abs_return"] = 0.01 * working["abn_volume"]
    working["abs_return_lag"] = 0.0
    working["log_return"] = working["abs_return"]
    working["log_return_lag"] = 0.0
    analysis._working = features.__class__.build(working)

    tables = analysis.analyze()
    coef, _ = _abn_coef(tables, "A")
    assert abs(coef - 1.0) <= 1e-6


def test_noisy_model_a_positive(tmp_path: Path) -> None:
    panel = _synthetic_panel(["AAA", "BBB", "CCC"], n=400, seed=7)
    features = add_volume_features(panel)
    feat = pd.DataFrame(features).copy()
    rng = np.random.default_rng(7)
    noise = rng.normal(0.0, 0.002, size=len(feat))
    # Positive linear link: |r| rises with abnormal turnover, plus small noise.
    planted_abs = 0.02 + 0.05 * feat["abn_volume"] + noise
    feat["abs_return"] = planted_abs
    feat["log_return"] = planted_abs
    rebuilt_parts: list[pd.DataFrame] = []
    for _, chunk in feat.groupby("ticker", sort=False):
        part = chunk.copy()
        log_return = part["log_return"].astype(float)
        part["abs_return_lag"] = part["abs_return"].shift(1)
        part["log_return_lag"] = log_return.shift(1)
        valid = log_return.notna().astype(float)
        full = valid.rolling(5, min_periods=5).sum() == 5.0
        rv = log_return.rolling(5, min_periods=5).std(ddof=1) * np.sqrt(252.0)
        part["rv_5"] = rv.where(full)
        part["future_rv_5"] = part["rv_5"].shift(-5)
        rebuilt_parts.append(part)
    planted = pd.concat(rebuilt_parts, ignore_index=True)

    analysis = VolumeImpact(
        VolumeConfig(
            min_rows=100,
            figure_path=tmp_path / "a.png",
            lead_vol_figure_path=tmp_path / "b.png",
            coefficients_figure_path=tmp_path / "c.png",
        )
    )
    analysis.prepare(panel)
    analysis._working = features.__class__.build(planted)
    tables = analysis.analyze()
    coef, p_value = _abn_coef(tables, "A")
    assert coef > 0
    assert p_value < 0.01


def test_future_rv_horizon_excludes_today() -> None:
    panel = _synthetic_panel(["AAA"], n=80, seed=3)
    features = add_volume_features(panel, horizon=5)
    frame = pd.DataFrame(features)
    assert frame["future_rv_5"].iloc[-5:].isna().all()

    returns = frame["log_return"].to_numpy(dtype=float)
    for index in range(len(frame) - 5):
        window = returns[index + 1 : index + 6]
        value = frame["future_rv_5"].iloc[index]
        if np.any(np.isnan(window)):
            assert np.isnan(value)
            continue
        expected = float(np.std(window, ddof=1) * np.sqrt(252.0))
        assert value == pytest.approx(expected, abs=1e-12)


def test_zero_volume_clears_abn_volume_only() -> None:
    panel = _synthetic_panel(
        ["AAA"],
        n=60,
        seed=4,
        zero_volume_at={"AAA": 30},
    )
    before = pd.DataFrame(panel)["log_return"].copy()
    features = add_volume_features(panel)
    frame = pd.DataFrame(features)
    assert pd.isna(frame.loc[30, "abn_volume"])
    assert pd.isna(frame.loc[30, "log_turnover"])
    pd.testing.assert_series_equal(
        frame["log_return"].reset_index(drop=True),
        before.reset_index(drop=True),
        check_names=False,
    )


def test_short_ticker_excluded_and_does_not_change_coefs(tmp_path: Path) -> None:
    long_panel = _synthetic_panel(["LONG"], n=300, seed=5)
    # Short ticker: few bars so joint rows stay well below min_rows=100.
    short_panel = _synthetic_panel(["SHORT"], n=80, seed=6)
    combined = PanelFrame.build(
        pd.concat([pd.DataFrame(long_panel), pd.DataFrame(short_panel)], ignore_index=True)
    )

    alone = VolumeImpact(VolumeConfig(min_rows=100, figure_path=tmp_path / "x.png"))
    alone.prepare(long_panel)
    alone_tables = alone.analyze()
    alone_coef, _ = _abn_coef(alone_tables, "A")

    both = VolumeImpact(
        VolumeConfig(
            min_rows=100,
            figure_path=tmp_path / "a.png",
            lead_vol_figure_path=tmp_path / "b.png",
            coefficients_figure_path=tmp_path / "c.png",
        )
    )
    both.prepare(combined)
    both_tables = both.analyze()
    both_coef, _ = _abn_coef(both_tables, "A")

    excluded = pd.DataFrame(both_tables.excluded)
    assert "SHORT" in set(excluded["ticker"])
    assert (excluded.loc[excluded["ticker"] == "SHORT", "reason"] == "too_few_volume_rows").all()
    assert alone_coef == pytest.approx(both_coef, abs=1e-10)
