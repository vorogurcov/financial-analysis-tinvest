"""Offline checks for task-4 temporal hypothesis tests."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from tinvest.analytics.hypotheses import (
    HypothesisConfig,
    ReturnHypotheses,
    apply_holm,
    half_life,
    _weekday_groups,
)
from tinvest.research.panel import PanelFrame

_MOSCOW = ZoneInfo("Europe/Moscow")


def _trading_days(n: int, start: datetime | None = None) -> pd.DatetimeIndex:
    start = start or datetime(2020, 1, 2, tzinfo=timezone.utc)
    return pd.bdate_range(start=start, periods=n, tz="UTC")


def _panel_from_returns(
    returns: dict[str, np.ndarray],
    *,
    times: pd.DatetimeIndex | None = None,
    sectors: dict[str, str] | None = None,
) -> PanelFrame:
    n = len(next(iter(returns.values())))
    index = times if times is not None else _trading_days(n)
    parts: list[pd.DataFrame] = []
    for ticker, log_return in returns.items():
        close = np.exp(np.nancumsum(np.nan_to_num(log_return, nan=0.0)) + np.log(100.0))
        parts.append(
            pd.DataFrame(
                {
                    "ticker": ticker,
                    "time": index,
                    "open": close.astype(float),
                    "high": (close * 1.001).astype(float),
                    "low": (close * 0.999).astype(float),
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


def _analysis(panel: PanelFrame, tmp_path: Path, **overrides: object) -> ReturnHypotheses:
    figures = tmp_path / "figures"
    config = HypothesisConfig(
        acf_path=figures / "task4_acf.png",
        reject_share_path=figures / "task4_reject_share.png",
        weekday_path=figures / "task4_weekday.png",
        **overrides,  # type: ignore[arg-type]
    )
    analysis = ReturnHypotheses(config)
    analysis.prepare(panel)
    return analysis


def _iid_returns(n: int = 500, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.0, 1.0, size=n)


def _clustered_returns(n: int = 500, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    e = rng.normal(0.0, 1.0, size=n)
    z = rng.normal(0.0, 1.0, size=n)
    x = np.zeros(n, dtype=float)
    for t in range(1, n):
        x[t] = 0.9 * x[t - 1] + e[t]
    return z * np.exp(0.5 * x)


def test_iid_lb_return_and_arch_not_rejected(tmp_path: Path) -> None:
    panel = _panel_from_returns({"IID": _iid_returns()})
    tables = _analysis(panel, tmp_path, min_tickers=1).analyze()
    tests = pd.DataFrame(tables.tests)
    lb = tests.loc[tests["test"] == "lb_return"].iloc[0]
    arch = tests.loc[tests["test"] == "arch_lm"].iloc[0]
    assert float(lb["p_value"]) > 0.05
    assert float(arch["p_value"]) > 0.05


def test_clustered_arch_rejected(tmp_path: Path) -> None:
    panel = _panel_from_returns({"SV": _clustered_returns()})
    tables = _analysis(panel, tmp_path, min_tickers=1).analyze()
    tests = pd.DataFrame(tables.tests)
    arch = tests.loc[tests["test"] == "arch_lm"].iloc[0]
    assert float(arch["p_value"]) < 0.01


def test_half_life_formula() -> None:
    assert half_life(0.5) == pytest.approx(1.0)
    assert np.isnan(half_life(0.0))
    assert np.isnan(half_life(1.1))


def test_holm_five_equal_pvalues() -> None:
    adjusted = apply_holm([0.01] * 5, alpha=0.05)
    assert list(adjusted) == sorted(adjusted, reverse=True)
    assert float(np.min(adjusted)) >= 0.01


def test_saturday_excluded_from_weekday_groups() -> None:
    # Friday 2020-01-03 and Saturday 2020-01-04 in Moscow (UTC+3).
    friday = datetime(2020, 1, 3, 12, 0, tzinfo=_MOSCOW).astimezone(timezone.utc)
    saturday = datetime(2020, 1, 4, 12, 0, tzinfo=_MOSCOW).astimezone(timezone.utc)
    frame = pd.DataFrame(
        {
            "time": [friday, saturday],
            "r": [0.01, 0.02],
        }
    )
    groups, n_obs = _weekday_groups(frame)
    assert n_obs == 1
    assert len(groups) == 1
    assert groups[0] == pytest.approx([0.01])
