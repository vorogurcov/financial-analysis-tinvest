"""Pairwise return correlations and their link to equal-weight market volatility."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy import stats
import statsmodels.api as sm

from tinvest.analytics.base import Analysis
from tinvest.analytics.distribution import assign_sector_groups
from tinvest.charts.spec import ChartSpec
from tinvest.common.frame import TypedFrame
from tinvest.research.panel import MarketFrame, PanelFrame

_SQRT_252 = float(np.sqrt(252.0))
_MIN_WITHIN_PAIRS = 10
_HAC_MAXLAGS = 63
_OTHER_SECTOR = "прочие"


class CorrelationPairsFrame(TypedFrame):
    """Upper-triangle pairwise Pearson and Spearman correlations."""

    ticker_a: str
    ticker_b: str
    sector_a: str
    sector_b: str
    pearson: float
    spearman: float
    n_overlap: int


class CorrelationRollingFrame(TypedFrame):
    """Rolling mean pairwise correlation and market realized volatility."""

    time: datetime
    pair_corr: float
    market_rv: float
    n_names: int


class CorrelationRegimesFrame(TypedFrame):
    """Volatility terciles and stress-day correlation summary."""

    regime: str
    n_days: int
    mean_pair_corr: float
    mean_market_rv: float
    note: str


class CorrelationSectorFrame(TypedFrame):
    """Within- and between-sector mean Pearson correlations."""

    slice: str
    mean_pearson: float
    n_pairs: int
    note: str


class CorrelationRegressionFrame(TypedFrame):
    """HAC OLS of rolling pair_corr on market_rv."""

    term: str
    coef: float
    std_err: float
    p_value: float
    n: int
    r_squared: float
    cov: str
    note: str


class CompareDailyFrame(TypedFrame):
    """Stock and market log returns on days where both are present."""

    time: datetime
    stock_return: float
    market_return: float


class CompareRollingFrame(TypedFrame):
    """Rolling Pearson correlation and OLS beta of one stock on the market."""

    time: datetime
    corr: float
    beta: float


class CompareCumulativeFrame(TypedFrame):
    """Cumulative log returns of one stock and the market on common days."""

    time: datetime
    stock_cum: float
    market_cum: float


@dataclass(frozen=True)
class CorrelationConfig:
    """Windows, floors, and figure destinations for the correlation analysis."""

    min_overlap: int = 60
    window: int = 63
    min_names_in_window: int = 20
    min_returns_in_window: int = 50
    stress_threshold: float = -0.02
    min_stress_days: int = 15
    min_regression_rows: int = 100
    heatmap_path: Path | None = None
    rolling_path: Path | None = None
    regimes_path: Path | None = None
    ticker: str | None = None
    compare_scatter_path: Path | None = None
    compare_cumulative_path: Path | None = None
    compare_rolling_path: Path | None = None

    def __post_init__(self) -> None:
        if self.min_overlap < 2:
            raise ValueError("min_overlap must be at least 2")
        if self.window < 2:
            raise ValueError("window must be at least 2")
        if self.min_names_in_window < 2:
            raise ValueError("min_names_in_window must be at least 2")
        if self.min_returns_in_window < 2:
            raise ValueError("min_returns_in_window must be at least 2")
        if self.min_stress_days < 1:
            raise ValueError("min_stress_days must be at least 1")
        if self.min_regression_rows < 1:
            raise ValueError("min_regression_rows must be at least 1")
        raw_ticker = self.ticker.strip().upper() if isinstance(self.ticker, str) else ""
        object.__setattr__(self, "ticker", raw_ticker or None)
        for name in (
            "heatmap_path",
            "rolling_path",
            "regimes_path",
            "compare_scatter_path",
            "compare_cumulative_path",
            "compare_rolling_path",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, Path(value))


@dataclass(frozen=True)
class CompareSummary:
    """One stock measured against the equal-weight market."""

    ticker: str
    n_overlap: int
    pearson: float | None
    spearman: float | None
    alpha: float | None
    beta: float | None
    beta_std_err: float | None
    beta_p_value: float | None
    r_squared: float | None
    resid_vol: float | None
    beta_low_vol: float | None
    beta_mid_vol: float | None
    beta_high_vol: float | None
    beta_stress: float | None
    note: str


@dataclass(frozen=True)
class CorrelationSummary:
    """Compact log line for the correlation analysis."""

    mean_pair_corr: float | None
    coef_b: float | None
    p_value_b: float | None
    mean_pair_corr_low_vol: float | None
    mean_pair_corr_high_vol: float | None
    n_pairs: int
    compare: CompareSummary | None = None


@dataclass(frozen=True)
class CorrelationTables:
    """All tables produced by the correlation analysis."""

    pairs: CorrelationPairsFrame
    rolling: CorrelationRollingFrame
    regimes: CorrelationRegimesFrame
    sector_corr: CorrelationSectorFrame
    regression: CorrelationRegressionFrame
    compare_daily: CompareDailyFrame
    compare_rolling: CompareRollingFrame
    compare_cumulative: CompareCumulativeFrame


class ReturnCorrelation(Analysis[CorrelationTables]):
    """Full-sample pairs, rolling windows, regimes, HAC regression, and figures."""

    def __init__(self, config: CorrelationConfig) -> None:
        super().__init__(config)
        self.config: CorrelationConfig = config
        self._panel: PanelFrame | None = None
        self._market: MarketFrame | None = None
        self._summary: CorrelationSummary | None = None
        self._wide: pd.DataFrame | None = None
        self._ticker_meta: pd.DataFrame | None = None
        self._pearson_matrix: pd.DataFrame | None = None

    def prepare(  # type: ignore[override]
        self,
        panel: PanelFrame,
        market: MarketFrame | None = None,
    ) -> PanelFrame:
        if panel.empty:
            raise ValueError("выборка пуста")
        if market is None:
            raise ValueError("MarketFrame is required")
        self._panel = panel
        self._market = market
        self._working = panel
        self._results = None
        self._summary = None
        self._wide = None
        self._ticker_meta = None
        self._pearson_matrix = None
        return panel

    def analyze(self) -> CorrelationTables:
        panel = self._panel
        market = self._market
        if panel is None or market is None:
            raise RuntimeError("Call prepare() before analyze()")

        wide, meta = _wide_returns(panel)
        self._wide = wide
        self._ticker_meta = meta

        pairs, pearson_matrix = _full_sample_pairs(
            wide,
            meta,
            min_overlap=self.config.min_overlap,
        )
        self._pearson_matrix = pearson_matrix
        sector_corr = _sector_slices(pairs, meta)

        rolling = _rolling_windows(
            wide,
            market,
            window=self.config.window,
            min_names=self.config.min_names_in_window,
            min_returns=self.config.min_returns_in_window,
        )
        regimes = _regime_table(
            rolling,
            wide,
            market,
            meta,
            min_overlap=self.config.min_overlap,
            stress_threshold=self.config.stress_threshold,
            min_stress_days=self.config.min_stress_days,
        )
        regression = _fit_hac_regression(
            rolling,
            min_rows=self.config.min_regression_rows,
        )
        compare, compare_daily, compare_rolling, compare_cumulative = _stock_compare(
            wide,
            market,
            rolling,
            ticker=self.config.ticker,
            min_overlap=self.config.min_overlap,
            window=self.config.window,
            min_returns=self.config.min_returns_in_window,
            stress_threshold=self.config.stress_threshold,
            min_stress_days=self.config.min_stress_days,
        )

        tables = CorrelationTables(
            pairs=pairs,
            rolling=rolling,
            regimes=regimes,
            sector_corr=sector_corr,
            regression=regression,
            compare_daily=compare_daily,
            compare_rolling=compare_rolling,
            compare_cumulative=compare_cumulative,
        )
        self._results = tables
        self._summary = _build_summary(pairs, regimes, regression, compare=compare)
        return tables

    def get_summary(self) -> CorrelationSummary:
        if self._summary is None:
            raise RuntimeError("Call analyze() before get_summary()")
        return self._summary

    def chart_specs(self) -> list[ChartSpec]:
        from tinvest.charts.correlation import correlation_charts

        if self._results is None:
            raise RuntimeError("Call analyze() before chart_specs()")
        compare = None if self._summary is None else self._summary.compare
        if compare is not None and compare.note == "":
            return correlation_charts(
                pearson=self._pearson_matrix,
                meta=self._ticker_meta,
                rolling=pd.DataFrame(self._results.rolling),
                regimes=pd.DataFrame(self._results.regimes),
                compare_daily=pd.DataFrame(self._results.compare_daily),
                compare_cumulative=pd.DataFrame(self._results.compare_cumulative),
                compare_rolling=pd.DataFrame(self._results.compare_rolling),
                compare_ticker=compare.ticker,
                compare_alpha=compare.alpha,
                compare_beta=compare.beta,
            )
        return correlation_charts(
            pearson=self._pearson_matrix,
            meta=self._ticker_meta,
            rolling=pd.DataFrame(self._results.rolling),
            regimes=pd.DataFrame(self._results.regimes),
        )

    def plot(self) -> Figure:
        if self._results is None:
            raise RuntimeError("Call analyze() before plot()")

        heat = _plot_heatmap(self._pearson_matrix, self._ticker_meta)
        _save_figure(heat, self.config.heatmap_path)

        rolling_fig = _plot_rolling(pd.DataFrame(self._results.rolling))
        _save_figure(rolling_fig, self.config.rolling_path)

        regimes = pd.DataFrame(self._results.regimes)
        joint = pd.DataFrame(self._results.rolling).dropna(subset=["pair_corr", "market_rv"])
        if not joint.empty:
            bars = _plot_regime_bars(regimes)
            _save_figure(bars, self.config.regimes_path)

        compare = None if self._summary is None else self._summary.compare
        if compare is not None and compare.note == "":
            daily = pd.DataFrame(self._results.compare_daily)
            cumulative = pd.DataFrame(self._results.compare_cumulative)
            compare_rolling = pd.DataFrame(self._results.compare_rolling)
            _save_figure(
                _plot_compare_scatter(daily, compare.ticker, compare.alpha, compare.beta),
                self.config.compare_scatter_path,
            )
            _save_figure(
                _plot_compare_cumulative(cumulative, compare.ticker),
                self.config.compare_cumulative_path,
            )
            _save_figure(
                _plot_compare_rolling(compare_rolling, compare.ticker),
                self.config.compare_rolling_path,
            )

        return heat


def _wide_returns(panel: PanelFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    working = pd.DataFrame(panel).copy()
    working["time"] = pd.to_datetime(working["time"], utc=True)
    meta = (
        working[["ticker", "sector"]]
        .drop_duplicates(subset=["ticker"], keep="last")
        .sort_values("ticker", kind="mergesort")
        .reset_index(drop=True)
    )
    meta["sector"] = meta["sector"].astype(str)
    meta["sector_group"] = assign_sector_groups(meta["sector"])
    wide = working.pivot_table(
        index="time",
        columns="ticker",
        values="log_return",
        aggfunc="last",
    ).sort_index()
    wide = wide.reindex(columns=meta["ticker"].tolist())
    return wide, meta


def _full_sample_pairs(
    wide: pd.DataFrame,
    meta: pd.DataFrame,
    *,
    min_overlap: int,
) -> tuple[CorrelationPairsFrame, pd.DataFrame]:
    tickers = [str(t) for t in wide.columns]
    n = len(tickers)
    sector_by = meta.set_index("ticker")["sector"].astype(str).to_dict()
    pearson = pd.DataFrame(np.nan, index=tickers, columns=tickers, dtype=float)
    for ticker in tickers:
        pearson.loc[ticker, ticker] = 1.0

    rows: list[dict[str, object]] = []
    for i in range(n):
        for j in range(i + 1, n):
            a = tickers[i]
            b = tickers[j]
            pair = wide[[a, b]].dropna()
            n_overlap = int(len(pair))
            if n_overlap < min_overlap:
                continue
            x = pair[a].to_numpy(dtype=float)
            y = pair[b].to_numpy(dtype=float)
            if np.std(x, ddof=1) == 0.0 or np.std(y, ddof=1) == 0.0:
                pearson_val = float("nan")
                spearman_val = float("nan")
            else:
                pearson_val = float(np.corrcoef(x, y)[0, 1])
                spearman_val = float(stats.spearmanr(x, y).statistic)
            pearson.loc[a, b] = pearson_val
            pearson.loc[b, a] = pearson_val
            rows.append(
                {
                    "ticker_a": a,
                    "ticker_b": b,
                    "sector_a": sector_by.get(a, ""),
                    "sector_b": sector_by.get(b, ""),
                    "pearson": pearson_val,
                    "spearman": spearman_val,
                    "n_overlap": n_overlap,
                }
            )

    if not rows:
        return CorrelationPairsFrame.blank(), pearson
    out = pd.DataFrame(rows)
    out = out.sort_values(["ticker_a", "ticker_b"], kind="mergesort").reset_index(drop=True)
    out["n_overlap"] = out["n_overlap"].astype("int64")
    for column in ("pearson", "spearman"):
        out[column] = out[column].astype("float64")
    for column in ("ticker_a", "ticker_b", "sector_a", "sector_b"):
        out[column] = out[column].astype(str)
    return CorrelationPairsFrame.build(out), pearson


def _sector_slices(
    pairs: CorrelationPairsFrame,
    meta: pd.DataFrame,
) -> CorrelationSectorFrame:
    group_by = meta.set_index("ticker")["sector_group"].astype(str).to_dict()
    frame = pd.DataFrame(pairs)
    if frame.empty:
        rows = [
            {"slice": "within", "mean_pearson": float("nan"), "n_pairs": 0, "note": "too_few_pairs"},
            {"slice": "between", "mean_pearson": float("nan"), "n_pairs": 0, "note": ""},
        ]
        return _sector_frame(rows)

    frame = frame.copy()
    frame["group_a"] = frame["ticker_a"].map(group_by)
    frame["group_b"] = frame["ticker_b"].map(group_by)

    within_mask = (
        (frame["group_a"] == frame["group_b"])
        & (frame["group_a"] != _OTHER_SECTOR)
        & (frame["group_b"] != _OTHER_SECTOR)
    )
    between_mask = (
        (frame["group_a"] != frame["group_b"])
        & (frame["group_a"] != _OTHER_SECTOR)
        & (frame["group_b"] != _OTHER_SECTOR)
    )
    within = frame.loc[within_mask]
    between = frame.loc[between_mask]

    within_n = int(len(within))
    within_note = "too_few_pairs" if within_n < _MIN_WITHIN_PAIRS else ""
    within_mean = float(within["pearson"].mean()) if within_n > 0 else float("nan")
    between_n = int(len(between))
    between_mean = float(between["pearson"].mean()) if between_n > 0 else float("nan")

    rows = [
        {
            "slice": "within",
            "mean_pearson": within_mean,
            "n_pairs": within_n,
            "note": within_note,
        },
        {
            "slice": "between",
            "mean_pearson": between_mean,
            "n_pairs": between_n,
            "note": "",
        },
    ]
    return _sector_frame(rows)


def _sector_frame(rows: list[dict[str, object]]) -> CorrelationSectorFrame:
    out = pd.DataFrame(rows)
    out["n_pairs"] = out["n_pairs"].astype("int64")
    out["mean_pearson"] = out["mean_pearson"].astype("float64")
    out["slice"] = out["slice"].astype(str)
    out["note"] = out["note"].astype(str)
    return CorrelationSectorFrame.build(out)


def _mean_pairwise_pearson(window: pd.DataFrame) -> float | None:
    if window.shape[1] < 2:
        return None
    corr = window.corr(method="pearson")
    values = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).to_numpy(dtype=float)
    flat = values[np.isfinite(values)]
    if flat.size == 0:
        return None
    return float(np.mean(flat))


def _rolling_windows(
    wide: pd.DataFrame,
    market: MarketFrame,
    *,
    window: int,
    min_names: int,
    min_returns: int,
) -> CorrelationRollingFrame:
    dates = list(wide.index)
    if len(dates) < window:
        return CorrelationRollingFrame.blank()

    market_series = (
        pd.DataFrame(market)
        .assign(time=lambda df: pd.to_datetime(df["time"], utc=True))
        .set_index("time")["market_return"]
        .astype(float)
    )

    rows: list[dict[str, object]] = []
    for end in range(window - 1, len(dates)):
        start = end - window + 1
        window_dates = dates[start : end + 1]
        chunk = wide.loc[window_dates]
        counts = chunk.notna().sum(axis=0)
        keep = counts[counts >= min_returns].index.tolist()
        n_names = int(len(keep))
        if n_names >= min_names:
            pair_corr = _mean_pairwise_pearson(chunk[keep])
            pair_value = float("nan") if pair_corr is None else pair_corr
        else:
            pair_value = float("nan")

        market_window = market_series.reindex(window_dates)
        if int(market_window.notna().sum()) == window:
            market_rv = float(market_window.std(ddof=1) * _SQRT_252)
        else:
            market_rv = float("nan")

        rows.append(
            {
                "time": window_dates[-1],
                "pair_corr": pair_value,
                "market_rv": market_rv,
                "n_names": n_names,
            }
        )

    if not rows:
        return CorrelationRollingFrame.blank()
    out = pd.DataFrame(rows)
    out["time"] = pd.to_datetime(out["time"], utc=True)
    out["n_names"] = out["n_names"].astype("int64")
    out["pair_corr"] = out["pair_corr"].astype("float64")
    out["market_rv"] = out["market_rv"].astype("float64")
    return CorrelationRollingFrame.build(out)


def _regime_table(
    rolling: CorrelationRollingFrame,
    wide: pd.DataFrame,
    market: MarketFrame,
    meta: pd.DataFrame,
    *,
    min_overlap: int,
    stress_threshold: float,
    min_stress_days: int,
) -> CorrelationRegimesFrame:
    roll = pd.DataFrame(rolling)
    joint = roll.dropna(subset=["pair_corr", "market_rv"]).copy()

    regime_rows: list[dict[str, object]] = []
    if joint.empty:
        for name in ("low_vol", "mid_vol", "high_vol"):
            regime_rows.append(
                {
                    "regime": name,
                    "n_days": 0,
                    "mean_pair_corr": float("nan"),
                    "mean_market_rv": float("nan"),
                    "note": "",
                }
            )
    else:
        q1 = float(joint["market_rv"].quantile(1.0 / 3.0))
        q2 = float(joint["market_rv"].quantile(2.0 / 3.0))
        labels = {
            "low_vol": joint.loc[joint["market_rv"] <= q1],
            "mid_vol": joint.loc[(joint["market_rv"] > q1) & (joint["market_rv"] <= q2)],
            "high_vol": joint.loc[joint["market_rv"] > q2],
        }
        for name, chunk in labels.items():
            n_days = int(len(chunk))
            regime_rows.append(
                {
                    "regime": name,
                    "n_days": n_days,
                    "mean_pair_corr": float(chunk["pair_corr"].mean()) if n_days else float("nan"),
                    "mean_market_rv": float(chunk["market_rv"].mean()) if n_days else float("nan"),
                    "note": "",
                }
            )

    market_df = pd.DataFrame(market).copy()
    market_df["time"] = pd.to_datetime(market_df["time"], utc=True)
    stress_days = market_df.loc[market_df["market_return"] <= stress_threshold, "time"]
    n_stress = int(len(stress_days))
    if n_stress < min_stress_days:
        regime_rows.append(
            {
                "regime": "stress",
                "n_days": n_stress,
                "mean_pair_corr": float("nan"),
                "mean_market_rv": float("nan"),
                "note": "too_few_stress_days",
            }
        )
    else:
        stress_wide = wide.loc[wide.index.isin(stress_days)]
        pairs, _ = _full_sample_pairs(stress_wide, meta, min_overlap=min_overlap)
        pair_df = pd.DataFrame(pairs)
        mean_corr = (
            float(pair_df["pearson"].mean()) if not pair_df.empty else float("nan")
        )
        regime_rows.append(
            {
                "regime": "stress",
                "n_days": n_stress,
                "mean_pair_corr": mean_corr,
                "mean_market_rv": float("nan"),
                "note": "",
            }
        )

    out = pd.DataFrame(regime_rows)
    out["n_days"] = out["n_days"].astype("int64")
    out["mean_pair_corr"] = out["mean_pair_corr"].astype("float64")
    out["mean_market_rv"] = out["mean_market_rv"].astype("float64")
    out["regime"] = out["regime"].astype(str)
    out["note"] = out["note"].astype(str)
    return CorrelationRegimesFrame.build(out)


def _fit_hac_regression(
    rolling: CorrelationRollingFrame,
    *,
    min_rows: int,
) -> CorrelationRegressionFrame:
    joint = pd.DataFrame(rolling).dropna(subset=["pair_corr", "market_rv"]).copy()
    n = int(len(joint))
    if n < min_rows:
        out = pd.DataFrame(
            [
                {
                    "term": "",
                    "coef": float("nan"),
                    "std_err": float("nan"),
                    "p_value": float("nan"),
                    "n": n,
                    "r_squared": float("nan"),
                    "cov": "",
                    "note": "too_few_windows",
                }
            ]
        )
        return _regression_frame(out)

    y = joint["pair_corr"].to_numpy(dtype=float)
    x = sm.add_constant(joint["market_rv"].to_numpy(dtype=float), has_constant="add")
    model = sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": _HAC_MAXLAGS})
    term_names = ["Intercept", "market_rv"]
    rows: list[dict[str, object]] = []
    for index, term in enumerate(term_names):
        rows.append(
            {
                "term": term,
                "coef": float(model.params[index]),
                "std_err": float(model.bse[index]),
                "p_value": float(model.pvalues[index]),
                "n": n,
                "r_squared": float(model.rsquared),
                "cov": "HAC_63",
                "note": "",
            }
        )
    return _regression_frame(pd.DataFrame(rows))


def _regression_frame(frame: pd.DataFrame) -> CorrelationRegressionFrame:
    out = frame.copy()
    out["n"] = out["n"].astype("int64")
    for column in ("coef", "std_err", "p_value", "r_squared"):
        out[column] = out[column].astype("float64")
    for column in ("term", "cov", "note"):
        out[column] = out[column].astype(str)
    return CorrelationRegressionFrame.build(out)


def _build_summary(
    pairs: CorrelationPairsFrame,
    regimes: CorrelationRegimesFrame,
    regression: CorrelationRegressionFrame,
    *,
    compare: CompareSummary | None = None,
) -> CorrelationSummary:
    pair_df = pd.DataFrame(pairs)
    n_pairs = int(len(pair_df))
    mean_pair = float(pair_df["pearson"].mean()) if n_pairs else None

    regime_df = pd.DataFrame(regimes)
    low = regime_df.loc[regime_df["regime"] == "low_vol"]
    high = regime_df.loc[regime_df["regime"] == "high_vol"]
    mean_low = (
        float(low["mean_pair_corr"].iloc[0])
        if not low.empty and pd.notna(low["mean_pair_corr"].iloc[0])
        else None
    )
    mean_high = (
        float(high["mean_pair_corr"].iloc[0])
        if not high.empty and pd.notna(high["mean_pair_corr"].iloc[0])
        else None
    )

    reg = pd.DataFrame(regression)
    slope = reg.loc[reg["term"] == "market_rv"]
    if slope.empty or pd.isna(slope["coef"].iloc[0]):
        coef_b = None
        p_value_b = None
    else:
        coef_b = float(slope["coef"].iloc[0])
        p_value_b = float(slope["p_value"].iloc[0])

    return CorrelationSummary(
        mean_pair_corr=mean_pair,
        coef_b=coef_b,
        p_value_b=p_value_b,
        mean_pair_corr_low_vol=mean_low,
        mean_pair_corr_high_vol=mean_high,
        n_pairs=n_pairs,
        compare=compare,
    )


def _blank_compare() -> tuple[CompareDailyFrame, CompareRollingFrame, CompareCumulativeFrame]:
    return (
        CompareDailyFrame.blank(),
        CompareRollingFrame.blank(),
        CompareCumulativeFrame.blank(),
    )


def _empty_compare(ticker: str, *, n_overlap: int, note: str) -> CompareSummary:
    return CompareSummary(
        ticker=ticker,
        n_overlap=n_overlap,
        pearson=None,
        spearman=None,
        alpha=None,
        beta=None,
        beta_std_err=None,
        beta_p_value=None,
        r_squared=None,
        resid_vol=None,
        beta_low_vol=None,
        beta_mid_vol=None,
        beta_high_vol=None,
        beta_stress=None,
        note=note,
    )


def _finite(value: float) -> float | None:
    if not np.isfinite(value):
        return None
    return float(value)


def _market_return_series(market: MarketFrame) -> pd.Series:
    frame = pd.DataFrame(market).copy()
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    return frame.set_index("time")["market_return"].astype(float)


def _hac_maxlags(n_obs: int) -> int:
    return min(_HAC_MAXLAGS, max(n_obs - 2, 1))


def _hac_stock_on_market(stock: np.ndarray, market: np.ndarray) -> dict[str, float | None]:
    empty: dict[str, float | None] = {
        "alpha": None,
        "beta": None,
        "beta_std_err": None,
        "beta_p_value": None,
        "r_squared": None,
        "resid_vol": None,
    }
    n_obs = int(len(stock))
    if n_obs < 3:
        return empty
    if float(np.std(stock, ddof=1)) == 0.0 or float(np.std(market, ddof=1)) == 0.0:
        return empty
    y = np.asarray(stock, dtype=float)
    x = sm.add_constant(np.asarray(market, dtype=float), has_constant="add")
    model = sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": _hac_maxlags(n_obs)})
    resid = np.asarray(model.resid, dtype=float)
    return {
        "alpha": float(model.params[0]),
        "beta": float(model.params[1]),
        "beta_std_err": float(model.bse[1]),
        "beta_p_value": float(model.pvalues[1]),
        "r_squared": float(model.rsquared),
        "resid_vol": float(np.std(resid, ddof=1) * _SQRT_252),
    }


def _corr_pair(stock: np.ndarray, market: np.ndarray) -> tuple[float | None, float | None]:
    if len(stock) < 3:
        return None, None
    if float(np.std(stock, ddof=1)) == 0.0 or float(np.std(market, ddof=1)) == 0.0:
        return None, None
    pearson = float(np.corrcoef(stock, market)[0, 1])
    spearman = float(stats.spearmanr(stock, market).statistic)
    return _finite(pearson), _finite(spearman)


def _rolling_stock_beta(
    stock: pd.Series,
    market: pd.Series,
    *,
    window: int,
    min_returns: int,
) -> CompareRollingFrame:
    dates = list(stock.index)
    if len(dates) < window:
        return CompareRollingFrame.blank()

    stock_values = stock.to_numpy(dtype=float)
    market_values = market.reindex(stock.index).to_numpy(dtype=float)
    rows: list[dict[str, object]] = []
    for end in range(window - 1, len(dates)):
        start = end - window + 1
        stock_window = stock_values[start : end + 1]
        market_window = market_values[start : end + 1]
        mask = np.isfinite(stock_window) & np.isfinite(market_window)
        if int(mask.sum()) < min_returns:
            corr = float("nan")
            beta = float("nan")
        else:
            stock_obs = stock_window[mask]
            market_obs = market_window[mask]
            if float(np.std(stock_obs, ddof=1)) == 0.0 or float(np.std(market_obs, ddof=1)) == 0.0:
                corr = float("nan")
                beta = float("nan")
            else:
                corr = float(np.corrcoef(stock_obs, market_obs)[0, 1])
                covariance = float(np.cov(stock_obs, market_obs, ddof=1)[0, 1])
                variance = float(np.var(market_obs, ddof=1))
                beta = covariance / variance if variance != 0.0 else float("nan")
        rows.append({"time": dates[end], "corr": corr, "beta": beta})

    out = pd.DataFrame(rows)
    out["time"] = pd.to_datetime(out["time"], utc=True)
    out["corr"] = out["corr"].astype("float64")
    out["beta"] = out["beta"].astype("float64")
    return CompareRollingFrame.build(out)


def _regime_betas(
    compare_rolling: CompareRollingFrame,
    market_rolling: CorrelationRollingFrame,
) -> dict[str, float | None]:
    result = {"beta_low_vol": None, "beta_mid_vol": None, "beta_high_vol": None}
    market_df = pd.DataFrame(market_rolling)
    beta_df = pd.DataFrame(compare_rolling)
    if market_df.empty or beta_df.empty:
        return result
    joint = market_df.dropna(subset=["pair_corr", "market_rv"]).copy()
    if joint.empty:
        return result
    q1 = float(joint["market_rv"].quantile(1.0 / 3.0))
    q2 = float(joint["market_rv"].quantile(2.0 / 3.0))
    merged = joint.merge(beta_df[["time", "beta"]], on="time", how="left")
    masks = {
        "beta_low_vol": merged["market_rv"] <= q1,
        "beta_mid_vol": (merged["market_rv"] > q1) & (merged["market_rv"] <= q2),
        "beta_high_vol": merged["market_rv"] > q2,
    }
    for key, mask in masks.items():
        values = merged.loc[mask, "beta"].dropna()
        result[key] = float(values.mean()) if len(values) else None
    return result


def _stress_beta(
    aligned: pd.DataFrame,
    market: MarketFrame,
    *,
    stress_threshold: float,
    min_stress_days: int,
) -> float | None:
    market_df = pd.DataFrame(market).copy()
    market_df["time"] = pd.to_datetime(market_df["time"], utc=True)
    stress_times = pd.to_datetime(
        market_df.loc[market_df["market_return"] <= stress_threshold, "time"],
        utc=True,
    )
    if int(len(stress_times)) < min_stress_days:
        return None
    aligned_time = pd.to_datetime(aligned["time"], utc=True)
    chunk = aligned.loc[aligned_time.isin(stress_times)]
    fit = _hac_stock_on_market(
        chunk["stock_return"].to_numpy(dtype=float),
        chunk["market_return"].to_numpy(dtype=float),
    )
    return fit["beta"]


def _stock_compare(
    wide: pd.DataFrame,
    market: MarketFrame,
    market_rolling: CorrelationRollingFrame,
    *,
    ticker: str | None,
    min_overlap: int,
    window: int,
    min_returns: int,
    stress_threshold: float,
    min_stress_days: int,
) -> tuple[CompareSummary | None, CompareDailyFrame, CompareRollingFrame, CompareCumulativeFrame]:
    blank_daily, blank_rolling, blank_cumulative = _blank_compare()
    if ticker is None:
        return None, blank_daily, blank_rolling, blank_cumulative

    columns = [str(name) for name in wide.columns]
    if ticker not in columns:
        summary = _empty_compare(ticker, n_overlap=0, note="ticker_not_in_panel")
        return summary, blank_daily, blank_rolling, blank_cumulative

    stock = wide[ticker].astype(float)
    stock.index = pd.to_datetime(stock.index, utc=True)
    market_series = _market_return_series(market)
    aligned = pd.concat(
        [stock.rename("stock_return"), market_series.rename("market_return")],
        axis=1,
        join="inner",
    ).dropna()
    aligned.index.name = "time"
    n_overlap = int(len(aligned))
    if n_overlap < min_overlap:
        summary = _empty_compare(ticker, n_overlap=n_overlap, note="too_few_overlap")
        return summary, blank_daily, blank_rolling, blank_cumulative

    daily_out = aligned.reset_index()
    daily_out["time"] = pd.to_datetime(daily_out["time"], utc=True)
    daily_out["stock_return"] = daily_out["stock_return"].astype("float64")
    daily_out["market_return"] = daily_out["market_return"].astype("float64")
    daily = CompareDailyFrame.build(daily_out[["time", "stock_return", "market_return"]])

    cumulative_out = daily_out.copy()
    cumulative_out["stock_cum"] = cumulative_out["stock_return"].cumsum().astype("float64")
    cumulative_out["market_cum"] = cumulative_out["market_return"].cumsum().astype("float64")
    cumulative = CompareCumulativeFrame.build(
        cumulative_out[["time", "stock_cum", "market_cum"]]
    )

    stock_values = daily_out["stock_return"].to_numpy(dtype=float)
    market_values = daily_out["market_return"].to_numpy(dtype=float)
    pearson, spearman = _corr_pair(stock_values, market_values)
    fit = _hac_stock_on_market(stock_values, market_values)
    compare_rolling = _rolling_stock_beta(
        stock,
        market_series,
        window=window,
        min_returns=min_returns,
    )
    regime_betas = _regime_betas(compare_rolling, market_rolling)
    stress_beta = _stress_beta(
        daily_out,
        market,
        stress_threshold=stress_threshold,
        min_stress_days=min_stress_days,
    )
    summary = CompareSummary(
        ticker=ticker,
        n_overlap=n_overlap,
        pearson=pearson,
        spearman=spearman,
        alpha=fit["alpha"],
        beta=fit["beta"],
        beta_std_err=fit["beta_std_err"],
        beta_p_value=fit["beta_p_value"],
        r_squared=fit["r_squared"],
        resid_vol=fit["resid_vol"],
        beta_low_vol=regime_betas["beta_low_vol"],
        beta_mid_vol=regime_betas["beta_mid_vol"],
        beta_high_vol=regime_betas["beta_high_vol"],
        beta_stress=stress_beta,
        note="",
    )
    return summary, daily, compare_rolling, cumulative


def _save_figure(fig: Figure, path: Path | None) -> None:
    if path is None:
        return
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_heatmap(
    pearson: pd.DataFrame | None,
    meta: pd.DataFrame | None,
) -> Figure:
    fig, ax = plt.subplots(figsize=(10, 8))
    if pearson is None or meta is None or pearson.empty:
        ax.set_axis_off()
        return fig

    order_meta = meta.sort_values(["sector", "ticker"], kind="mergesort")
    order = order_meta["ticker"].astype(str).tolist()
    matrix = pearson.reindex(index=order, columns=order)
    labels = [f"{row.sector} | {row.ticker}" for row in order_meta.itertuples(index=False)]

    image = ax.imshow(matrix.to_numpy(dtype=float), cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=6, rotation=90)
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_title("Попарная корреляция Пирсона лог-доходностей")
    fig.tight_layout()
    return fig


def _plot_rolling(rolling: pd.DataFrame) -> Figure:
    fig, ax1 = plt.subplots(figsize=(12, 5))
    if rolling.empty:
        ax1.set_axis_off()
        return fig

    times = pd.to_datetime(rolling["time"], utc=True)
    ax1.plot(times, rolling["pair_corr"], color="steelblue", label="Средняя попарная корреляция")
    ax1.set_xlabel("Дата")
    ax1.set_ylabel("Средняя попарная корреляция", color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")

    ax2 = ax1.twinx()
    ax2.plot(times, rolling["market_rv"], color="darkorange", label="Рыночная волатильность")
    ax2.set_ylabel("Рыночная RV (годовая)", color="darkorange")
    ax2.tick_params(axis="y", labelcolor="darkorange")

    ax1.set_title("Скользящая корреляция и волатильность рынка")
    ax1.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def _plot_regime_bars(regimes: pd.DataFrame) -> Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    order = ["low_vol", "mid_vol", "high_vol"]
    subset = regimes.loc[regimes["regime"].isin(order)].copy()
    subset["regime"] = pd.Categorical(subset["regime"], categories=order, ordered=True)
    subset = subset.sort_values("regime")
    labels = ["Низкая RV", "Средняя RV", "Высокая RV"]
    values = subset["mean_pair_corr"].to_numpy(dtype=float)
    ax.bar(labels, values, color=["#4C78A8", "#F58518", "#E45756"])
    ax.set_title("Средняя попарная корреляция по терцилям рыночной RV")
    ax.set_ylabel("Средняя pair_corr")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def _plot_compare_scatter(
    daily: pd.DataFrame,
    ticker: str,
    alpha: float | None,
    beta: float | None,
) -> Figure:
    fig, ax = plt.subplots(figsize=(8, 6))
    if daily.empty:
        ax.set_axis_off()
        return fig
    market = daily["market_return"].to_numpy(dtype=float)
    stock = daily["stock_return"].to_numpy(dtype=float)
    ax.scatter(market, stock, s=12, alpha=0.7, color="steelblue", label=ticker)
    if alpha is not None and beta is not None and len(market) > 0:
        x0 = float(np.min(market))
        x1 = float(np.max(market))
        ax.plot(
            [x0, x1],
            [alpha + beta * x0, alpha + beta * x1],
            color="black",
            linewidth=1.5,
            label="Регрессия",
        )
    ax.set_xlabel("Доходность рынка")
    ax.set_ylabel(f"Доходность {ticker}")
    ax.set_title(f"Дневные доходности: {ticker} и рынок")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def _plot_compare_cumulative(cumulative: pd.DataFrame, ticker: str) -> Figure:
    fig, ax = plt.subplots(figsize=(12, 5))
    if cumulative.empty:
        ax.set_axis_off()
        return fig
    times = pd.to_datetime(cumulative["time"], utc=True)
    ax.plot(times, cumulative["stock_cum"], color="steelblue", label=ticker)
    ax.plot(times, cumulative["market_cum"], color="darkorange", label="Рынок")
    ax.set_xlabel("Дата")
    ax.set_ylabel("Накопленная лог-доходность")
    ax.set_title(f"Накопленная лог-доходность: {ticker} и рынок")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def _plot_compare_rolling(rolling: pd.DataFrame, ticker: str) -> Figure:
    fig, ax1 = plt.subplots(figsize=(12, 5))
    if rolling.empty:
        ax1.set_axis_off()
        return fig
    times = pd.to_datetime(rolling["time"], utc=True)
    ax1.plot(times, rolling["corr"], color="steelblue", label="Корреляция")
    ax1.set_xlabel("Дата")
    ax1.set_ylabel("Корреляция", color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")

    ax2 = ax1.twinx()
    ax2.plot(times, rolling["beta"], color="darkorange", label="Бета")
    ax2.set_ylabel("Бета", color="darkorange")
    ax2.tick_params(axis="y", labelcolor="darkorange")

    ax1.set_title(f"Скользящие корреляция и бета: {ticker}")
    ax1.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig
