"""Temporal hypothesis tests on daily log returns: ACF, ARCH, weekday, RV persistence."""

from __future__ import annotations

from dataclasses import dataclass
from math import log
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib.figure import Figure
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.multitest import multipletests

from tinvest.analytics.base import Analysis
from tinvest.charts.spec import ChartSpec
from tinvest.common.frame import TypedFrame
from tinvest.research.panel import PanelFrame

_SQRT_252 = float(np.sqrt(252.0))
_MOSCOW = ZoneInfo("Europe/Moscow")
_TEST_ORDER = ("lb_return", "lb_square", "arch_lm", "weekday", "rv_persist")
_WEEKDAY_LABELS = ("Пн", "Вт", "Ср", "Чт", "Пт")


class HypothesisTestsFrame(TypedFrame):
    """Per-ticker hypothesis test results."""

    ticker: str
    test: str
    n: int
    statistic: float
    p_value: float
    p_holm: float
    phi: float
    half_life: float
    note: str


class HypothesisSummaryFrame(TypedFrame):
    """One summary row per hypothesis code."""

    test: str
    n: int
    median_p: float
    reject_share: float
    median_phi: float
    median_half_life: float
    persistent_share: float
    note: str


@dataclass(frozen=True)
class HypothesisConfig:
    """Lags, floors, and figure destinations for temporal hypothesis tests."""

    ljung_lags: int = 10
    arch_lags: int = 5
    rv_window: int = 21
    hac_lags: int = 21
    min_returns: int = 200
    min_tickers: int = 15
    alpha: float = 0.05
    acf_path: Path | None = None
    reject_share_path: Path | None = None
    weekday_path: Path | None = None

    def __post_init__(self) -> None:
        if self.ljung_lags < 1:
            raise ValueError("ljung_lags must be at least 1")
        if self.arch_lags < 1:
            raise ValueError("arch_lags must be at least 1")
        if self.rv_window < 2:
            raise ValueError("rv_window must be at least 2")
        if self.hac_lags < 1:
            raise ValueError("hac_lags must be at least 1")
        if self.min_returns < 1:
            raise ValueError("min_returns must be at least 1")
        if self.min_tickers < 1:
            raise ValueError("min_tickers must be at least 1")
        if not (0.0 < self.alpha < 1.0):
            raise ValueError("alpha must be in (0, 1)")
        for name in ("acf_path", "reject_share_path", "weekday_path"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, Path(value))


@dataclass(frozen=True)
class HypothesisSummary:
    """Compact log line for one hypothesis code."""

    test: str
    n: int
    median_p: float | None
    reject_share: float | None
    median_phi: float | None = None
    median_half_life: float | None = None
    persistent_share: float | None = None
    note: str = ""


@dataclass(frozen=True)
class HypothesisTables:
    """All tables produced by the hypothesis analysis."""

    tests: HypothesisTestsFrame
    summary: HypothesisSummaryFrame


def half_life(phi: float) -> float:
    """AR(1) half-life; empty when phi is outside (0, 1)."""
    if phi <= 0.0 or phi >= 1.0:
        return float("nan")
    return float(log(0.5) / log(phi))


def apply_holm(p_values: list[float] | np.ndarray, *, alpha: float = 0.05) -> np.ndarray:
    """Holm step-down adjusted p-values for a family of tests."""
    values = np.asarray(p_values, dtype=float)
    if values.size == 0:
        return values
    _reject, adjusted, _a, _b = multipletests(values, alpha=alpha, method="holm")
    return np.asarray(adjusted, dtype=float)


class ReturnHypotheses(Analysis[HypothesisTables]):
    """Five pre-registered tests on each ticker, with Holm correction by family."""

    def __init__(self, config: HypothesisConfig) -> None:
        super().__init__(config)
        self.config: HypothesisConfig = config
        self._panel: PanelFrame | None = None
        self._series: dict[str, pd.DataFrame] | None = None
        self._summaries: list[HypothesisSummary] | None = None

    def prepare(self, panel: PanelFrame) -> PanelFrame:  # type: ignore[override]
        if panel.empty:
            raise ValueError("выборка пуста")
        self._panel = panel
        self._working = panel
        self._results = None
        self._summaries = None
        self._series = _prepare_series(
            panel,
            rv_window=self.config.rv_window,
        )
        return panel

    def analyze(self) -> HypothesisTables:
        if self._series is None:
            raise RuntimeError("Call prepare() before analyze()")

        rows = _compute_tests(
            self._series,
            ljung_lags=self.config.ljung_lags,
            arch_lags=self.config.arch_lags,
            hac_lags=self.config.hac_lags,
            min_returns=self.config.min_returns,
        )
        tests_df = _apply_holm_families(
            rows,
            min_tickers=self.config.min_tickers,
            alpha=self.config.alpha,
        )
        tests = _tests_frame(tests_df)
        summaries = _build_summaries(
            tests_df,
            alpha=self.config.alpha,
            min_tickers=self.config.min_tickers,
        )
        summary = _summary_frame(summaries)

        tables = HypothesisTables(tests=tests, summary=summary)
        self._results = tables
        self._summaries = summaries
        return tables

    def get_summary(self) -> list[HypothesisSummary]:
        if self._summaries is None:
            raise RuntimeError("Call analyze() before get_summary()")
        return list(self._summaries)

    def chart_specs(self) -> list[ChartSpec]:
        from tinvest.charts.hypotheses import hypotheses_charts

        if self._results is None or self._series is None:
            raise RuntimeError("Call analyze() before chart_specs()")
        return hypotheses_charts(
            self._series,
            pd.DataFrame(self._results.tests),
            ljung_lags=self.config.ljung_lags,
            min_returns=self.config.min_returns,
            min_tickers=self.config.min_tickers,
            alpha=self.config.alpha,
        )

    def plot(self) -> Figure:
        if self._results is None or self._series is None:
            raise RuntimeError("Call analyze() before plot()")

        acf_fig = _plot_acf(
            self._series,
            lags=self.config.ljung_lags,
            min_returns=self.config.min_returns,
        )
        _save_figure(acf_fig, self.config.acf_path)

        reject_fig = _plot_reject_share(
            pd.DataFrame(self._results.tests),
            alpha=self.config.alpha,
            min_tickers=self.config.min_tickers,
        )
        _save_figure(reject_fig, self.config.reject_share_path)

        weekday_fig = _plot_weekday(self._series, min_returns=self.config.min_returns)
        _save_figure(weekday_fig, self.config.weekday_path)

        return acf_fig


def _prepare_series(panel: PanelFrame, *, rv_window: int) -> dict[str, pd.DataFrame]:
    working = pd.DataFrame(panel).copy()
    working["time"] = pd.to_datetime(working["time"], utc=True)
    series: dict[str, pd.DataFrame] = {}
    for ticker, chunk in working.groupby("ticker", sort=True):
        ordered = chunk.sort_values("time", kind="mergesort")
        mask = ordered["log_return"].notna()
        subset = ordered.loc[mask, ["time", "log_return"]].copy()
        r = subset["log_return"].astype(float)
        rv = r.rolling(window=rv_window, min_periods=rv_window).std(ddof=1) * _SQRT_252
        out = pd.DataFrame(
            {
                "time": subset["time"].to_numpy(),
                "r": r.to_numpy(dtype=float),
                "rv_21": rv.to_numpy(dtype=float),
            }
        )
        series[str(ticker)] = out.reset_index(drop=True)
    return series


def _empty_test_row(ticker: str, test: str, n: int, note: str) -> dict[str, object]:
    return {
        "ticker": ticker,
        "test": test,
        "n": n,
        "statistic": float("nan"),
        "p_value": float("nan"),
        "p_holm": float("nan"),
        "phi": float("nan"),
        "half_life": float("nan"),
        "note": note,
    }


def _ljung_box(series: np.ndarray, *, lags: int) -> tuple[float, float]:
    result = acorr_ljungbox(series, lags=lags, return_df=True)
    return float(result["lb_stat"].iloc[-1]), float(result["lb_pvalue"].iloc[-1])


def _arch_lm(series: np.ndarray, *, nlags: int) -> tuple[float, float]:
    result = het_arch(series, nlags=nlags, result_object=True)
    return float(result.lm), float(result.lmpval)


def _weekday_groups(frame: pd.DataFrame) -> tuple[list[np.ndarray], int]:
    times = pd.to_datetime(frame["time"], utc=True)
    moscow = times.dt.tz_convert(_MOSCOW)
    weekday = moscow.dt.day_of_week
    abs_r = frame["r"].abs().to_numpy(dtype=float)
    groups: list[np.ndarray] = []
    n_obs = 0
    for day in range(5):
        values = abs_r[weekday.to_numpy() == day]
        if values.size > 0:
            groups.append(values)
            n_obs += int(values.size)
    return groups, n_obs


def _rv_persist(
    frame: pd.DataFrame,
    *,
    hac_lags: int,
) -> tuple[float, float, float, float, int]:
    rv = frame["rv_21"].astype(float)
    lag = rv.shift(1)
    joint = pd.DataFrame({"y": rv, "x": lag}).dropna()
    n = int(len(joint))
    if n < 3:
        return float("nan"), float("nan"), float("nan"), float("nan"), n

    y = joint["y"].to_numpy(dtype=float)
    x = sm.add_constant(joint["x"].to_numpy(dtype=float), has_constant="add")
    model = sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    phi = float(model.params[1])
    t_stat = float(model.tvalues[1])
    p_value = float(model.pvalues[1])
    return t_stat, p_value, phi, half_life(phi), n


def _compute_tests(
    series: dict[str, pd.DataFrame],
    *,
    ljung_lags: int,
    arch_lags: int,
    hac_lags: int,
    min_returns: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for ticker in sorted(series):
        frame = series[ticker]
        r = frame["r"].to_numpy(dtype=float)
        n = int(r.size)

        if n < min_returns:
            for test in _TEST_ORDER:
                rows.append(_empty_test_row(ticker, test, n, "too_few_returns"))
            continue

        if float(np.std(r, ddof=1)) == 0.0:
            for test in _TEST_ORDER:
                rows.append(_empty_test_row(ticker, test, n, "zero_variance"))
            continue

        lb_stat, lb_p = _ljung_box(r, lags=ljung_lags)
        rows.append(
            {
                "ticker": ticker,
                "test": "lb_return",
                "n": n,
                "statistic": lb_stat,
                "p_value": lb_p,
                "p_holm": float("nan"),
                "phi": float("nan"),
                "half_life": float("nan"),
                "note": "",
            }
        )

        sq_stat, sq_p = _ljung_box(r * r, lags=ljung_lags)
        rows.append(
            {
                "ticker": ticker,
                "test": "lb_square",
                "n": n,
                "statistic": sq_stat,
                "p_value": sq_p,
                "p_holm": float("nan"),
                "phi": float("nan"),
                "half_life": float("nan"),
                "note": "",
            }
        )

        arch_stat, arch_p = _arch_lm(r, nlags=arch_lags)
        rows.append(
            {
                "ticker": ticker,
                "test": "arch_lm",
                "n": n,
                "statistic": arch_stat,
                "p_value": arch_p,
                "p_holm": float("nan"),
                "phi": float("nan"),
                "half_life": float("nan"),
                "note": "",
            }
        )

        groups, n_weekday = _weekday_groups(frame)
        if len(groups) < 2:
            rows.append(_empty_test_row(ticker, "weekday", n_weekday, "too_few_weekdays"))
        else:
            kw_stat, kw_p = stats.kruskal(*groups)
            rows.append(
                {
                    "ticker": ticker,
                    "test": "weekday",
                    "n": n_weekday,
                    "statistic": float(kw_stat),
                    "p_value": float(kw_p),
                    "p_holm": float("nan"),
                    "phi": float("nan"),
                    "half_life": float("nan"),
                    "note": "",
                }
            )

        t_stat, p_value, phi, hl, n_rv = _rv_persist(frame, hac_lags=hac_lags)
        if np.isnan(p_value):
            rows.append(_empty_test_row(ticker, "rv_persist", n_rv, "too_few_returns"))
        else:
            rows.append(
                {
                    "ticker": ticker,
                    "test": "rv_persist",
                    "n": n_rv,
                    "statistic": t_stat,
                    "p_value": p_value,
                    "p_holm": float("nan"),
                    "phi": phi,
                    "half_life": hl,
                    "note": "",
                }
            )
    return rows


def _apply_holm_families(
    rows: list[dict[str, object]],
    *,
    min_tickers: int,
    alpha: float,
) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame

    frame["p_holm"] = float("nan")
    for test in _TEST_ORDER:
        mask = (frame["test"] == test) & frame["p_value"].notna()
        indices = frame.index[mask]
        if len(indices) < min_tickers:
            continue
        adjusted = apply_holm(
            frame.loc[indices, "p_value"].to_numpy(dtype=float),
            alpha=alpha,
        )
        frame.loc[indices, "p_holm"] = adjusted
    return frame


def _build_summaries(
    tests: pd.DataFrame,
    *,
    alpha: float,
    min_tickers: int,
) -> list[HypothesisSummary]:
    summaries: list[HypothesisSummary] = []
    for test in _TEST_ORDER:
        subset = tests.loc[tests["test"] == test].copy()
        computed = subset.loc[subset["p_value"].notna()]
        n = int(len(computed))
        if n == 0:
            summaries.append(
                HypothesisSummary(
                    test=test,
                    n=0,
                    median_p=None,
                    reject_share=None,
                    note="too_few_tickers",
                )
            )
            continue

        median_p = float(computed["p_value"].median())
        holm_ok = n >= min_tickers and computed["p_holm"].notna().all()
        if holm_ok:
            reject_share = float((computed["p_holm"] < alpha).mean())
            note = ""
        else:
            reject_share = None
            note = "too_few_tickers"

        median_phi = None
        median_half_life = None
        persistent_share = None
        if test == "rv_persist":
            median_phi = float(computed["phi"].median())
            hl = computed.loc[
                (computed["phi"] > 0.0) & (computed["phi"] < 1.0),
                "half_life",
            ]
            median_half_life = float(hl.median()) if not hl.empty else None
            if holm_ok:
                persistent_share = float(
                    ((computed["p_holm"] < alpha) & (computed["phi"] > 0.0)).mean()
                )

        summaries.append(
            HypothesisSummary(
                test=test,
                n=n,
                median_p=median_p,
                reject_share=reject_share,
                median_phi=median_phi,
                median_half_life=median_half_life,
                persistent_share=persistent_share,
                note=note,
            )
        )
    return summaries


def _tests_frame(frame: pd.DataFrame) -> HypothesisTestsFrame:
    if frame.empty:
        return HypothesisTestsFrame.blank()
    out = frame.copy()
    out["n"] = out["n"].astype("int64")
    for column in ("statistic", "p_value", "p_holm", "phi", "half_life"):
        out[column] = out[column].astype("float64")
    for column in ("ticker", "test", "note"):
        out[column] = out[column].astype(str)
    order = {name: index for index, name in enumerate(_TEST_ORDER)}
    out["_order"] = out["test"].map(order)
    out = out.sort_values(["ticker", "_order"], kind="mergesort").drop(columns="_order")
    return HypothesisTestsFrame.build(out.reset_index(drop=True))


def _summary_frame(summaries: list[HypothesisSummary]) -> HypothesisSummaryFrame:
    rows: list[dict[str, object]] = []
    for item in summaries:
        rows.append(
            {
                "test": item.test,
                "n": item.n,
                "median_p": item.median_p if item.median_p is not None else float("nan"),
                "reject_share": (
                    item.reject_share if item.reject_share is not None else float("nan")
                ),
                "median_phi": (
                    item.median_phi if item.median_phi is not None else float("nan")
                ),
                "median_half_life": (
                    item.median_half_life
                    if item.median_half_life is not None
                    else float("nan")
                ),
                "persistent_share": (
                    item.persistent_share
                    if item.persistent_share is not None
                    else float("nan")
                ),
                "note": item.note,
            }
        )
    out = pd.DataFrame(rows)
    out["n"] = out["n"].astype("int64")
    for column in (
        "median_p",
        "reject_share",
        "median_phi",
        "median_half_life",
        "persistent_share",
    ):
        out[column] = out[column].astype("float64")
    for column in ("test", "note"):
        out[column] = out[column].astype(str)
    return HypothesisSummaryFrame.build(out)


def _acf_at_lags(series: np.ndarray, lags: int) -> np.ndarray:
    n = series.size
    centered = series - series.mean()
    denom = float(np.dot(centered, centered))
    if denom == 0.0 or n <= lags:
        return np.full(lags, np.nan, dtype=float)
    values = np.empty(lags, dtype=float)
    for lag in range(1, lags + 1):
        values[lag - 1] = float(np.dot(centered[lag:], centered[:-lag]) / denom)
    return values


def _plot_acf(
    series: dict[str, pd.DataFrame],
    *,
    lags: int,
    min_returns: int,
) -> Figure:
    fig, ax = plt.subplots(figsize=(10, 5))
    r_rows: list[np.ndarray] = []
    sq_rows: list[np.ndarray] = []
    for frame in series.values():
        r = frame["r"].to_numpy(dtype=float)
        if r.size < min_returns:
            continue
        if float(np.std(r, ddof=1)) == 0.0:
            continue
        r_rows.append(_acf_at_lags(r, lags))
        sq_rows.append(_acf_at_lags(r * r, lags))

    if not r_rows:
        ax.set_axis_off()
        return fig

    lag_axis = np.arange(1, lags + 1)
    for values, color, label in (
        (np.vstack(r_rows), "steelblue", "Доходности"),
        (np.vstack(sq_rows), "darkorange", "Квадраты доходностей"),
    ):
        median = np.nanmedian(values, axis=0)
        q25 = np.nanpercentile(values, 25, axis=0)
        q75 = np.nanpercentile(values, 75, axis=0)
        ax.plot(lag_axis, median, color=color, marker="o", label=label)
        ax.fill_between(lag_axis, q25, q75, color=color, alpha=0.2, label="25–75% тикеров")

    handles, labels = ax.get_legend_handles_labels()
    seen: set[str] = set()
    unique = []
    for handle, label in zip(handles, labels, strict=True):
        if label in seen:
            continue
        seen.add(label)
        unique.append((handle, label))
    ax.legend(
        [item[0] for item in unique],
        [item[1] for item in unique],
        frameon=False,
    )
    ax.set_xlabel("Лаг")
    ax.set_ylabel("АКФ")
    ax.set_title("Медианная АКФ доходностей и квадратов")
    ax.set_xticks(lag_axis)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def _plot_reject_share(
    tests: pd.DataFrame,
    *,
    alpha: float,
    min_tickers: int,
) -> Figure:
    fig, ax = plt.subplots(figsize=(10, 5))
    raw_shares: list[float] = []
    holm_shares: list[float] = []
    for test in _TEST_ORDER:
        subset = tests.loc[(tests["test"] == test) & tests["p_value"].notna()]
        if subset.empty:
            raw_shares.append(float("nan"))
            holm_shares.append(float("nan"))
            continue
        raw_shares.append(float((subset["p_value"] < alpha).mean()))
        if len(subset) >= min_tickers and subset["p_holm"].notna().all():
            holm_shares.append(float((subset["p_holm"] < alpha).mean()))
        else:
            holm_shares.append(float("nan"))

    x = np.arange(len(_TEST_ORDER))
    width = 0.35
    ax.bar(x - width / 2, raw_shares, width, label="Сырая доля p < 0,05", color="#4C78A8")
    ax.bar(x + width / 2, holm_shares, width, label="Доля после Холма", color="#F58518")
    ax.set_xticks(x)
    ax.set_xticklabels(list(_TEST_ORDER))
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Доля отклонений")
    ax.set_title("Доли отклонений гипотез до и после Холма")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def _plot_weekday(
    series: dict[str, pd.DataFrame],
    *,
    min_returns: int,
) -> Figure:
    fig, ax = plt.subplots(figsize=(10, 5))
    buckets: dict[int, list[float]] = {day: [] for day in range(5)}
    for frame in series.values():
        r = frame["r"].to_numpy(dtype=float)
        if r.size < min_returns:
            continue
        mean_abs = float(np.mean(np.abs(r)))
        if mean_abs == 0.0:
            continue
        times = pd.to_datetime(frame["time"], utc=True)
        weekday = times.dt.tz_convert(_MOSCOW).dt.day_of_week.to_numpy()
        scaled = np.abs(r) / mean_abs
        for day in range(5):
            values = scaled[weekday == day]
            if values.size:
                buckets[day].extend(values.tolist())

    data = [buckets[day] for day in range(5)]
    if not any(data):
        ax.set_axis_off()
        return fig

    ax.boxplot(data, tick_labels=list(_WEEKDAY_LABELS), showfliers=False)
    ax.set_xlabel("День недели")
    ax.set_ylabel("Стандартизованный |r|")
    ax.set_title("Стандартизованный модуль доходности по дням недели")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def _save_figure(fig: Figure, path: Path | None) -> None:
    if path is None:
        return
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(fig)
