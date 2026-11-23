"""Abnormal turnover versus same-day moves and near-term realized volatility."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy import stats
import statsmodels.formula.api as smf

from tinvest.analytics.base import Analysis
from tinvest.charts.spec import ChartSpec
from tinvest.charts.stats import decile_means
from tinvest.common.frame import TypedFrame
from tinvest.research.panel import PanelFrame

logger = logging.getLogger(__name__)

_SQRT_252 = float(np.sqrt(252.0))
_VOLUME_MIN_PERIODS = 15
_JOINT_COLUMNS = ("abn_volume", "abs_return", "abs_return_lag", "rv_5", "future_rv_5")
_KEEP_TERMS = frozenset(
    {"abn_volume", "abs_return_lag", "rv_5", "log_return_lag"}
)


class VolumePanelFrame(PanelFrame):
    """Panel bars enriched with turnover and short-horizon volatility features."""

    turnover: float
    log_turnover: float
    abn_volume: float
    abs_return: float
    abs_return_lag: float
    log_return_lag: float
    rv_5: float
    future_rv_5: float
    range: float


class VolumeCoefficientsFrame(TypedFrame):
    """OLS HC1 coefficients for volume-impact models."""

    model: str
    term: str
    coef: float
    std_err: float
    p_value: float
    n: int
    r_squared: float


class VolumeSpearmanFrame(TypedFrame):
    """Within-ticker Spearman between abnormal volume and absolute return."""

    ticker: str
    rho: float
    p_value: float
    n: int


class VolumeExcludedFrame(TypedFrame):
    """Tickers dropped from the volume regressions."""

    ticker: str
    n_rows: int
    reason: str


@dataclass(frozen=True)
class VolumeConfig:
    """Windows, sample floor, and figure destinations for the volume analysis."""

    horizon: int = 5
    volume_window: int = 21
    min_rows: int = 100
    figure_path: Path | None = None
    lead_vol_figure_path: Path | None = None
    coefficients_figure_path: Path | None = None

    def __post_init__(self) -> None:
        if self.horizon < 2:
            raise ValueError("horizon must be at least 2")
        if self.volume_window < 2:
            raise ValueError("volume_window must be at least 2")
        if self.min_rows < 1:
            raise ValueError("min_rows must be at least 1")
        if self.figure_path is not None:
            object.__setattr__(self, "figure_path", Path(self.figure_path))
        if self.lead_vol_figure_path is not None:
            object.__setattr__(self, "lead_vol_figure_path", Path(self.lead_vol_figure_path))
        if self.coefficients_figure_path is not None:
            object.__setattr__(
                self, "coefficients_figure_path", Path(self.coefficients_figure_path)
            )


@dataclass(frozen=True)
class VolumeSummary:
    """Compact log line for the volume-impact analysis."""

    n_tickers: int
    coef_a: float | None
    p_value_a: float | None
    coef_b: float | None
    p_value_b: float | None
    coef_c: float | None
    p_value_c: float | None
    spearman_median: float | None
    spearman_positive_share: float | None


@dataclass(frozen=True)
class VolumeTables:
    """All tables produced by the volume-impact analysis."""

    coefficients: VolumeCoefficientsFrame
    spearman: VolumeSpearmanFrame
    excluded: VolumeExcludedFrame


class VolumeImpact(Analysis[VolumeTables]):
    """Abnormal turnover features, panel OLS, Spearman, and three figures."""

    def __init__(self, config: VolumeConfig) -> None:
        super().__init__(config)
        self.config: VolumeConfig = config
        self._summary: VolumeSummary | None = None
        self._sample: pd.DataFrame | None = None

    def prepare(self, panel: PanelFrame) -> VolumePanelFrame:  # type: ignore[override]
        if panel.empty:
            raise ValueError("выборка пуста")
        features = add_volume_features(
            panel,
            horizon=self.config.horizon,
            volume_window=self.config.volume_window,
        )
        self._working = features
        self._results = None
        self._summary = None
        self._sample = None
        return features

    def analyze(self) -> VolumeTables:
        working = pd.DataFrame(self._require_working()).copy()
        joint = working.dropna(subset=list(_JOINT_COLUMNS))
        counts = joint.groupby("ticker", sort=True).size()
        keep = set(counts[counts >= self.config.min_rows].index.astype(str))
        all_tickers = working["ticker"].drop_duplicates().astype(str)

        excluded_rows: list[dict[str, object]] = []
        for ticker in all_tickers:
            n_rows = int(counts.get(ticker, 0))
            if ticker not in keep:
                excluded_rows.append(
                    {
                        "ticker": ticker,
                        "n_rows": n_rows,
                        "reason": "too_few_volume_rows",
                    }
                )
        excluded = _excluded_frame(excluded_rows)

        if not keep:
            tables = VolumeTables(
                coefficients=VolumeCoefficientsFrame.blank(),
                spearman=VolumeSpearmanFrame.blank(),
                excluded=excluded,
            )
            self._results = tables
            self._sample = None
            self._summary = VolumeSummary(
                n_tickers=0,
                coef_a=None,
                p_value_a=None,
                coef_b=None,
                p_value_b=None,
                coef_c=None,
                p_value_c=None,
                spearman_median=None,
                spearman_positive_share=None,
            )
            return tables

        sample = joint.loc[joint["ticker"].isin(keep)].copy()
        self._sample = sample

        parts = [
            _fit_model(
                sample,
                model_name="A",
                columns=("abs_return", "abn_volume", "abs_return_lag"),
            ),
            _fit_model(
                sample,
                model_name="B",
                columns=("future_rv_5", "abn_volume", "rv_5"),
            ),
            _fit_model(
                sample,
                model_name="C",
                columns=("log_return", "abn_volume", "log_return_lag"),
            ),
        ]

        threshold = float(np.quantile(sample["abs_return"].to_numpy(dtype=float), 0.99))
        trimmed = sample.loc[sample["abs_return"] <= threshold].copy()
        if len(trimmed) < len(sample) and not trimmed.empty:
            parts.append(
                _fit_model(
                    trimmed,
                    model_name="A_trim",
                    columns=("abs_return", "abn_volume", "abs_return_lag"),
                )
            )
            parts.append(
                _fit_model(
                    trimmed,
                    model_name="B_trim",
                    columns=("future_rv_5", "abn_volume", "rv_5"),
                )
            )

        coefficients = _coefficients_frame(pd.concat(parts, ignore_index=True))
        spearman = _spearman_by_ticker(sample)
        tables = VolumeTables(
            coefficients=coefficients,
            spearman=spearman,
            excluded=excluded,
        )
        self._results = tables
        self._summary = _build_summary(coefficients, spearman, n_tickers=len(keep))
        return tables

    def get_summary(self) -> VolumeSummary:
        if self._summary is None:
            raise RuntimeError("Call analyze() before get_summary()")
        return self._summary

    def chart_specs(self) -> list[ChartSpec]:
        from tinvest.charts.volume import volume_charts

        summary = self.get_summary()
        if summary.n_tickers == 0 or self._sample is None or self._results is None:
            return []
        return volume_charts(self._sample, pd.DataFrame(self._results.coefficients))

    def plot(self) -> Figure:
        summary = self.get_summary()
        if summary.n_tickers == 0 or self._sample is None or self._results is None:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.set_axis_off()
            return fig

        sample = self._sample
        coefficients = pd.DataFrame(self._results.coefficients)

        fig = _plot_decile_means(
            sample,
            x_col="abn_volume",
            y_col="abs_return",
            title="Средний модуль доходности по децилям аномального оборота",
            ylabel="Средний |r|",
        )
        self._save(fig)

        lead_fig = _plot_decile_means(
            sample,
            x_col="abn_volume",
            y_col="future_rv_5",
            title="Средняя будущая 5-дневная волатильность по децилям оборота",
            ylabel="Средняя future_rv_5",
        )
        _save_figure(lead_fig, self.config.lead_vol_figure_path)

        coef_fig = _plot_coefficients(coefficients)
        _save_figure(coef_fig, self.config.coefficients_figure_path)
        return fig


def add_volume_features(
    panel: PanelFrame | pd.DataFrame,
    *,
    horizon: int = 5,
    volume_window: int = 21,
) -> VolumePanelFrame:
    """Compute section-4 columns inside each ticker, ordered by time."""
    working = pd.DataFrame(panel).copy()
    if working.empty:
        return VolumePanelFrame.blank()

    working = working.sort_values(["ticker", "time"], kind="mergesort").reset_index(drop=True)
    parts: list[pd.DataFrame] = []
    for _, chunk in working.groupby("ticker", sort=False):
        parts.append(
            _features_one_ticker(chunk, horizon=horizon, volume_window=volume_window)
        )
    frame = pd.concat(parts, ignore_index=True)
    return _volume_panel_frame(frame)


def _features_one_ticker(
    chunk: pd.DataFrame,
    *,
    horizon: int,
    volume_window: int,
) -> pd.DataFrame:
    out = chunk.copy()
    turnover = out["volume"].astype(float) * out["lot"].astype(float) * out["close"].astype(float)
    out["turnover"] = turnover
    log_turnover = pd.Series(np.nan, index=out.index, dtype="float64")
    positive = turnover > 0
    log_turnover.loc[positive] = np.log(turnover.loc[positive])
    out["log_turnover"] = log_turnover

    rolling_median = log_turnover.rolling(
        window=volume_window,
        min_periods=_VOLUME_MIN_PERIODS,
    ).median()
    out["abn_volume"] = log_turnover - rolling_median

    log_return = out["log_return"].astype(float)
    abs_return = log_return.abs()
    out["abs_return"] = abs_return
    out["abs_return_lag"] = abs_return.shift(1)
    out["log_return_lag"] = log_return.shift(1)

    # Rolling std skips NaNs by default; require a full window of non-null returns.
    valid = log_return.notna().astype(float)
    full_window = valid.rolling(window=horizon, min_periods=horizon).sum() == float(horizon)
    rv = log_return.rolling(window=horizon, min_periods=horizon).std(ddof=1) * _SQRT_252
    out["rv_5"] = rv.where(full_window)
    out["future_rv_5"] = out["rv_5"].shift(-horizon)

    close = out["close"].astype(float)
    out["range"] = (out["high"].astype(float) - out["low"].astype(float)) / close
    return out


def _volume_panel_frame(frame: pd.DataFrame) -> VolumePanelFrame:
    ordered = frame.copy()
    ordered["time"] = pd.to_datetime(ordered["time"], utc=True)
    ordered["volume"] = ordered["volume"].astype("int64")
    ordered["lot"] = ordered["lot"].astype("int64")
    for column in (
        "open",
        "high",
        "low",
        "close",
        "log_return",
        "turnover",
        "log_turnover",
        "abn_volume",
        "abs_return",
        "abs_return_lag",
        "log_return_lag",
        "rv_5",
        "future_rv_5",
        "range",
    ):
        ordered[column] = ordered[column].astype("float64")
    for column in ("ticker", "sector", "share_type"):
        ordered[column] = ordered[column].astype(str)
    return VolumePanelFrame.build(ordered)


def _standardize_within_ticker(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        def _scale(series: pd.Series) -> pd.Series:
            std = float(series.std(ddof=1))
            if not np.isfinite(std) or std == 0.0:
                return pd.Series(0.0, index=series.index)
            return (series - series.mean()) / std

        out[column] = out.groupby("ticker", sort=False)[column].transform(_scale)
    return out


def _fit_model(
    sample: pd.DataFrame,
    *,
    model_name: str,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    scaled = _standardize_within_ticker(sample, columns)
    # Drop controls that are constant after within-ticker scaling (collinear with FE).
    active_columns = [
        column
        for column in columns
        if column == columns[0] or float(scaled[column].std(ddof=1) or 0.0) > 0.0
    ]
    response = active_columns[0]
    rhs = " + ".join([*active_columns[1:], "C(ticker)"])
    use_formula = f"{response} ~ {rhs}"
    model = smf.ols(use_formula, data=scaled).fit(cov_type="HC1")
    rows: list[dict[str, object]] = []
    for term in model.params.index:
        name = str(term)
        if name == "Intercept" or name.startswith("C(ticker)"):
            continue
        if name not in _KEEP_TERMS:
            continue
        rows.append(
            {
                "model": model_name,
                "term": name,
                "coef": float(model.params[term]),
                "std_err": float(model.bse[term]),
                "p_value": float(model.pvalues[term]),
                "n": int(model.nobs),
                "r_squared": float(model.rsquared),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=["model", "term", "coef", "std_err", "p_value", "n", "r_squared"]
        )
    return pd.DataFrame(rows)


def _coefficients_frame(frame: pd.DataFrame) -> VolumeCoefficientsFrame:
    if frame.empty:
        return VolumeCoefficientsFrame.blank()
    out = frame.copy()
    out["n"] = out["n"].astype("int64")
    for column in ("coef", "std_err", "p_value", "r_squared"):
        out[column] = out[column].astype("float64")
    out["model"] = out["model"].astype(str)
    out["term"] = out["term"].astype(str)
    return VolumeCoefficientsFrame.build(out)


def _excluded_frame(rows: list[dict[str, object]]) -> VolumeExcludedFrame:
    if not rows:
        return VolumeExcludedFrame.blank()
    out = pd.DataFrame(rows)
    out["n_rows"] = out["n_rows"].astype("int64")
    out["ticker"] = out["ticker"].astype(str)
    out["reason"] = out["reason"].astype(str)
    return VolumeExcludedFrame.build(out)


def _spearman_by_ticker(sample: pd.DataFrame) -> VolumeSpearmanFrame:
    rows: list[dict[str, object]] = []
    for ticker, chunk in sample.groupby("ticker", sort=True):
        x = chunk["abn_volume"].to_numpy(dtype=float)
        y = chunk["abs_return"].to_numpy(dtype=float)
        if len(x) < 3:
            continue
        rho, p_value = stats.spearmanr(x, y)
        rows.append(
            {
                "ticker": str(ticker),
                "rho": float(rho),
                "p_value": float(p_value),
                "n": int(len(x)),
            }
        )
    if not rows:
        return VolumeSpearmanFrame.blank()
    out = pd.DataFrame(rows)
    out["n"] = out["n"].astype("int64")
    for column in ("rho", "p_value"):
        out[column] = out[column].astype("float64")
    out["ticker"] = out["ticker"].astype(str)
    return VolumeSpearmanFrame.build(out)


def _abn_coef(coefficients: pd.DataFrame, model_name: str) -> tuple[float | None, float | None]:
    mask = (coefficients["model"] == model_name) & (coefficients["term"] == "abn_volume")
    subset = coefficients.loc[mask]
    if subset.empty:
        return None, None
    row = subset.iloc[0]
    return float(row["coef"]), float(row["p_value"])


def _build_summary(
    coefficients: VolumeCoefficientsFrame,
    spearman: VolumeSpearmanFrame,
    *,
    n_tickers: int,
) -> VolumeSummary:
    coef_df = pd.DataFrame(coefficients)
    coef_a, p_a = _abn_coef(coef_df, "A")
    coef_b, p_b = _abn_coef(coef_df, "B")
    coef_c, p_c = _abn_coef(coef_df, "C")

    spearman_df = pd.DataFrame(spearman)
    if spearman_df.empty:
        median_rho = None
        positive_share = None
    else:
        median_rho = float(spearman_df["rho"].median())
        positive_share = float((spearman_df["rho"] > 0).mean())

    return VolumeSummary(
        n_tickers=n_tickers,
        coef_a=coef_a,
        p_value_a=p_a,
        coef_b=coef_b,
        p_value_b=p_b,
        coef_c=coef_c,
        p_value_c=p_c,
        spearman_median=median_rho,
        spearman_positive_share=positive_share,
    )


def _save_figure(fig: Figure, path: Path | None) -> None:
    if path is None:
        return
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_decile_means(
    sample: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    title: str,
    ylabel: str,
) -> Figure:
    points = decile_means(sample, x_col=x_col, y_col=y_col)
    fig, ax = plt.subplots(figsize=(10, 5))
    if not points.empty:
        ax.errorbar(
            points["decile"],
            points["mean"],
            yerr=[points["mean"] - points["low"], points["high"] - points["mean"]],
            fmt="o",
            capsize=4,
            color="steelblue",
        )
        ax.set_xticks(points["decile"].astype(int))
    ax.set_title(title)
    ax.set_xlabel("дециль аномального оборота")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def _plot_coefficients(coefficients: pd.DataFrame) -> Figure:
    order = ["A", "B", "C", "A_trim", "B_trim"]
    subset = coefficients.loc[coefficients["term"] == "abn_volume"].copy()
    subset["model"] = pd.Categorical(subset["model"], categories=order, ordered=True)
    subset = subset.dropna(subset=["model"]).sort_values("model")

    fig, ax = plt.subplots(figsize=(10, 5))
    if not subset.empty:
        xs = np.arange(len(subset))
        coefs = subset["coef"].to_numpy(dtype=float)
        errs = 1.96 * subset["std_err"].to_numpy(dtype=float)
        ax.errorbar(xs, coefs, yerr=errs, fmt="o", capsize=4, color="darkorange")
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xticks(xs)
        ax.set_xticklabels(subset["model"].astype(str).tolist())
    ax.set_title("Коэффициенты abn_volume (HC1, 95% интервал)")
    ax.set_xlabel("Модель")
    ax.set_ylabel("Коэффициент")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
