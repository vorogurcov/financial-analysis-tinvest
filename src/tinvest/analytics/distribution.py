"""Distribution of daily log returns and annualized volatility."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy import stats
import statsmodels.formula.api as smf

from tinvest.analytics.base import Analysis
from tinvest.common.frame import TypedFrame
from tinvest.research.panel import PanelFrame
from tinvest.research.universe import UniverseFrame


class InstrumentMetricsFrame(TypedFrame):
    """One row of return and volatility metrics per included ticker."""

    ticker: str
    sector: str
    sector_group: str
    share_type: str
    div_yield_flag: bool
    liquidity_flag: bool
    short_enabled_flag: bool
    lot: int
    issue_size: int
    n_returns: int
    mean_log_return: float
    ann_mean_return: float
    ann_vol: float
    skewness: float
    excess_kurtosis: float
    median_turnover_rub: float
    log_issue_size: float
    log_median_turnover: float
    age_years: float
    jarque_bera_stat: float
    jarque_bera_pvalue: float
    shapiro_stat: float
    shapiro_pvalue: float
    shapiro_note: str


class SpearmanFrame(TypedFrame):
    """Rank correlation between instrument characteristics and metrics."""

    characteristic: str
    metric: str
    rho: float
    p_value: float
    n: int


class GroupTestsFrame(TypedFrame):
    """Non-parametric group comparison of annualized volatility."""

    test: str
    statistic: float
    p_value: float
    n_groups: int
    note: str


class VolRegressionFrame(TypedFrame):
    """OLS coefficients for annualized volatility models."""

    model: str
    term: str
    coef: float
    std_err: float
    p_value: float
    n: int
    r_squared: float
    base_sector: str


@dataclass(frozen=True)
class DistributionConfig:
    """Paths for figures and the universe catalog."""

    figures_dir: Path
    universe_path: Path | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "figures_dir", Path(self.figures_dir))
        if self.universe_path is not None:
            object.__setattr__(self, "universe_path", Path(self.universe_path))


@dataclass(frozen=True)
class DistributionSummary:
    """Compact log line for the distribution analysis."""

    n_tickers: int
    median_ann_vol: float
    highest_vol_sector_group: str
    base_r_squared: float
    jarque_bera_reject_share: float


@dataclass(frozen=True)
class DistributionTables:
    """All tables produced by the distribution analysis."""

    metrics: InstrumentMetricsFrame
    spearman: SpearmanFrame
    group_tests: GroupTestsFrame
    regression: VolRegressionFrame


CHARACTERISTICS = (
    "log_issue_size",
    "log_median_turnover",
    "log_lot",
    "age_years",
    "div_yield_flag",
    "short_enabled_flag",
    "liquidity_flag",
)
METRICS = ("ann_mean_return", "ann_vol", "skewness", "excess_kurtosis")


def assign_sector_groups(sectors: pd.Series) -> pd.Series:
    """Keep sector names with at least five tickers; otherwise ``прочие``."""
    counts = sectors.value_counts()
    large = set(counts[counts >= 5].index.astype(str))
    return sectors.map(lambda value: value if value in large else "прочие")


def regression_eligible_mask(sector_group: pd.Series) -> pd.Series:
    """Tickers in groups with size >= 5 (including large ``прочие``)."""
    counts = sector_group.value_counts()
    large = set(counts[counts >= 5].index.astype(str))
    return sector_group.isin(large)


class ReturnDistribution(Analysis[DistributionTables]):
    """Instrument metrics, Spearman, Kruskal–Wallis, OLS, and four figures."""

    def __init__(self, config: DistributionConfig) -> None:
        super().__init__(config)
        self.config: DistributionConfig = config
        self._panel: PanelFrame | None = None
        self._metrics: InstrumentMetricsFrame | None = None
        self._summary: DistributionSummary | None = None

    def prepare(self, panel: PanelFrame) -> InstrumentMetricsFrame:  # type: ignore[override]
        if panel.empty:
            raise ValueError("выборка пуста")
        if self.config.universe_path is None:
            raise ValueError("DistributionConfig.universe_path is required")

        universe = _read_universe_file(self.config.universe_path)
        metrics = _compute_metrics(panel, universe)
        self._panel = panel
        self._working = metrics
        self._metrics = metrics
        self._results = None
        self._summary = None
        return metrics

    def analyze(self) -> DistributionTables:
        metrics = self._metrics
        if metrics is None:
            raise RuntimeError("Call prepare() before analyze()")

        frame = pd.DataFrame(metrics).copy()
        frame["log_lot"] = np.log(frame["lot"].astype(float))
        frame["div_yield_flag"] = frame["div_yield_flag"].astype(int)
        frame["short_enabled_flag"] = frame["short_enabled_flag"].astype(int)
        frame["liquidity_flag"] = frame["liquidity_flag"].astype(int)

        spearman = _spearman_table(frame)
        group_tests = _kruskal_table(frame)
        regression = _regression_table(frame)

        tables = DistributionTables(
            metrics=metrics,
            spearman=spearman,
            group_tests=group_tests,
            regression=regression,
        )
        self._results = tables
        self._summary = _build_summary(frame, regression)
        return tables

    def get_summary(self) -> DistributionSummary:
        if self._summary is None:
            raise RuntimeError("Call analyze() before get_summary()")
        return self._summary

    def plot(self) -> Figure:
        if self._panel is None or self._metrics is None:
            raise RuntimeError("Call prepare() and analyze() before plot()")
        figures_dir = self.config.figures_dir
        figures_dir.mkdir(parents=True, exist_ok=True)

        hist_fig = _plot_return_hist(self._panel, figures_dir / "task1_return_hist.png")
        _plot_qq(self._panel, self._metrics, figures_dir / "task1_qq.png")
        _plot_vol_by_sector(self._metrics, figures_dir / "task1_vol_by_sector.png")
        _plot_vol_vs_size(self._metrics, figures_dir / "task1_vol_vs_size.png")
        return hist_fig


def _read_universe_file(path: Path) -> UniverseFrame:
    frame = pd.read_parquet(path)
    frame["ipo_date"] = pd.to_datetime(frame["ipo_date"], utc=True)
    frame["lot"] = frame["lot"].astype("int64")
    frame["issue_size"] = frame["issue_size"].astype("int64")
    frame["nominal"] = frame["nominal"].astype("float64")
    for flag in ("div_yield_flag", "liquidity_flag", "short_enabled_flag", "included"):
        frame[flag] = frame[flag].astype(bool)
    for column in (
        "ticker",
        "figi",
        "name",
        "class_code",
        "currency",
        "country_of_risk",
        "sector",
        "share_type",
        "exclude_reason",
    ):
        frame[column] = frame[column].astype(str)
    return UniverseFrame.build(frame)


def _compute_metrics(panel: PanelFrame, universe: UniverseFrame) -> InstrumentMetricsFrame:
    panel_df = pd.DataFrame(panel)
    catalog = pd.DataFrame(universe)
    catalog = catalog.loc[catalog["included"]].copy()
    catalog["sector_group"] = assign_sector_groups(catalog["sector"])

    rows: list[dict[str, object]] = []
    for ticker, chunk in panel_df.groupby("ticker", sort=True):
        meta = catalog.loc[catalog["ticker"] == ticker]
        if meta.empty:
            continue
        meta_row = meta.iloc[0]
        returns = chunk["log_return"].dropna()
        n_returns = int(len(returns))
        if n_returns < 2:
            continue

        mean_r = float(returns.mean())
        std_r = float(returns.std(ddof=1))
        turnover = chunk["volume"].astype(float) * float(meta_row["lot"]) * chunk["close"].astype(
            float
        )
        positive_turnover = turnover[turnover > 0]
        median_turnover = float(positive_turnover.median()) if not positive_turnover.empty else 0.0

        ipo = pd.Timestamp(meta_row["ipo_date"])
        last_time = pd.Timestamp(chunk["time"].max())
        if ipo.year < 1990:
            age_years = float("nan")
        else:
            age_years = float((last_time - ipo).days / 365.25)

        jb_stat, jb_p = stats.jarque_bera(returns.to_numpy(dtype=float))
        if 20 <= n_returns <= 5000:
            shapiro_stat, shapiro_p = stats.shapiro(returns.to_numpy(dtype=float))
            shapiro_note = ""
        else:
            shapiro_stat, shapiro_p = float("nan"), float("nan")
            shapiro_note = "skipped_n"

        issue_size = int(meta_row["issue_size"])
        rows.append(
            {
                "ticker": str(ticker),
                "sector": str(meta_row["sector"]),
                "sector_group": str(meta_row["sector_group"]),
                "share_type": str(meta_row["share_type"]),
                "div_yield_flag": bool(meta_row["div_yield_flag"]),
                "liquidity_flag": bool(meta_row["liquidity_flag"]),
                "short_enabled_flag": bool(meta_row["short_enabled_flag"]),
                "lot": int(meta_row["lot"]),
                "issue_size": issue_size,
                "n_returns": n_returns,
                "mean_log_return": mean_r,
                "ann_mean_return": mean_r * 252.0,
                "ann_vol": std_r * float(np.sqrt(252.0)),
                "skewness": float(stats.skew(returns, bias=False)),
                "excess_kurtosis": float(stats.kurtosis(returns, bias=False, fisher=True)),
                "median_turnover_rub": median_turnover,
                "log_issue_size": float(np.log(issue_size)),
                "log_median_turnover": float(np.log(median_turnover))
                if median_turnover > 0
                else float("nan"),
                "age_years": age_years,
                "jarque_bera_stat": float(jb_stat),
                "jarque_bera_pvalue": float(jb_p),
                "shapiro_stat": float(shapiro_stat),
                "shapiro_pvalue": float(shapiro_p),
                "shapiro_note": shapiro_note,
            }
        )

    if not rows:
        raise ValueError("выборка пуста")

    frame = pd.DataFrame(rows)
    frame["lot"] = frame["lot"].astype("int64")
    frame["issue_size"] = frame["issue_size"].astype("int64")
    frame["n_returns"] = frame["n_returns"].astype("int64")
    for flag in ("div_yield_flag", "liquidity_flag", "short_enabled_flag"):
        frame[flag] = frame[flag].astype(bool)
    for column in (
        "mean_log_return",
        "ann_mean_return",
        "ann_vol",
        "skewness",
        "excess_kurtosis",
        "median_turnover_rub",
        "log_issue_size",
        "log_median_turnover",
        "age_years",
        "jarque_bera_stat",
        "jarque_bera_pvalue",
        "shapiro_stat",
        "shapiro_pvalue",
    ):
        frame[column] = frame[column].astype("float64")
    for column in ("ticker", "sector", "sector_group", "share_type", "shapiro_note"):
        frame[column] = frame[column].astype(str)
    return InstrumentMetricsFrame.build(frame)


def _spearman_table(frame: pd.DataFrame) -> SpearmanFrame:
    rows: list[dict[str, object]] = []
    for characteristic in CHARACTERISTICS:
        for metric in METRICS:
            pair = frame[[characteristic, metric]].dropna()
            n = int(len(pair))
            x = pair[characteristic]
            y = pair[metric]
            # Constant inputs make Spearman undefined; avoid scipy ConstantInputWarning.
            if n < 3 or x.nunique(dropna=True) < 2 or y.nunique(dropna=True) < 2:
                rho, p_value = float("nan"), float("nan")
            else:
                rho_val, p_val = stats.spearmanr(x, y)
                rho, p_value = float(rho_val), float(p_val)
            rows.append(
                {
                    "characteristic": characteristic,
                    "metric": metric,
                    "rho": rho,
                    "p_value": p_value,
                    "n": n,
                }
            )
    out = pd.DataFrame(rows)
    out["n"] = out["n"].astype("int64")
    out["rho"] = out["rho"].astype("float64")
    out["p_value"] = out["p_value"].astype("float64")
    out["characteristic"] = out["characteristic"].astype(str)
    out["metric"] = out["metric"].astype(str)
    return SpearmanFrame.build(out)


def _kruskal_table(frame: pd.DataFrame) -> GroupTestsFrame:
    eligible = frame.loc[regression_eligible_mask(frame["sector_group"])].copy()
    groups = []
    for _name, chunk in eligible.groupby("sector_group"):
        if len(chunk) >= 5:
            groups.append(chunk["ann_vol"].to_numpy(dtype=float))
    if len(groups) < 2:
        out = pd.DataFrame(
            [
                {
                    "test": "kruskal_wallis_ann_vol",
                    "statistic": float("nan"),
                    "p_value": float("nan"),
                    "n_groups": int(len(groups)),
                    "note": "too_few_groups",
                }
            ]
        )
    else:
        statistic, p_value = stats.kruskal(*groups)
        out = pd.DataFrame(
            [
                {
                    "test": "kruskal_wallis_ann_vol",
                    "statistic": float(statistic),
                    "p_value": float(p_value),
                    "n_groups": int(len(groups)),
                    "note": "",
                }
            ]
        )
    out["statistic"] = out["statistic"].astype("float64")
    out["p_value"] = out["p_value"].astype("float64")
    out["n_groups"] = out["n_groups"].astype("int64")
    out["test"] = out["test"].astype(str)
    out["note"] = out["note"].astype(str)
    return GroupTestsFrame.build(out)


def _normalize_term(term: str) -> str:
    """Map statsmodels ``C(col, Treatment(...))[T.x]`` to ``col[T.x]``."""
    prefix = "C("
    if not term.startswith(prefix) or ")[T." not in term:
        return term
    inner, _, level = term.partition(")[T.")
    column = inner[len(prefix) :].split(",", 1)[0].strip()
    level_name = level[:-1] if level.endswith("]") else level
    return f"{column}[T.{level_name}]"


def _fit_ols(frame: pd.DataFrame, formula: str, base_sector: str, model_name: str) -> pd.DataFrame:
    model = smf.ols(formula, data=frame).fit(cov_type="HC3")
    rows: list[dict[str, object]] = []
    for term in model.params.index:
        rows.append(
            {
                "model": model_name,
                "term": _normalize_term(str(term)),
                "coef": float(model.params[term]),
                "std_err": float(model.bse[term]),
                "p_value": float(model.pvalues[term]),
                "n": int(model.nobs),
                "r_squared": float(model.rsquared),
                "base_sector": base_sector,
            }
        )
    return pd.DataFrame(rows)


def _regression_table(frame: pd.DataFrame) -> VolRegressionFrame:
    eligible = frame.loc[regression_eligible_mask(frame["sector_group"])].copy()
    if eligible.empty:
        blank = pd.DataFrame(
            columns=[
                "model",
                "term",
                "coef",
                "std_err",
                "p_value",
                "n",
                "r_squared",
                "base_sector",
            ]
        )
        # TypedFrame cannot be empty without proper dtypes — synthesize a placeholder row set
        # only when we truly cannot estimate; acceptance expects a real regression on live data.
        return VolRegressionFrame.blank()

    counts = eligible["sector_group"].value_counts()
    base_sector = str(counts.idxmax())
    base_formula = (
        "ann_vol ~ log_issue_size + log_median_turnover "
        "+ C(share_type, Treatment(reference='common')) "
        f"+ C(sector_group, Treatment(reference='{base_sector}'))"
    )
    parts = [_fit_ols(eligible, base_formula, base_sector, "base")]

    extended = eligible.dropna(subset=["age_years"]).copy()
    if len(extended) >= 5 and extended["share_type"].nunique() >= 1:
        ext_counts = extended["sector_group"].value_counts()
        ext_base = str(ext_counts.idxmax())
        ext_formula = (
            "ann_vol ~ log_issue_size + log_median_turnover + log_lot + age_years "
            "+ div_yield_flag + short_enabled_flag "
            "+ C(share_type, Treatment(reference='common')) "
            f"+ C(sector_group, Treatment(reference='{ext_base}'))"
        )
        parts.append(_fit_ols(extended, ext_formula, ext_base, "extended"))

    out = pd.concat(parts, ignore_index=True)
    out["n"] = out["n"].astype("int64")
    for column in ("coef", "std_err", "p_value", "r_squared"):
        out[column] = out[column].astype("float64")
    for column in ("model", "term", "base_sector"):
        out[column] = out[column].astype(str)
    return VolRegressionFrame.build(out)


def _build_summary(frame: pd.DataFrame, regression: VolRegressionFrame) -> DistributionSummary:
    med_vol = float(frame["ann_vol"].median())
    sector_medians = frame.groupby("sector_group")["ann_vol"].median().sort_values(ascending=False)
    highest = str(sector_medians.index[0]) if not sector_medians.empty else ""
    base_rows = pd.DataFrame(regression)
    if not base_rows.empty:
        base = base_rows.loc[base_rows["model"] == "base"]
        r_squared = float(base["r_squared"].iloc[0]) if not base.empty else float("nan")
    else:
        r_squared = float("nan")
    reject_share = float((frame["jarque_bera_pvalue"] < 0.05).mean())
    return DistributionSummary(
        n_tickers=int(len(frame)),
        median_ann_vol=med_vol,
        highest_vol_sector_group=highest,
        base_r_squared=r_squared,
        jarque_bera_reject_share=reject_share,
    )


def _plot_return_hist(panel: PanelFrame, path: Path) -> Figure:
    working = pd.DataFrame(panel)
    parts: list[pd.Series] = []
    for _, chunk in working.groupby("ticker"):
        returns = chunk["log_return"].dropna()
        if len(returns) < 2:
            continue
        std = returns.std(ddof=1)
        if std == 0 or np.isnan(std):
            continue
        parts.append((returns - returns.mean()) / std)
    if parts:
        standardized = pd.concat(parts, ignore_index=True)
    else:
        standardized = pd.Series(dtype=float)

    fig, ax = plt.subplots(figsize=(10, 5))
    if not standardized.empty:
        ax.hist(standardized, bins=60, density=True, alpha=0.65, color="steelblue", label="Данные")
        xs = np.linspace(-4, 4, 400)
        ax.plot(xs, stats.norm.pdf(xs), color="black", linewidth=1.5, label="N(0, 1)")
    ax.set_title("Стандартизованные дневные лог-доходности")
    ax.set_xlabel("Стандартизованная доходность")
    ax.set_ylabel("Плотность")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return fig


def _plot_qq(panel: PanelFrame, metrics: InstrumentMetricsFrame, path: Path) -> Figure:
    metrics_df = pd.DataFrame(metrics).sort_values("median_turnover_rub", ascending=False)
    tickers = metrics_df["ticker"].head(16).tolist()
    n = len(tickers)
    ncols = 4
    nrows = int(np.ceil(n / ncols)) if n else 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3 * nrows))
    axes_arr = np.atleast_1d(axes).ravel()
    panel_df = pd.DataFrame(panel)
    for index, ticker in enumerate(tickers):
        ax = axes_arr[index]
        returns = panel_df.loc[panel_df["ticker"] == ticker, "log_return"].dropna()
        std = returns.std(ddof=1)
        if len(returns) < 2 or std == 0 or np.isnan(std):
            ax.set_visible(False)
            continue
        standardized = (returns - returns.mean()) / std
        stats.probplot(standardized, dist="norm", plot=ax)
        ax.set_title(ticker)
        ax.set_xlabel("")
        ax.set_ylabel("")
    for ax in axes_arr[n:]:
        ax.set_visible(False)
    fig.suptitle("QQ-графики стандартизованных доходностей", y=1.01)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return fig


def _plot_vol_by_sector(metrics: InstrumentMetricsFrame, path: Path) -> Figure:
    frame = pd.DataFrame(metrics)
    plot_df = frame.loc[regression_eligible_mask(frame["sector_group"])].copy()
    fig, ax = plt.subplots(figsize=(10, 5))
    if not plot_df.empty:
        grouped = list(plot_df.groupby("sector_group"))
        groups = [chunk["ann_vol"].to_numpy(dtype=float) for _, chunk in grouped]
        labels = [str(name) for name, _ in grouped]
        ax.boxplot(groups, showfliers=True)
        ax.set_xticklabels(labels)
    ax.set_title("Годовая волатильность по секторным группам")
    ax.set_xlabel("Секторная группа")
    ax.set_ylabel("Годовая волатильность")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return fig


def _plot_vol_vs_size(metrics: InstrumentMetricsFrame, path: Path) -> Figure:
    frame = pd.DataFrame(metrics)
    fig, ax = plt.subplots(figsize=(10, 5))
    groups = list(frame.groupby("sector_group"))
    cmap = plt.get_cmap("tab10")
    for index, (name, chunk) in enumerate(groups):
        ax.scatter(
            chunk["log_issue_size"],
            chunk["ann_vol"],
            label=str(name),
            alpha=0.75,
            color=cmap(index % 10),
        )
    ax.set_title("Годовая волатильность и размер выпуска")
    ax.set_xlabel("ln(размер выпуска)")
    ax.set_ylabel("Годовая волатильность")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return fig
