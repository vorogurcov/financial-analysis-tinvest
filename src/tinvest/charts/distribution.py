"""Chart series for the return-distribution experiment."""

from __future__ import annotations

import pandas as pd

from tinvest.analytics.distribution import regression_eligible_mask
from tinvest.charts.spec import ChartSeries, ChartSpec
from tinvest.charts.stats import (
    boxplot_stats,
    density_histogram,
    normal_pdf_curve,
    qq_points,
    standardize_returns,
)
from tinvest.research.panel import PanelFrame


def distribution_charts(panel: PanelFrame, metrics: pd.DataFrame) -> list[ChartSpec]:
    panel_df = pd.DataFrame(panel)
    metrics_df = pd.DataFrame(metrics)
    return [
        _return_hist(panel_df),
        _qq_grid(panel_df, metrics_df),
        _vol_by_sector(metrics_df),
        _vol_vs_size(metrics_df),
    ]


def _return_hist(panel: pd.DataFrame) -> ChartSpec:
    standardized = standardize_returns(panel)
    return ChartSpec(
        id="return_hist",
        title="Стандартизованные дневные лог-доходности",
        kind="histogram",
        x_label="Стандартизованная доходность",
        y_label="Плотность",
        bins=density_histogram(standardized, bins=60),
        reference_line=normal_pdf_curve(),
        series=[ChartSeries(name="Данные")],
    )


def _qq_grid(panel: pd.DataFrame, metrics: pd.DataFrame) -> ChartSpec:
    ordered = metrics.sort_values("median_turnover_rub", ascending=False)
    tickers = ordered["ticker"].head(16).astype(str).tolist()
    panels: list[ChartSpec] = []
    for ticker in tickers:
        returns = panel.loc[panel["ticker"] == ticker, "log_return"].dropna()
        std = returns.std(ddof=1)
        if len(returns) < 2 or std == 0 or pd.isna(std):
            continue
        standardized = (returns - returns.mean()) / std
        theoretical, sample = qq_points(standardized)
        panels.append(
            ChartSpec(
                id=f"qq_{ticker}",
                title=ticker,
                kind="scatter",
                x_label="Теоретические квантили",
                y_label="Выборочные квантили",
                series=[
                    ChartSeries(
                        name=ticker,
                        x=[float(v) for v in theoretical],
                        values=[float(v) for v in sample],
                    )
                ],
            )
        )
    return ChartSpec(
        id="qq_grid",
        title="QQ-графики стандартизованных доходностей",
        kind="qq_grid",
        panels=panels,
    )


def _vol_by_sector(metrics: pd.DataFrame) -> ChartSpec:
    plot_df = metrics.loc[regression_eligible_mask(metrics["sector_group"])].copy()
    boxes = []
    if not plot_df.empty:
        for name, chunk in plot_df.groupby("sector_group"):
            boxes.append(
                boxplot_stats(
                    chunk["ann_vol"].to_numpy(dtype=float),
                    label=str(name),
                    show_outliers=True,
                )
            )
    return ChartSpec(
        id="vol_by_sector",
        title="Годовая волатильность по секторным группам",
        kind="boxplot",
        x_label="Секторная группа",
        y_label="Годовая волатильность",
        boxes=boxes,
    )


def _vol_vs_size(metrics: pd.DataFrame) -> ChartSpec:
    series: list[ChartSeries] = []
    for name, chunk in metrics.groupby("sector_group"):
        series.append(
            ChartSeries(
                name=str(name),
                x=[float(v) for v in chunk["log_issue_size"]],
                values=[float(v) for v in chunk["ann_vol"]],
            )
        )
    return ChartSpec(
        id="vol_vs_size",
        title="Годовая волатильность и размер выпуска",
        kind="scatter",
        x_label="ln(размер выпуска)",
        y_label="Годовая волатильность",
        series=series,
    )
