"""Chart series for the volume-impact experiment."""

from __future__ import annotations

import pandas as pd

from tinvest.charts.spec import ChartSeries, ChartSpec
from tinvest.charts.stats import decile_means


def volume_charts(sample: pd.DataFrame, coefficients: pd.DataFrame) -> list[ChartSpec]:
    if sample.empty:
        return []
    return [
        _decile_chart(
            sample,
            x_col="abn_volume",
            y_col="abs_return",
            chart_id="abs_return_vs_volume",
            title="Средний модуль доходности по децилям аномального оборота",
            ylabel="Средний |r|",
        ),
        _decile_chart(
            sample,
            x_col="abn_volume",
            y_col="future_rv_5",
            chart_id="lead_vol",
            title="Средняя будущая 5-дневная волатильность по децилям оборота",
            ylabel="Средняя future_rv_5",
        ),
        _coefficients_chart(coefficients),
    ]


def _decile_chart(
    sample: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    chart_id: str,
    title: str,
    ylabel: str,
) -> ChartSpec:
    points = decile_means(sample, x_col=x_col, y_col=y_col)
    if points.empty:
        return ChartSpec(
            id=chart_id,
            title=title,
            kind="errorbar",
            x_label="дециль аномального оборота",
            y_label=ylabel,
            series=[],
        )
    means = [float(v) for v in points["mean"]]
    return ChartSpec(
        id=chart_id,
        title=title,
        kind="errorbar",
        x=[int(v) for v in points["decile"]],
        x_label="дециль аномального оборота",
        y_label=ylabel,
        series=[
            ChartSeries(
                name=ylabel,
                values=means,
                yerr_low=[float(m - low) for m, low in zip(means, points["low"], strict=True)],
                yerr_high=[float(high - m) for m, high in zip(means, points["high"], strict=True)],
            )
        ],
    )


def _coefficients_chart(coefficients: pd.DataFrame) -> ChartSpec:
    order = ["A", "B", "C", "A_trim", "B_trim"]
    subset = coefficients.loc[coefficients["term"] == "abn_volume"].copy()
    subset["model"] = pd.Categorical(subset["model"], categories=order, ordered=True)
    subset = subset.dropna(subset=["model"]).sort_values("model")
    if subset.empty:
        return ChartSpec(
            id="coefficients",
            title="Коэффициенты abn_volume (HC1, 95% интервал)",
            kind="errorbar",
            x_label="Модель",
            y_label="Коэффициент",
            series=[],
        )
    coefs = [float(v) for v in subset["coef"]]
    errs = [1.96 * float(v) for v in subset["std_err"]]
    return ChartSpec(
        id="coefficients",
        title="Коэффициенты abn_volume (HC1, 95% интервал)",
        kind="errorbar",
        x=[str(v) for v in subset["model"]],
        x_label="Модель",
        y_label="Коэффициент",
        series=[
            ChartSeries(
                name="abn_volume",
                values=coefs,
                yerr_low=errs,
                yerr_high=errs,
            )
        ],
    )
