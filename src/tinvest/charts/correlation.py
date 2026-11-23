"""Chart series for the return-correlation experiment."""

from __future__ import annotations

import pandas as pd

from tinvest.charts.spec import ChartSeries, ChartSpec


def correlation_charts(
    *,
    pearson: pd.DataFrame | None,
    meta: pd.DataFrame | None,
    rolling: pd.DataFrame,
    regimes: pd.DataFrame,
    compare_daily: pd.DataFrame | None = None,
    compare_cumulative: pd.DataFrame | None = None,
    compare_rolling: pd.DataFrame | None = None,
    compare_ticker: str | None = None,
    compare_alpha: float | None = None,
    compare_beta: float | None = None,
) -> list[ChartSpec]:
    charts = [_heatmap(pearson, meta), _rolling(rolling)]
    joint = rolling.dropna(subset=["pair_corr", "market_rv"])
    if not joint.empty:
        charts.append(_regime_bars(regimes))
    if compare_ticker:
        charts.append(_compare_scatter(compare_daily, compare_ticker, compare_alpha, compare_beta))
        charts.append(_compare_cumulative(compare_cumulative, compare_ticker))
        charts.append(_compare_rolling_beta(compare_rolling, compare_ticker))
    return charts


def _heatmap(pearson: pd.DataFrame | None, meta: pd.DataFrame | None) -> ChartSpec:
    if pearson is None or meta is None or pearson.empty:
        return ChartSpec(
            id="heatmap",
            title="Попарная корреляция Пирсона лог-доходностей",
            kind="heatmap",
            matrix=[],
            x_labels=[],
            y_labels=[],
        )
    order_meta = meta.sort_values(["sector", "ticker"], kind="mergesort")
    order = order_meta["ticker"].astype(str).tolist()
    matrix = pearson.reindex(index=order, columns=order)
    labels = [f"{row.sector} | {row.ticker}" for row in order_meta.itertuples(index=False)]
    values: list[list[float | None]] = []
    for row in matrix.to_numpy(dtype=float):
        values.append([None if pd.isna(v) else float(v) for v in row])
    return ChartSpec(
        id="heatmap",
        title="Попарная корреляция Пирсона лог-доходностей",
        kind="heatmap",
        matrix=values,
        x_labels=labels,
        y_labels=labels,
    )


def _rolling(rolling: pd.DataFrame) -> ChartSpec:
    if rolling.empty:
        return ChartSpec(
            id="rolling",
            title="Скользящая корреляция и волатильность рынка",
            kind="dual_line",
            series=[],
        )
    times = [pd.Timestamp(t).isoformat() for t in pd.to_datetime(rolling["time"], utc=True)]
    return ChartSpec(
        id="rolling",
        title="Скользящая корреляция и волатильность рынка",
        kind="dual_line",
        x=times,
        x_label="Дата",
        y_label="Средняя попарная корреляция",
        y_label_right="Рыночная RV (годовая)",
        series=[
            ChartSeries(
                name="Средняя попарная корреляция",
                axis="left",
                values=[_f(v) for v in rolling["pair_corr"]],
            ),
            ChartSeries(
                name="Рыночная волатильность",
                axis="right",
                values=[_f(v) for v in rolling["market_rv"]],
            ),
        ],
    )


def _regime_bars(regimes: pd.DataFrame) -> ChartSpec:
    order = ["low_vol", "mid_vol", "high_vol"]
    labels = ["Низкая RV", "Средняя RV", "Высокая RV"]
    subset = regimes.loc[regimes["regime"].isin(order)].copy()
    subset["regime"] = pd.Categorical(subset["regime"], categories=order, ordered=True)
    subset = subset.sort_values("regime")
    return ChartSpec(
        id="regimes",
        title="Средняя попарная корреляция по терцилям рыночной RV",
        kind="bar",
        categories=labels,
        values=[_f(v) for v in subset["mean_pair_corr"]],
        y_label="Средняя pair_corr",
    )


def _compare_scatter(
    daily: pd.DataFrame | None,
    ticker: str,
    alpha: float | None,
    beta: float | None,
) -> ChartSpec:
    title = f"Дневные доходности: {ticker} и рынок"
    if daily is None or daily.empty:
        return ChartSpec(id="compare_scatter", title=title, kind="scatter", series=[])
    x_values = [_f(value) for value in daily["market_return"]]
    y_values = [_f(value) for value in daily["stock_return"]]
    finite_x = [value for value in x_values if value is not None]
    reference: list[dict[str, float]] | None = None
    if alpha is not None and beta is not None and finite_x:
        x0 = min(finite_x)
        x1 = max(finite_x)
        reference = [
            {"x": x0, "y": alpha + beta * x0},
            {"x": x1, "y": alpha + beta * x1},
        ]
    return ChartSpec(
        id="compare_scatter",
        title=title,
        kind="scatter",
        x_label="Доходность рынка",
        y_label=f"Доходность {ticker}",
        series=[ChartSeries(name=ticker, x=x_values, values=y_values)],
        reference_line=reference,
    )


def _compare_cumulative(cumulative: pd.DataFrame | None, ticker: str) -> ChartSpec:
    title = f"Накопленная лог-доходность: {ticker} и рынок"
    if cumulative is None or cumulative.empty:
        return ChartSpec(id="compare_cumulative", title=title, kind="dual_line", series=[])
    times = [pd.Timestamp(value).isoformat() for value in pd.to_datetime(cumulative["time"], utc=True)]
    return ChartSpec(
        id="compare_cumulative",
        title=title,
        kind="dual_line",
        x=times,
        x_label="Дата",
        y_label="Накопленная лог-доходность",
        series=[
            ChartSeries(name=ticker, axis="left", values=[_f(value) for value in cumulative["stock_cum"]]),
            ChartSeries(
                name="Рынок",
                axis="left",
                values=[_f(value) for value in cumulative["market_cum"]],
            ),
        ],
    )


def _compare_rolling_beta(rolling: pd.DataFrame | None, ticker: str) -> ChartSpec:
    title = f"Скользящие корреляция и бета: {ticker}"
    if rolling is None or rolling.empty:
        return ChartSpec(id="compare_rolling", title=title, kind="dual_line", series=[])
    times = [pd.Timestamp(value).isoformat() for value in pd.to_datetime(rolling["time"], utc=True)]
    return ChartSpec(
        id="compare_rolling",
        title=title,
        kind="dual_line",
        x=times,
        x_label="Дата",
        y_label="Корреляция",
        y_label_right="Бета",
        series=[
            ChartSeries(name="Корреляция", axis="left", values=[_f(value) for value in rolling["corr"]]),
            ChartSeries(name="Бета", axis="right", values=[_f(value) for value in rolling["beta"]]),
        ],
    )


def _f(value: object) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if pd.isna(number):
        return None
    return number
