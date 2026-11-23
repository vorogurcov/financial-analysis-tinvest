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
) -> list[ChartSpec]:
    charts = [_heatmap(pearson, meta), _rolling(rolling)]
    joint = rolling.dropna(subset=["pair_corr", "market_rv"])
    if not joint.empty:
        charts.append(_regime_bars(regimes))
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
