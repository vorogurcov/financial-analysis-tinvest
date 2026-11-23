"""Chart series for the temporal-hypotheses experiment."""

from __future__ import annotations

from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from tinvest.charts.spec import ChartSeries, ChartSpec
from tinvest.charts.stats import boxplot_stats

_MOSCOW = ZoneInfo("Europe/Moscow")
_TEST_ORDER = ("lb_return", "lb_square", "arch_lm", "weekday", "rv_persist")
_WEEKDAY_LABELS = ("Пн", "Вт", "Ср", "Чт", "Пт")


def hypotheses_charts(
    series: dict[str, pd.DataFrame],
    tests: pd.DataFrame,
    *,
    ljung_lags: int,
    min_returns: int,
    min_tickers: int,
    alpha: float,
) -> list[ChartSpec]:
    return [
        _acf_chart(series, lags=ljung_lags, min_returns=min_returns),
        _reject_share_chart(tests, alpha=alpha, min_tickers=min_tickers),
        _weekday_chart(series, min_returns=min_returns),
    ]


def _acf_at_lags(values: np.ndarray, lags: int) -> np.ndarray:
    n = values.size
    centered = values - values.mean()
    denom = float(np.dot(centered, centered))
    if denom == 0.0 or n <= lags:
        return np.full(lags, np.nan, dtype=float)
    out = np.empty(lags, dtype=float)
    for lag in range(1, lags + 1):
        out[lag - 1] = float(np.dot(centered[lag:], centered[:-lag]) / denom)
    return out


def _acf_chart(
    series: dict[str, pd.DataFrame],
    *,
    lags: int,
    min_returns: int,
) -> ChartSpec:
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

    lag_axis = list(range(1, lags + 1))
    if not r_rows:
        return ChartSpec(
            id="acf",
            title="Медианная АКФ доходностей и квадратов",
            kind="line",
            x=lag_axis,
            x_label="Лаг",
            y_label="АКФ",
            series=[],
        )

    chart_series: list[ChartSeries] = []
    for rows, name in ((r_rows, "Доходности"), (sq_rows, "Квадраты доходностей")):
        stacked = np.vstack(rows)
        median = np.nanmedian(stacked, axis=0)
        q25 = np.nanpercentile(stacked, 25, axis=0)
        q75 = np.nanpercentile(stacked, 75, axis=0)
        chart_series.append(
            ChartSeries(
                name=name,
                values=[float(v) for v in median],
                band_low=[float(v) for v in q25],
                band_high=[float(v) for v in q75],
            )
        )
    return ChartSpec(
        id="acf",
        title="Медианная АКФ доходностей и квадратов",
        kind="line",
        x=lag_axis,
        x_label="Лаг",
        y_label="АКФ",
        series=chart_series,
    )


def _reject_share_chart(
    tests: pd.DataFrame,
    *,
    alpha: float,
    min_tickers: int,
) -> ChartSpec:
    raw_shares: list[float | None] = []
    holm_shares: list[float | None] = []
    for test in _TEST_ORDER:
        subset = tests.loc[(tests["test"] == test) & tests["p_value"].notna()]
        if subset.empty:
            raw_shares.append(None)
            holm_shares.append(None)
            continue
        raw_shares.append(float((subset["p_value"] < alpha).mean()))
        if len(subset) >= min_tickers and subset["p_holm"].notna().all():
            holm_shares.append(float((subset["p_holm"] < alpha).mean()))
        else:
            holm_shares.append(None)
    return ChartSpec(
        id="reject_share",
        title="Доли отклонений гипотез до и после Холма",
        kind="grouped_bar",
        categories=list(_TEST_ORDER),
        y_label="Доля отклонений",
        groups=[
            {"name": "Сырая доля p < 0,05", "values": raw_shares},
            {"name": "Доля после Холма", "values": holm_shares},
        ],
    )


def _weekday_chart(series: dict[str, pd.DataFrame], *, min_returns: int) -> ChartSpec:
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

    boxes = [
        boxplot_stats(np.asarray(buckets[day], dtype=float), label=_WEEKDAY_LABELS[day], show_outliers=False)
        for day in range(5)
        if buckets[day]
    ]
    return ChartSpec(
        id="weekday",
        title="Стандартизованный модуль доходности по дням недели",
        kind="boxplot",
        x_label="День недели",
        y_label="Стандартизованный |r|",
        boxes=boxes,
    )
