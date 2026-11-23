"""Chart series for the volatility-forecast experiment."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from tinvest.charts.spec import ChartSeries, ChartSpec

_CALIBRATION_BINS = 10


def vol_forecast_charts(
    *,
    gate_passed: bool,
    y_true: np.ndarray | None,
    scores: dict[str, np.ndarray | None],
    aucs: dict[str, float],
    importance: pd.DataFrame,
    probability: np.ndarray | None,
    calibration: str,
) -> list[ChartSpec]:
    if not gate_passed or y_true is None:
        return []
    charts = [_roc(y_true=y_true, scores=scores, aucs=aucs), _importance(importance)]
    if calibration == "ok" and probability is not None:
        charts.append(_calibration(y_true, probability))
    return charts


def _roc(
    *,
    y_true: np.ndarray,
    scores: dict[str, np.ndarray | None],
    aucs: dict[str, float],
) -> ChartSpec:
    series: list[ChartSeries] = [
        ChartSeries(name="диагональ", x=[0.0, 1.0], values=[0.0, 1.0]),
    ]
    for name in ("rv_21", "logit", "hgb"):
        score = scores.get(name)
        if score is None:
            continue
        fpr, tpr, _ = roc_curve(y_true, score)
        series.append(
            ChartSeries(
                name=f"{name} AUC={aucs[name]:.3f}",
                x=[float(v) for v in fpr],
                values=[float(v) for v in tpr],
            )
        )
    return ChartSpec(
        id="roc",
        title="ROC: прогноз повышенной волатильности",
        kind="line",
        x_label="FPR",
        y_label="TPR",
        series=series,
    )


def _importance(importance: pd.DataFrame) -> ChartSpec:
    if importance.empty:
        return ChartSpec(
            id="importance",
            title="Перестановочная важность HGB",
            kind="bar",
            categories=[],
            values=[],
            y_label="Среднее падение ROC-AUC",
        )
    ordered = importance.sort_values("importance_mean", ascending=True, kind="mergesort")
    return ChartSpec(
        id="importance",
        title="Перестановочная важность HGB",
        kind="bar",
        categories=[str(v) for v in ordered["feature"]],
        values=[float(v) for v in ordered["importance_mean"]],
        y_label="Среднее падение ROC-AUC",
        horizontal=True,
    )


def _calibration(y_true: np.ndarray, probability: np.ndarray) -> ChartSpec:
    edges = np.linspace(0.0, 1.0, _CALIBRATION_BINS + 1)
    centers: list[float] = []
    rates: list[float] = []
    for left, right in zip(edges[:-1], edges[1:], strict=True):
        if right == 1.0:
            mask = (probability >= left) & (probability <= right)
        else:
            mask = (probability >= left) & (probability < right)
        if not np.any(mask):
            continue
        centers.append(float((left + right) / 2.0))
        rates.append(float(y_true[mask].mean()))
    return ChartSpec(
        id="calibration",
        title="Калибровка HGB",
        kind="line",
        x_label="Предсказанная вероятность",
        y_label="Доля положительных",
        series=[
            ChartSeries(name="идеал", x=[0.0, 1.0], values=[0.0, 1.0]),
            ChartSeries(
                name="HGB",
                x=centers,
                values=rates,
            ),
        ],
    )
