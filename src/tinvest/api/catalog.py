"""Experiment catalog and form field schemas for the web UI."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

FieldType = Literal["number", "integer", "boolean", "string"]


@dataclass(frozen=True)
class FieldSpec:
    name: str
    label: str
    type: FieldType
    default: Any
    min: float | int | None = None
    max: float | int | None = None
    step: float | int | None = None
    required: bool = True


SAMPLE_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("lookback_days", "Глубина истории, дни", "integer", 1095, min=30, max=5000),
    FieldSpec("min_bars", "Минимум баров", "integer", 252, min=2, max=5000),
    FieldSpec("max_names", "Максимум тикеров", "integer", 80, min=1, max=500),
    FieldSpec("return_jump", "Порог скачка доходности", "number", 0.5, min=0.01, max=5.0, step=0.01),
    FieldSpec("refresh", "Обновить свечи с API", "boolean", False),
)


EXPERIMENTS: tuple[dict[str, Any], ...] = (
    {
        "id": "distribution",
        "title": "Распределение доходностей",
        "description": "Волатильность, нормальность и связь с характеристиками инструментов.",
        "needs_sample": True,
        "fields": [],
    },
    {
        "id": "volume",
        "title": "Влияние объёма",
        "description": "Аномальный оборот против модуля доходности и будущей волатильности.",
        "needs_sample": True,
        "fields": [
            asdict(FieldSpec("horizon", "Горизонт RV", "integer", 5, min=2, max=60)),
            asdict(FieldSpec("volume_window", "Окно оборота", "integer", 21, min=2, max=120)),
            asdict(FieldSpec("min_rows", "Минимум строк на тикер", "integer", 100, min=1, max=5000)),
        ],
    },
    {
        "id": "correlation",
        "title": "Корреляции",
        "description": "Попарные корреляции и их зависимость от рыночной волатильности.",
        "needs_sample": True,
        "fields": [
            asdict(FieldSpec("min_overlap", "Минимум перекрытия", "integer", 60, min=2, max=1000)),
            asdict(FieldSpec("window", "Скользящее окно", "integer", 63, min=2, max=500)),
            asdict(FieldSpec("min_names_in_window", "Минимум имён в окне", "integer", 20, min=2, max=500)),
            asdict(
                FieldSpec("min_returns_in_window", "Минимум доходностей в окне", "integer", 50, min=2, max=500)
            ),
            asdict(
                FieldSpec(
                    "stress_threshold",
                    "Порог стресс-дня",
                    "number",
                    -0.02,
                    min=-0.2,
                    max=0.0,
                    step=0.001,
                )
            ),
            asdict(FieldSpec("min_stress_days", "Минимум стресс-дней", "integer", 15, min=1, max=500)),
            asdict(
                FieldSpec("min_regression_rows", "Минимум строк регрессии", "integer", 100, min=1, max=5000)
            ),
        ],
    },
    {
        "id": "hypotheses",
        "title": "Гипотезы",
        "description": "Автокорреляция, ARCH, день недели и устойчивость волатильности.",
        "needs_sample": True,
        "fields": [
            asdict(FieldSpec("ljung_lags", "Лаги Ljung–Box", "integer", 10, min=1, max=40)),
            asdict(FieldSpec("arch_lags", "Лаги ARCH", "integer", 5, min=1, max=40)),
            asdict(FieldSpec("rv_window", "Окно RV", "integer", 21, min=2, max=120)),
            asdict(FieldSpec("hac_lags", "Лаги HAC", "integer", 21, min=1, max=120)),
            asdict(FieldSpec("min_returns", "Минимум доходностей", "integer", 200, min=1, max=5000)),
            asdict(FieldSpec("min_tickers", "Минимум тикеров", "integer", 15, min=1, max=500)),
            asdict(FieldSpec("alpha", "Уровень значимости", "number", 0.05, min=0.001, max=0.2, step=0.001)),
        ],
    },
    {
        "id": "vol_forecast",
        "title": "Прогноз волатильности",
        "description": "Классификация повышенной 5-дневной реализованной волатильности.",
        "needs_sample": True,
        "fields": [
            asdict(FieldSpec("horizon", "Горизонт", "integer", 5, min=2, max=60)),
            asdict(FieldSpec("quantile", "Квантиль метки", "number", 0.8, min=0.5, max=0.95, step=0.01)),
            asdict(FieldSpec("min_past_labels", "Минимум прошлых меток", "integer", 120, min=1, max=5000)),
            asdict(
                FieldSpec("train_fraction", "Доля обучения", "number", 0.8, min=0.5, max=0.95, step=0.01)
            ),
            asdict(FieldSpec("min_tickers", "Минимум тикеров", "integer", 40, min=1, max=500)),
            asdict(FieldSpec("min_market_days", "Минимум рыночных дней", "integer", 400, min=1, max=5000)),
            asdict(
                FieldSpec(
                    "max_null_return_share",
                    "Макс. доля пустых доходностей",
                    "number",
                    0.05,
                    min=0.0,
                    max=0.5,
                    step=0.01,
                )
            ),
            asdict(FieldSpec("min_median_labels", "Медиана меток", "integer", 150, min=1, max=5000)),
            asdict(FieldSpec("min_test_dates", "Минимум дат теста", "integer", 60, min=1, max=5000)),
            asdict(FieldSpec("auc_floor", "Пол AUC", "number", 0.5, min=0.0, max=1.0, step=0.01)),
            asdict(FieldSpec("auc_margin", "Запас AUC", "number", 0.02, min=0.0, max=0.2, step=0.01)),
            asdict(FieldSpec("random_state", "Случайное зерно", "integer", 0, min=0, max=10_000)),
        ],
    },
)


def list_experiments() -> list[dict[str, Any]]:
    return [
        {
            **item,
            "sample_fields": [asdict(field) for field in SAMPLE_FIELDS] if item["needs_sample"] else [],
        }
        for item in EXPERIMENTS
    ]


def get_experiment(experiment_id: str) -> dict[str, Any] | None:
    for item in list_experiments():
        if item["id"] == experiment_id:
            return item
    return None
