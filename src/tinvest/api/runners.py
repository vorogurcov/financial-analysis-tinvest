"""Execute one experiment and return summary + chart series."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
import tempfile
from typing import Any

from tinvest.analytics.correlation import CorrelationConfig, ReturnCorrelation
from tinvest.analytics.distribution import DistributionConfig, ReturnDistribution
from tinvest.analytics.hypotheses import HypothesisConfig, ReturnHypotheses
from tinvest.analytics.vol_forecast import VolForecastConfig, VolatilityForecast
from tinvest.analytics.volume import VolumeConfig, VolumeImpact
from tinvest.api.sample import SampleBundle, SampleKey, sample_cache
from tinvest.api.summaries import summary_to_dict
from tinvest.charts.spec import charts_to_dicts
from tinvest.common.paths import project_root
from tinvest.research.universe import universe_path


def run_experiment(
    *,
    experiment: str,
    token: str,
    params: dict[str, Any],
    sample: dict[str, Any] | None,
    on_phase: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    if sample is None:
        raise ValueError("sample parameters are required")
    key = SampleKey(
        lookback_days=int(sample.get("lookback_days", 1095)),
        min_bars=int(sample.get("min_bars", 252)),
        max_names=int(sample.get("max_names", 80)),
        return_jump=float(sample.get("return_jump", 0.5)),
        refresh=bool(sample.get("refresh", False)),
    )
    if on_phase is not None:
        on_phase("preparing_sample")
    bundle = sample_cache.get_or_build(token, key)
    if on_phase is not None:
        on_phase("running")

    runners = {
        "distribution": _run_distribution,
        "volume": _run_volume,
        "correlation": _run_correlation,
        "hypotheses": _run_hypotheses,
        "vol_forecast": _run_vol_forecast,
    }
    runner = runners.get(experiment)
    if runner is None:
        raise ValueError(f"Unknown experiment: {experiment}")
    return runner(bundle=bundle, params=params)


def _run_distribution(*, bundle: SampleBundle, params: dict[str, Any]) -> dict[str, Any]:
    del params
    with tempfile.TemporaryDirectory() as tmp:
        figures = Path(tmp) / "figures"
        figures.mkdir(parents=True, exist_ok=True)
        analysis = ReturnDistribution(
            DistributionConfig(
                figures_dir=figures,
                universe_path=universe_path(project_root()),
            )
        )
        analysis.prepare(bundle.panel)
        analysis.analyze()
        return {
            "summary": summary_to_dict(analysis.get_summary()),
            "charts": charts_to_dicts(analysis.chart_specs()),
            "message": None,
        }


def _run_volume(*, bundle: SampleBundle, params: dict[str, Any]) -> dict[str, Any]:
    analysis = VolumeImpact(
        VolumeConfig(
            horizon=int(params.get("horizon", 5)),
            volume_window=int(params.get("volume_window", 21)),
            min_rows=int(params.get("min_rows", 100)),
        )
    )
    analysis.prepare(bundle.panel)
    analysis.analyze()
    summary = analysis.get_summary()
    charts = analysis.chart_specs()
    message = None if summary.n_tickers else "Модели объёма не оценены: недостаточно строк."
    return {
        "summary": summary_to_dict(summary),
        "charts": charts_to_dicts(charts),
        "message": message,
    }


def _run_correlation(*, bundle: SampleBundle, params: dict[str, Any]) -> dict[str, Any]:
    analysis = ReturnCorrelation(
        CorrelationConfig(
            min_overlap=int(params.get("min_overlap", 60)),
            window=int(params.get("window", 63)),
            min_names_in_window=int(params.get("min_names_in_window", 20)),
            min_returns_in_window=int(params.get("min_returns_in_window", 50)),
            stress_threshold=float(params.get("stress_threshold", -0.02)),
            min_stress_days=int(params.get("min_stress_days", 15)),
            min_regression_rows=int(params.get("min_regression_rows", 100)),
        )
    )
    analysis.prepare(bundle.panel, bundle.market)
    analysis.analyze()
    return {
        "summary": summary_to_dict(analysis.get_summary()),
        "charts": charts_to_dicts(analysis.chart_specs()),
        "message": None,
    }


def _run_hypotheses(*, bundle: SampleBundle, params: dict[str, Any]) -> dict[str, Any]:
    analysis = ReturnHypotheses(
        HypothesisConfig(
            ljung_lags=int(params.get("ljung_lags", 10)),
            arch_lags=int(params.get("arch_lags", 5)),
            rv_window=int(params.get("rv_window", 21)),
            hac_lags=int(params.get("hac_lags", 21)),
            min_returns=int(params.get("min_returns", 200)),
            min_tickers=int(params.get("min_tickers", 15)),
            alpha=float(params.get("alpha", 0.05)),
        )
    )
    analysis.prepare(bundle.panel)
    analysis.analyze()
    summaries = analysis.get_summary()
    return {
        "summary": [asdict(item) for item in summaries],
        "charts": charts_to_dicts(analysis.chart_specs()),
        "message": None,
    }


def _run_vol_forecast(*, bundle: SampleBundle, params: dict[str, Any]) -> dict[str, Any]:
    analysis = VolatilityForecast(
        VolForecastConfig(
            horizon=int(params.get("horizon", 5)),
            quantile=float(params.get("quantile", 0.8)),
            min_past_labels=int(params.get("min_past_labels", 120)),
            train_fraction=float(params.get("train_fraction", 0.8)),
            min_tickers=int(params.get("min_tickers", 40)),
            min_market_days=int(params.get("min_market_days", 400)),
            max_null_return_share=float(params.get("max_null_return_share", 0.05)),
            min_median_labels=int(params.get("min_median_labels", 150)),
            min_test_dates=int(params.get("min_test_dates", 60)),
            auc_floor=float(params.get("auc_floor", 0.5)),
            auc_margin=float(params.get("auc_margin", 0.02)),
            random_state=int(params.get("random_state", 0)),
        )
    )
    analysis.prepare(bundle.panel, bundle.market)
    analysis.analyze()
    summary = analysis.get_summary()
    charts = analysis.chart_specs()
    message = None
    if not summary.gate_passed:
        message = "Прогноз пропущен: порог качества данных не пройден."
    elif summary.hgb_beats_baseline is False:
        message = "Модель не улучшает опору на уже наблюдаемую волатильность."
    return {
        "summary": summary_to_dict(summary),
        "charts": charts_to_dicts(charts),
        "message": message,
    }
