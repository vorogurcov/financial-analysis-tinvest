"""Collect the research universe and run distribution, volume, correlation, hypotheses, forecast."""

from __future__ import annotations

import logging
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg", force=True)

from dotenv import load_dotenv

from tinvest.analytics.correlation import CorrelationConfig, ReturnCorrelation
from tinvest.analytics.distribution import DistributionConfig, ReturnDistribution
from tinvest.analytics.hypotheses import HypothesisConfig, ReturnHypotheses
from tinvest.analytics.vol_forecast import VolForecastConfig, VolatilityForecast
from tinvest.analytics.volume import VolumeConfig, VolumeImpact
from tinvest.common.paths import project_root
from tinvest.research.config import load_research_settings
from tinvest.research.panel import equal_weight_market, load_panel
from tinvest.research.universe import build_universe, universe_path

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    root = project_root()
    env_path = root / ".env"
    if not env_path.is_file():
        raise FileNotFoundError(
            f"Missing {env_path}. Copy .env.example to .env and set TINVEST_TOKEN."
        )
    load_dotenv(env_path)
    token = os.getenv("TINVEST_TOKEN", "").strip()
    if not token:
        raise ValueError("TINVEST_TOKEN is empty. Set it in .env.")

    settings = load_research_settings(env_path)
    output = Path(settings.output)
    figures = output / "figures"
    output.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    build_universe(token, settings, root=root)
    panel = load_panel(root, return_jump=settings.return_jump)

    analysis = ReturnDistribution(
        DistributionConfig(
            figures_dir=figures,
            universe_path=universe_path(root),
        )
    )
    analysis.prepare(panel)
    tables = analysis.analyze()
    summary = analysis.get_summary()
    analysis.plot()

    tables.metrics.to_csv(output / "task1_instrument_metrics.csv", index=False, encoding="utf-8-sig")
    tables.spearman.to_csv(output / "task1_spearman.csv", index=False, encoding="utf-8-sig")
    tables.group_tests.to_csv(output / "task1_group_tests.csv", index=False, encoding="utf-8-sig")
    tables.regression.to_csv(output / "task1_vol_regression.csv", index=False, encoding="utf-8-sig")

    logger.info(
        "Тикеров=%s, медиана годовой волатильности=%.4f, "
        "группа с наибольшей медианой=%s, R² базовой регрессии=%.4f",
        summary.n_tickers,
        summary.median_ann_vol,
        summary.highest_vol_sector_group,
        summary.base_r_squared,
    )

    volume = VolumeImpact(
        VolumeConfig(
            figure_path=figures / "task2_abs_return_vs_volume.png",
            lead_vol_figure_path=figures / "task2_lead_vol.png",
            coefficients_figure_path=figures / "task2_coefficients.png",
        )
    )
    volume.prepare(panel)
    volume_tables = volume.analyze()
    volume_summary = volume.get_summary()
    volume.plot()

    volume_tables.coefficients.to_csv(
        output / "task2_coefficients.csv", index=False, encoding="utf-8-sig"
    )
    volume_tables.spearman.to_csv(
        output / "task2_spearman_by_ticker.csv", index=False, encoding="utf-8-sig"
    )
    volume_tables.excluded.to_csv(
        output / "task2_excluded.csv", index=False, encoding="utf-8-sig"
    )

    if volume_summary.n_tickers == 0:
        logger.info("Модели объёма не оценены")
    else:
        logger.info(
            "Объём: coef_A=%.4f, coef_B=%.4f, тикеров=%s, медиана Спирмена=%.4f",
            volume_summary.coef_a,
            volume_summary.coef_b,
            volume_summary.n_tickers,
            volume_summary.spearman_median,
        )

    market = equal_weight_market(panel)
    correlation = ReturnCorrelation(
        CorrelationConfig(
            heatmap_path=figures / "task3_corr_heatmap.png",
            rolling_path=figures / "task3_rolling_corr.png",
            regimes_path=figures / "task3_regime_bars.png",
        )
    )
    correlation.prepare(panel, market)
    corr_tables = correlation.analyze()
    corr_summary = correlation.get_summary()
    correlation.plot()

    corr_tables.pairs.to_csv(output / "task3_pairs.csv", index=False, encoding="utf-8-sig")
    corr_tables.rolling.to_csv(output / "task3_rolling.csv", index=False, encoding="utf-8-sig")
    corr_tables.regimes.to_csv(output / "task3_regimes.csv", index=False, encoding="utf-8-sig")
    corr_tables.sector_corr.to_csv(
        output / "task3_sector_corr.csv", index=False, encoding="utf-8-sig"
    )
    corr_tables.regression.to_csv(
        output / "task3_corr_on_vol.csv", index=False, encoding="utf-8-sig"
    )

    logger.info(
        "Корреляции: средняя попарная=%.4f, b=%.4f, pair_corr low/high=%.4f/%.4f, пар=%s",
        corr_summary.mean_pair_corr if corr_summary.mean_pair_corr is not None else float("nan"),
        corr_summary.coef_b if corr_summary.coef_b is not None else float("nan"),
        (
            corr_summary.mean_pair_corr_low_vol
            if corr_summary.mean_pair_corr_low_vol is not None
            else float("nan")
        ),
        (
            corr_summary.mean_pair_corr_high_vol
            if corr_summary.mean_pair_corr_high_vol is not None
            else float("nan")
        ),
        corr_summary.n_pairs,
    )

    hypotheses = ReturnHypotheses(
        HypothesisConfig(
            acf_path=figures / "task4_acf.png",
            reject_share_path=figures / "task4_reject_share.png",
            weekday_path=figures / "task4_weekday.png",
        )
    )
    hypotheses.prepare(panel)
    hyp_tables = hypotheses.analyze()
    hyp_summaries = hypotheses.get_summary()
    hypotheses.plot()

    hyp_tables.tests.to_csv(output / "task4_tests.csv", index=False, encoding="utf-8-sig")
    hyp_tables.summary.to_csv(output / "task4_summary.csv", index=False, encoding="utf-8-sig")

    for item in hyp_summaries:
        logger.info(
            "%s, n=%s, median_p=%s, reject_share=%s",
            item.test,
            item.n,
            f"{item.median_p:.4f}" if item.median_p is not None else "nan",
            f"{item.reject_share:.4f}" if item.reject_share is not None else "nan",
        )

    figures_dir = figures
    forecast = VolatilityForecast(
        VolForecastConfig(
            output_dir=output,
            roc_path=figures_dir / "task5_roc.png",
            importance_path=figures_dir / "task5_importance.png",
            calibration_path=figures_dir / "task5_calibration.png",
        )
    )
    forecast.prepare(panel, market)
    forecast.analyze()
    forecast_summary = forecast.get_summary()
    forecast.plot()

    if not forecast_summary.gate_passed:
        logger.info(
            "Прогноз волатильности пропущен: порог данных не пройден (каталог=%s)",
            forecast_summary.output_dir,
        )
    else:
        logger.info(
            "Прогноз волатильности: AUC constant=%.3f rv_21=%.3f logit=%.3f hgb=%.3f, "
            "logit_beats_baseline=%s, hgb_beats_baseline=%s, positive_rate=%.3f",
            forecast_summary.auc_constant if forecast_summary.auc_constant is not None else float("nan"),
            forecast_summary.auc_rv_21 if forecast_summary.auc_rv_21 is not None else float("nan"),
            forecast_summary.auc_logit if forecast_summary.auc_logit is not None else float("nan"),
            forecast_summary.auc_hgb if forecast_summary.auc_hgb is not None else float("nan"),
            forecast_summary.logit_beats_baseline,
            forecast_summary.hgb_beats_baseline,
            (
                forecast_summary.positive_rate_test
                if forecast_summary.positive_rate_test is not None
                else float("nan")
            ),
        )
        if not forecast_summary.hgb_beats_baseline:
            logger.info(
                "модель не улучшает опору на уже наблюдаемую волатильность"
            )

    logger.info("Wrote results: %s", output)


if __name__ == "__main__":
    main()
