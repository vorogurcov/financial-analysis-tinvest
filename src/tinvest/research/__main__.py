"""Collect the research universe and run task-1 distribution analysis."""

from __future__ import annotations

import logging
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg", force=True)

from dotenv import load_dotenv

from tinvest.analytics.distribution import DistributionConfig, ReturnDistribution
from tinvest.common.paths import project_root
from tinvest.research.config import load_research_settings
from tinvest.research.panel import load_panel
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
    logger.info("Wrote results: %s", output)


if __name__ == "__main__":
    main()
