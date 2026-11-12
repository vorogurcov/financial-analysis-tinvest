"""Collect candles, then write a timestamped analysis folder."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import logging
import os
from pathlib import Path
from zoneinfo import ZoneInfo

os.environ.setdefault("MPLBACKEND", "Agg")

from dotenv import load_dotenv

from tinvest.analytics import LinearTrend, TrendConfig
from tinvest.client import CandleFrame, FetchConfig, fetch_candles
from tinvest.storage import candles_dir, read_candles, write_candles

logger = logging.getLogger(__name__)
SPB = ZoneInfo("Europe/Moscow")


def project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file() and (parent / "notebooks").is_dir():
            return parent
    return Path.cwd()


def results_root() -> Path:
    return Path(__file__).resolve().parent / "results"


def default_output_dir() -> Path:
    """``src/tinvest/results/DD_MM_YYYY_HH_MM`` in Saint Petersburg time."""
    stamp = datetime.now(SPB).strftime("%d_%m_%Y_%H_%M")
    return results_root() / stamp


def default_processed_dir(ticker: str) -> Path:
    return candles_dir(project_root(), ticker)


@dataclass(frozen=True)
class PipelineConfig:
    """Env-driven ticker window in, a new timestamped folder under ``src/tinvest/results`` out."""

    ticker: str = "SBER"
    lookback_days: int = 365
    processed: Path | None = None
    output: Path = field(default_factory=default_output_dir)
    refresh: bool = True

    def __post_init__(self) -> None:
        ticker = self.ticker.strip().upper()
        if not ticker:
            raise ValueError("ticker must be non-empty")
        object.__setattr__(self, "ticker", ticker)
        if self.lookback_days < 1:
            raise ValueError("lookback_days must be at least 1")
        processed = self.processed if self.processed is not None else default_processed_dir(ticker)
        object.__setattr__(self, "processed", Path(processed))
        object.__setattr__(self, "output", Path(self.output))


def load_settings(env_path: Path | None = None) -> PipelineConfig:
    """Read ``TINVEST_*`` from ``.env`` at the project root."""
    root = project_root()
    path = env_path if env_path is not None else root / ".env"
    if path.is_file():
        load_dotenv(path)
    ticker = (os.getenv("TINVEST_TICKER", "SBER") or "SBER").strip().upper() or "SBER"
    lookback = int(os.getenv("TINVEST_LOOKBACK_DAYS", "365") or "365")
    return PipelineConfig(ticker=ticker, lookback_days=lookback)


def collect(token: str, config: PipelineConfig | None = None) -> CandleFrame:
    """Download candles and write them under ``data/processed/<TICKER>/``.

    Analogous to dreambank ``scrape``: a separate collect stage that leaves files for ``run``.
    """
    settings = config if config is not None else load_settings()
    if not token.strip():
        raise ValueError("TINVEST_TOKEN is empty")

    parquet = settings.processed / "candles.parquet"
    if parquet.is_file() and not settings.refresh:
        logger.info("Using existing candles: %s", parquet)
        return read_candles(settings.processed)

    candles = fetch_candles(
        token,
        FetchConfig(ticker=settings.ticker, lookback_days=settings.lookback_days),
    )
    write_candles(candles, settings.processed)
    logger.info(
        "Saved %s candles for %s → %s",
        len(candles),
        settings.ticker,
        settings.processed,
    )
    return candles


def run(config: PipelineConfig | None = None) -> Path:
    """Read saved candles and write the trend table and figure into a timestamped folder."""
    settings = config if config is not None else load_settings()
    output = Path(settings.output)
    figures = output / "figures"
    output.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    candles = read_candles(settings.processed)
    logger.info(
        "Loaded %s candles for %s from %s",
        len(candles),
        settings.ticker,
        settings.processed,
    )

    analysis = LinearTrend(
        TrendConfig(
            ticker=settings.ticker,
            figure_path=figures / "01_trend.png",
        )
    )
    analysis.prepare(candles)
    result = analysis.analyze()
    summary = analysis.get_summary()
    result.to_csv(output / "trend_candles.csv", index=False, encoding="utf-8-sig")
    analysis.plot()

    logger.info(
        "Trend slope=%.4f (%s), period return=%.2f%%, candles=%s",
        summary.slope,
        summary.label,
        summary.period_return * 100,
        summary.candle_count,
    )
    logger.info("Wrote results: %s", output)
    return output


def main() -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
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

    settings = load_settings(env_path)
    collect(token, settings)
    run(settings)


if __name__ == "__main__":
    main()
