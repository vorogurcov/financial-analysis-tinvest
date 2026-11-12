"""Persist candle frames as CSV and parquet, and read parquet back."""

from pathlib import Path

import pandas as pd

from tinvest.client.frames import CandleFrame


def candles_dir(root: Path | str, ticker: str) -> Path:
    """Return ``{root}/data/processed/{TICKER}`` for the given repository root."""
    return Path(root) / "data" / "processed" / ticker.strip().upper()


def write_candles(frame: CandleFrame, directory: Path | str) -> Path:
    """Write ``candles.csv`` (utf-8-sig) and ``candles.parquet`` into ``directory``."""
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    frame.to_csv(folder / "candles.csv", index=False, encoding="utf-8-sig")
    frame.to_parquet(folder / "candles.parquet", index=False)
    return folder


def read_candles(directory: Path | str) -> CandleFrame:
    """Read ``candles.parquet`` and rebuild the typed candle frame."""
    path = Path(directory) / "candles.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"Candle parquet not found: {path}")
    frame = pd.read_parquet(path)
    if "time" in frame.columns:
        frame["time"] = pd.to_datetime(frame["time"], utc=True)
    if "volume" in frame.columns:
        frame["volume"] = frame["volume"].astype("int64")
    for column in ("open", "high", "low", "close"):
        if column in frame.columns:
            frame[column] = frame[column].astype("float64")
    return CandleFrame.build(frame)
