"""Project and results path helpers."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

SPB = ZoneInfo("Europe/Moscow")


def project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file() and (parent / "src" / "tinvest").is_dir():
            return parent
    return Path.cwd()


def results_root() -> Path:
    return Path(__file__).resolve().parents[1] / "results"


def default_output_dir() -> Path:
    """``src/tinvest/results/DD_MM_YYYY_HH_MM`` in Saint Petersburg time."""
    stamp = datetime.now(SPB).strftime("%d_%m_%Y_%H_%M")
    return results_root() / stamp
