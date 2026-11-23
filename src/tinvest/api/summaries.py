"""Convert typed experiment summaries into JSON-safe dicts."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def summary_to_dict(summary: object) -> dict[str, Any] | list[Any]:
    if isinstance(summary, list):
        return [summary_to_dict(item) for item in summary]
    if is_dataclass(summary) and not isinstance(summary, type):
        raw = asdict(summary)
        return {key: _jsonable(value) for key, value in raw.items()}
    if isinstance(summary, dict):
        return {key: _jsonable(value) for key, value in summary.items()}
    raise TypeError(f"Unsupported summary type: {type(summary)!r}")


def _jsonable(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value
