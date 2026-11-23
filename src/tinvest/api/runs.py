"""Single-flight run store for experiment jobs."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import threading
import traceback
from typing import Any
from uuid import uuid4

from tinvest.api.runners import run_experiment


@dataclass
class RunRecord:
    id: str
    experiment: str
    status: str
    created_at: str
    updated_at: str
    error: str | None = None
    summary: dict[str, Any] | list[Any] | None = None
    charts: list[dict[str, Any]] = field(default_factory=list)
    message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "experiment": self.experiment,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "error": self.error,
            "summary": self.summary,
            "charts": self.charts,
            "message": self.message,
        }


class RunManager:
    """At most one active experiment run at a time."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active_id: str | None = None
        self._runs: dict[str, RunRecord] = {}

    def start(
        self,
        *,
        experiment: str,
        token: str,
        params: dict[str, Any],
        sample: dict[str, Any] | None,
    ) -> RunRecord:
        with self._lock:
            if self._active_id is not None:
                active = self._runs.get(self._active_id)
                if active is not None and active.status in {
                    "queued",
                    "preparing_sample",
                    "running",
                }:
                    raise RuntimeError("Another experiment is already running")
            now = _now()
            run_id = str(uuid4())
            record = RunRecord(
                id=run_id,
                experiment=experiment,
                status="queued",
                created_at=now,
                updated_at=now,
            )
            self._runs[run_id] = record
            self._active_id = run_id

        thread = threading.Thread(
            target=self._execute,
            kwargs={
                "run_id": run_id,
                "experiment": experiment,
                "token": token,
                "params": params,
                "sample": sample,
            },
            daemon=True,
        )
        thread.start()
        return record

    def get(self, run_id: str) -> RunRecord | None:
        with self._lock:
            return self._runs.get(run_id)

    def _execute(
        self,
        *,
        run_id: str,
        experiment: str,
        token: str,
        params: dict[str, Any],
        sample: dict[str, Any] | None,
    ) -> None:
        def on_phase(phase: str) -> None:
            self._update(run_id, status=phase)

        try:
            self._update(run_id, status="preparing_sample")
            result = run_experiment(
                experiment=experiment,
                token=token,
                params=params,
                sample=sample,
                on_phase=on_phase,
            )
            self._update(
                run_id,
                status="done",
                summary=result["summary"],
                charts=result["charts"],
                message=result.get("message"),
                error=None,
            )
        except Exception as exc:  # noqa: BLE001 — surface any failure to the UI
            self._update(
                run_id,
                status="error",
                error=f"{exc}\n{traceback.format_exc()}",
            )
        finally:
            with self._lock:
                if self._active_id == run_id:
                    self._active_id = None

    def _update(self, run_id: str, **fields: Any) -> None:
        with self._lock:
            record = self._runs[run_id]
            for key, value in fields.items():
                setattr(record, key, value)
            record.updated_at = _now()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


run_manager = RunManager()
