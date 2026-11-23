"""FastAPI application for the experiments web UI."""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from tinvest.api.catalog import get_experiment, list_experiments
from tinvest.api.runs import run_manager
from tinvest.common.paths import project_root
from tinvest.research.universe import read_universe, universe_path


class RunRequest(BaseModel):
    experiment: str
    params: dict[str, Any] = Field(default_factory=dict)
    sample: dict[str, Any] | None = None


def _load_token() -> str:
    root = project_root()
    env_path = root / ".env"
    if env_path.is_file():
        load_dotenv(env_path)
    token = os.getenv("TINVEST_TOKEN", "").strip()
    if not token:
        raise HTTPException(status_code=500, detail="TINVEST_TOKEN is empty. Set it in .env.")
    return token


def create_app() -> FastAPI:
    app = FastAPI(title="Tinvest Experiments", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/experiments")
    def experiments() -> list[dict[str, Any]]:
        return list_experiments()

    @app.get("/api/tickers")
    def tickers() -> list[dict[str, str]]:
        path = universe_path(project_root())
        if not path.is_file():
            return []
        universe = pd.DataFrame(read_universe(project_root()))
        included = universe.loc[universe["included"].astype(bool)].sort_values(
            "ticker",
            kind="mergesort",
        )
        return [
            {
                "ticker": str(row.ticker),
                "name": str(row.name),
                "sector": str(row.sector),
            }
            for row in included.itertuples(index=False)
        ]

    @app.post("/api/runs")
    def start_run(body: RunRequest) -> dict[str, str]:
        experiment = get_experiment(body.experiment)
        if experiment is None:
            raise HTTPException(status_code=404, detail=f"Unknown experiment: {body.experiment}")
        if body.sample is None:
            raise HTTPException(status_code=400, detail="sample parameters are required")
        try:
            record = run_manager.start(
                experiment=body.experiment,
                token=_load_token(),
                params=body.params,
                sample=body.sample,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"run_id": record.id}

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        record = run_manager.get(run_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return record.to_dict()

    return app


app = create_app()
