"""Run the local experiments API: ``python -m tinvest.api``."""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")


def main() -> None:
    import uvicorn

    uvicorn.run(
        "tinvest.api.app:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    main()
