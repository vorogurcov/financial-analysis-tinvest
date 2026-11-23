"""In-memory research sample cache keyed by ResearchConfig fields."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import threading

from tinvest.research.config import ResearchConfig
from tinvest.research.panel import MarketFrame, PanelFrame, equal_weight_market, load_panel
from tinvest.research.universe import UniverseFrame, build_universe
from tinvest.common.paths import project_root

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SampleKey:
    lookback_days: int
    min_bars: int
    max_names: int
    return_jump: float
    refresh: bool


@dataclass
class SampleBundle:
    key: SampleKey
    universe: UniverseFrame
    panel: PanelFrame
    market: MarketFrame


class SampleCache:
    """Build universe/panel once and reuse until sample parameters change."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._bundle: SampleBundle | None = None

    def get_or_build(self, token: str, key: SampleKey) -> SampleBundle:
        with self._lock:
            if self._bundle is not None and self._bundle.key == key and not key.refresh:
                logger.info("Reusing cached research sample")
                return self._bundle

            root = project_root()
            config = ResearchConfig(
                lookback_days=key.lookback_days,
                min_bars=key.min_bars,
                max_names=key.max_names,
                return_jump=key.return_jump,
                refresh=key.refresh,
            )
            logger.info(
                "Building research sample lookback=%s max_names=%s refresh=%s",
                key.lookback_days,
                key.max_names,
                key.refresh,
            )
            universe = build_universe(token, config, root=root)
            panel = load_panel(root, return_jump=key.return_jump)
            market = equal_weight_market(panel)
            # Cache with refresh=False so a later identical request reuses candles.
            cached_key = SampleKey(
                lookback_days=key.lookback_days,
                min_bars=key.min_bars,
                max_names=key.max_names,
                return_jump=key.return_jump,
                refresh=False,
            )
            self._bundle = SampleBundle(
                key=cached_key,
                universe=universe,
                panel=panel,
                market=market,
            )
            return self._bundle


sample_cache = SampleCache()
