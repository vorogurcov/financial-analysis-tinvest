"""Build the TQBR equity universe catalog and download candles."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import logging
from pathlib import Path
import time
from zoneinfo import ZoneInfo

import pandas as pd
from t_tech.invest import Client, InstrumentStatus, ShareType
from t_tech.invest.utils import quotation_to_decimal

from tinvest.client import CandleFrame, FetchConfig, fetch_candles
from tinvest.common.frame import TypedFrame
from tinvest.common.paths import project_root
from tinvest.research.config import ResearchConfig
from tinvest.storage import candles_dir, read_candles, write_candles

logger = logging.getLogger(__name__)
MOSCOW = ZoneInfo("Europe/Moscow")

EXCLUDE_FILTER = "filter"
EXCLUDE_FETCH_ERROR = "fetch_error"
EXCLUDE_TOO_FEW_BARS = "too_few_bars"
EXCLUDE_ZERO_TURNOVER = "zero_turnover"
EXCLUDE_OVER_CAP = "over_cap"

RETRY_PAUSES_S = (1.0, 2.0, 4.0)
BETWEEN_INSTRUMENTS_S = 0.2


class UniverseFrame(TypedFrame):
    """Catalog of TQBR equity candidates and inclusion flags."""

    ticker: str
    figi: str
    name: str
    class_code: str
    currency: str
    country_of_risk: str
    lot: int
    sector: str
    share_type: str
    issue_size: int
    nominal: float
    div_yield_flag: bool
    liquidity_flag: bool
    short_enabled_flag: bool
    ipo_date: datetime
    included: bool
    exclude_reason: str


@dataclass(frozen=True)
class CandidateEval:
    """Per-ticker bar count and median turnover after candle cleaning."""

    ticker: str
    figi: str
    n_bars: int
    median_turnover_rub: float


def processed_dir(root: Path | None = None) -> Path:
    base = root if root is not None else project_root()
    return Path(base) / "data" / "processed"


def universe_path(root: Path | None = None) -> Path:
    return processed_dir(root) / "universe.parquet"


def fetch_errors_path(root: Path | None = None) -> Path:
    return processed_dir(root) / "fetch_errors.csv"


def moscow_today() -> datetime.date:
    return datetime.now(MOSCOW).date()


def clean_ohlcv(frame: pd.DataFrame, today: datetime.date | None = None) -> pd.DataFrame:
    """Drop bad prices, today's unfinished bar, and duplicate timestamps."""
    if frame.empty:
        return frame.copy()
    working = frame.copy()
    working["time"] = pd.to_datetime(working["time"], utc=True)
    bad = (
        (working["close"] <= 0)
        | (working["high"] <= 0)
        | (working["low"] <= 0)
        | (working["high"] < working["low"])
    )
    working = working.loc[~bad].copy()
    cutoff = today if today is not None else moscow_today()
    local_dates = working["time"].dt.tz_convert(MOSCOW).dt.date
    working = working.loc[local_dates != cutoff].copy()
    working = working.sort_values("time").drop_duplicates("time", keep="last")
    return working.reset_index(drop=True)


def median_turnover_rub(candles: pd.DataFrame, lot: int) -> float:
    """Median ruble turnover over bars with positive turnover."""
    if candles.empty:
        return 0.0
    turnover = candles["volume"].astype(float) * float(lot) * candles["close"].astype(float)
    positive = turnover[turnover > 0]
    if positive.empty:
        return 0.0
    return float(positive.median())


def _share_type_label(share_type: ShareType) -> str | None:
    if share_type == ShareType.SHARE_TYPE_COMMON:
        return "common"
    if share_type == ShareType.SHARE_TYPE_PREFERRED:
        return "preferred"
    return None


def passes_card_filter(share: object) -> bool:
    """Return True when a share card matches the TQBR research filter."""
    share_type = _share_type_label(share.share_type)
    if share_type is None:
        return False
    return (
        share.class_code == "TQBR"
        and str(share.currency).lower() == "rub"
        and share.country_of_risk == "RU"
        and share.otc_flag is False
        and share.blocked_tca_flag is False
        and share.for_qual_investor_flag is False
        and share.api_trade_available_flag is True
        and share.buy_available_flag is True
        and share.sell_available_flag is True
        and int(share.issue_size) > 0
    )


def card_row(share: object) -> dict[str, object]:
    sector = str(share.sector or "").strip() or "не указан"
    share_type = _share_type_label(share.share_type)
    assert share_type is not None
    ipo = share.ipo_date
    if ipo is None:
        ipo = pd.Timestamp("1970-01-01", tz="UTC").to_pydatetime()
    else:
        ipo = pd.Timestamp(ipo)
        if ipo.tzinfo is None:
            ipo = ipo.tz_localize("UTC")
        else:
            ipo = ipo.tz_convert("UTC")
        ipo = ipo.to_pydatetime()
    return {
        "ticker": str(share.ticker).strip().upper(),
        "figi": str(share.figi),
        "name": str(share.name),
        "class_code": str(share.class_code),
        "currency": str(share.currency).lower(),
        "country_of_risk": str(share.country_of_risk),
        "lot": int(share.lot),
        "sector": sector,
        "share_type": share_type,
        "issue_size": int(share.issue_size),
        "nominal": float(quotation_to_decimal(share.nominal)),
        "div_yield_flag": bool(share.div_yield_flag),
        "liquidity_flag": bool(share.liquidity_flag),
        "short_enabled_flag": bool(share.short_enabled_flag),
        "ipo_date": ipo,
        "included": False,
        "exclude_reason": "",
    }


def select_included(
    evaluations: list[CandidateEval],
    *,
    min_bars: int,
    max_names: int,
) -> dict[str, str]:
    """Map ticker → exclude_reason (empty string means included).

    Candidates must already have been fetched successfully. Reasons are
    ``too_few_bars``, ``zero_turnover``, ``over_cap``, or ``""``.
    """
    reasons: dict[str, str] = {}
    eligible: list[CandidateEval] = []
    for item in evaluations:
        if item.n_bars < min_bars:
            reasons[item.ticker] = EXCLUDE_TOO_FEW_BARS
            continue
        if item.median_turnover_rub <= 0:
            reasons[item.ticker] = EXCLUDE_ZERO_TURNOVER
            continue
        eligible.append(item)
    eligible.sort(key=lambda row: (-row.median_turnover_rub, row.ticker))
    for index, item in enumerate(eligible):
        if index < max_names:
            reasons[item.ticker] = ""
        else:
            reasons[item.ticker] = EXCLUDE_OVER_CAP
    return reasons


def _load_or_fetch_candles(
    token: str,
    *,
    ticker: str,
    figi: str,
    lookback_days: int,
    refresh: bool,
    root: Path,
) -> CandleFrame:
    directory = candles_dir(root, ticker)
    parquet = directory / "candles.parquet"
    if parquet.is_file() and not refresh:
        logger.info("Using existing candles: %s", parquet)
        return read_candles(directory)

    last_error: Exception | None = None
    attempts = 1 + len(RETRY_PAUSES_S)
    for attempt in range(attempts):
        try:
            candles = fetch_candles(
                token,
                FetchConfig(ticker=ticker, lookback_days=lookback_days, figi=figi),
            )
            write_candles(candles, directory)
            return candles
        except Exception as exc:  # noqa: BLE001 — one ticker must not stop the sweep
            last_error = exc
            if attempt < len(RETRY_PAUSES_S):
                time.sleep(RETRY_PAUSES_S[attempt])
    assert last_error is not None
    raise last_error


def build_universe(
    token: str,
    config: ResearchConfig | None = None,
    *,
    root: Path | None = None,
) -> UniverseFrame:
    """Filter TQBR shares, fetch candles, and write ``universe.parquet``."""
    settings = config if config is not None else ResearchConfig()
    base = Path(root) if root is not None else project_root()
    out_dir = processed_dir(base)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not settings.refresh:
        logger.info(
            "refresh=false: existing candle parquet files are reused; "
            "changing lookback_days does not extend saved history"
        )

    with Client(token) as client:
        response = client.instruments.shares(
            instrument_status=InstrumentStatus.INSTRUMENT_STATUS_BASE
        )
        shares = list(response.instruments)

    cards = [card_row(share) for share in shares if passes_card_filter(share)]
    cards.sort(key=lambda row: str(row["ticker"]))

    evaluations: list[CandidateEval] = []
    fetch_errors: list[dict[str, str]] = []
    reasons: dict[str, str] = {}
    today = moscow_today()

    for card in cards:
        ticker = str(card["ticker"])
        figi = str(card["figi"])
        lot = int(card["lot"])
        parquet = candles_dir(base, ticker) / "candles.parquet"
        reused = parquet.is_file() and not settings.refresh
        try:
            candles = _load_or_fetch_candles(
                token,
                ticker=ticker,
                figi=figi,
                lookback_days=settings.lookback_days,
                refresh=settings.refresh,
                root=base,
            )
        except Exception as exc:  # noqa: BLE001 — one ticker must not stop the sweep
            fetch_errors.append({"ticker": ticker, "figi": figi, "error": str(exc)})
            reasons[ticker] = EXCLUDE_FETCH_ERROR
            candles = None

        if not reused:
            time.sleep(BETWEEN_INSTRUMENTS_S)

        if candles is None:
            continue

        cleaned = clean_ohlcv(pd.DataFrame(candles), today=today)
        turnover = median_turnover_rub(cleaned, lot)
        evaluations.append(
            CandidateEval(
                ticker=ticker,
                figi=figi,
                n_bars=len(cleaned),
                median_turnover_rub=turnover,
            )
        )

    selection = select_included(
        evaluations,
        min_bars=settings.min_bars,
        max_names=settings.max_names,
    )
    reasons.update(selection)

    rows: list[dict[str, object]] = []
    for card in cards:
        ticker = str(card["ticker"])
        reason = reasons.get(ticker, EXCLUDE_FETCH_ERROR)
        row = dict(card)
        row["included"] = reason == ""
        row["exclude_reason"] = reason
        rows.append(row)

    if fetch_errors:
        errors = pd.DataFrame(fetch_errors, columns=["ticker", "figi", "error"])
        errors.to_csv(fetch_errors_path(base), index=False, encoding="utf-8-sig")
    elif fetch_errors_path(base).is_file():
        fetch_errors_path(base).unlink()

    frame = pd.DataFrame(rows)
    if frame.empty:
        catalog = UniverseFrame.blank()
    else:
        frame["ipo_date"] = pd.to_datetime(frame["ipo_date"], utc=True)
        frame["lot"] = frame["lot"].astype("int64")
        frame["issue_size"] = frame["issue_size"].astype("int64")
        frame["nominal"] = frame["nominal"].astype("float64")
        for flag in ("div_yield_flag", "liquidity_flag", "short_enabled_flag", "included"):
            frame[flag] = frame[flag].astype(bool)
        catalog = UniverseFrame.build(frame)

    path = universe_path(base)
    catalog.to_parquet(path, index=False)
    logger.info(
        "Wrote universe: %s (%s included of %s)",
        path,
        int(catalog["included"].sum()) if not catalog.empty else 0,
        len(catalog),
    )
    return catalog


def read_universe(root: Path | None = None) -> UniverseFrame:
    path = universe_path(root)
    if not path.is_file():
        raise FileNotFoundError(f"Universe parquet not found: {path}")
    frame = pd.read_parquet(path)
    frame["ipo_date"] = pd.to_datetime(frame["ipo_date"], utc=True)
    frame["lot"] = frame["lot"].astype("int64")
    frame["issue_size"] = frame["issue_size"].astype("int64")
    frame["nominal"] = frame["nominal"].astype("float64")
    for flag in ("div_yield_flag", "liquidity_flag", "short_enabled_flag", "included"):
        frame[flag] = frame[flag].astype(bool)
    for column in (
        "ticker",
        "figi",
        "name",
        "class_code",
        "currency",
        "country_of_risk",
        "sector",
        "share_type",
        "exclude_reason",
    ):
        frame[column] = frame[column].astype(str)
    return UniverseFrame.build(frame)
