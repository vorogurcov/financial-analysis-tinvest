"""Forecast elevated five-day realized volatility with logistic and HGB classifiers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import floor, log as math_log
from pathlib import Path
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler

from tinvest.analytics.base import Analysis
from tinvest.analytics.distribution import assign_sector_groups
from tinvest.analytics.volume import add_volume_features
from tinvest.common.frame import TypedFrame
from tinvest.research.panel import MarketFrame, PanelFrame

logger = logging.getLogger(__name__)

_SQRT_252 = float(np.sqrt(252.0))
_LN2 = math_log(2.0)
_VOLUME_WINDOW = 21
_VOLUME_MIN_PERIODS = 15
_PARKINSON_WINDOW = 21
_SMA_WINDOW = 21
_MARKET_RV_WINDOW = 21
_RV_21 = 21
_RV_63 = 63
_PERM_REPEATS = 5
_CALIBRATION_BINS = 10
_MIN_POSITIVES_CALIBRATION = 30

_NUMERIC_FEATURES = (
    "rv_5",
    "rv_21",
    "rv_63",
    "log_return",
    "abs_return",
    "abn_volume",
    "parkinson_21",
    "range",
    "sma_gap",
    "market_rv_21",
)
_FEATURE_COLUMNS = _NUMERIC_FEATURES + ("sector_group",)
_MODEL_CODES = ("constant", "rv_21", "logit", "hgb")

_SUCCESS_FILES = (
    "task5_metrics.csv",
    "task5_coefficients.csv",
    "task5_importance.csv",
    "task5_predictions.csv",
)
_SUCCESS_FIGURES = (
    "task5_roc.png",
    "task5_importance.png",
    "task5_calibration.png",
)


class ForecastSampleFrame(TypedFrame):
    """Model sample rows with features, label, and future volatility."""

    ticker: str
    time: datetime
    y: int
    future_rv_5: float
    rv_5: float
    rv_21: float
    rv_63: float
    log_return: float
    abs_return: float
    abn_volume: float
    parkinson_21: float
    range: float
    sma_gap: float
    market_rv_21: float
    sector_group: str


class ForecastMetricsFrame(TypedFrame):
    """Test-set metrics for each scoring rule."""

    model: str
    roc_auc: float
    average_precision: float
    brier: float
    positive_rate_test: float


class ForecastCoefficientsFrame(TypedFrame):
    """Logit coefficients on the original feature scale."""

    feature: str
    coef: float


class ForecastImportanceFrame(TypedFrame):
    """Permutation importance of HGB features on the test set."""

    feature: str
    importance_mean: float
    importance_std: float


class ForecastPredictionsFrame(TypedFrame):
    """Test-set predictions for logit and HGB."""

    ticker: str
    time: datetime
    y: int
    p_logit: float
    p_hgb: float
    rv_21: float
    future_rv_5: float


class ForecastGateFrame(TypedFrame):
    """Data-quality gate checks before training."""

    check: str
    value: float
    threshold: float
    passed: bool


@dataclass(frozen=True)
class VolForecastConfig:
    """Windows, gates, and output paths for the volatility forecast."""

    horizon: int = 5
    quantile: float = 0.8
    min_past_labels: int = 120
    train_fraction: float = 0.8
    min_tickers: int = 40
    min_market_days: int = 400
    max_null_return_share: float = 0.05
    min_median_labels: int = 150
    min_test_dates: int = 60
    auc_floor: float = 0.5
    auc_margin: float = 0.02
    random_state: int = 0
    output_dir: Path | None = None
    roc_path: Path | None = None
    importance_path: Path | None = None
    calibration_path: Path | None = None

    def __post_init__(self) -> None:
        if self.horizon < 2:
            raise ValueError("horizon must be at least 2")
        if not 0.0 < self.quantile < 1.0:
            raise ValueError("quantile must be in (0, 1)")
        if self.min_past_labels < 1:
            raise ValueError("min_past_labels must be at least 1")
        if not 0.0 < self.train_fraction < 1.0:
            raise ValueError("train_fraction must be in (0, 1)")
        for name in (
            "output_dir",
            "roc_path",
            "importance_path",
            "calibration_path",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, Path(value))


@dataclass(frozen=True)
class VolForecastSummary:
    """Compact log line and gate values for the volatility forecast."""

    n_tickers: float
    n_market_days: float
    null_return_share: float
    median_labels: float
    n_test_dates: float
    both_classes: float
    gate_passed: bool
    auc_constant: float | None
    auc_rv_21: float | None
    auc_logit: float | None
    auc_hgb: float | None
    logit_beats_baseline: bool | None
    hgb_beats_baseline: bool | None
    positive_rate_test: float | None
    calibration: str
    output_dir: Path | None


@dataclass(frozen=True)
class VolForecastTables:
    """All tables produced by the volatility forecast."""

    metrics: ForecastMetricsFrame
    coefficients: ForecastCoefficientsFrame
    importance: ForecastImportanceFrame
    predictions: ForecastPredictionsFrame
    gate: ForecastGateFrame


def beats_baseline(
    model_auc: float,
    baseline_auc: float,
    *,
    auc_floor: float = 0.5,
    auc_margin: float = 0.02,
) -> bool:
    """True when the model clears the floor and beats the rv_21 score by the margin."""
    return bool(model_auc > auc_floor and model_auc >= baseline_auc + auc_margin)


def time_split_dates(
    dates: pd.DatetimeIndex | np.ndarray | list,
    *,
    train_fraction: float = 0.8,
) -> tuple[pd.Timestamp, pd.DatetimeIndex, pd.DatetimeIndex]:
    """Return cut date, train dates, and test dates for sorted unique dates."""
    unique = pd.DatetimeIndex(pd.to_datetime(pd.unique(dates))).sort_values()
    n = len(unique)
    if n < 2:
        raise ValueError("need at least two distinct dates for a time split")
    cut_index = floor(train_fraction * n)
    if cut_index <= 0:
        cut_index = 1
    if cut_index >= n:
        cut_index = n - 1
    cut = unique[cut_index]
    train = unique[:cut_index]
    test = unique[cut_index:]
    return pd.Timestamp(cut), train, test


def rolling_rv(log_return: pd.Series, window: int) -> pd.Series:
    """Annualized std over a full window of non-empty returns."""
    valid = log_return.notna().astype(float)
    full = valid.rolling(window=window, min_periods=window).sum() == float(window)
    rv = log_return.rolling(window=window, min_periods=window).std(ddof=1) * _SQRT_252
    return rv.where(full)


def build_label_threshold(
    future_rv: pd.Series,
    *,
    horizon: int,
    quantile: float,
    min_past_labels: int,
) -> tuple[pd.Series, pd.Series]:
    """Causal high-vol labels: threshold uses only fully completed past windows."""
    values = future_rv.astype(float).to_numpy()
    n = len(values)
    thresholds = np.full(n, np.nan, dtype=float)
    labels = np.full(n, np.nan, dtype=float)
    for index in range(n):
        current = values[index]
        if not np.isfinite(current):
            continue
        end = index - horizon
        if end <= 0:
            continue
        past = values[:end]
        past = past[np.isfinite(past)]
        if past.size < min_past_labels:
            continue
        threshold = float(np.quantile(past, quantile))
        thresholds[index] = threshold
        labels[index] = 1.0 if current >= threshold else 0.0
    return (
        pd.Series(labels, index=future_rv.index, dtype="float64"),
        pd.Series(thresholds, index=future_rv.index, dtype="float64"),
    )


class VolatilityForecast(Analysis[VolForecastTables]):
    """Prepare features, gate the sample, fit baselines and classifiers."""

    def __init__(self, config: VolForecastConfig) -> None:
        super().__init__(config)
        self.config: VolForecastConfig = config
        self._panel: PanelFrame | None = None
        self._market: MarketFrame | None = None
        self._sample: pd.DataFrame | None = None
        self._labeled: pd.DataFrame | None = None
        self._summary: VolForecastSummary | None = None
        self._train: pd.DataFrame | None = None
        self._test: pd.DataFrame | None = None
        self._p_rv21: np.ndarray | None = None
        self._p_logit: np.ndarray | None = None
        self._p_hgb: np.ndarray | None = None
        self._score_rv21: np.ndarray | None = None
        self._score_logit: np.ndarray | None = None
        self._score_hgb: np.ndarray | None = None

    def prepare(  # type: ignore[override]
        self,
        panel: PanelFrame,
        market: MarketFrame | None = None,
    ) -> ForecastSampleFrame:
        if panel.empty:
            raise ValueError("выборка пуста")
        if market is None:
            raise ValueError("MarketFrame is required")
        self._panel = panel
        self._market = market
        labeled, sample = _prepare_sample(
            panel,
            market,
            horizon=self.config.horizon,
            quantile=self.config.quantile,
            min_past_labels=self.config.min_past_labels,
        )
        self._labeled = labeled
        self._sample = sample
        self._working = _sample_frame(sample) if not sample.empty else ForecastSampleFrame.blank()
        self._results = None
        self._summary = None
        self._train = None
        self._test = None
        self._p_rv21 = None
        self._p_logit = None
        self._p_hgb = None
        self._score_rv21 = None
        self._score_logit = None
        self._score_hgb = None
        return self._working  # type: ignore[return-value]

    def analyze(self) -> VolForecastTables:
        if self._panel is None or self._market is None or self._sample is None:
            raise RuntimeError("Call prepare() before analyze()")
        if self._labeled is None:
            raise RuntimeError("Call prepare() before analyze()")

        gate_df, gate_passed, gate_values = _evaluate_gate(
            panel=self._panel,
            market=self._market,
            labeled=self._labeled,
            sample=self._sample,
            config=self.config,
        )
        gate = _gate_frame(gate_df)
        output_dir = self.config.output_dir

        if not gate_passed:
            _write_gate_outputs(output_dir, gate_df, skipped=True)
            _remove_success_outputs(self.config)
            tables = VolForecastTables(
                metrics=ForecastMetricsFrame.blank(),
                coefficients=ForecastCoefficientsFrame.blank(),
                importance=ForecastImportanceFrame.blank(),
                predictions=ForecastPredictionsFrame.blank(),
                gate=gate,
            )
            self._results = tables
            self._summary = VolForecastSummary(
                n_tickers=gate_values["n_tickers"],
                n_market_days=gate_values["n_market_days"],
                null_return_share=gate_values["null_return_share"],
                median_labels=gate_values["median_labels"],
                n_test_dates=gate_values["n_test_dates"],
                both_classes=gate_values["both_classes"],
                gate_passed=False,
                auc_constant=None,
                auc_rv_21=None,
                auc_logit=None,
                auc_hgb=None,
                logit_beats_baseline=None,
                hgb_beats_baseline=None,
                positive_rate_test=None,
                calibration="skipped_gate",
                output_dir=output_dir,
            )
            return tables

        train, test, _cut = _split_sample(self._sample, train_fraction=self.config.train_fraction)
        self._train = train
        self._test = test

        fitted = _fit_models(
            train,
            test,
            random_state=self.config.random_state,
        )
        metrics_df = fitted["metrics"]
        coefficients_df = fitted["coefficients"]
        importance_df = fitted["importance"]
        predictions_df = fitted["predictions"]
        self._p_rv21 = fitted["p_rv21"]
        self._p_logit = fitted["p_logit"]
        self._p_hgb = fitted["p_hgb"]
        self._score_rv21 = fitted["score_rv21"]
        self._score_logit = fitted["score_logit"]
        self._score_hgb = fitted["score_hgb"]

        metrics = _metrics_frame(metrics_df)
        coefficients = _coefficients_frame(coefficients_df)
        importance = _importance_frame(importance_df)
        predictions = _predictions_frame(predictions_df)

        auc_by_model = {
            str(row["model"]): float(row["roc_auc"]) for _, row in metrics_df.iterrows()
        }
        auc_rv21 = auc_by_model["rv_21"]
        auc_logit = auc_by_model["logit"]
        auc_hgb = auc_by_model["hgb"]
        logit_beats = beats_baseline(
            auc_logit,
            auc_rv21,
            auc_floor=self.config.auc_floor,
            auc_margin=self.config.auc_margin,
        )
        hgb_beats = beats_baseline(
            auc_hgb,
            auc_rv21,
            auc_floor=self.config.auc_floor,
            auc_margin=self.config.auc_margin,
        )
        positive_rate = float(metrics_df["positive_rate_test"].iloc[0])
        n_positives = int(test["y"].sum())
        calibration = (
            "ok" if n_positives >= _MIN_POSITIVES_CALIBRATION else "skipped_few_positives"
        )

        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            _write_gate_outputs(output_dir, gate_df, skipped=False)
            metrics.to_csv(output_dir / "task5_metrics.csv", index=False, encoding="utf-8-sig")
            coefficients.to_csv(
                output_dir / "task5_coefficients.csv", index=False, encoding="utf-8-sig"
            )
            importance.to_csv(
                output_dir / "task5_importance.csv", index=False, encoding="utf-8-sig"
            )
            predictions.to_csv(
                output_dir / "task5_predictions.csv", index=False, encoding="utf-8-sig"
            )
            skipped_path = output_dir / "task5_skipped.txt"
            if skipped_path.is_file():
                skipped_path.unlink()

        tables = VolForecastTables(
            metrics=metrics,
            coefficients=coefficients,
            importance=importance,
            predictions=predictions,
            gate=gate,
        )
        self._results = tables
        self._summary = VolForecastSummary(
            n_tickers=gate_values["n_tickers"],
            n_market_days=gate_values["n_market_days"],
            null_return_share=gate_values["null_return_share"],
            median_labels=gate_values["median_labels"],
            n_test_dates=gate_values["n_test_dates"],
            both_classes=gate_values["both_classes"],
            gate_passed=True,
            auc_constant=auc_by_model["constant"],
            auc_rv_21=auc_rv21,
            auc_logit=auc_logit,
            auc_hgb=auc_hgb,
            logit_beats_baseline=logit_beats,
            hgb_beats_baseline=hgb_beats,
            positive_rate_test=positive_rate,
            calibration=calibration,
            output_dir=output_dir,
        )
        return tables

    def get_summary(self) -> VolForecastSummary:
        if self._summary is None:
            raise RuntimeError("Call analyze() before get_summary()")
        return self._summary

    def plot(self) -> Figure:
        if self._results is None or self._summary is None:
            raise RuntimeError("Call analyze() before plot()")

        if not self._summary.gate_passed or self._test is None:
            _remove_figure(self.config.roc_path)
            _remove_figure(self.config.importance_path)
            _remove_figure(self.config.calibration_path)
            fig = plt.figure()
            fig.clf()
            return fig

        y_true = self._test["y"].to_numpy(dtype=int)
        roc_fig = _plot_roc(
            y_true=y_true,
            scores={
                "rv_21": self._score_rv21,
                "logit": self._score_logit,
                "hgb": self._score_hgb,
            },
            aucs={
                "rv_21": self._summary.auc_rv_21 or 0.0,
                "logit": self._summary.auc_logit or 0.0,
                "hgb": self._summary.auc_hgb or 0.0,
            },
        )
        _save_figure(roc_fig, self.config.roc_path)

        importance = pd.DataFrame(self._results.importance)
        imp_fig = _plot_importance(importance)
        _save_figure(imp_fig, self.config.importance_path)

        if self._summary.calibration == "ok" and self._p_hgb is not None:
            cal_fig = _plot_calibration(y_true, self._p_hgb)
            _save_figure(cal_fig, self.config.calibration_path)
        else:
            _remove_figure(self.config.calibration_path)

        return roc_fig


def _prepare_sample(
    panel: PanelFrame,
    market: MarketFrame,
    *,
    horizon: int,
    quantile: float,
    min_past_labels: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    volume = pd.DataFrame(
        add_volume_features(panel, horizon=horizon, volume_window=_VOLUME_WINDOW)
    )
    market_rv = _market_rv_series(market, window=_MARKET_RV_WINDOW)

    ticker_sector = (
        pd.DataFrame(panel)[["ticker", "sector"]]
        .drop_duplicates(subset=["ticker"], keep="first")
        .copy()
    )
    ticker_sector["sector_group"] = assign_sector_groups(ticker_sector["sector"].astype(str))
    sector_map = ticker_sector.set_index("ticker")["sector_group"].astype(str).to_dict()

    parts: list[pd.DataFrame] = []
    for ticker, chunk in volume.groupby("ticker", sort=True):
        ordered = chunk.sort_values("time", kind="mergesort").copy()
        log_return = ordered["log_return"].astype(float)
        ordered["rv_21"] = rolling_rv(log_return, _RV_21)
        ordered["rv_63"] = rolling_rv(log_return, _RV_63)
        ordered["parkinson_21"] = _parkinson(ordered, window=_PARKINSON_WINDOW)
        close = ordered["close"].astype(float)
        sma = close.rolling(window=_SMA_WINDOW, min_periods=_SMA_WINDOW).mean()
        ordered["sma_gap"] = close / sma - 1.0
        ordered["sector_group"] = str(sector_map.get(str(ticker), "прочие"))
        y, _threshold = build_label_threshold(
            ordered["future_rv_5"],
            horizon=horizon,
            quantile=quantile,
            min_past_labels=min_past_labels,
        )
        ordered["y"] = y
        parts.append(ordered)

    if not parts:
        empty = pd.DataFrame(
            columns=["ticker", "time", "y", "future_rv_5", *_FEATURE_COLUMNS]
        )
        return empty, empty

    frame = pd.concat(parts, ignore_index=True)
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    frame = frame.merge(market_rv, on="time", how="left")

    labeled = frame.loc[frame["y"].notna()].copy()
    labeled["y"] = labeled["y"].astype(int)

    model_cols = ["ticker", "time", "y", "future_rv_5", *_FEATURE_COLUMNS]
    sample = labeled.dropna(subset=list(_FEATURE_COLUMNS)).loc[:, model_cols].copy()
    sample["y"] = sample["y"].astype(int)
    sample["sector_group"] = sample["sector_group"].astype(str)
    return labeled, sample.reset_index(drop=True)


def _market_rv_series(market: MarketFrame, *, window: int) -> pd.DataFrame:
    if market.empty:
        return pd.DataFrame(columns=["time", "market_rv_21"])
    working = pd.DataFrame(market).copy()
    working["time"] = pd.to_datetime(working["time"], utc=True)
    working = working.sort_values("time", kind="mergesort")
    returns = working["market_return"].astype(float)
    valid = returns.notna().astype(float)
    full = valid.rolling(window=window, min_periods=window).sum() == float(window)
    rv = returns.rolling(window=window, min_periods=window).std(ddof=1) * _SQRT_252
    working["market_rv_21"] = rv.where(full)
    return working.loc[:, ["time", "market_rv_21"]]


def _parkinson(chunk: pd.DataFrame, *, window: int) -> pd.Series:
    high = chunk["high"].astype(float)
    low = chunk["low"].astype(float)
    ratio = np.log(high / low)
    daily = (ratio * ratio) / (4.0 * _LN2)
    valid = daily.notna().astype(float)
    full = valid.rolling(window=window, min_periods=window).sum() == float(window)
    mean = daily.rolling(window=window, min_periods=window).mean()
    return np.sqrt(mean).where(full) * _SQRT_252


def _evaluate_gate(
    *,
    panel: PanelFrame,
    market: MarketFrame,
    labeled: pd.DataFrame,
    sample: pd.DataFrame,
    config: VolForecastConfig,
) -> tuple[pd.DataFrame, bool, dict[str, float]]:
    n_tickers = float(pd.DataFrame(panel)["ticker"].nunique())
    n_market_days = float(len(pd.DataFrame(market)))
    returns = pd.DataFrame(panel)["log_return"]
    null_share = float(returns.isna().mean()) if len(returns) else 1.0

    if labeled.empty:
        median_labels = 0.0
    else:
        counts = labeled.groupby("ticker", sort=False).size()
        median_labels = float(counts.median()) if len(counts) else 0.0

    if sample.empty or sample["time"].nunique() < 2:
        n_test_dates = 0.0
        both_classes = 0.0
        train = sample.iloc[0:0]
        test = sample.iloc[0:0]
    else:
        train, test, _cut = _split_sample(sample, train_fraction=config.train_fraction)
        n_test_dates = float(test["time"].nunique())
        both_classes = float(
            train["y"].nunique() >= 2 and test["y"].nunique() >= 2 and len(train) > 0 and len(test) > 0
        )

    checks = [
        ("n_tickers", n_tickers, float(config.min_tickers), n_tickers >= config.min_tickers),
        (
            "n_market_days",
            n_market_days,
            float(config.min_market_days),
            n_market_days >= config.min_market_days,
        ),
        (
            "null_return_share",
            null_share,
            float(config.max_null_return_share),
            null_share <= config.max_null_return_share,
        ),
        (
            "median_labels",
            median_labels,
            float(config.min_median_labels),
            median_labels >= config.min_median_labels,
        ),
        (
            "n_test_dates",
            n_test_dates,
            float(config.min_test_dates),
            n_test_dates >= config.min_test_dates,
        ),
        ("both_classes", both_classes, 1.0, both_classes >= 1.0),
    ]
    rows = [
        {
            "check": name,
            "value": value,
            "threshold": threshold,
            "passed": bool(passed),
        }
        for name, value, threshold, passed in checks
    ]
    gate_df = pd.DataFrame(rows)
    gate_passed = bool(gate_df["passed"].all())
    values = {
        "n_tickers": n_tickers,
        "n_market_days": n_market_days,
        "null_return_share": null_share,
        "median_labels": median_labels,
        "n_test_dates": n_test_dates,
        "both_classes": both_classes,
    }
    return gate_df, gate_passed, values


def _split_sample(
    sample: pd.DataFrame,
    *,
    train_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    dates = pd.DatetimeIndex(sample["time"].drop_duplicates().sort_values())
    cut, train_dates, test_dates = time_split_dates(dates, train_fraction=train_fraction)
    train = sample.loc[sample["time"].isin(train_dates)].copy()
    test = sample.loc[sample["time"].isin(test_dates)].copy()
    return train, test, cut


def _fit_models(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    random_state: int,
) -> dict[str, object]:
    y_train = train["y"].to_numpy(dtype=int)
    y_test = test["y"].to_numpy(dtype=int)
    positive_rate = float(y_test.mean()) if len(y_test) else float("nan")

    p_constant = float(y_train.mean()) if len(y_train) else 0.0
    score_constant = np.full(len(test), p_constant, dtype=float)
    p_constant_arr = score_constant.copy()

    score_rv21 = test["rv_21"].to_numpy(dtype=float)
    rv_model = LogisticRegression(max_iter=500, random_state=random_state)
    rv_model.fit(train[["rv_21"]].to_numpy(dtype=float), y_train)
    p_rv21 = rv_model.predict_proba(test[["rv_21"]].to_numpy(dtype=float))[:, 1]

    x_train_num = train.loc[:, list(_NUMERIC_FEATURES)].to_numpy(dtype=float)
    x_test_num = test.loc[:, list(_NUMERIC_FEATURES)].to_numpy(dtype=float)
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train_num)
    x_test_scaled = scaler.transform(x_test_num)

    train_sectors = train["sector_group"].astype(str)
    test_sectors = test["sector_group"].astype(str)
    sector_levels = sorted(train_sectors.unique().tolist())
    train_dummy = _sector_dummies(train_sectors, sector_levels)
    test_dummy = _sector_dummies(test_sectors, sector_levels)

    x_train_logit = np.hstack([x_train_scaled, train_dummy])
    x_test_logit = np.hstack([x_test_scaled, test_dummy])
    logit = LogisticRegression(
        class_weight="balanced",
        max_iter=500,
        random_state=random_state,
    )
    logit.fit(x_train_logit, y_train)
    p_logit = logit.predict_proba(x_test_logit)[:, 1]
    score_logit = p_logit

    feature_names = list(_NUMERIC_FEATURES) + [f"sector_group_{level}" for level in sector_levels]
    coef_raw = logit.coef_.ravel()
    coef_out: list[dict[str, object]] = []
    for index, name in enumerate(feature_names):
        value = float(coef_raw[index])
        if index < len(_NUMERIC_FEATURES):
            scale = float(scaler.scale_[index])
            if scale != 0.0 and np.isfinite(scale):
                value = value / scale
        coef_out.append({"feature": name, "coef": value})
    coefficients_df = pd.DataFrame(coef_out)
    coefficients_df["abs_coef"] = coefficients_df["coef"].abs()
    coefficients_df = coefficients_df.sort_values(
        "abs_coef", ascending=False, kind="mergesort"
    ).drop(columns=["abs_coef"])

    train_cat, test_cat, cat_codes = _encode_sector_for_hgb(train_sectors, test_sectors)
    x_train_hgb = np.hstack([x_train_num, train_cat.reshape(-1, 1)])
    x_test_hgb = np.hstack([x_test_num, test_cat.reshape(-1, 1)])
    hgb_features = list(_NUMERIC_FEATURES) + ["sector_group"]
    hgb = HistGradientBoostingClassifier(
        max_depth=3,
        learning_rate=0.1,
        max_iter=200,
        random_state=random_state,
        class_weight="balanced",
        categorical_features=[len(_NUMERIC_FEATURES)],
    )
    hgb.fit(x_train_hgb, y_train)
    p_hgb = hgb.predict_proba(x_test_hgb)[:, 1]
    score_hgb = p_hgb

    perm = permutation_importance(
        hgb,
        x_test_hgb,
        y_test,
        n_repeats=_PERM_REPEATS,
        scoring="roc_auc",
        random_state=random_state,
    )
    importance_df = pd.DataFrame(
        {
            "feature": hgb_features,
            "importance_mean": perm.importances_mean.astype(float),
            "importance_std": perm.importances_std.astype(float),
        }
    )

    metrics_rows = [
        _metric_row("constant", y_test, score_constant, p_constant_arr, positive_rate),
        _metric_row("rv_21", y_test, score_rv21, p_rv21, positive_rate),
        _metric_row("logit", y_test, score_logit, p_logit, positive_rate),
        _metric_row("hgb", y_test, score_hgb, p_hgb, positive_rate),
    ]
    metrics_df = pd.DataFrame(metrics_rows)

    predictions_df = pd.DataFrame(
        {
            "ticker": test["ticker"].astype(str).to_numpy(),
            "time": pd.to_datetime(test["time"], utc=True),
            "y": y_test.astype(int),
            "p_logit": p_logit.astype(float),
            "p_hgb": p_hgb.astype(float),
            "rv_21": test["rv_21"].to_numpy(dtype=float),
            "future_rv_5": test["future_rv_5"].to_numpy(dtype=float),
        }
    )

    return {
        "metrics": metrics_df,
        "coefficients": coefficients_df.reset_index(drop=True),
        "importance": importance_df,
        "predictions": predictions_df,
        "p_rv21": p_rv21,
        "p_logit": p_logit,
        "p_hgb": p_hgb,
        "score_rv21": score_rv21,
        "score_logit": score_logit,
        "score_hgb": score_hgb,
        "scaler": scaler,
        "sector_levels": sector_levels,
        "cat_codes": cat_codes,
    }


def _metric_row(
    model: str,
    y_true: np.ndarray,
    score: np.ndarray,
    probability: np.ndarray,
    positive_rate: float,
) -> dict[str, object]:
    if model == "constant":
        auc = 0.5
    else:
        auc = float(roc_auc_score(y_true, score))
    return {
        "model": model,
        "roc_auc": auc,
        "average_precision": float(average_precision_score(y_true, score)),
        "brier": float(brier_score_loss(y_true, probability)),
        "positive_rate_test": positive_rate,
    }


def _sector_dummies(sectors: pd.Series, levels: list[str]) -> np.ndarray:
    if not levels:
        return np.zeros((len(sectors), 0), dtype=float)
    matrix = np.zeros((len(sectors), len(levels)), dtype=float)
    index_map = {level: index for index, level in enumerate(levels)}
    for row_index, value in enumerate(sectors.astype(str).to_numpy()):
        column = index_map.get(value)
        if column is not None:
            matrix[row_index, column] = 1.0
    return matrix


def _encode_sector_for_hgb(
    train_sectors: pd.Series,
    test_sectors: pd.Series,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    levels = sorted(train_sectors.astype(str).unique().tolist())
    codes = {level: float(index) for index, level in enumerate(levels)}
    train = np.array(
        [codes[str(value)] for value in train_sectors.astype(str)],
        dtype=float,
    )
    test = np.array(
        [codes.get(str(value), np.nan) for value in test_sectors.astype(str)],
        dtype=float,
    )
    return train, test, codes


def _write_gate_outputs(
    output_dir: Path | None,
    gate_df: pd.DataFrame,
    *,
    skipped: bool,
) -> None:
    if output_dir is None:
        return
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    gate_df.to_csv(destination / "task5_gate.csv", index=False, encoding="utf-8-sig")
    skipped_path = destination / "task5_skipped.txt"
    if skipped:
        lines = [
            f"{row.check}: value={row.value}, threshold={row.threshold}, passed={row.passed}"
            for row in gate_df.itertuples(index=False)
        ]
        skipped_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    elif skipped_path.is_file():
        skipped_path.unlink()


def _remove_success_outputs(config: VolForecastConfig) -> None:
    if config.output_dir is not None:
        root = Path(config.output_dir)
        for name in _SUCCESS_FILES:
            path = root / name
            if path.is_file():
                path.unlink()
    for path in (config.roc_path, config.importance_path, config.calibration_path):
        _remove_figure(path)


def _remove_figure(path: Path | None) -> None:
    if path is None:
        return
    destination = Path(path)
    if destination.is_file():
        destination.unlink()


def _plot_roc(
    *,
    y_true: np.ndarray,
    scores: dict[str, np.ndarray | None],
    aucs: dict[str, float],
) -> Figure:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="gray", label="диагональ")
    for name in ("rv_21", "logit", "hgb"):
        score = scores.get(name)
        if score is None:
            continue
        fpr, tpr, _ = roc_curve(y_true, score)
        ax.plot(fpr, tpr, label=f"{name} AUC={aucs[name]:.3f}")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("ROC: прогноз повышенной волатильности")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def _plot_importance(importance: pd.DataFrame) -> Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    if importance.empty:
        ax.set_axis_off()
        return fig
    ordered = importance.sort_values("importance_mean", ascending=True, kind="mergesort")
    ax.barh(ordered["feature"].astype(str), ordered["importance_mean"].astype(float))
    ax.set_xlabel("Среднее падение ROC-AUC")
    ax.set_title("Перестановочная важность HGB")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


def _plot_calibration(y_true: np.ndarray, probability: np.ndarray) -> Figure:
    fig, ax = plt.subplots(figsize=(6, 6))
    edges = np.linspace(0.0, 1.0, _CALIBRATION_BINS + 1)
    centers: list[float] = []
    rates: list[float] = []
    for left, right in zip(edges[:-1], edges[1:]):
        if right == 1.0:
            mask = (probability >= left) & (probability <= right)
        else:
            mask = (probability >= left) & (probability < right)
        if not np.any(mask):
            continue
        centers.append(float((left + right) / 2.0))
        rates.append(float(y_true[mask].mean()))
    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="gray")
    if centers:
        ax.plot(centers, rates, marker="o")
    ax.set_xlabel("Предсказанная вероятность")
    ax.set_ylabel("Доля положительных")
    ax.set_title("Калибровка HGB")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def _save_figure(fig: Figure, path: Path | None) -> None:
    if path is None:
        return
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _sample_frame(frame: pd.DataFrame) -> ForecastSampleFrame:
    ordered = frame.copy()
    ordered["time"] = pd.to_datetime(ordered["time"], utc=True)
    ordered["y"] = ordered["y"].astype("int64")
    ordered["ticker"] = ordered["ticker"].astype(str)
    ordered["sector_group"] = ordered["sector_group"].astype(str)
    for column in ("future_rv_5", *_NUMERIC_FEATURES):
        ordered[column] = ordered[column].astype("float64")
    columns = [
        "ticker",
        "time",
        "y",
        "future_rv_5",
        "rv_5",
        "rv_21",
        "rv_63",
        "log_return",
        "abs_return",
        "abn_volume",
        "parkinson_21",
        "range",
        "sma_gap",
        "market_rv_21",
        "sector_group",
    ]
    return ForecastSampleFrame.build(ordered.loc[:, columns])


def _metrics_frame(frame: pd.DataFrame) -> ForecastMetricsFrame:
    ordered = frame.copy()
    ordered["model"] = ordered["model"].astype(str)
    for column in ("roc_auc", "average_precision", "brier", "positive_rate_test"):
        ordered[column] = ordered[column].astype("float64")
    return ForecastMetricsFrame.build(ordered)


def _coefficients_frame(frame: pd.DataFrame) -> ForecastCoefficientsFrame:
    ordered = frame.copy()
    ordered["feature"] = ordered["feature"].astype(str)
    ordered["coef"] = ordered["coef"].astype("float64")
    return ForecastCoefficientsFrame.build(ordered)


def _importance_frame(frame: pd.DataFrame) -> ForecastImportanceFrame:
    ordered = frame.copy()
    ordered["feature"] = ordered["feature"].astype(str)
    ordered["importance_mean"] = ordered["importance_mean"].astype("float64")
    ordered["importance_std"] = ordered["importance_std"].astype("float64")
    return ForecastImportanceFrame.build(ordered)


def _predictions_frame(frame: pd.DataFrame) -> ForecastPredictionsFrame:
    ordered = frame.copy()
    ordered["time"] = pd.to_datetime(ordered["time"], utc=True)
    ordered["ticker"] = ordered["ticker"].astype(str)
    ordered["y"] = ordered["y"].astype("int64")
    for column in ("p_logit", "p_hgb", "rv_21", "future_rv_5"):
        ordered[column] = ordered[column].astype("float64")
    return ForecastPredictionsFrame.build(ordered)


def _gate_frame(frame: pd.DataFrame) -> ForecastGateFrame:
    ordered = frame.copy()
    ordered["check"] = ordered["check"].astype(str)
    ordered["value"] = ordered["value"].astype("float64")
    ordered["threshold"] = ordered["threshold"].astype("float64")
    ordered["passed"] = ordered["passed"].astype(bool)
    return ForecastGateFrame.build(ordered)
