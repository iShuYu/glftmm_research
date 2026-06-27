from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from itertools import product
from multiprocessing import get_context
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from sampler.resample import (
    DEFAULT_BOOKTICKER_ROOT,
    DEFAULT_OUTPUT_ROOT as DEFAULT_TICKER_CACHE_ROOT,
    TickerResampler,
    atomic_write_parquet,
    config_paths,
    day_timestamp_grid,
    ensure_list,
    freq_path_component,
    generate_dates,
    input_candidate_paths,
    input_path as raw_input_path,
    normalize_scheme_shift,
    normalize_scheme_shift_list,
    normalize_category,
    output_path as sampled_ticker_path,
    previous_date_str,
    previous_day_ticker_tail,
    resolve_existing_input_path,
    scheme_shift_path_component,
    shifted_bucket_timestamps,
)


DEFAULT_DATA_ROOT = DEFAULT_BOOKTICKER_ROOT.parent
DEFAULT_TRADE_ROOT = DEFAULT_DATA_ROOT / "TRADE"
DEFAULT_VOLATILITY_ROOT = DEFAULT_DATA_ROOT / "VOLATILITY"
SUPPORTED_INDICATORS = ("sigma", "atr", "rv", "bv", "parkinson", "gk")
RAW_TRADE_TIME_COLUMN = "exchange_timestamp"
RAW_TRADE_COLUMNS = (RAW_TRADE_TIME_COLUMN, "price", "volume")
TICKER_COLUMNS = ("timestamp", "best_bid_price", "best_ask_price")
BASE_FRAME_CACHE: dict[tuple[Any, ...], pd.DataFrame] = {}


@dataclass(frozen=True)
class Task:
    symbol: str
    date: str
    freq_ms: int
    scheme_shift_ms: int
    indicator: str
    lookback: int
    ticker_cache_root: Path
    trade_roots: tuple[Path, ...]
    output_root: Path
    bookticker_roots: tuple[Path, ...]
    ticker_category: str
    trade_category: str
    auto_resample: bool
    overwrite: bool
    strict_validate: bool
    annualize: bool
    trading_minutes_per_year: int
    min_periods: int
    compression: str


@dataclass(frozen=True)
class VolatilityConfig:
    lookback: int
    freq_ms: int
    annualize: bool = False
    trading_minutes_per_year: int = 365 * 24 * 60
    min_periods: int = 1

    @property
    def annualization_factor(self) -> float:
        if not self.annualize:
            return 1.0
        periods_per_minute = 60000.0 / float(self.freq_ms)
        periods_per_year = periods_per_minute * float(self.trading_minutes_per_year)
        return float(np.sqrt(periods_per_year))


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def trade_input_path(root: Path, symbol: str, date_str: str, category: str = "TRADE") -> Path:
    return raw_input_path(root=root, symbol=symbol, date_str=date_str, category=category)


def volatility_output_path(
    root: Path,
    symbol: str,
    indicator: str,
    freq_ms: int,
    lookback: int,
    date_str: str,
    scheme_shift_ms: int = 0,
) -> Path:
    return (
        root
        / symbol
        / "volatility"
        / freq_path_component(freq_ms)
        / scheme_shift_path_component(freq_ms, scheme_shift_ms)
        / indicator
        / f"lookback_{lookback}"
        / f"{date_str}.parquet"
    )


def normalize_trade_frame(raw_df: pd.DataFrame) -> pd.DataFrame:
    missing = set(RAW_TRADE_COLUMNS) - set(raw_df.columns)
    if missing:
        raise ValueError(f"trade frame missing columns: {sorted(missing)}")

    frame = raw_df.loc[:, RAW_TRADE_COLUMNS].rename(
        columns={RAW_TRADE_TIME_COLUMN: "timestamp"}
    )
    frame = frame.dropna(subset=["timestamp", "price", "volume"])
    frame = frame[(frame["price"] > 0.0) & (frame["volume"] > 0.0)]
    if frame.empty:
        return pd.DataFrame(
            {
                "timestamp": pd.Series(dtype="int64"),
                "price": pd.Series(dtype="float64"),
            }
        )

    frame["timestamp"] = frame["timestamp"].astype("int64")
    frame["price"] = frame["price"].astype("float64")
    if not frame["timestamp"].is_monotonic_increasing:
        frame = frame.sort_values("timestamp", kind="mergesort")

    return frame.loc[:, ["timestamp", "price"]].reset_index(drop=True)


def read_trade_frame(
    root: Any,
    symbol: str,
    date_str: str,
    category: str = "TRADE",
) -> pd.DataFrame:
    path = resolve_existing_input_path(
        roots=root,
        symbol=symbol,
        date_str=date_str,
        category=category,
    )
    raw_df = pd.read_parquet(path, columns=list(RAW_TRADE_COLUMNS))
    return normalize_trade_frame(raw_df)


def read_sampled_ticker(
    root: Path,
    symbol: str,
    date_str: str,
    freq_ms: int,
    scheme_shift_ms: int = 0,
) -> pd.DataFrame:
    path = sampled_ticker_path(
        root=root,
        symbol=symbol,
        freq_ms=freq_ms,
        date_str=date_str,
        scheme_shift_ms=scheme_shift_ms,
    )
    if not path.exists():
        raise FileNotFoundError(path)

    frame = pd.read_parquet(path, columns=list(TICKER_COLUMNS))
    missing = set(TICKER_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"sampled ticker missing columns: {sorted(missing)}")

    frame["timestamp"] = frame["timestamp"].astype("int64")
    frame["best_bid_price"] = frame["best_bid_price"].astype("float64")
    frame["best_ask_price"] = frame["best_ask_price"].astype("float64")
    return frame.sort_values("timestamp", kind="mergesort", ignore_index=True)


def read_or_build_sampled_ticker(
    ticker_cache_root: Path,
    bookticker_roots: Any,
    symbol: str,
    date_str: str,
    freq_ms: int,
    scheme_shift_ms: int,
    ticker_category: str,
    auto_resample: bool,
    compression: str,
) -> pd.DataFrame:
    try:
        return read_sampled_ticker(
            root=ticker_cache_root,
            symbol=symbol,
            date_str=date_str,
            freq_ms=freq_ms,
            scheme_shift_ms=scheme_shift_ms,
        )
    except FileNotFoundError:
        if not auto_resample:
            raise

    in_path = resolve_existing_input_path(
        roots=bookticker_roots,
        symbol=symbol,
        date_str=date_str,
        category=ticker_category,
    )

    prev_tail = previous_day_ticker_tail(
        root=bookticker_roots,
        symbol=symbol,
        date_str=date_str,
        category=ticker_category,
        freq_ms=freq_ms,
        scheme_shift_ms=scheme_shift_ms,
    )
    frame = TickerResampler(
        freq_ms=freq_ms,
        scheme_shift_ms=scheme_shift_ms,
    ).resample_file(
        path=in_path,
        date_str=date_str,
        prev_tail=prev_tail,
    )
    path = sampled_ticker_path(
        root=ticker_cache_root,
        symbol=symbol,
        freq_ms=freq_ms,
        date_str=date_str,
        scheme_shift_ms=scheme_shift_ms,
    )
    atomic_write_parquet(frame, path, compression=compression)
    return frame.loc[:, TICKER_COLUMNS]


def build_base_frame(
    symbol: str,
    date: str,
    freq_ms: int,
    ticker_cache_root: Path,
    trade_roots: Any,
    bookticker_roots: Any = (DEFAULT_BOOKTICKER_ROOT,),
    ticker_category: str = "BOOKTICKER",
    trade_category: str = "TRADE",
    auto_resample: bool = False,
    compression: str = "snappy",
    scheme_shift_ms: int = 0,
) -> pd.DataFrame:
    scheme_shift_ms = normalize_scheme_shift(scheme_shift_ms, freq_ms)
    cache_key = (
        symbol,
        date,
        freq_ms,
        scheme_shift_ms,
        str(ticker_cache_root),
        tuple(str(path) for path in input_candidate_paths(trade_roots, symbol, date, trade_category)),
        tuple(str(path) for path in input_candidate_paths(bookticker_roots, symbol, date, ticker_category)),
        ticker_category,
        trade_category,
        auto_resample,
    )
    if cache_key in BASE_FRAME_CACHE:
        return BASE_FRAME_CACHE[cache_key].copy()

    prev_date = previous_date_str(date)
    date_list = [prev_date, date]

    ticker_frames: list[pd.DataFrame] = []
    for one_date in date_list:
        try:
            ticker = read_or_build_sampled_ticker(
                ticker_cache_root=ticker_cache_root,
                bookticker_roots=bookticker_roots,
                symbol=symbol,
                date_str=one_date,
                freq_ms=freq_ms,
                scheme_shift_ms=scheme_shift_ms,
                ticker_category=ticker_category,
                auto_resample=auto_resample,
                compression=compression,
            )
        except FileNotFoundError:
            if one_date == date:
                raise
            continue
        if not ticker.empty:
            ticker_frames.append(ticker.loc[:, TICKER_COLUMNS].copy())

    if not ticker_frames:
        return pd.DataFrame({"timestamp": pd.Series(dtype="int64")})

    ticker = (
        pd.concat(ticker_frames, ignore_index=True)
        .drop_duplicates(subset=["timestamp"], keep="last")
        .sort_values("timestamp", kind="mergesort", ignore_index=True)
    )
    ticker["best_bid_price"] = ticker["best_bid_price"].astype("float64")
    ticker["best_ask_price"] = ticker["best_ask_price"].astype("float64")
    ticker["mid"] = 0.5 * (ticker["best_bid_price"] + ticker["best_ask_price"])

    trade_frames: list[pd.DataFrame] = []
    for one_date in date_list:
        try:
            trades = read_trade_frame(
                root=trade_roots,
                symbol=symbol,
                date_str=one_date,
                category=trade_category,
            )
        except FileNotFoundError:
            if one_date == date:
                raise
            continue
        if not trades.empty:
            trade_frames.append(trades)

    if trade_frames:
        trades = pd.concat(trade_frames, ignore_index=True)
        trades["bucket_ts"] = shifted_bucket_timestamps(
            trades["timestamp"],
            freq_ms=freq_ms,
            scheme_shift_ms=scheme_shift_ms,
        )
        ohlc = (
            trades.groupby("bucket_ts", as_index=False, sort=True)["price"]
            .agg(
                high=("max"),
                low=("min"),
                close=("last"),
                open=("first"),
            )
            .rename(columns={"bucket_ts": "timestamp"})
        )
        base = ticker.merge(ohlc, on="timestamp", how="left")
    else:
        base = ticker.copy()
        base["high"] = np.nan
        base["low"] = np.nan
        base["close"] = np.nan
        base["open"] = np.nan

    base["close"] = base["close"].fillna(base["mid"])
    base["high"] = base["high"].fillna(base["close"])
    base["low"] = base["low"].fillna(base["close"])
    base["open"] = base["open"].fillna(base["close"])

    day_grid = day_timestamp_grid(date, freq_ms, scheme_shift_ms)
    base = base[base["timestamp"].isin(day_grid)].sort_values(
        "timestamp",
        kind="mergesort",
        ignore_index=True,
    )

    base["timestamp"] = base["timestamp"].astype("int64")
    for col in ("mid", "high", "low", "close", "open"):
        base[col] = base[col].astype("float64")

    base = base.loc[:, ["timestamp", "mid", "high", "low", "close", "open"]]
    BASE_FRAME_CACHE[cache_key] = base.copy()
    return base


class VolatilityCalculator:
    def __init__(self, base: pd.DataFrame, freq_ms: int) -> None:
        self.base = base
        self.freq_ms = int(freq_ms)
        self._precompute_returns()

    def _precompute_returns(self) -> None:
        mid = self.base["mid"].to_numpy(dtype="float64")
        close = self.base["close"].to_numpy(dtype="float64")
        high = self.base["high"].to_numpy(dtype="float64")
        low = self.base["low"].to_numpy(dtype="float64")
        open_ = self.base["open"].to_numpy(dtype="float64")

        self.mid_return = np.zeros(len(mid), dtype="float64")
        self.log_return = np.zeros(len(mid), dtype="float64")
        if len(mid) > 1:
            prev_mid = mid[:-1]
            valid_prev = np.isfinite(prev_mid) & (prev_mid > 0.0)
            valid_curr = np.isfinite(mid[1:]) & (mid[1:] > 0.0)
            self.mid_return[1:] = np.divide(
                np.diff(mid),
                prev_mid,
                out=np.zeros(len(mid) - 1, dtype="float64"),
                where=valid_prev & valid_curr,
            )
            ratio = np.divide(
                mid[1:],
                prev_mid,
                out=np.ones(len(mid) - 1, dtype="float64"),
                where=valid_prev & valid_curr,
            )
            self.log_return[1:] = np.log(ratio)

        self.mid_return = np.nan_to_num(
            self.mid_return,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        self.log_return = np.nan_to_num(
            self.log_return,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        self.price_scale = np.nan_to_num(
            mid,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        self.up_mask = self.mid_return > 0.0
        self.down_mask = self.mid_return < 0.0
        self.nonzero_mask = self.mid_return != 0.0

        tr1 = np.maximum(high - low, 0.0)
        prev_close = np.zeros(len(close), dtype="float64")
        if len(close):
            prev_close[0] = close[0]
            prev_close[1:] = close[:-1]
        tr2 = np.abs(high - prev_close)
        tr3 = np.abs(low - prev_close)
        self.true_range = np.maximum(tr1, np.maximum(tr2, tr3))
        self.true_range = np.nan_to_num(
            self.true_range,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        with np.errstate(divide="ignore", invalid="ignore"):
            valid_hl = (
                np.isfinite(high)
                & np.isfinite(low)
                & (high > 0.0)
                & (low > 0.0)
            )
            valid_co = (
                np.isfinite(close)
                & np.isfinite(open_)
                & (close > 0.0)
                & (open_ > 0.0)
            )
            high_low_ratio = np.divide(
                high,
                low,
                out=np.ones(len(high), dtype="float64"),
                where=valid_hl,
            )
            close_open_ratio = np.divide(
                close,
                open_,
                out=np.ones(len(close), dtype="float64"),
                where=valid_co,
            )
            high_low = np.log(high_low_ratio)
            close_open = np.log(close_open_ratio)
            self.parkinson = (high_low**2) / (4.0 * np.log(2.0))
            self.gk = 0.5 * high_low**2 - (2.0 * np.log(2.0) - 1.0) * close_open**2

        self.parkinson = np.nan_to_num(
            self.parkinson,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        self.gk = np.maximum(
            np.nan_to_num(self.gk, nan=0.0, posinf=0.0, neginf=0.0),
            0.0,
        )

    @staticmethod
    def _clean_values(values: np.ndarray) -> np.ndarray:
        return np.nan_to_num(
            np.asarray(values, dtype="float64"),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    @staticmethod
    def _finalize_volatility(
        values: pd.Series,
        config: VolatilityConfig,
        annualize: bool = True,
        lag: bool = True,
    ) -> np.ndarray:
        out = values.shift(1) if lag else values.copy()
        if annualize and config.annualize:
            out *= config.annualization_factor
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out = out.clip(lower=0.0)
        return out.to_numpy(dtype="float64")

    @classmethod
    def _rolling_std(cls, values: np.ndarray, config: VolatilityConfig) -> np.ndarray:
        series = pd.Series(cls._clean_values(values))
        vol = series.rolling(
            window=config.lookback,
            min_periods=config.min_periods,
        ).std()
        return cls._finalize_volatility(vol, config=config, annualize=True, lag=False)

    @classmethod
    def _rolling_mean(
        cls,
        values: np.ndarray,
        config: VolatilityConfig,
        lag: bool = True,
    ) -> np.ndarray:
        mean = pd.Series(cls._clean_values(values)).rolling(
            window=config.lookback,
            min_periods=config.min_periods,
        ).mean()
        return cls._finalize_volatility(mean, config=config, annualize=False, lag=lag)

    @classmethod
    def _rolling_sqrt_mean(
        cls,
        values: np.ndarray,
        config: VolatilityConfig,
        lag: bool = True,
    ) -> np.ndarray:
        clean = np.maximum(cls._clean_values(values), 0.0)
        mean = pd.Series(clean).rolling(
            window=config.lookback,
            min_periods=config.min_periods,
        ).mean()
        vol = np.sqrt(mean.clip(lower=0.0))
        return cls._finalize_volatility(vol, config=config, annualize=True, lag=lag)

    def compute_sigma(self, config: VolatilityConfig) -> pd.DataFrame:
        nonzero_returns = np.where(self.nonzero_mask, self.mid_return, 0.0)
        up_returns = np.where(self.up_mask, self.mid_return, 0.0)
        down_returns = np.where(self.down_mask, -self.mid_return, 0.0)

        return pd.DataFrame(
            {
                "timestamp": self.base["timestamp"],
                "volatility": self._rolling_std(nonzero_returns, config) * self.price_scale,
                "volatility_up": self._rolling_std(up_returns, config) * self.price_scale,
                "volatility_down": self._rolling_std(down_returns, config) * self.price_scale,
            }
        )

    def compute_atr(self, config: VolatilityConfig) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "timestamp": self.base["timestamp"],
                "volatility": self._rolling_mean(self.true_range, config),
            }
        )

    def compute_rv(self, config: VolatilityConfig) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "timestamp": self.base["timestamp"],
                "volatility": self._rolling_sqrt_mean(
                    self.log_return**2,
                    config,
                    lag=False,
                )
                * self.price_scale,
            }
        )

    def compute_bv(self, config: VolatilityConfig) -> pd.DataFrame:
        abs_returns = np.abs(self.log_return)
        bipower = (np.pi / 2.0) * abs_returns * np.roll(abs_returns, 1)
        if len(bipower):
            bipower[0] = 0.0
        return pd.DataFrame(
            {
                "timestamp": self.base["timestamp"],
                "volatility": self._rolling_sqrt_mean(bipower, config, lag=False)
                * self.price_scale,
            }
        )

    def compute_parkinson(self, config: VolatilityConfig) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "timestamp": self.base["timestamp"],
                "volatility": self._rolling_sqrt_mean(self.parkinson, config)
                * self.price_scale,
            }
        )

    def compute_gk(self, config: VolatilityConfig) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "timestamp": self.base["timestamp"],
                "volatility": self._rolling_sqrt_mean(self.gk, config)
                * self.price_scale,
            }
        )

    def compute(self, indicator: str, config: VolatilityConfig) -> pd.DataFrame:
        compute_func = {
            "sigma": self.compute_sigma,
            "atr": self.compute_atr,
            "rv": self.compute_rv,
            "bv": self.compute_bv,
            "parkinson": self.compute_parkinson,
            "gk": self.compute_gk,
        }.get(indicator)
        if compute_func is None:
            raise ValueError(f"unsupported volatility indicator: {indicator}")
        frame = compute_func(config)
        frame["timestamp"] = frame["timestamp"].astype("int64")
        for col in frame.columns:
            if col != "timestamp":
                frame[col] = frame[col].astype("float64")
        return frame


def validate_volatility_frame(
    df: pd.DataFrame,
    date_str: str,
    freq_ms: int,
    scheme_shift_ms: int = 0,
) -> None:
    required = {"timestamp", "volatility"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"volatility frame missing columns: {sorted(missing)}")

    if df.empty:
        return

    timestamps = df["timestamp"].to_numpy(dtype="int64")
    if not np.all(np.diff(timestamps) >= 0):
        raise ValueError("volatility timestamp is not monotonic increasing")

    steps = np.diff(timestamps)
    if len(steps) > 0 and not np.all(steps == freq_ms):
        bad_steps = steps[steps != freq_ms][:5]
        raise ValueError(
            f"volatility timestamp step is not constant {freq_ms}ms; "
            f"found examples: {bad_steps.tolist()}"
        )

    expected_grid = day_timestamp_grid(date_str, freq_ms, scheme_shift_ms)
    if len(df) != len(expected_grid):
        raise ValueError(f"expected {len(expected_grid)} rows, got {len(df)}")
    if not np.array_equal(timestamps, expected_grid):
        raise ValueError("volatility timestamp grid does not match scheme_shift")

    value_cols = [col for col in df.columns if col != "timestamp"]
    values = df.loc[:, value_cols].to_numpy(dtype="float64")
    if not np.isfinite(values).all():
        raise ValueError("volatility frame contains non-finite values")
    if (values < 0.0).any():
        raise ValueError("volatility frame contains negative values")


def run_one(task: Task) -> str:
    path = volatility_output_path(
        root=task.output_root,
        symbol=task.symbol,
        indicator=task.indicator,
        freq_ms=task.freq_ms,
        lookback=task.lookback,
        date_str=task.date,
        scheme_shift_ms=task.scheme_shift_ms,
    )
    if path.exists() and not task.overwrite:
        return (
            f"[skip] {task.indicator} {task.symbol} {task.date} "
            f"freq={task.freq_ms} scheme_shift={task.scheme_shift_ms} "
            f"lookback={task.lookback} -> {path}"
        )

    try:
        base = build_base_frame(
            symbol=task.symbol,
            date=task.date,
            freq_ms=task.freq_ms,
            ticker_cache_root=task.ticker_cache_root,
            trade_roots=task.trade_roots,
            bookticker_roots=task.bookticker_roots,
            ticker_category=task.ticker_category,
            trade_category=task.trade_category,
            auto_resample=task.auto_resample,
            compression=task.compression,
            scheme_shift_ms=task.scheme_shift_ms,
        )
        if base.empty:
            return (
                f"[empty] {task.indicator} {task.symbol} {task.date} "
                f"freq={task.freq_ms} lookback={task.lookback} -> no data"
            )

        config = VolatilityConfig(
            lookback=task.lookback,
            freq_ms=task.freq_ms,
            annualize=task.annualize,
            trading_minutes_per_year=task.trading_minutes_per_year,
            min_periods=task.min_periods,
        )
        frame = VolatilityCalculator(base=base, freq_ms=task.freq_ms).compute(
            indicator=task.indicator,
            config=config,
        )

        if task.strict_validate:
            validate_volatility_frame(
                df=frame,
                date_str=task.date,
                freq_ms=task.freq_ms,
                scheme_shift_ms=task.scheme_shift_ms,
            )

        atomic_write_parquet(frame, path, compression=task.compression)
        return (
            f"[done] {task.indicator} {task.symbol} {task.date} "
            f"freq={task.freq_ms} scheme_shift={task.scheme_shift_ms} "
            f"lookback={task.lookback} rows={len(frame)} -> {path}"
        )
    except Exception as exc:
        return (
            f"[error] {task.indicator} {task.symbol} {task.date} "
            f"freq={task.freq_ms} scheme_shift={task.scheme_shift_ms} "
            f"lookback={task.lookback}: {exc}"
        )


def config_path(cfg: dict[str, Any], key: str, default: Path) -> Path:
    return config_paths(cfg, key, default=default)[0]


def parse_indicator_lookbacks(vol_cfg: dict[str, Any]) -> dict[str, list[int]]:
    indicators_raw = vol_cfg.get("indicators")
    if isinstance(indicators_raw, dict) and indicators_raw:
        indicator_lbs: dict[str, list[int]] = {}
        for name, item in indicators_raw.items():
            key = str(name).strip().lower()
            if key not in SUPPORTED_INDICATORS:
                raise ValueError(
                    f"unsupported volatility indicator: {name}, "
                    f"supported={SUPPORTED_INDICATORS}"
                )
            if not isinstance(item, dict):
                raise ValueError(f"indicator config must be object: {name}")
            if not bool(item.get("enabled", True)):
                continue
            lookbacks = [int(x) for x in ensure_list(item.get("lookback", []))]
            lookbacks = [x for x in lookbacks if x > 0]
            if not lookbacks:
                raise ValueError(f"indicator {name} requires positive lookback list")
            indicator_lbs[key] = lookbacks
        return indicator_lbs

    indicators = [
        str(x).strip().lower()
        for x in ensure_list(vol_cfg.get("indicator", vol_cfg.get("indicators", "sigma")))
    ]
    lookbacks = [int(x) for x in ensure_list(vol_cfg.get("lookback", 60))]
    lookbacks = [x for x in lookbacks if x > 0]
    if not lookbacks:
        raise ValueError("volatility.lookback requires positive values")

    indicator_lbs = {}
    for indicator in indicators:
        if indicator not in SUPPORTED_INDICATORS:
            raise ValueError(
                f"unsupported volatility indicator: {indicator}, "
                f"supported={SUPPORTED_INDICATORS}"
            )
        indicator_lbs[indicator] = lookbacks
    return indicator_lbs


def build_tasks(cfg: dict[str, Any]) -> list[Task]:
    symbols = [str(s).upper() for s in ensure_list(cfg.get("symbols"))]
    if not symbols:
        raise ValueError("config requires symbols")

    date_start = cfg.get("date_start")
    date_end = cfg.get("date_end", date_start)
    if not date_start:
        raise ValueError("config requires date_start")

    vol_cfg = cfg.get("volatility", {})
    if not isinstance(vol_cfg, dict):
        raise ValueError("config requires volatility section")

    freqs = [int(v) for v in ensure_list(vol_cfg.get("freq", cfg.get("freq", 1000)))]
    if not freqs:
        raise ValueError("volatility.freq must not be empty")
    for freq in freqs:
        if freq <= 0:
            raise ValueError(f"volatility.freq must be positive integer ms, got {freq}")

    indicator_lbs = parse_indicator_lookbacks(vol_cfg)
    if not indicator_lbs:
        raise ValueError("no enabled indicators found in volatility config")

    ticker_category = normalize_category(
        vol_cfg.get("ticker_category", cfg.get("ticker_category", "BOOKTICKER"))
    )
    trade_category = normalize_category(
        vol_cfg.get("trade_category", cfg.get("trade_category", "TRADE")),
        default="TRADE",
    )
    trade_root_default = DEFAULT_DATA_ROOT / trade_category

    ticker_cache_root = config_path(
        cfg,
        "ticker_cache_root",
        default=DEFAULT_TICKER_CACHE_ROOT,
    )
    bookticker_roots = config_paths(
        cfg,
        "bookticker_roots",
        default=DEFAULT_BOOKTICKER_ROOT,
    )
    trade_roots = config_paths(
        cfg,
        "trade_roots",
        default=trade_root_default,
    )
    output_root = config_path(
        cfg,
        "output_root",
        default=DEFAULT_VOLATILITY_ROOT,
    )

    overwrite = bool(vol_cfg.get("overwrite", cfg.get("overwrite", False)))
    strict_validate = bool(vol_cfg.get("strict_validate", cfg.get("strict_validate", True)))
    auto_resample = bool(vol_cfg.get("auto_resample", cfg.get("auto_resample", False)))
    annualize = bool(vol_cfg.get("annualize", cfg.get("annualize", False)))
    min_periods = max(1, int(vol_cfg.get("min_periods", cfg.get("min_periods", 1))))
    trading_minutes_per_year = int(
        vol_cfg.get("trading_minutes_per_year", cfg.get("trading_minutes_per_year", 365 * 24 * 60))
    )
    compression = str(vol_cfg.get("compression", cfg.get("compression", "snappy")))
    raw_scheme_shift = vol_cfg.get("scheme_shift", cfg.get("scheme_shift", [0]))

    tasks: list[Task] = []
    for symbol, date_str, freq_ms, indicator in product(
        symbols,
        generate_dates(date_start, date_end),
        freqs,
        sorted(indicator_lbs.keys()),
    ):
        for scheme_shift_ms in normalize_scheme_shift_list(raw_scheme_shift, freq_ms):
            for lookback in indicator_lbs[indicator]:
                tasks.append(
                    Task(
                        symbol=symbol,
                        date=date_str,
                        freq_ms=freq_ms,
                        scheme_shift_ms=scheme_shift_ms,
                        indicator=indicator,
                        lookback=lookback,
                        ticker_cache_root=ticker_cache_root,
                        trade_roots=trade_roots,
                        output_root=output_root,
                        bookticker_roots=bookticker_roots,
                        ticker_category=ticker_category,
                        trade_category=trade_category,
                        auto_resample=auto_resample,
                        overwrite=overwrite,
                        strict_validate=strict_validate,
                        annualize=annualize,
                        trading_minutes_per_year=trading_minutes_per_year,
                        min_periods=min_periods,
                        compression=compression,
                    )
                )
    return tasks


def run_all(cfg: dict[str, Any]) -> None:
    tasks = build_tasks(cfg)
    print(f"total volatility tasks: {len(tasks)}")
    if not tasks:
        return

    parallel_cfg = cfg.get("parallel", {})
    workers = max(1, int(parallel_cfg.get("num_workers", cfg.get("workers", 1))))
    start_method = str(parallel_cfg.get("start_method", "spawn")).strip().lower()
    allowed_start_methods = {"fork", "spawn", "forkserver"}
    if start_method not in allowed_start_methods:
        raise ValueError(
            f"parallel.start_method must be one of {sorted(allowed_start_methods)}, "
            f"got: {start_method}"
        )

    raw_maxtasksperchild = parallel_cfg.get("maxtasksperchild")
    maxtasksperchild = (
        None
        if raw_maxtasksperchild in (None, 0)
        else max(1, int(raw_maxtasksperchild))
    )

    print(
        f"parallel config: workers={workers}, start_method={start_method}, "
        f"maxtasksperchild={maxtasksperchild}"
    )

    if workers > 1:
        ctx = get_context(start_method)
        errors: list[str] = []
        with ctx.Pool(processes=workers, maxtasksperchild=maxtasksperchild) as pool:
            for idx, msg in enumerate(pool.imap_unordered(run_one, tasks, chunksize=1), start=1):
                print(f"[{idx}/{len(tasks)}] {msg}", flush=True)
                if msg.startswith("[error]"):
                    errors.append(msg)
    else:
        errors = []
        for idx, task in enumerate(tasks, start=1):
            msg = run_one(task)
            print(f"[{idx}/{len(tasks)}] {msg}", flush=True)
            if msg.startswith("[error]"):
                errors.append(msg)

    if errors:
        preview = "\n".join(errors[:10])
        if len(errors) > 10:
            preview = f"{preview}\n... {len(errors) - 10} more errors"
        raise RuntimeError(f"{len(errors)} volatility task(s) failed:\n{preview}")


def build_config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    if args.config:
        cfg = load_config(args.config)
    else:
        cfg = {
            "symbols": args.symbols,
            "date_start": args.date_start,
            "date_end": args.date_end,
            "paths": {
                "ticker_cache_root": args.ticker_cache_root or str(DEFAULT_TICKER_CACHE_ROOT),
                "bookticker_roots": [
                    path
                    for path in (
                        args.bookticker_root or str(DEFAULT_BOOKTICKER_ROOT),
                        args.bookticker_backup_root,
                    )
                    if path
                ],
                "trade_roots": [
                    path
                    for path in (
                        args.trade_root or str(DEFAULT_TRADE_ROOT),
                        args.trade_backup_root,
                    )
                    if path
                ],
                "output_root": args.output_root or str(DEFAULT_VOLATILITY_ROOT),
            },
            "volatility": {
                "freq": args.freq,
                "scheme_shift": [0] if args.scheme_shift is None else args.scheme_shift,
                "indicator": args.indicators,
                "lookback": args.lookback,
                "ticker_category": args.ticker_category or "BOOKTICKER",
                "trade_category": args.trade_category or "TRADE",
                "auto_resample": args.auto_resample,
                "overwrite": args.overwrite,
                "strict_validate": not args.no_strict_validate,
                "annualize": args.annualize,
                "trading_minutes_per_year": args.trading_minutes_per_year,
                "min_periods": args.min_periods,
                "compression": args.compression,
            },
            "parallel": {
                "num_workers": args.workers,
                "start_method": args.start_method,
                "maxtasksperchild": args.maxtasksperchild,
            },
        }

    cfg.setdefault("paths", {})
    if args.ticker_cache_root is not None:
        cfg["paths"]["ticker_cache_root"] = args.ticker_cache_root
    if args.bookticker_root is not None or args.bookticker_backup_root is not None:
        roots = []
        if args.bookticker_root is not None:
            roots.append(Path(args.bookticker_root))
        else:
            roots.extend(
                config_paths(
                    cfg,
                    "bookticker_roots",
                    default=DEFAULT_BOOKTICKER_ROOT,
                )
            )
        if args.bookticker_backup_root is not None:
            roots.append(Path(args.bookticker_backup_root))
        cfg["paths"]["bookticker_roots"] = [str(path) for path in dict.fromkeys(roots)]
    if args.trade_root is not None or args.trade_backup_root is not None:
        roots = []
        if args.trade_root is not None:
            roots.append(Path(args.trade_root))
        else:
            roots.extend(
                config_paths(
                    cfg,
                    "trade_roots",
                    default=DEFAULT_TRADE_ROOT,
                )
            )
        if args.trade_backup_root is not None:
            roots.append(Path(args.trade_backup_root))
        cfg["paths"]["trade_roots"] = [str(path) for path in dict.fromkeys(roots)]
    if args.output_root is not None:
        cfg["paths"]["output_root"] = args.output_root

    cfg.setdefault("volatility", {})
    if args.ticker_category is not None:
        cfg["volatility"]["ticker_category"] = args.ticker_category
    if args.trade_category is not None:
        cfg["volatility"]["trade_category"] = args.trade_category
    if args.scheme_shift is not None:
        cfg["volatility"]["scheme_shift"] = args.scheme_shift
    if args.overwrite:
        cfg["volatility"]["overwrite"] = True
    if args.auto_resample:
        cfg["volatility"]["auto_resample"] = True
    if args.annualize:
        cfg["volatility"]["annualize"] = True
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute volatility from sampled ticker and trade data")
    parser.add_argument("--config", help="optional JSON config path")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT"])
    parser.add_argument("--date-start")
    parser.add_argument("--date-end")
    parser.add_argument("--freq", nargs="+", type=int, default=[1000])
    parser.add_argument("--scheme-shift", nargs="+", type=int)
    parser.add_argument("--indicators", nargs="+", choices=SUPPORTED_INDICATORS, default=["sigma"])
    parser.add_argument("--lookback", nargs="+", type=int, default=[60])
    parser.add_argument("--ticker-cache-root")
    parser.add_argument("--bookticker-root")
    parser.add_argument("--bookticker-backup-root")
    parser.add_argument("--ticker-category")
    parser.add_argument("--trade-root")
    parser.add_argument("--trade-backup-root")
    parser.add_argument("--trade-category")
    parser.add_argument("--output-root")
    parser.add_argument("--compression", default="snappy")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--start-method", default="spawn")
    parser.add_argument("--maxtasksperchild", type=int)
    parser.add_argument("--min-periods", type=int, default=1)
    parser.add_argument("--trading-minutes-per-year", type=int, default=365 * 24 * 60)
    parser.add_argument("--annualize", action="store_true")
    parser.add_argument("--auto-resample", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-strict-validate", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = build_config_from_args(args)

    if args.dry_run:
        tasks = build_tasks(cfg)
        print(f"would process {len(tasks)} tasks")
        for task in tasks[:10]:
            print(
                f"  - {task.indicator} {task.symbol} {task.date} "
                f"freq={task.freq_ms}ms scheme_shift={task.scheme_shift_ms}ms "
                f"lookback={task.lookback} "
                f"ticker={sampled_ticker_path(task.ticker_cache_root, task.symbol, task.freq_ms, task.date, task.scheme_shift_ms)} "
                f"trade={input_candidate_paths(task.trade_roots, task.symbol, task.date, task.trade_category)} "
                f"output={volatility_output_path(task.output_root, task.symbol, task.indicator, task.freq_ms, task.lookback, task.date, task.scheme_shift_ms)}"
            )
        if len(tasks) > 10:
            print(f"  ... and {len(tasks) - 10} more")
        return

    run_all(cfg)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"volatility failed: {exc}", file=sys.stderr)
        raise
