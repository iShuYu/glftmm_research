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

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from sampler.resample import (  # noqa: E402
    DATE_FMT_COMPACT,
    DATE_FMT_DASH,
    DEFAULT_BOOKTICKER_ROOT,
    DEFAULT_OUTPUT_ROOT as DEFAULT_TICKER_CACHE_ROOT,
    TICKER_VALUE_COLUMNS,
    TickerResampler,
    atomic_write_parquet,
    config_paths,
    day_timestamp_grid,
    ensure_list,
    generate_dates,
    input_candidate_paths,
    input_path as raw_input_path,
    normalize_category,
    output_path as sampled_ticker_path,
    previous_date_str,
    previous_day_ticker_tail,
    resolve_existing_input_path,
)


DEFAULT_DATA_ROOT = DEFAULT_BOOKTICKER_ROOT.parent
DEFAULT_TRADE_ROOT = DEFAULT_DATA_ROOT / "TRADE"
DEFAULT_INTENSITY_ROOT = DEFAULT_DATA_ROOT / "TRADE_INTENSITY"
SUPPORTED_INDICATORS = ("k", "k_decay", "k_median")
TICKER_COLUMNS = ("timestamp", "best_bid_price", "best_ask_price")
RAW_TRADE_TIME_COLUMN = "exchange_timestamp"
RAW_TRADE_COLUMNS = (RAW_TRADE_TIME_COLUMN, "price", "volume")


@dataclass(frozen=True)
class Task:
    symbol: str
    date: str
    freq_ms: int
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
    compression: str


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def trade_input_path(root: Path, symbol: str, date_str: str, category: str = "TRADE") -> Path:
    return raw_input_path(root=root, symbol=symbol, date_str=date_str, category=category)


def intensity_output_path(
    root: Path,
    symbol: str,
    indicator: str,
    freq_ms: int,
    lookback: int,
    date_str: str,
) -> Path:
    return (
        root
        / symbol
        / "intensity"
        / f"freq_{freq_ms}ms"
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
) -> pd.DataFrame:
    path = sampled_ticker_path(
        root=root,
        symbol=symbol,
        freq_ms=freq_ms,
        date_str=date_str,
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
    )
    frame = TickerResampler(freq_ms=freq_ms).resample_file(
        path=in_path,
        date_str=date_str,
        prev_tail=prev_tail,
    )
    path = sampled_ticker_path(
        root=ticker_cache_root,
        symbol=symbol,
        freq_ms=freq_ms,
        date_str=date_str,
    )
    atomic_write_parquet(frame, path, compression=compression)
    return frame.loc[:, TICKER_COLUMNS]


def compute_k_decay(break_dist: pd.Series, lookback: int) -> pd.Series:
    if lookback <= 0:
        raise ValueError(f"lookback must be positive integer, got {lookback}")

    carry = 0.5 ** (1.0 / lookback)
    alpha = 1.0 - carry
    values = np.abs(break_dist.to_numpy(dtype="float64", copy=False))
    intensity = np.empty(len(values), dtype="float64")

    prev = 0.0
    for idx, value in enumerate(values):
        prev = alpha * float(value) + carry * prev
        intensity[idx] = prev

    return pd.Series(intensity, index=break_dist.index)


def rolling_mean_excluding_zeros(values: pd.Series, lookback: int) -> pd.Series:
    if lookback <= 0:
        raise ValueError(f"lookback must be positive integer, got {lookback}")

    vals = pd.to_numeric(values, errors="coerce").fillna(0.0).astype("float64")
    vals = vals.where(vals > 0.0, 0.0)
    sum_roll = vals.rolling(window=lookback, min_periods=1).sum()
    cnt_roll = (vals > 0.0).astype("float64").rolling(window=lookback, min_periods=1).sum()
    return (sum_roll / cnt_roll.where(cnt_roll > 0.0)).fillna(0.0)


def rolling_median_excluding_zeros(values: pd.Series, lookback: int) -> pd.Series:
    if lookback <= 0:
        raise ValueError(f"lookback must be positive integer, got {lookback}")

    vals = pd.to_numeric(values, errors="coerce").fillna(0.0).astype("float64")
    vals = vals.where(vals > 0.0)
    return vals.rolling(window=lookback, min_periods=1).median().fillna(0.0)


def build_trade_intensity_frame(
    symbol: str,
    date: str,
    freq_ms: int,
    lookback: int,
    indicator: str,
    ticker_cache_root: Path,
    trade_roots: Any,
    bookticker_roots: Any = (DEFAULT_BOOKTICKER_ROOT,),
    ticker_category: str = "BOOKTICKER",
    trade_category: str = "TRADE",
    auto_resample: bool = False,
    compression: str = "snappy",
) -> pd.DataFrame:
    prev_date = previous_date_str(date)
    date_list = [prev_date, date]

    ticker_frames: list[pd.DataFrame] = []
    for one_date in date_list:
        try:
            tdf = read_or_build_sampled_ticker(
                ticker_cache_root=ticker_cache_root,
                bookticker_roots=bookticker_roots,
                symbol=symbol,
                date_str=one_date,
                freq_ms=freq_ms,
                ticker_category=ticker_category,
                auto_resample=auto_resample,
                compression=compression,
            )
        except FileNotFoundError:
            if one_date == date:
                raise
            continue
        if not tdf.empty:
            ticker_frames.append(tdf.loc[:, TICKER_COLUMNS].copy())

    if not ticker_frames:
        return pd.DataFrame(
            {
                "timestamp": pd.Series(dtype="int64"),
                "intensity": pd.Series(dtype="float64"),
            }
        )

    ticker = (
        pd.concat(ticker_frames, ignore_index=True)
        .drop_duplicates(subset=["timestamp"], keep="last")
        .sort_values("timestamp", kind="mergesort", ignore_index=True)
    )
    ticker["best_bid_price"] = ticker["best_bid_price"].astype("float64")
    ticker["best_ask_price"] = ticker["best_ask_price"].astype("float64")

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

    out = ticker.copy()
    out["bucket_break_dist_max"] = 0.0

    if trade_frames:
        trades = pd.concat(trade_frames, ignore_index=True)
        trades["bucket_ts"] = ((trades["timestamp"] // freq_ms) * freq_ms).astype("int64")

        bbo = ticker.set_index("timestamp").loc[:, ["best_bid_price", "best_ask_price"]]
        bucket_index = bbo.index.get_indexer(trades["bucket_ts"].to_numpy(dtype="int64"))
        valid = bucket_index >= 0

        if valid.any():
            trade_prices = trades.loc[valid, "price"].to_numpy(dtype="float64")
            trade_buckets = trades.loc[valid, "bucket_ts"].to_numpy(dtype="int64")
            bid_prices = bbo["best_bid_price"].to_numpy(dtype="float64")[bucket_index[valid]]
            ask_prices = bbo["best_ask_price"].to_numpy(dtype="float64")[bucket_index[valid]]

            ask_break = np.where(trade_prices > ask_prices, trade_prices - ask_prices, 0.0)
            bid_break = np.where(trade_prices < bid_prices, bid_prices - trade_prices, 0.0)
            break_dist = np.maximum(ask_break, bid_break)

            bucket = (
                pd.DataFrame(
                    {
                        "timestamp": trade_buckets,
                        "bucket_break_dist_max": break_dist,
                    }
                )
                .groupby("timestamp", as_index=False, sort=True)
                .agg(bucket_break_dist_max=("bucket_break_dist_max", "max"))
            )
            out = out.drop(columns=["bucket_break_dist_max"]).merge(
                bucket,
                on="timestamp",
                how="left",
            )
            out["bucket_break_dist_max"] = out["bucket_break_dist_max"].fillna(0.0)

    # Feature at t uses breakouts observed in [t - freq, t).
    out["bucket_break_dist_max"] = (
        out["bucket_break_dist_max"].shift(1).fillna(0.0).astype("float64")
    )

    if indicator == "k":
        out["intensity"] = rolling_mean_excluding_zeros(
            values=out["bucket_break_dist_max"],
            lookback=lookback,
        )
    elif indicator == "k_decay":
        out["intensity"] = compute_k_decay(
            break_dist=out["bucket_break_dist_max"],
            lookback=lookback,
        )
    elif indicator == "k_median":
        out["intensity"] = rolling_median_excluding_zeros(
            values=out["bucket_break_dist_max"],
            lookback=lookback,
        )
    else:
        raise ValueError(f"unsupported intensity indicator: {indicator}")

    day_grid = day_timestamp_grid(date, freq_ms)
    out = out[out["timestamp"].isin(day_grid)].sort_values(
        "timestamp",
        kind="mergesort",
        ignore_index=True,
    )
    out["timestamp"] = out["timestamp"].astype("int64")
    out["intensity"] = out["intensity"].astype("float64")
    return out.loc[:, ["timestamp", "intensity"]]


def validate_intensity_frame(df: pd.DataFrame, date_str: str, freq_ms: int) -> None:
    required = {"timestamp", "intensity"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"trade-intensity frame missing columns: {sorted(missing)}")

    if df.empty:
        return

    timestamps = df["timestamp"].to_numpy(dtype="int64")
    if not np.all(np.diff(timestamps) >= 0):
        raise ValueError("trade-intensity timestamp is not monotonic increasing")

    steps = np.diff(timestamps)
    if len(steps) > 0 and not np.all(steps == freq_ms):
        bad_steps = steps[steps != freq_ms][:5]
        raise ValueError(
            f"trade-intensity timestamp step is not constant {freq_ms}ms; "
            f"found examples: {bad_steps.tolist()}"
        )

    expected_rows = len(day_timestamp_grid(date_str, freq_ms))
    if len(df) != expected_rows:
        raise ValueError(f"expected {expected_rows} rows, got {len(df)}")

    values = df["intensity"].to_numpy(dtype="float64")
    if np.isnan(values).any():
        raise ValueError("trade-intensity contains NaN")
    if (values < 0.0).any():
        raise ValueError("trade-intensity contains negative values")


def run_one(task: Task) -> str:
    path = intensity_output_path(
        root=task.output_root,
        symbol=task.symbol,
        indicator=task.indicator,
        freq_ms=task.freq_ms,
        lookback=task.lookback,
        date_str=task.date,
    )
    if path.exists() and not task.overwrite:
        return (
            f"[skip] {task.indicator} {task.symbol} {task.date} "
            f"freq={task.freq_ms} lookback={task.lookback} -> {path}"
        )

    try:
        frame = build_trade_intensity_frame(
            symbol=task.symbol,
            date=task.date,
            freq_ms=task.freq_ms,
            lookback=task.lookback,
            indicator=task.indicator,
            ticker_cache_root=task.ticker_cache_root,
            trade_roots=task.trade_roots,
            bookticker_roots=task.bookticker_roots,
            ticker_category=task.ticker_category,
            trade_category=task.trade_category,
            auto_resample=task.auto_resample,
            compression=task.compression,
        )

        if task.strict_validate:
            validate_intensity_frame(df=frame, date_str=task.date, freq_ms=task.freq_ms)

        atomic_write_parquet(frame, path, compression=task.compression)
        return (
            f"[done] {task.indicator} {task.symbol} {task.date} "
            f"freq={task.freq_ms} lookback={task.lookback} rows={len(frame)} -> {path}"
        )
    except Exception as exc:
        return (
            f"[error] {task.indicator} {task.symbol} {task.date} "
            f"freq={task.freq_ms} lookback={task.lookback}: {exc}"
        )


def config_path(cfg: dict[str, Any], *keys: str, default: Path) -> Path:
    return config_paths(cfg, *keys, default=default)[0]


def parse_indicator_lookbacks(intensity_cfg: dict[str, Any]) -> dict[str, list[int]]:
    indicators_raw = intensity_cfg.get("indicators")
    if isinstance(indicators_raw, dict) and indicators_raw:
        indicator_lbs: dict[str, list[int]] = {}
        for name, item in indicators_raw.items():
            key = str(name).strip().lower()
            if key not in SUPPORTED_INDICATORS:
                raise ValueError(
                    f"unsupported intensity indicator: {name}, "
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
        for x in ensure_list(intensity_cfg.get("indicator", intensity_cfg.get("indicators", "k")))
    ]
    lookbacks = [int(x) for x in ensure_list(intensity_cfg.get("lookback", 60))]
    lookbacks = [x for x in lookbacks if x > 0]
    if not lookbacks:
        raise ValueError("intensity.lookback requires positive values")

    indicator_lbs = {}
    for indicator in indicators:
        if indicator not in SUPPORTED_INDICATORS:
            raise ValueError(
                f"unsupported intensity indicator: {indicator}, "
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

    intensity_cfg = cfg.get("intensity", cfg.get("trade_intensity", {}))
    if not isinstance(intensity_cfg, dict):
        raise ValueError("config requires intensity section")

    freqs = [int(v) for v in ensure_list(intensity_cfg.get("freq", cfg.get("freq", 1000)))]
    if not freqs:
        raise ValueError("intensity.freq must not be empty")
    for freq in freqs:
        if freq <= 0:
            raise ValueError(f"intensity.freq must be positive integer ms, got {freq}")

    indicator_lbs = parse_indicator_lookbacks(intensity_cfg)
    if not indicator_lbs:
        raise ValueError("no enabled indicators found in intensity config")

    ticker_category = normalize_category(
        intensity_cfg.get("ticker_category", cfg.get("ticker_category", "BOOKTICKER"))
    )
    trade_category = normalize_category(
        intensity_cfg.get("trade_category", cfg.get("trade_category", "TRADE")),
        default="TRADE",
    )
    trade_root_default = DEFAULT_DATA_ROOT / trade_category

    ticker_cache_root = config_path(
        cfg,
        "ticker_cache_root",
        "sampled_ticker_root",
        default=DEFAULT_TICKER_CACHE_ROOT,
    )
    bookticker_roots = config_paths(
        cfg,
        "bookticker_roots",
        "bookticker_root",
        default=DEFAULT_BOOKTICKER_ROOT,
    )
    trade_roots = config_paths(
        cfg,
        "trade_roots",
        "trade_root",
        "trades_root",
        default=trade_root_default,
    )
    output_root = config_path(
        cfg,
        "intensity_cache_root",
        "trade_intensity_cache_root",
        "output_root",
        default=DEFAULT_INTENSITY_ROOT,
    )

    overwrite = bool(intensity_cfg.get("overwrite", cfg.get("overwrite", False)))
    strict_validate = bool(
        intensity_cfg.get(
            "strict_validate",
            intensity_cfg.get("strict_unleak_check", cfg.get("strict_validate", True)),
        )
    )
    auto_resample = bool(intensity_cfg.get("auto_resample", cfg.get("auto_resample", False)))
    compression = str(intensity_cfg.get("compression", cfg.get("compression", "snappy")))

    tasks: list[Task] = []
    for symbol, date_str, freq_ms, indicator in product(
        symbols,
        generate_dates(date_start, date_end),
        freqs,
        sorted(indicator_lbs.keys()),
    ):
        for lookback in indicator_lbs[indicator]:
            tasks.append(
                Task(
                    symbol=symbol,
                    date=date_str,
                    freq_ms=freq_ms,
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
                    compression=compression,
                )
            )
    return tasks


def run_all(cfg: dict[str, Any]) -> None:
    tasks = build_tasks(cfg)
    print(f"total intensity tasks: {len(tasks)}")
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
        raise RuntimeError(f"{len(errors)} intensity task(s) failed:\n{preview}")


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
                "output_root": args.output_root or str(DEFAULT_INTENSITY_ROOT),
            },
            "intensity": {
                "freq": args.freq,
                "indicator": args.indicators,
                "lookback": args.lookback,
                "ticker_category": args.ticker_category or "BOOKTICKER",
                "trade_category": args.trade_category or "TRADE",
                "auto_resample": args.auto_resample,
                "overwrite": args.overwrite,
                "strict_validate": not args.no_strict_validate,
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
                    "bookticker_root",
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
                    "trade_root",
                    "trades_root",
                    default=DEFAULT_TRADE_ROOT,
                )
            )
        if args.trade_backup_root is not None:
            roots.append(Path(args.trade_backup_root))
        cfg["paths"]["trade_roots"] = [str(path) for path in dict.fromkeys(roots)]
    if args.output_root is not None:
        cfg["paths"]["output_root"] = args.output_root

    cfg.setdefault("intensity", {})
    if args.ticker_category is not None:
        cfg["intensity"]["ticker_category"] = args.ticker_category
    if args.trade_category is not None:
        cfg["intensity"]["trade_category"] = args.trade_category
    if args.overwrite:
        cfg["intensity"]["overwrite"] = True
    if args.auto_resample:
        cfg["intensity"]["auto_resample"] = True
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute trade intensity from sampled ticker data")
    parser.add_argument("--config", help="optional JSON config path")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT"])
    parser.add_argument("--date-start")
    parser.add_argument("--date-end")
    parser.add_argument("--freq", nargs="+", type=int, default=[1000])
    parser.add_argument("--indicators", nargs="+", choices=SUPPORTED_INDICATORS, default=["k"])
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
                f"freq={task.freq_ms}ms lookback={task.lookback} "
                f"ticker={sampled_ticker_path(task.ticker_cache_root, task.symbol, task.freq_ms, task.date)} "
                f"trade={input_candidate_paths(task.trade_roots, task.symbol, task.date, task.trade_category)} "
                f"output={intensity_output_path(task.output_root, task.symbol, task.indicator, task.freq_ms, task.lookback, task.date)}"
            )
        if len(tasks) > 10:
            print(f"  ... and {len(tasks) - 10} more")
        return

    run_all(cfg)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"intensity failed: {exc}", file=sys.stderr)
        raise
