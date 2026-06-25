from __future__ import annotations

import argparse
import json
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
    DEFAULT_BOOKTICKER_ROOT,
    TICKER_VALUE_COLUMNS,
    atomic_write_parquet,
    config_paths,
    day_timestamp_grid,
    ensure_list,
    freq_path_component,
    generate_dates,
    input_candidate_paths,
    input_path as raw_input_path,
    normalize_category,
    normalize_scheme_shift,
    normalize_scheme_shift_list,
    output_path as sampled_ticker_path,
    previous_date_str,
    resolve_existing_input_path,
    scheme_shift_path_component,
    shifted_bucket_timestamps,
)


DEFAULT_DATA_ROOT = DEFAULT_BOOKTICKER_ROOT.parent
DEFAULT_TRADE_ROOT = DEFAULT_DATA_ROOT / "TRADE"
DEFAULT_INSTRUCTOR_ROOT = DEFAULT_DATA_ROOT / "INSTRUCTOR"
SUPPORTED_INSTRUCTORS = ("trade_imbalance", "bbo_imbalance")
RAW_TRADE_TIME_COLUMN = "exchange_timestamp"
RAW_TRADE_COLUMNS = (RAW_TRADE_TIME_COLUMN, "price", "volume", "is_buyer_maker")
EPS = 1e-8


@dataclass(frozen=True)
class Task:
    symbol: str
    date: str
    freq_ms: int
    scheme_shift_ms: int
    indicator: str
    lookback: int
    trade_roots: tuple[Path, ...]
    ticker_cache_root: Path
    output_root: Path
    trade_category: str
    overwrite: bool
    strict_validate: bool
    compression: str


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def trade_input_path(root: Path, symbol: str, date_str: str, category: str = "TRADE") -> Path:
    return raw_input_path(root=root, symbol=symbol, date_str=date_str, category=category)


def instructor_output_path(
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
        / "instructor"
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
    frame = frame.dropna(subset=["timestamp", "price", "volume", "is_buyer_maker"])
    frame = frame[(frame["price"] > 0.0) & (frame["volume"] > 0.0)]
    if frame.empty:
        return pd.DataFrame(
            {
                "timestamp": pd.Series(dtype="int64"),
                "price": pd.Series(dtype="float64"),
                "volume": pd.Series(dtype="float64"),
                "is_buyer_maker": pd.Series(dtype="bool"),
            }
        )

    frame["timestamp"] = frame["timestamp"].astype("int64")
    frame["price"] = frame["price"].astype("float64")
    frame["volume"] = frame["volume"].astype("float64")
    frame["is_buyer_maker"] = frame["is_buyer_maker"].astype("bool")
    if not frame["timestamp"].is_monotonic_increasing:
        frame = frame.sort_values("timestamp", kind="mergesort")

    return frame.loc[:, ["timestamp", "volume", "is_buyer_maker"]].reset_index(drop=True)


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


def read_sampled_ticker_frame(
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
        raise FileNotFoundError(f"missing sampled ticker: {path}")

    columns = ("timestamp", *TICKER_VALUE_COLUMNS)
    frame = pd.read_parquet(path, columns=list(columns))
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"sampled ticker missing columns: {sorted(missing)}")

    frame["timestamp"] = frame["timestamp"].astype("int64")
    for column in TICKER_VALUE_COLUMNS:
        frame[column] = frame[column].astype("float64")
    return frame.sort_values("timestamp", kind="mergesort", ignore_index=True)


def lookback_periods(lookback: int, freq_ms: int) -> int:
    if lookback <= 0:
        raise ValueError(f"lookback must be positive integer seconds, got {lookback}")
    if freq_ms <= 0:
        raise ValueError(f"freq_ms must be positive integer ms, got {freq_ms}")
    return max(1, int(np.ceil(float(lookback) * 1000.0 / float(freq_ms))))


def warmup_timestamp_grid(
    date_str: str,
    freq_ms: int,
    periods: int,
    scheme_shift_ms: int = 0,
) -> np.ndarray:
    day_grid = day_timestamp_grid(date_str, freq_ms, scheme_shift_ms)
    if len(day_grid) == 0:
        return day_grid
    start = int(day_grid[0]) - periods * int(freq_ms)
    stop = int(day_grid[-1]) + int(freq_ms)
    return np.arange(start, stop, freq_ms, dtype=np.int64)


def build_trade_imbalance_frame(
    symbol: str,
    date: str,
    freq_ms: int,
    lookback: int,
    trade_roots: Any,
    trade_category: str = "TRADE",
    scheme_shift_ms: int = 0,
) -> pd.DataFrame:
    scheme_shift_ms = normalize_scheme_shift(scheme_shift_ms, freq_ms)
    periods = lookback_periods(lookback, freq_ms)
    date_list = [previous_date_str(date), date]
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

    grid = warmup_timestamp_grid(date, freq_ms, periods, scheme_shift_ms)
    out = pd.DataFrame({"timestamp": grid})
    out["trade_qty_bid"] = 0.0
    out["trade_qty_ask"] = 0.0

    if trade_frames:
        trades = pd.concat(trade_frames, ignore_index=True)
        trades["bucket_ts"] = shifted_bucket_timestamps(
            trades["timestamp"],
            freq_ms=freq_ms,
            scheme_shift_ms=scheme_shift_ms,
        )
        trades = trades[
            (trades["bucket_ts"] >= int(grid[0]))
            & (trades["bucket_ts"] <= int(grid[-1]))
        ]
        if not trades.empty:
            is_bid = trades["is_buyer_maker"].to_numpy(dtype="bool")
            volume = trades["volume"].to_numpy(dtype="float64")
            trades["trade_qty_bid"] = np.where(is_bid, volume, 0.0)
            trades["trade_qty_ask"] = np.where(is_bid, 0.0, volume)
            bucket = (
                trades.groupby("bucket_ts", as_index=False, sort=True)
                .agg(
                    trade_qty_bid=("trade_qty_bid", "sum"),
                    trade_qty_ask=("trade_qty_ask", "sum"),
                )
                .rename(columns={"bucket_ts": "timestamp"})
            )
            out = out.drop(columns=["trade_qty_bid", "trade_qty_ask"]).merge(
                bucket,
                on="timestamp",
                how="left",
            )
            out[["trade_qty_bid", "trade_qty_ask"]] = out[
                ["trade_qty_bid", "trade_qty_ask"]
            ].fillna(0.0)

    bid_roll = (
        out["trade_qty_bid"]
        .shift(1)
        .fillna(0.0)
        .rolling(window=periods, min_periods=1)
        .sum()
    )
    ask_roll = (
        out["trade_qty_ask"]
        .shift(1)
        .fillna(0.0)
        .rolling(window=periods, min_periods=1)
        .sum()
    )
    out["instructor"] = (bid_roll - ask_roll) / (bid_roll + ask_roll + EPS)

    day_grid = day_timestamp_grid(date, freq_ms, scheme_shift_ms)
    out = out[out["timestamp"].isin(day_grid)].sort_values(
        "timestamp",
        kind="mergesort",
        ignore_index=True,
    )
    out["timestamp"] = out["timestamp"].astype("int64")
    out["instructor"] = out["instructor"].astype("float64")
    return out.loc[:, ["timestamp", "instructor"]]


def build_bbo_imbalance_frame(
    symbol: str,
    date: str,
    freq_ms: int,
    lookback: int,
    ticker_cache_root: Path,
    scheme_shift_ms: int = 0,
) -> pd.DataFrame:
    if int(lookback) != 0:
        raise ValueError("bbo_imbalance only supports lookback 0")
    scheme_shift_ms = normalize_scheme_shift(scheme_shift_ms, freq_ms)
    ticker = read_sampled_ticker_frame(
        root=ticker_cache_root,
        symbol=symbol,
        date_str=date,
        freq_ms=freq_ms,
        scheme_shift_ms=scheme_shift_ms,
    )
    bid_qty = ticker["best_bid_qty"].to_numpy(dtype="float64")
    ask_qty = ticker["best_ask_qty"].to_numpy(dtype="float64")

    out = ticker.loc[:, ["timestamp"]].copy()
    out["instructor"] = (bid_qty - ask_qty) / (bid_qty + ask_qty + EPS)
    out["timestamp"] = out["timestamp"].astype("int64")
    out["instructor"] = out["instructor"].astype("float64")
    return out.loc[:, ["timestamp", "instructor"]]


def build_trade_instructor_frame(
    symbol: str,
    date: str,
    freq_ms: int,
    lookback: int,
    indicator: str,
    trade_roots: Any,
    trade_category: str = "TRADE",
    ticker_cache_root: Path | None = None,
    scheme_shift_ms: int = 0,
) -> pd.DataFrame:
    scheme_shift_ms = normalize_scheme_shift(scheme_shift_ms, freq_ms)
    if indicator == "trade_imbalance":
        return build_trade_imbalance_frame(
            symbol=symbol,
            date=date,
            freq_ms=freq_ms,
            lookback=lookback,
            trade_roots=trade_roots,
            trade_category=trade_category,
            scheme_shift_ms=scheme_shift_ms,
        )
    if indicator == "bbo_imbalance":
        if ticker_cache_root is None:
            raise ValueError("ticker_cache_root is required for bbo_imbalance")
        return build_bbo_imbalance_frame(
            symbol=symbol,
            date=date,
            freq_ms=freq_ms,
            lookback=lookback,
            ticker_cache_root=ticker_cache_root,
            scheme_shift_ms=scheme_shift_ms,
        )
    raise ValueError(f"unsupported instructor indicator: {indicator}")


def validate_instructor_frame(
    df: pd.DataFrame,
    date_str: str,
    freq_ms: int,
    scheme_shift_ms: int = 0,
) -> None:
    required = {"timestamp", "instructor"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"instructor frame missing columns: {sorted(missing)}")

    if df.empty:
        return

    timestamps = df["timestamp"].to_numpy(dtype="int64")
    if not np.all(np.diff(timestamps) >= 0):
        raise ValueError("instructor timestamp is not monotonic increasing")

    steps = np.diff(timestamps)
    if len(steps) > 0 and not np.all(steps == freq_ms):
        bad_steps = steps[steps != freq_ms][:5]
        raise ValueError(
            f"instructor timestamp step is not constant {freq_ms}ms; "
            f"found examples: {bad_steps.tolist()}"
        )

    expected_grid = day_timestamp_grid(date_str, freq_ms, scheme_shift_ms)
    if len(df) != len(expected_grid):
        raise ValueError(f"expected {len(expected_grid)} rows, got {len(df)}")
    if not np.array_equal(timestamps, expected_grid):
        raise ValueError("instructor timestamp grid does not match scheme_shift")

    values = df["instructor"].to_numpy(dtype="float64")
    if not np.isfinite(values).all():
        raise ValueError("instructor frame contains non-finite values")
    if (np.abs(values) > 1.0 + 1e-9).any():
        raise ValueError("instructor must be inside [-1, 1]")


def run_one(task: Task) -> str:
    path = instructor_output_path(
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
        frame = build_trade_instructor_frame(
            symbol=task.symbol,
            date=task.date,
            freq_ms=task.freq_ms,
            lookback=task.lookback,
            indicator=task.indicator,
            trade_roots=task.trade_roots,
            trade_category=task.trade_category,
            ticker_cache_root=task.ticker_cache_root,
            scheme_shift_ms=task.scheme_shift_ms,
        )

        if task.strict_validate:
            validate_instructor_frame(
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


def config_path(cfg: dict[str, Any], *keys: str, default: Path) -> Path:
    return config_paths(cfg, *keys, default=default)[0]


def normalize_instructor_lookbacks(indicator: str, raw_lookbacks: Any) -> list[int]:
    lookbacks = [int(x) for x in ensure_list(raw_lookbacks)]
    if indicator == "bbo_imbalance":
        if not lookbacks:
            raise ValueError("indicator bbo_imbalance requires lookback 0")
        if any(x != 0 for x in lookbacks):
            raise ValueError("indicator bbo_imbalance only supports lookback 0")
        return [0]

    lookbacks = [x for x in lookbacks if x > 0]
    if not lookbacks:
        raise ValueError(f"indicator {indicator} requires positive lookback list")
    return lookbacks


def parse_indicator_lookbacks(instructor_cfg: dict[str, Any]) -> dict[str, list[int]]:
    indicators_raw = instructor_cfg.get("indicators")
    if isinstance(indicators_raw, dict) and indicators_raw:
        indicator_lbs: dict[str, list[int]] = {}
        for name, item in indicators_raw.items():
            key = str(name).strip().lower()
            if key not in SUPPORTED_INSTRUCTORS:
                raise ValueError(
                    f"unsupported instructor indicator: {name}, "
                    f"supported={SUPPORTED_INSTRUCTORS}"
                )
            if not isinstance(item, dict):
                raise ValueError(f"indicator config must be object: {name}")
            if not bool(item.get("enabled", True)):
                continue
            indicator_lbs[key] = normalize_instructor_lookbacks(
                key,
                item.get("lookback", []),
            )
        return indicator_lbs

    indicators = [
        str(x).strip().lower()
        for x in ensure_list(
            instructor_cfg.get(
                "indicator",
                instructor_cfg.get("instructors", "trade_imbalance"),
            )
        )
        if str(x).strip()
    ]
    indicator_lbs = {}
    for indicator in indicators:
        if indicator not in SUPPORTED_INSTRUCTORS:
            raise ValueError(
                f"unsupported instructor indicator: {indicator}, "
                f"supported={SUPPORTED_INSTRUCTORS}"
            )
        indicator_lbs[indicator] = normalize_instructor_lookbacks(
            indicator,
            instructor_cfg.get("lookback", instructor_cfg.get("lookback_instructor", [])),
        )
    return indicator_lbs


def build_tasks(cfg: dict[str, Any]) -> list[Task]:
    symbols = [str(s).upper() for s in ensure_list(cfg.get("symbols"))]
    if not symbols:
        raise ValueError("config requires symbols")

    date_start = cfg.get("date_start")
    date_end = cfg.get("date_end", date_start)
    if not date_start:
        raise ValueError("config requires date_start")

    instructor_cfg = cfg.get("instructor", {})
    if not isinstance(instructor_cfg, dict):
        raise ValueError("config requires instructor section")

    freqs = [int(v) for v in ensure_list(instructor_cfg.get("freq", cfg.get("freq", 1000)))]
    if not freqs:
        raise ValueError("instructor.freq must not be empty")
    for freq in freqs:
        if freq <= 0:
            raise ValueError(f"instructor.freq must be positive integer ms, got {freq}")

    indicator_lbs = parse_indicator_lookbacks(instructor_cfg)
    if not indicator_lbs:
        raise ValueError("no enabled indicators found in instructor config")

    output_root = config_path(
        cfg,
        "instructor_cache_root",
        "output_root",
        default=DEFAULT_INSTRUCTOR_ROOT,
    )
    ticker_cache_root = config_path(
        cfg,
        "ticker_cache_root",
        "sampled_ticker_root",
        default=output_root,
    )
    trade_category = normalize_category(
        instructor_cfg.get("trade_category", cfg.get("trade_category", "TRADE")),
        default="TRADE",
    )
    trade_root_default = DEFAULT_DATA_ROOT / trade_category
    trade_roots = config_paths(
        cfg,
        "trade_roots",
        "trade_root",
        "trades_root",
        default=trade_root_default,
    )

    overwrite = bool(instructor_cfg.get("overwrite", cfg.get("overwrite", False)))
    strict_validate = bool(instructor_cfg.get("strict_validate", cfg.get("strict_validate", True)))
    compression = str(instructor_cfg.get("compression", cfg.get("compression", "snappy")))
    raw_scheme_shift = instructor_cfg.get("scheme_shift", cfg.get("scheme_shift", [0]))

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
                        trade_roots=trade_roots,
                        ticker_cache_root=ticker_cache_root,
                        output_root=output_root,
                        trade_category=trade_category,
                        overwrite=overwrite,
                        strict_validate=strict_validate,
                        compression=compression,
                    )
                )
    return tasks


def run_all(cfg: dict[str, Any]) -> None:
    tasks = build_tasks(cfg)
    print(f"total instructor tasks: {len(tasks)}")
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
        raise RuntimeError(f"{len(errors)} instructor task(s) failed:\n{preview}")


def build_config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    raw_scheme_shift = getattr(args, "scheme_shift", None)
    if args.config:
        cfg = load_config(args.config)
    else:
        cfg = {
            "symbols": args.symbols,
            "date_start": args.date_start,
            "date_end": args.date_end,
            "paths": {
                "trade_roots": [
                    path
                    for path in (
                        args.trade_root or str(DEFAULT_TRADE_ROOT),
                        args.trade_backup_root,
                    )
                    if path
                ],
                "ticker_cache_root": (
                    args.ticker_cache_root
                    or args.output_root
                    or str(DEFAULT_INSTRUCTOR_ROOT)
                ),
                "output_root": args.output_root or str(DEFAULT_INSTRUCTOR_ROOT),
            },
            "instructor": {
                "freq": args.freq,
                "scheme_shift": [0] if raw_scheme_shift is None else raw_scheme_shift,
                "indicator": args.indicators,
                "lookback": args.lookback_instructor,
                "trade_category": args.trade_category or "TRADE",
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
    if args.ticker_cache_root is not None:
        cfg["paths"]["ticker_cache_root"] = args.ticker_cache_root

    cfg.setdefault("instructor", {})
    if args.trade_category is not None:
        cfg["instructor"]["trade_category"] = args.trade_category
    if raw_scheme_shift is not None:
        cfg["instructor"]["scheme_shift"] = raw_scheme_shift
    if args.overwrite:
        cfg["instructor"]["overwrite"] = True
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute instructor features")
    parser.add_argument("--config", help="optional JSON config path")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT"])
    parser.add_argument("--date-start")
    parser.add_argument("--date-end")
    parser.add_argument("--freq", nargs="+", type=int, default=[1000])
    parser.add_argument("--scheme-shift", nargs="+", type=int)
    parser.add_argument(
        "--indicators",
        nargs="+",
        choices=SUPPORTED_INSTRUCTORS,
        default=["trade_imbalance"],
    )
    parser.add_argument("--lookback-instructor", "--lookback", nargs="+", type=int, default=[60])
    parser.add_argument("--trade-root")
    parser.add_argument("--trade-backup-root")
    parser.add_argument("--trade-category")
    parser.add_argument("--ticker-cache-root")
    parser.add_argument("--output-root")
    parser.add_argument("--compression", default="snappy")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--start-method", default="spawn")
    parser.add_argument("--maxtasksperchild", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-strict-validate", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = build_config_from_args(args)

    if args.dry_run:
        tasks = build_tasks(cfg)
        print(f"would process {len(tasks)} tasks")
        for task in tasks[:10]:
            if task.indicator == "bbo_imbalance":
                input_desc = "ticker=" + str(
                    sampled_ticker_path(
                        task.ticker_cache_root,
                        task.symbol,
                        task.freq_ms,
                        task.date,
                        task.scheme_shift_ms,
                    )
                )
            else:
                input_desc = "trade=" + str(
                    input_candidate_paths(
                        task.trade_roots,
                        task.symbol,
                        task.date,
                        task.trade_category,
                    )
                )
            output_path = instructor_output_path(
                task.output_root,
                task.symbol,
                task.indicator,
                task.freq_ms,
                task.lookback,
                task.date,
                task.scheme_shift_ms,
            )
            print(
                f"  - {task.indicator} {task.symbol} {task.date} "
                f"freq={task.freq_ms}ms scheme_shift={task.scheme_shift_ms}ms "
                f"lookback={task.lookback} "
                f"{input_desc} "
                f"output={output_path}"
            )
        if len(tasks) > 10:
            print(f"  ... and {len(tasks) - 10} more")
        return

    run_all(cfg)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"instructor failed: {exc}", file=sys.stderr)
        raise
