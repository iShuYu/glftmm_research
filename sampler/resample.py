from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from itertools import product
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DATE_FMT_DASH = "%Y-%m-%d"
DATE_FMT_COMPACT = "%Y%m%d"
DEFAULT_BOOKTICKER_ROOT = Path(
    "/data/users/data-helper/PROCESSED/TARDIS/BINANCE/UFUTURES/BOOKTICKER"
)
DEFAULT_OUTPUT_ROOT = DEFAULT_BOOKTICKER_ROOT.parent / "BOOKTICKER_SAMPLED"

RAW_TIME_COLUMN = "exchange_timestamp"
RAW_TICKER_COLUMNS = (
    "exchange_timestamp",
    "price_bid",
    "price_ask",
    "volume_bid",
    "volume_ask",
)
RAW_TO_SAMPLE_COLUMNS = {
    "exchange_timestamp": "timestamp",
    "price_bid": "best_bid_price",
    "price_ask": "best_ask_price",
    "volume_bid": "best_bid_qty",
    "volume_ask": "best_ask_qty",
}
TICKER_VALUE_COLUMNS = (
    "best_bid_price",
    "best_ask_price",
    "best_bid_qty",
    "best_ask_qty",
)

MS_IN_SECOND = 1000
MS_IN_DAY = 24 * 60 * 60 * MS_IN_SECOND


@dataclass(frozen=True)
class Task:
    symbol: str
    date: str
    freq_ms: int
    scheme_shift_ms: int
    bookticker_roots: tuple[Path, ...]
    ticker_category: str
    output_root: Path
    overwrite: bool
    strict_validate: bool
    compression: str


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str) and "," in value:
        return [x.strip() for x in value.split(",") if x.strip()]
    return [value]


def _is_empty_config_value(value: Any) -> bool:
    return value in (None, "", [])


def _split_stage_freqs(cfg: dict[str, Any]) -> list[int]:
    values: list[int] = []
    for key in ("freq_ms_instructor", "freq_ms_intensity", "freq_ms_volatility"):
        raw = cfg.get(key)
        if _is_empty_config_value(raw):
            continue
        values.extend(int(item) for item in ensure_list(raw))
    return sorted(set(values))


def normalize_category(value: Any, default: str = "BOOKTICKER") -> str:
    category = str(value if value is not None else default).strip().upper()
    if not category:
        raise ValueError("category must not be empty")
    return category


def ensure_paths(value: Any) -> tuple[Path, ...]:
    if value is None:
        return ()
    if isinstance(value, Path):
        items = [value]
    elif isinstance(value, str):
        items = [value]
    else:
        items = ensure_list(value)

    paths: list[Path] = []
    seen: set[str] = set()
    for item in items:
        if item is None:
            continue
        text = str(item).strip()
        if not text:
            continue
        path = Path(text)
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        paths.append(path)
    return tuple(paths)


def config_paths(cfg: dict[str, Any], *keys: str, default: Path) -> tuple[Path, ...]:
    paths_cfg = cfg.get("paths", {})
    for key in keys:
        if key in paths_cfg:
            roots = ensure_paths(paths_cfg[key])
            if roots:
                return roots
        if key in cfg:
            roots = ensure_paths(cfg[key])
            if roots:
                return roots
    return (default,)


def parse_date_input(value: str) -> datetime:
    for fmt in (DATE_FMT_DASH, DATE_FMT_COMPACT):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    raise ValueError(
        f"unsupported date format: {value}, expected YYYY-MM-DD or YYYYMMDD"
    )


def generate_dates(start: str, end: str) -> list[str]:
    start_dt = parse_date_input(start)
    end_dt = parse_date_input(end)
    if end_dt < start_dt:
        raise ValueError(f"date_end must be >= date_start, got {end} < {start}")
    return pd.date_range(start=start_dt, end=end_dt, freq="D").strftime(DATE_FMT_DASH).tolist()


def input_path(
    root: Path,
    symbol: str,
    date_str: str,
    category: str = "BOOKTICKER",
) -> Path:
    category = normalize_category(category)
    return root / symbol / f"{symbol}--{category}--{date_str}.parquet"


def input_candidate_paths(
    roots: Any,
    symbol: str,
    date_str: str,
    category: str = "BOOKTICKER",
) -> list[Path]:
    return [
        input_path(root=root, symbol=symbol, date_str=date_str, category=category)
        for root in ensure_paths(roots)
    ]


def resolve_existing_input_path(
    roots: Any,
    symbol: str,
    date_str: str,
    category: str = "BOOKTICKER",
) -> Path:
    candidates = input_candidate_paths(
        roots=roots,
        symbol=symbol,
        date_str=date_str,
        category=category,
    )
    for path in candidates:
        if path.exists():
            return path
    if not candidates:
        raise FileNotFoundError("no input roots configured")
    raise FileNotFoundError(
        "missing input file; tried: " + ", ".join(str(path) for path in candidates)
    )


def normalize_scheme_shift(value: Any, freq_ms: int) -> int:
    freq = int(freq_ms)
    if freq <= 0:
        raise ValueError(f"freq_ms must be positive, got {freq_ms}")
    shift = 0 if value in (None, "", []) else int(value)
    if shift < 0:
        raise ValueError(f"scheme_shift must be >= 0, got {shift}")
    if shift >= freq:
        raise ValueError(
            f"scheme_shift must be < freq_ms, got scheme_shift={shift}, freq_ms={freq}"
        )
    return shift


def normalize_scheme_shift_list(value: Any, freq_ms: int) -> list[int]:
    shifts: list[int] = []
    seen: set[int] = set()
    raw_values = [0] if value in (None, "", []) else ensure_list(value)
    for raw in raw_values:
        shift = normalize_scheme_shift(raw, freq_ms)
        if shift in seen:
            continue
        seen.add(shift)
        shifts.append(shift)
    if not shifts:
        shifts.append(0)
    return shifts


def freq_path_component(freq_ms: int) -> str:
    return f"freq_{int(freq_ms)}ms"


def scheme_shift_path_component(freq_ms: int, scheme_shift_ms: int = 0) -> str:
    shift = normalize_scheme_shift(scheme_shift_ms, freq_ms)
    return f"scheme_shift_{shift}ms"


def output_path(
    root: Path,
    symbol: str,
    freq_ms: int,
    date_str: str,
    scheme_shift_ms: int = 0,
) -> Path:
    return (
        root
        / symbol
        / "resample"
        / freq_path_component(freq_ms)
        / scheme_shift_path_component(freq_ms, scheme_shift_ms)
        / f"{date_str}.parquet"
    )


def previous_date_str(date_str: str) -> str:
    current = dt.date.fromisoformat(date_str)
    return (current - dt.timedelta(days=1)).isoformat()


def day_timestamp_grid(
    date_str: str,
    freq_ms: int,
    scheme_shift_ms: int = 0,
) -> np.ndarray:
    shift = normalize_scheme_shift(scheme_shift_ms, freq_ms)
    date_obj = dt.date.fromisoformat(date_str)
    day_start = dt.datetime.combine(date_obj, dt.time.min)
    epoch = dt.datetime(1970, 1, 1)
    day_start_ms = int((day_start - epoch).total_seconds() * MS_IN_SECOND)
    next_day_start_ms = day_start_ms + MS_IN_DAY
    return np.arange(
        day_start_ms + shift,
        next_day_start_ms + shift,
        int(freq_ms),
        dtype=np.int64,
    )


def shifted_bucket_timestamps(
    timestamps: Any,
    freq_ms: int,
    scheme_shift_ms: int = 0,
) -> np.ndarray:
    shift = normalize_scheme_shift(scheme_shift_ms, freq_ms)
    ts = np.asarray(timestamps, dtype="int64")
    return (((ts - shift) // int(freq_ms)) * int(freq_ms) + shift).astype("int64")


def normalize_bookticker_frame(raw_df: pd.DataFrame) -> pd.DataFrame:
    missing = set(RAW_TICKER_COLUMNS) - set(raw_df.columns)
    if missing:
        raise ValueError(f"bookticker frame missing columns: {sorted(missing)}")

    frame = raw_df.loc[:, RAW_TICKER_COLUMNS].rename(columns=RAW_TO_SAMPLE_COLUMNS)
    frame = frame.dropna(subset=["timestamp"])
    if frame.empty:
        return pd.DataFrame(columns=["timestamp", *TICKER_VALUE_COLUMNS])

    frame["timestamp"] = frame["timestamp"].astype("int64")
    for col in TICKER_VALUE_COLUMNS:
        frame[col] = frame[col].astype("float64")

    valid = valid_ticker_values_mask(
        frame.loc[:, TICKER_VALUE_COLUMNS].to_numpy(dtype="float64", copy=False)
    )
    frame = frame.loc[valid]
    if frame.empty:
        return pd.DataFrame(columns=["timestamp", *TICKER_VALUE_COLUMNS])

    if not frame["timestamp"].is_monotonic_increasing:
        frame = frame.sort_values("timestamp", kind="mergesort")

    return frame.reset_index(drop=True)


def valid_ticker_values_mask(values: np.ndarray) -> np.ndarray:
    if values.ndim != 2 or values.shape[1] != len(TICKER_VALUE_COLUMNS):
        raise ValueError(
            f"ticker values must be N x {len(TICKER_VALUE_COLUMNS)}, got {values.shape}"
        )
    if len(values) == 0:
        return np.zeros(0, dtype=bool)

    bid = values[:, 0]
    ask = values[:, 1]
    bid_qty = values[:, 2]
    ask_qty = values[:, 3]
    return (
        np.isfinite(values).all(axis=1)
        & (bid > 0.0)
        & (ask > 0.0)
        & (ask >= bid)
        & (bid_qty >= 0.0)
        & (ask_qty >= 0.0)
    )


def read_bookticker_frame(
    root: Any,
    symbol: str,
    date_str: str,
    category: str = "BOOKTICKER",
) -> pd.DataFrame:
    path = resolve_existing_input_path(
        roots=root,
        symbol=symbol,
        date_str=date_str,
        category=category,
    )
    raw_df = pd.read_parquet(path, columns=list(RAW_TICKER_COLUMNS))
    return normalize_bookticker_frame(raw_df)


def iter_bookticker_frames(path: Path) -> Any:
    try:
        import pyarrow.parquet as pq
    except ImportError:
        yield normalize_bookticker_frame(pd.read_parquet(path, columns=list(RAW_TICKER_COLUMNS)))
        return

    parquet_file = pq.ParquetFile(path)
    for row_group_idx in range(parquet_file.num_row_groups):
        table = parquet_file.read_row_group(
            row_group_idx,
            columns=list(RAW_TICKER_COLUMNS),
        )
        frame = normalize_bookticker_frame(table.to_pandas())
        if not frame.empty:
            yield frame


def iter_bookticker_arrays(path: Path) -> Any:
    value_raw_columns = ("price_bid", "price_ask", "volume_bid", "volume_ask")

    try:
        import pyarrow.parquet as pq
    except ImportError:
        for frame in iter_bookticker_frames(path):
            ts = frame["timestamp"].to_numpy(dtype="int64")
            values = frame.loc[:, TICKER_VALUE_COLUMNS].to_numpy(dtype="float64")
            yield ts, values
        return

    parquet_file = pq.ParquetFile(path)
    for row_group_idx in range(parquet_file.num_row_groups):
        table = parquet_file.read_row_group(
            row_group_idx,
            columns=list(RAW_TICKER_COLUMNS),
        )
        if table.num_rows == 0:
            continue

        ts = (
            table[RAW_TIME_COLUMN]
            .combine_chunks()
            .to_numpy(zero_copy_only=False)
            .astype("int64", copy=False)
        )
        values = np.column_stack(
            [
                table[col]
                .combine_chunks()
                .to_numpy(zero_copy_only=False)
                .astype("float64", copy=False)
                for col in value_raw_columns
            ]
        )

        valid = valid_ticker_values_mask(values)
        if not valid.all():
            ts = ts[valid]
            values = values[valid]
        if len(ts) == 0:
            continue

        if len(ts) > 1 and np.any(ts[1:] < ts[:-1]):
            order = np.argsort(ts, kind="mergesort")
            ts = ts[order]
            values = values[order]

        yield ts, values


def read_bookticker_tail(
    root: Any,
    symbol: str,
    date_str: str,
    category: str = "BOOKTICKER",
    before_timestamp_ms: int | None = None,
) -> pd.DataFrame | None:
    try:
        path = resolve_existing_input_path(
            roots=root,
            symbol=symbol,
            date_str=date_str,
            category=category,
        )
    except FileNotFoundError:
        return None

    try:
        import pyarrow.parquet as pq

        parquet_file = pq.ParquetFile(path)
        for row_group_idx in range(parquet_file.num_row_groups - 1, -1, -1):
            table = parquet_file.read_row_group(
                row_group_idx,
                columns=list(RAW_TICKER_COLUMNS),
            )
            tail = normalize_bookticker_frame(table.to_pandas())
            if before_timestamp_ms is not None and not tail.empty:
                tail = tail[tail["timestamp"] <= int(before_timestamp_ms)]
            if not tail.empty:
                return tail.tail(1).reset_index(drop=True)
        return None
    except Exception:
        raw_tail = pd.read_parquet(path, columns=list(RAW_TICKER_COLUMNS))

    if raw_tail.empty:
        return None

    tail = normalize_bookticker_frame(raw_tail)
    if before_timestamp_ms is not None and not tail.empty:
        tail = tail[tail["timestamp"] <= int(before_timestamp_ms)]
    if tail.empty:
        return None
    return tail.tail(1).reset_index(drop=True)


def previous_day_ticker_tail(
    root: Any,
    symbol: str,
    date_str: str,
    category: str = "BOOKTICKER",
    freq_ms: int | None = None,
    scheme_shift_ms: int = 0,
) -> pd.DataFrame | None:
    before_timestamp_ms = None
    if freq_ms is not None:
        grid = day_timestamp_grid(date_str, freq_ms, scheme_shift_ms)
        before_timestamp_ms = int(grid[0]) if len(grid) else None
    return read_bookticker_tail(
        root=root,
        symbol=symbol,
        date_str=previous_date_str(date_str),
        category=category,
        before_timestamp_ms=before_timestamp_ms,
    )


def atomic_write_parquet(
    df: pd.DataFrame,
    path: Path,
    compression: str = "snappy",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / f".{path.name}.tmp.{os.getpid()}.parquet"

    try:
        df.to_parquet(tmp_path, index=False, compression=compression)
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


class TickerResampler:
    """Resample bookticker snapshots without looking inside the future bucket."""

    def __init__(
        self,
        freq_ms: int,
        scheme_shift_ms: int = 0,
        value_columns: tuple[str, ...] = TICKER_VALUE_COLUMNS,
    ) -> None:
        if freq_ms <= 0:
            raise ValueError(f"freq_ms must be positive, got {freq_ms}")
        self.freq_ms = int(freq_ms)
        self.scheme_shift_ms = normalize_scheme_shift(scheme_shift_ms, self.freq_ms)
        self.value_columns = value_columns

    def resample(
        self,
        raw_df: pd.DataFrame,
        date_str: str,
        prev_tail: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        full_grid = pd.DataFrame(
            {"timestamp": day_timestamp_grid(date_str, self.freq_ms, self.scheme_shift_ms)}
        )
        if full_grid.empty:
            return self._empty_result(full_grid)

        ticker = raw_df
        if prev_tail is not None and not prev_tail.empty:
            ticker = pd.concat([prev_tail, ticker], ignore_index=True)

        if ticker.empty:
            return self._empty_result(full_grid)

        ticker = ticker.loc[:, ["timestamp", *self.value_columns]].dropna(subset=["timestamp"])
        ticker["timestamp"] = ticker["timestamp"].astype("int64")
        ticker = ticker.sort_values("timestamp", kind="mergesort")

        # Each sampled timestamp gets the latest known ticker at or before that
        # instant. It never uses the last update inside [t, t + freq).
        sampled = pd.merge_asof(
            full_grid,
            ticker,
            on="timestamp",
            direction="backward",
            allow_exact_matches=True,
        )
        return self._cast_result(sampled)

    def resample_file(
        self,
        path: Path,
        date_str: str,
        prev_tail: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        grid = day_timestamp_grid(date_str, self.freq_ms, self.scheme_shift_ms)
        sampled_values = np.full(
            (len(grid), len(self.value_columns)),
            np.nan,
            dtype="float64",
        )
        next_grid_pos = 0
        last_values: np.ndarray | None = None
        carry_ts: np.ndarray | None = None
        carry_values: np.ndarray | None = None

        if prev_tail is not None and not prev_tail.empty:
            tail_ts = int(prev_tail["timestamp"].iloc[-1])
            if len(grid) > 0 and tail_ts <= int(grid[0]):
                last_values = prev_tail.loc[:, self.value_columns].tail(1).to_numpy(dtype="float64")[0]
            else:
                carry_ts = np.array([tail_ts], dtype="int64")
                carry_values = prev_tail.loc[:, self.value_columns].tail(1).to_numpy(dtype="float64")

        def consume_arrays(ts: np.ndarray, values: np.ndarray, end_pos: int) -> None:
            nonlocal last_values, next_grid_pos

            if len(ts) == 0:
                return

            if end_pos > next_grid_pos:
                grid_slice = grid[next_grid_pos:end_pos]
                right_idx = np.searchsorted(ts, grid_slice, side="right") - 1
                has_current = right_idx >= 0
                target = sampled_values[next_grid_pos:end_pos]

                if has_current.any():
                    target[has_current] = values[right_idx[has_current]]
                if last_values is not None and (~has_current).any():
                    target[~has_current] = last_values

                next_grid_pos = end_pos

            last_values = values[-1]

        for ts, values in iter_bookticker_arrays(path):
            if carry_ts is not None and carry_values is not None:
                ts = np.concatenate([carry_ts, ts])
                values = np.vstack([carry_values, values])

            if len(ts) == 0:
                carry_ts = None
                carry_values = None
                continue

            if len(ts) > 1 and np.any(ts[1:] < ts[:-1]):
                order = np.argsort(ts, kind="mergesort")
                ts = ts[order]
                values = values[order]

            max_ts = int(ts[-1])
            stable_end = int(np.searchsorted(ts, max_ts, side="left"))
            stable_ts = ts[:stable_end]
            stable_values = values[:stable_end]
            carry_ts = ts[stable_end:]
            carry_values = values[stable_end:]

            if len(stable_ts) == 0:
                continue

            stable_end_pos = int(
                np.searchsorted(
                    grid,
                    int(stable_ts[-1]),
                    side="right",
                )
            )
            consume_arrays(stable_ts, stable_values, stable_end_pos)

        if carry_ts is not None and carry_values is not None and len(carry_ts) > 0:
            consume_arrays(carry_ts, carry_values, len(grid))
        elif last_values is not None and next_grid_pos < len(grid):
            sampled_values[next_grid_pos:] = last_values

        result = pd.DataFrame({"timestamp": grid})
        for idx, col in enumerate(self.value_columns):
            result[col] = sampled_values[:, idx]
        return self._cast_result(result)

    def _empty_result(self, grid: pd.DataFrame) -> pd.DataFrame:
        result = grid.copy()
        for col in self.value_columns:
            result[col] = np.nan
        return self._cast_result(result)

    def _cast_result(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.loc[:, ["timestamp", *self.value_columns]].copy()
        result["timestamp"] = result["timestamp"].astype("int64")
        for col in self.value_columns:
            result[col] = result[col].astype("float64")
        return result


def validate_sampled_ticker(
    df: pd.DataFrame,
    freq_ms: int,
    require_values: bool = True,
    date_str: str | None = None,
    scheme_shift_ms: int = 0,
) -> None:
    required = {"timestamp", *TICKER_VALUE_COLUMNS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"sampled ticker missing columns: {sorted(missing)}")

    if df.empty:
        return

    timestamps = df["timestamp"].to_numpy()
    if not np.all(np.diff(timestamps) >= 0):
        raise ValueError("sampled ticker timestamp is not monotonic increasing")

    steps = np.diff(timestamps)
    if len(steps) > 0 and not np.all(steps == freq_ms):
        bad_steps = steps[steps != freq_ms][:5]
        raise ValueError(
            f"timestamp step is not constant {freq_ms}ms; "
            f"found examples: {bad_steps.tolist()}"
        )

    if date_str is not None:
        expected_grid = day_timestamp_grid(date_str, freq_ms, scheme_shift_ms)
        if len(df) != len(expected_grid):
            raise ValueError(f"expected {len(expected_grid)} rows, got {len(df)}")
        if not np.array_equal(timestamps, expected_grid):
            raise ValueError("sampled ticker timestamp grid does not match scheme_shift")

    if require_values and df.loc[:, TICKER_VALUE_COLUMNS].isna().any().any():
        nan_counts = df.loc[:, TICKER_VALUE_COLUMNS].isna().sum()
        bad = {col: int(count) for col, count in nan_counts.items() if count > 0}
        raise ValueError(f"sampled ticker contains NaN values: {bad}")

    for col in ("best_bid_price", "best_ask_price"):
        if (df[col] <= 0).any():
            raise ValueError(f"found non-positive {col}")

    crossed = df["best_ask_price"] < df["best_bid_price"]
    if crossed.any():
        raise ValueError(f"found ask price below bid price in {int(crossed.sum())} rows")

    for col in ("best_bid_qty", "best_ask_qty"):
        if (df[col] < 0).any():
            raise ValueError(f"found negative {col}")


def run_one(task: Task) -> str:
    out_path = output_path(
        root=task.output_root,
        symbol=task.symbol,
        freq_ms=task.freq_ms,
        date_str=task.date,
        scheme_shift_ms=task.scheme_shift_ms,
    )

    if out_path.exists() and not task.overwrite:
        return (
            f"[skip] {task.symbol} {task.date} freq={task.freq_ms} "
            f"scheme_shift={task.scheme_shift_ms} -> {out_path}"
        )

    try:
        in_path = resolve_existing_input_path(
            roots=task.bookticker_roots,
            symbol=task.symbol,
            date_str=task.date,
            category=task.ticker_category,
        )

        prev_tail = previous_day_ticker_tail(
            root=task.bookticker_roots,
            symbol=task.symbol,
            date_str=task.date,
            category=task.ticker_category,
            freq_ms=task.freq_ms,
            scheme_shift_ms=task.scheme_shift_ms,
        )
        frame = TickerResampler(
            freq_ms=task.freq_ms,
            scheme_shift_ms=task.scheme_shift_ms,
        ).resample_file(
            path=in_path,
            date_str=task.date,
            prev_tail=prev_tail,
        )

        if task.strict_validate:
            validate_sampled_ticker(
                df=frame,
                freq_ms=task.freq_ms,
                require_values=prev_tail is not None,
                date_str=task.date,
                scheme_shift_ms=task.scheme_shift_ms,
            )

        atomic_write_parquet(frame, out_path, compression=task.compression)
        size = out_path.stat().st_size if out_path.exists() else 0
        return (
            f"[done] {task.symbol} {task.date} freq={task.freq_ms} "
            f"scheme_shift={task.scheme_shift_ms} "
            f"rows={len(frame)} size={size} bytes -> {out_path}"
        )
    except Exception as exc:
        return (
            f"[error] {task.symbol} {task.date} freq={task.freq_ms} "
            f"scheme_shift={task.scheme_shift_ms}: {exc}"
        )


def _config_path(cfg: dict[str, Any], *keys: str, default: Path) -> Path:
    return config_paths(cfg, *keys, default=default)[0]


def build_tasks(cfg: dict[str, Any]) -> list[Task]:
    symbols = [str(s).upper() for s in ensure_list(cfg.get("symbols"))]
    if not symbols:
        raise ValueError("config requires symbols")

    date_start = cfg.get("date_start")
    date_end = cfg.get("date_end", date_start)
    if not date_start:
        raise ValueError("config requires date_start")

    sampler_cfg = cfg.get("sampler", {})
    if not isinstance(sampler_cfg, dict):
        raise ValueError("config sampler section must be a dict")

    raw_freqs = sampler_cfg.get("freq")
    if _is_empty_config_value(raw_freqs):
        raw_freqs = cfg.get("freq_ms")
    if _is_empty_config_value(raw_freqs):
        raw_freqs = _split_stage_freqs(cfg)
    if _is_empty_config_value(raw_freqs):
        raw_freqs = cfg.get("freq", 1000)

    freqs = [int(x) for x in ensure_list(raw_freqs)]
    if not freqs:
        raise ValueError("sampler.freq must not be empty")
    for freq in freqs:
        if freq <= 0:
            raise ValueError(f"sampler.freq must be positive integer ms, got {freq}")

    ticker_category = normalize_category(
        sampler_cfg.get("ticker_category", cfg.get("ticker_category", "BOOKTICKER"))
    )
    bookticker_roots = config_paths(
        cfg,
        "bookticker_roots",
        "bookticker_root",
        "input_root",
        default=DEFAULT_BOOKTICKER_ROOT,
    )
    output_root = _config_path(
        cfg,
        "ticker_cache_root",
        "output_root",
        default=DEFAULT_OUTPUT_ROOT,
    )
    overwrite = bool(sampler_cfg.get("overwrite", cfg.get("overwrite", False)))
    strict_validate = bool(sampler_cfg.get("strict_validate", cfg.get("strict_validate", True)))
    compression = str(sampler_cfg.get("compression", cfg.get("compression", "snappy")))
    raw_scheme_shift = sampler_cfg.get("scheme_shift", cfg.get("scheme_shift", [0]))

    tasks = []
    for symbol, date_str, freq_ms in product(symbols, generate_dates(date_start, date_end), freqs):
        for scheme_shift_ms in normalize_scheme_shift_list(raw_scheme_shift, freq_ms):
            tasks.append(
                Task(
                    symbol=symbol,
                    date=date_str,
                    freq_ms=freq_ms,
                    scheme_shift_ms=scheme_shift_ms,
                    bookticker_roots=bookticker_roots,
                    ticker_category=ticker_category,
                    output_root=output_root,
                    overwrite=overwrite,
                    strict_validate=strict_validate,
                    compression=compression,
                )
            )
    return tasks


def run_all(cfg: dict[str, Any]) -> None:
    tasks = build_tasks(cfg)
    print(f"total sampler tasks: {len(tasks)}")
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
        raise RuntimeError(f"{len(errors)} sampler task(s) failed:\n{preview}")


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
                "bookticker_roots": [
                    path
                    for path in (
                        args.bookticker_root or str(DEFAULT_BOOKTICKER_ROOT),
                        args.bookticker_backup_root,
                    )
                    if path
                ],
                "output_root": args.output_root or str(DEFAULT_OUTPUT_ROOT),
            },
            "sampler": {
                "freq": args.freq,
                "scheme_shift": [0] if raw_scheme_shift is None else raw_scheme_shift,
                "ticker_category": args.ticker_category or "BOOKTICKER",
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

    if args.output_root is not None:
        cfg.setdefault("paths", {})["output_root"] = args.output_root
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
                    "input_root",
                    default=DEFAULT_BOOKTICKER_ROOT,
                )
            )
        if args.bookticker_backup_root is not None:
            roots.append(Path(args.bookticker_backup_root))
        cfg.setdefault("paths", {})["bookticker_roots"] = [
            str(path) for path in dict.fromkeys(roots)
        ]
    cfg.setdefault("sampler", {})
    if args.ticker_category is not None:
        cfg["sampler"]["ticker_category"] = args.ticker_category
    if raw_scheme_shift is not None:
        cfg["sampler"]["scheme_shift"] = raw_scheme_shift
    if args.overwrite:
        cfg["sampler"]["overwrite"] = True
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Resample Binance bookticker data")
    parser.add_argument("--config", help="optional JSON config path")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT"])
    parser.add_argument("--date-start")
    parser.add_argument("--date-end")
    parser.add_argument("--freq", nargs="+", type=int, default=[1000])
    parser.add_argument("--scheme-shift", nargs="+", type=int)
    parser.add_argument("--bookticker-root")
    parser.add_argument("--bookticker-backup-root")
    parser.add_argument("--ticker-category")
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
            print(
                f"  - {task.symbol} {task.date} freq={task.freq_ms}ms "
                f"scheme_shift={task.scheme_shift_ms}ms "
                f"input={input_candidate_paths(task.bookticker_roots, task.symbol, task.date, task.ticker_category)} "
                f"output={output_path(task.output_root, task.symbol, task.freq_ms, task.date, task.scheme_shift_ms)}"
            )
        if len(tasks) > 10:
            print(f"  ... and {len(tasks) - 10} more")
        return

    run_all(cfg)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"resample failed: {exc}", file=sys.stderr)
        raise
