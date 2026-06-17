from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


_DATE_KEY_RE = re.compile(r"\d{4}-?\d{2}-?\d{2}")
_MAX_POSITION_RE = re.compile(r"(?:^|__)mp([0-9]+(?:p[0-9]+)?(?:e[+-]?[0-9]+)?)(?:$|__)")
_PNL_REPORT_COLUMNS = (
    "timestamp",
    "datetime",
    "total_pnl",
    "real_pnl",
    "realized_pnl",
    "unreal_pnl",
    "unrealized_pnl",
    "traded_volume",
)
_FULL_REPORT_COLUMNS = (
    *_PNL_REPORT_COLUMNS,
    "price",
    "best_bid_price",
    "best_ask_price",
    "mark_notional_usdt",
    "cost_notional_usdt",
    "gross_cost_notional_usdt",
    "inventory",
    "position",
)


@dataclass(slots=True)
class BacktestReport:
    metrics: dict[str, float] = field(default_factory=dict)
    orders: list[dict[str, Any]] = field(default_factory=list)
    fills: list[dict[str, Any]] = field(default_factory=list)
    equity_curve: list[tuple[int, float]] = field(default_factory=list)

    def add_metric(self, name: str, value: float) -> None:
        self.metrics[name] = value


def _normalize_date_key(value: str | None) -> str | None:
    if value is None:
        return None
    digits = re.sub(r"\D", "", str(value))
    if len(digits) < 8:
        raise ValueError(f"invalid date input: {value}, expected YYYY-MM-DD or YYYYMMDD")
    return digits[:8]


def _normalize_date_key_set(value: str | Sequence[str] | None) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        values = [value]
    else:
        values = list(value)

    out: set[str] = set()
    for item in values:
        key = _normalize_date_key(item)
        if key is not None:
            out.add(key)
    return out


def _extract_date_key_from_name(name: str) -> str | None:
    date_match = _DATE_KEY_RE.search(name)
    if date_match:
        return _normalize_date_key(date_match.group(0))

    digits = re.sub(r"\D", "", name)
    if len(digits) >= 8:
        return digits[:8]
    return None


def _as_folder_list(folders: str | Path | Sequence[str | Path]) -> list[str]:
    if isinstance(folders, (str, Path)):
        return [str(folders)]
    return [str(folder) for folder in folders]


def _list_filtered_parquet_files(
    folder: str | Path,
    srt: str | None = None,
    end: str | None = None,
    exclude_date: str | Sequence[str] | None = None,
) -> list[Path]:
    folder_path = Path(folder)
    files = sorted(path for path in folder_path.glob("*.parquet") if path.is_file())

    srt_key = _normalize_date_key(srt)
    end_key = _normalize_date_key(end)
    exclude_keys = _normalize_date_key_set(exclude_date)

    if srt_key is None and end_key is None and not exclude_keys:
        return files

    filtered: list[Path] = []
    for path in files:
        key = _extract_date_key_from_name(path.stem)
        if key is None:
            continue
        if key in exclude_keys:
            continue
        if srt_key is not None and key < srt_key:
            continue
        if end_key is not None and key > end_key:
            continue
        filtered.append(path)
    return filtered


def _normalize_report_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    renames: dict[str, str] = {}
    if "real_pnl" not in out.columns and "realized_pnl" in out.columns:
        renames["realized_pnl"] = "real_pnl"
    if "unreal_pnl" not in out.columns and "unrealized_pnl" in out.columns:
        renames["unrealized_pnl"] = "unreal_pnl"

    if renames:
        out = out.rename(columns=renames)

    if "total_pnl" not in out.columns:
        if {"real_pnl", "unreal_pnl"}.issubset(out.columns):
            out["total_pnl"] = out["real_pnl"] + out["unreal_pnl"]
        elif "real_pnl" in out.columns:
            out["total_pnl"] = out["real_pnl"]

    if "position" not in out.columns and "inventory" in out.columns:
        out["position"] = out["inventory"]

    if "price" not in out.columns and {"best_bid_price", "best_ask_price"}.issubset(out.columns):
        out["price"] = 0.5 * (out["best_bid_price"] + out["best_ask_price"])
    elif "price" not in out.columns and {"position", "unreal_pnl"}.issubset(out.columns):
        inv = out["position"].astype("float64")
        unreal = out["unreal_pnl"].astype("float64")
        valid = inv.abs() > 0.0
        approx = pd.Series(np.nan, index=out.index, dtype="float64")
        approx.loc[valid] = (unreal.loc[valid].diff() / inv.loc[valid]).fillna(0.0)
        out["price"] = approx.ffill().bfill()

    if "mark_notional_usdt" not in out.columns and {"position", "price"}.issubset(out.columns):
        out["mark_notional_usdt"] = (
            out["position"].astype("float64") * out["price"].astype("float64")
        )

    if (
        "cost_notional_usdt" not in out.columns
        and {"mark_notional_usdt", "unreal_pnl"}.issubset(out.columns)
    ):
        out["cost_notional_usdt"] = (
            out["mark_notional_usdt"].astype("float64")
            - out["unreal_pnl"].astype("float64")
        )

    if "gross_cost_notional_usdt" not in out.columns and "cost_notional_usdt" in out.columns:
        out["gross_cost_notional_usdt"] = out["cost_notional_usdt"].abs()

    if "cost_notional_usdt" in out.columns:
        out["inventory"] = out["cost_notional_usdt"]
    elif "inventory" not in out.columns and "position" in out.columns:
        out["inventory"] = out["position"]

    if "datetime" not in out.columns and "timestamp" in out.columns:
        out["datetime"] = pd.to_datetime(out["timestamp"], unit="ms")

    return out


def _parquet_columns(path: Path) -> set[str]:
    import pyarrow.parquet as pq

    return set(pq.read_schema(path).names)


def _read_report_parquet(path: Path, pnl_only: bool) -> pd.DataFrame:
    requested = _PNL_REPORT_COLUMNS if pnl_only else _FULL_REPORT_COLUMNS
    available = _parquet_columns(path)
    columns = [column for column in requested if column in available]
    if not columns:
        raise ValueError(f"no report columns found in parquet file: {path}")
    return pd.read_parquet(path, engine="pyarrow", columns=columns)


def _sample_report_frame(
    df: pd.DataFrame,
    every: int,
    row_offset: int,
) -> tuple[pd.DataFrame, int]:
    row_count = len(df)
    next_offset = (row_offset + row_count) % every
    if row_count == 0:
        return df, next_offset
    if every <= 1:
        return df.copy(), next_offset

    first_stride_row = (-row_offset) % every
    stride_rows = np.arange(first_stride_row, row_count, every, dtype=np.int64)
    keep_rows = np.unique(np.concatenate(([0, row_count - 1], stride_rows)))
    return df.iloc[keep_rows].copy(), next_offset


def load_report_frame(
    folder: str | Path,
    every: int = 100,
    srt: str | None = None,
    end: str | None = None,
    exclude_date: str | Sequence[str] | None = None,
    pnl_only: bool = False,
) -> pd.DataFrame:
    if every <= 0:
        raise ValueError(f"every must be > 0, got {every}")

    files = _list_filtered_parquet_files(folder=folder, srt=srt, end=end, exclude_date=exclude_date)
    if not files:
        raise FileNotFoundError(f"no matching parquet files found under: {folder}")

    frames: list[pd.DataFrame] = []
    row_offset = 0
    for path in files:
        frame = _read_report_parquet(path, pnl_only=pnl_only)
        frame, row_offset = _sample_report_frame(frame, every=every, row_offset=row_offset)
        if not frame.empty:
            frames.append(frame)

    if not frames:
        raise FileNotFoundError(f"no non-empty matching parquet files found under: {folder}")

    df = pd.concat(frames, ignore_index=True)

    return _normalize_report_columns(df)


def get_folders(root_dir: str | Path, symbol: str) -> list[str]:
    root = Path(root_dir)
    symbol_upper = symbol.upper()

    leaf_folders: list[str] = []
    for current_root, _dirs, files in os.walk(root):
        root_upper = str(current_root).upper()
        if symbol_upper in root_upper:
            parquet_names = [name for name in files if name.endswith(".parquet")]
            if parquet_names:
                leaf_folders.append(str(Path(current_root)))

    return sorted(leaf_folders)


def get_folders_with_counts(
    root_dir: str | Path,
    symbol: str,
    srt: str | None = None,
    end: str | None = None,
    exclude_date: str | Sequence[str] | None = None,
    keyword: str | None = None,
) -> tuple[list[str], list[int]]:
    root = Path(root_dir)
    symbol_upper = symbol.upper()
    srt_key = _normalize_date_key(srt)
    end_key = _normalize_date_key(end)
    exclude_keys = _normalize_date_key_set(exclude_date)

    folders: list[str] = []
    file_counts: list[int] = []

    for current_root, _dirs, files in os.walk(root):
        current_root_str = str(current_root)
        root_upper = current_root_str.upper()
        if symbol_upper not in root_upper:
            continue
        if keyword is not None and keyword not in current_root_str:
            continue

        parquet_names = [name for name in files if name.endswith(".parquet")]
        if not parquet_names:
            continue

        date_keys = {
            key for key in (_extract_date_key_from_name(Path(name).stem) for name in parquet_names)
            if key is not None
        }

        if srt_key is not None and srt_key not in date_keys:
            continue
        if end_key is not None and end_key not in date_keys:
            continue

        filtered_count = sum(
            1
            for name in parquet_names
            for key in [_extract_date_key_from_name(Path(name).stem)]
            if key is not None
            and key not in exclude_keys
            and (srt_key is None or key >= srt_key)
            and (end_key is None or key <= end_key)
        )
        if filtered_count == 0:
            continue

        folders.append(str(Path(current_root)))
        file_counts.append(filtered_count)

    sorted_pairs = sorted(zip(folders, file_counts), key=lambda x: x[0])
    if not sorted_pairs:
        return [], []

    sorted_folders, sorted_counts = zip(*sorted_pairs)
    return list(sorted_folders), list(sorted_counts)


def _line_exists(axis: Any) -> bool:
    return len(axis.lines) > 0


def _parse_max_position_from_folder(folder: str | Path) -> float:
    path = Path(folder)
    for part in reversed(path.parts):
        match = _MAX_POSITION_RE.search(part.lower())
        if match is None:
            continue
        token = match.group(1).replace("p", ".")
        value = float(token)
        if value == 0.0:
            raise ValueError(f"parsed max_position is zero from folder name: {folder}")
        return value
    raise ValueError(f"cannot parse max_position from folder path: {folder}")


def _compute_daily_total(df: pd.DataFrame) -> pd.Series:
    if "total_pnl" not in df.columns or "datetime" not in df.columns or df.empty:
        return pd.Series(dtype="float64")

    return (
        df.assign(_date=pd.to_datetime(df["datetime"]).dt.floor("D"))
        .groupby("_date", sort=True)["total_pnl"]
        .last()
        .astype("float64")
    )


def _compute_daily_pnl_from_total(daily_total: pd.Series) -> pd.Series:
    if daily_total.empty:
        return pd.Series(dtype="float64")
    return daily_total.diff().dropna().astype("float64")


def _compute_sharpe_from_daily_pnl(daily_pnl: pd.Series) -> float:
    if len(daily_pnl) < 2:
        return float("nan")

    std = float(daily_pnl.std(ddof=1))
    if std == 0.0 or not np.isfinite(std):
        return float("nan")
    return float(daily_pnl.mean() / std)


def _compute_max_drawdown_from_daily_total(daily_total: pd.Series) -> float:
    if daily_total.empty:
        return float("nan")

    values = daily_total.to_numpy(dtype="float64")
    if values.size < 2:
        return float("nan")

    running_max = np.maximum.accumulate(values)
    drawdowns = running_max - values
    max_drawdown = float(np.max(drawdowns))
    if not np.isfinite(max_drawdown):
        return float("nan")
    return max_drawdown


def _compute_annual_return_from_daily_pnl(
    daily_pnl: pd.Series,
    periods_per_year: float = 365.0,
) -> float:
    if daily_pnl.empty:
        return float("nan")
    return float(daily_pnl.mean() * periods_per_year)


def _compute_daily_traded_volume(df: pd.DataFrame) -> pd.Series:
    if "traded_volume" not in df.columns or "datetime" not in df.columns or df.empty:
        return pd.Series(dtype="float64")

    daily_turnover = (
        df.assign(_date=pd.to_datetime(df["datetime"]).dt.floor("D"))
        .groupby("_date", sort=True)["traded_volume"]
        .agg(lambda s: float(s.iloc[-1]) - float(s.iloc[0]))
        .astype("float64")
    )
    return daily_turnover


def _compute_spnl_curve(df: pd.DataFrame) -> pd.Series:
    if "total_pnl" not in df.columns or "traded_volume" not in df.columns or df.empty:
        return pd.Series(dtype="float64")

    pnl = df["total_pnl"].astype("float64")
    volume = df["traded_volume"].astype("float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        spnl = pnl / volume
    spnl = spnl.replace([np.inf, -np.inf], np.nan)
    return spnl.astype("float64")


def _compute_daily_series_from_states(
    folder: str | Path,
    srt: str | None = None,
    end: str | None = None,
    exclude_date: str | Sequence[str] | None = None,
) -> tuple[pd.Series, pd.Series]:
    state_dir = Path(folder) / "_state"
    if not state_dir.is_dir():
        raise FileNotFoundError(f"no _state directory found under: {folder}")

    files = sorted(path for path in state_dir.glob("*.json") if path.is_file())
    srt_key = _normalize_date_key(srt)
    end_key = _normalize_date_key(end)
    exclude_keys = _normalize_date_key_set(exclude_date)

    dates: list[str] = []
    total_pnls: list[float] = []
    traded_volumes: list[float] = []

    for path in files:
        key = _extract_date_key_from_name(path.stem)
        if key is None:
            continue
        if key in exclude_keys:
            continue
        if srt_key is not None and key < srt_key:
            continue
        if end_key is not None and key > end_key:
            continue

        with open(path, "r", encoding="utf-8") as fh:
            state = json.load(fh)

        total_raw = state.get("total_pnl")
        if total_raw is not None:
            try:
                total = float(total_raw)
                if not np.isfinite(total):
                    continue
            except (TypeError, ValueError):
                continue
        else:
            pos = state.get("position")
            if not isinstance(pos, dict):
                continue
            try:
                realized = float(pos.get("realized_pnl", 0.0))
                unrealized = float(pos.get("unrealized_pnl", 0.0))
                total = realized + unrealized
                if not np.isfinite(total):
                    continue
            except (TypeError, ValueError):
                continue

        vol_raw = state.get("traded_volume")
        vol = 0.0
        if vol_raw is not None:
            try:
                vol = float(vol_raw)
                if not np.isfinite(vol):
                    vol = 0.0
            except (TypeError, ValueError):
                vol = 0.0

        dates.append(key)
        total_pnls.append(total)
        traded_volumes.append(vol)

    if not dates:
        raise FileNotFoundError(f"no valid state data found in: {folder}/_state")

    sorted_tuples = sorted(zip(dates, total_pnls, traded_volumes))
    sorted_dates, sorted_totals, sorted_volumes = zip(*sorted_tuples)

    daily_total = pd.Series(sorted_totals, index=list(sorted_dates), dtype="float64")
    daily_volume_cum = pd.Series(sorted_volumes, index=list(sorted_dates), dtype="float64")
    return daily_total, daily_volume_cum


def _summarize_run_metrics(
    daily_total: pd.Series,
    daily_turnover_series: pd.Series,
    final_pnl: float,
    spnl: float,
) -> tuple[float, float, float, float, float, float, float]:
    daily_pnl = _compute_daily_pnl_from_total(daily_total)
    max_drawdown = _compute_max_drawdown_from_daily_total(daily_total)
    annual_return = _compute_annual_return_from_daily_pnl(daily_pnl)
    sharpe = _compute_sharpe_from_daily_pnl(daily_pnl)
    if max_drawdown > 0.0 and np.isfinite(max_drawdown) and np.isfinite(annual_return):
        calmar = float(annual_return / max_drawdown)
    else:
        calmar = float("nan")
    daily_turnover = (
        float(daily_turnover_series.mean())
        if not daily_turnover_series.empty
        else float("nan")
    )
    return final_pnl, sharpe, calmar, max_drawdown, annual_return, daily_turnover, spnl


def _load_pyplot() -> Any:
    import matplotlib.pyplot as plt

    return plt


def report(
    folders: str | Path | Sequence[str | Path],
    every: int = 100,
    pnl_only: bool = False,
    plot_flag: bool = True,
    srt: str | None = None,
    end: str | None = None,
    exclude_date: str | Sequence[str] | None = None,
    labels: Sequence[str] | None = None,
    normalize: bool = False,
    num_workers: int = 1,
) -> dict[str, np.ndarray]:
    folder_list = _as_folder_list(folders)
    if labels is not None and len(labels) != len(folder_list):
        raise ValueError("labels length must match folders length")
    if num_workers <= 0:
        raise ValueError(f"num_workers must be > 0, got {num_workers}")

    runs: list[tuple[str, pd.DataFrame, float]] = []
    final_pnls: list[float] = []
    sharpes: list[float] = []
    calmars: list[float] = []
    max_drawdowns: list[float] = []
    annual_returns: list[float] = []
    daily_turnovers: list[float] = []
    spnls: list[float] = []

    def append_metrics(summary: tuple[float, float, float, float, float, float, float]) -> None:
        final_pnl, sharpe, calmar, max_drawdown, annual_return, daily_turnover, spnl = summary
        max_drawdowns.append(max_drawdown)
        annual_returns.append(annual_return)
        sharpes.append(sharpe)
        calmars.append(calmar)
        daily_turnovers.append(daily_turnover)
        spnls.append(spnl)
        final_pnls.append(final_pnl)

    def prepare_plot_run(
        item: tuple[int, str],
    ) -> tuple[str, pd.DataFrame, float, tuple[float, float, float, float, float, float, float]] | None:
        i, folder = item
        label = labels[i] if labels is not None else str(i)
        pnl_scale = _parse_max_position_from_folder(folder) if normalize else 1.0

        try:
            df = load_report_frame(
                folder=folder,
                every=every,
                srt=srt,
                end=end,
                exclude_date=exclude_date,
                pnl_only=pnl_only,
            )
        except FileNotFoundError:
            return None

        try:
            daily_total, daily_volume_cum = _compute_daily_series_from_states(
                folder,
                srt=srt,
                end=end,
                exclude_date=exclude_date,
            )
            daily_turnover_series = daily_volume_cum.diff().dropna().astype("float64")
            final_pnl = float(daily_total.iloc[-1]) if not daily_total.empty else float("nan")
            last_vol = float(daily_volume_cum.iloc[-1]) if not daily_volume_cum.empty else 0.0
            if last_vol > 0 and np.isfinite(last_vol) and np.isfinite(final_pnl):
                spnl = final_pnl / last_vol
            else:
                spnl = float("nan")
        except (FileNotFoundError, ValueError):
            daily_total = _compute_daily_total(df)
            daily_turnover_series = _compute_daily_traded_volume(df)
            spnl_curve = _compute_spnl_curve(df)

            final_pnl = (
                float(df["total_pnl"].iloc[-1])
                if "total_pnl" in df.columns and not df.empty
                else float("nan")
            )
            spnl = (
                float(spnl_curve.iloc[-1])
                if not spnl_curve.empty and np.isfinite(spnl_curve.iloc[-1])
                else float("nan")
            )

        summary = _summarize_run_metrics(
            daily_total=daily_total,
            daily_turnover_series=daily_turnover_series,
            final_pnl=final_pnl,
            spnl=spnl,
        )
        return label, df, pnl_scale, summary

    if plot_flag:
        plot_items = list(enumerate(folder_list))
        if num_workers > 1 and len(plot_items) > 1:
            with ThreadPoolExecutor(max_workers=min(num_workers, len(plot_items))) as executor:
                prepared_runs = list(executor.map(prepare_plot_run, plot_items))
        else:
            prepared_runs = [prepare_plot_run(item) for item in plot_items]

        for prepared in prepared_runs:
            if prepared is None:
                continue
            label, df, pnl_scale, summary = prepared
            runs.append((label, df, pnl_scale))
            append_metrics(summary)
    else:
        for folder in folder_list:
            try:
                daily_total, daily_volume_cum = _compute_daily_series_from_states(
                    folder,
                    srt=srt,
                    end=end,
                    exclude_date=exclude_date,
                )
            except (FileNotFoundError, ValueError):
                continue

            daily_turnover_series = daily_volume_cum.diff().dropna().astype("float64")
            final_pnl = float(daily_total.iloc[-1]) if not daily_total.empty else float("nan")

            last_vol = float(daily_volume_cum.iloc[-1]) if not daily_volume_cum.empty else 0.0
            if last_vol > 0 and np.isfinite(last_vol) and np.isfinite(final_pnl):
                spnl = final_pnl / last_vol
            else:
                spnl = float("nan")

            summary = _summarize_run_metrics(
                daily_total=daily_total,
                daily_turnover_series=daily_turnover_series,
                final_pnl=final_pnl,
                spnl=spnl,
            )
            append_metrics(summary)

    if plot_flag and not runs:
        raise ValueError("no runs to report; check folders and date filters")
    if not plot_flag and not final_pnls:
        raise ValueError("no runs to report; check folders and date filters")

    if plot_flag:
        plt = _load_pyplot()
        if pnl_only:
            _fig, ax = plt.subplots(1, 1, figsize=(14, 4))
            axes = [ax]
        else:
            _fig, axes = plt.subplots(6, 1, figsize=(14, 12), sharex=True)

        for label, df, pnl_scale in runs:
            x = df["datetime"] if "datetime" in df.columns else np.arange(len(df))

            if pnl_only:
                if "total_pnl" in df.columns:
                    axes[0].plot(
                        x,
                        (df["total_pnl"] - df["total_pnl"].iloc[0]) / pnl_scale,
                        label=label,
                    )
                continue

            if "price" in df.columns:
                axes[0].plot(x, df["price"])

            if "cost_notional_usdt" in df.columns:
                axes[1].plot(x, df["cost_notional_usdt"])
            elif "inventory" in df.columns:
                axes[1].plot(x, df["inventory"])

            if "real_pnl" in df.columns:
                axes[2].plot(x, (df["real_pnl"] - df["real_pnl"].iloc[0]) / pnl_scale)

            if "unreal_pnl" in df.columns:
                axes[3].plot(x, df["unreal_pnl"] / pnl_scale)

            if "total_pnl" in df.columns:
                axes[4].plot(
                    x,
                    (df["total_pnl"] - df["total_pnl"].iloc[0]) / pnl_scale,
                    label=label,
                )

            if "traded_volume" in df.columns:
                axes[5].plot(x, df["traded_volume"] - df["traded_volume"].iloc[0], label=label)

        if pnl_only:
            axes[0].set_title("Total PnL / Max Position" if normalize else "Total PnL")
            axes[0].axhline(0, linestyle="--", alpha=0.5)
            if _line_exists(axes[0]):
                axes[0].legend(title="run_id")
        else:
            axes[0].set_title("Price")
            axes[1].set_title("Inventory Cost Notional")
            if normalize:
                axes[2].set_title("Realized PnL / Max Position")
                axes[3].set_title("Unrealized PnL / Max Position")
                axes[4].set_title("Total PnL / Max Position")
            else:
                axes[2].set_title("Realized PnL")
                axes[3].set_title("Unrealized PnL")
                axes[4].set_title("Total PnL")
            axes[4].axhline(0, linestyle="--", alpha=0.5)
            axes[5].set_title("Traded Volume")

            if _line_exists(axes[4]):
                axes[4].legend(title="run_id")
            if _line_exists(axes[5]):
                axes[5].legend(title="run_id")

        for axis in axes:
            axis.grid()

        plt.tight_layout()
        plt.show()

    return {
        "pnls": np.asarray(final_pnls, dtype="float64"),
        "sharpes": np.asarray(sharpes, dtype="float64"),
        "calmars": np.asarray(calmars, dtype="float64"),
        "max_drawdown": np.asarray(max_drawdowns, dtype="float64"),
        "annual_return": np.asarray(annual_returns, dtype="float64"),
        "calmar": np.asarray(calmars, dtype="float64"),
        "sharpe": np.asarray(sharpes, dtype="float64"),
        "daily_turnover": np.asarray(daily_turnovers, dtype="float64"),
        "spnl": np.asarray(spnls, dtype="float64"),
    }


__all__ = [
    "BacktestReport",
    "get_folders",
    "get_folders_with_counts",
    "load_report_frame",
    "report",
]
