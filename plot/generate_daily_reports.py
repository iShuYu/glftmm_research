from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
PLOT_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sim.report import report  # noqa: E402


def _date_key(value: str | None) -> str | None:
    if value is None:
        return None
    digits = re.sub(r"\D", "", str(value))
    if len(digits) < 8:
        raise ValueError(f"invalid date: {value}, expected YYYY-MM-DD or YYYYMMDD")
    return digits[:8]


def _date_dash(value: str) -> str:
    key = _date_key(value)
    assert key is not None
    return f"{key[:4]}-{key[4:6]}-{key[6:8]}"


def _load_config(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _default_result_root(config_path: Path) -> Path:
    cfg = _load_config(config_path)
    paths = cfg.get("paths")
    if not isinstance(paths, dict) or not paths.get("output_dir"):
        raise ValueError(f"config missing paths.output_dir: {config_path}")
    return Path(str(paths["output_dir"])).expanduser().resolve()


def _discover_folders(
    result_root: Path,
    symbols: set[str] | None,
    keyword: str | None,
) -> list[Path]:
    folders = sorted(path for path in result_root.glob("*/*/strat__all") if path.is_dir())
    out: list[Path] = []
    for folder in folders:
        symbol = folder.parts[-3]
        sim_dir = folder.parts[-2]
        if symbols is not None and symbol.upper() not in symbols:
            continue
        if keyword is not None and keyword not in sim_dir:
            continue
        out.append(folder)
    return out


def _completed_days(folder: Path) -> list[str]:
    parquet_days = {path.stem for path in folder.glob("*.parquet")}
    state_days = {path.stem for path in (folder / "_state").glob("*.json")}
    return sorted(parquet_days & state_days)


def _filter_days(
    days: list[str],
    *,
    start: str | None,
    end: str | None,
    exact_days: set[str] | None,
    limit: int | None,
) -> list[str]:
    start_key = _date_key(start)
    end_key = _date_key(end)
    exact_keys = {_date_key(day) for day in exact_days} if exact_days else None
    out: list[str] = []
    for day in days:
        key = _date_key(day)
        if key is None:
            continue
        if start_key is not None and key < start_key:
            continue
        if end_key is not None and key > end_key:
            continue
        if exact_keys is not None and key not in exact_keys:
            continue
        out.append(day)
    if limit is not None:
        out = out[:limit]
    return out


def _state_row(folder: Path, symbol: str, sim_dir: str, day: str, png_path: Path) -> dict[str, Any]:
    row: dict[str, Any] = {
        "symbol": symbol,
        "sim_dir": sim_dir,
        "day": day,
        "png": png_path.name,
        "png_path": str(png_path),
    }
    state_path = folder / "_state" / f"{day}.json"
    if not state_path.exists():
        return row
    with open(state_path, "r", encoding="utf-8") as fh:
        state = json.load(fh)
    row["total_pnl"] = float(state.get("total_pnl", np.nan))
    row["realized_pnl"] = float(state.get("realized_pnl", np.nan))
    row["unrealized_pnl"] = float(state.get("unrealized_pnl", np.nan))
    row["traded_volume"] = float(state.get("traded_volume", np.nan))
    pos = state.get("position", {}) if isinstance(state.get("position"), dict) else {}
    row["final_position"] = float(pos.get("qty", np.nan))
    row["mark_notional_usdt"] = float(pos.get("mark_notional_usdt", np.nan))
    row["cost_notional_usdt"] = float(pos.get("cost_notional_usdt", np.nan))
    return row


def _write_report_png(
    *,
    folder: Path,
    day: str,
    png_path: Path,
    every: int,
    pnl_only: bool,
    normalize: bool,
    max_plot_points: int | None,
    dpi: int,
) -> dict[str, float]:
    plt.close("all")
    stats = report(
        [folder],
        every=every,
        pnl_only=pnl_only,
        plot_flag=True,
        srt=day,
        end=day,
        labels=[day],
        normalize=normalize,
        num_workers=1,
        max_plot_points=max_plot_points,
    )
    fig = plt.gcf()
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return {key: float(value[0]) if len(value) else np.nan for key, value in stats.items()}


def _run_folder(folder: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    symbol = folder.parts[-3]
    sim_dir = folder.parts[-2]
    out_dir = args.output_dir / symbol / sim_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    days = _filter_days(
        _completed_days(folder),
        start=args.start,
        end=args.end,
        exact_days=set(args.date) if args.date else None,
        limit=args.limit,
    )

    manifest = {
        "symbol": symbol,
        "sim_dir": sim_dir,
        "source_folder": str(folder),
        "output_folder": str(out_dir),
        "completed_day_count_at_start": len(days),
        "first_day": days[0] if days else None,
        "last_day": days[-1] if days else None,
        "days": days,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"{symbol}/{sim_dir}: {len(days)} days", flush=True)
    rows: list[dict[str, Any]] = []
    for idx, day in enumerate(days, start=1):
        png_path = out_dir / f"{day}.png"
        row = _state_row(folder, symbol, sim_dir, day, png_path)
        try:
            if args.dry_run:
                row["status"] = "dry_run"
            elif png_path.exists() and not args.force:
                row["status"] = "exists"
            else:
                row.update(
                    _write_report_png(
                        folder=folder,
                        day=day,
                        png_path=png_path,
                        every=args.every,
                        pnl_only=args.pnl_only,
                        normalize=args.normalize,
                        max_plot_points=args.max_plot_points,
                        dpi=args.dpi,
                    )
                )
                row["status"] = "written"
        except Exception as exc:  # Keep going when one day is bad or still being written.
            row["status"] = "error"
            row["error"] = repr(exc)
        rows.append(row)
        if idx == 1 or idx == len(days) or idx % args.progress_every == 0:
            print(f"  [{idx}/{len(days)}] {day} {row['status']}", flush=True)

    pd.DataFrame(rows).to_csv(out_dir / "summary.csv", index=False)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate one daily report PNG per completed strategy day.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "config" / "config.json",
        help="Strategy config used to resolve paths.output_dir.",
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=None,
        help="Override result root. Defaults to paths.output_dir from --config.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PLOT_DIR,
        help="Where to write plots. Defaults to this plot directory.",
    )
    parser.add_argument("--symbol", action="append", help="Filter symbol, repeatable.")
    parser.add_argument("--keyword", help="Filter parameter directory by substring.")
    parser.add_argument("--start", help="Start date, YYYY-MM-DD or YYYYMMDD.")
    parser.add_argument("--end", help="End date, YYYY-MM-DD or YYYYMMDD.")
    parser.add_argument("--date", action="append", help="Exact date to plot, repeatable.")
    parser.add_argument("--limit", type=int, help="Limit days per parameter folder.")
    parser.add_argument("--every", type=int, default=50, help="Report row sampling stride.")
    parser.add_argument("--max-plot-points", type=int, default=50_000)
    parser.add_argument("--dpi", type=int, default=130)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--force", action="store_true", help="Regenerate existing PNGs.")
    parser.add_argument("--dry-run", action="store_true", help="List work without plotting.")
    parser.add_argument("--pnl-only", action="store_true", help="Plot only total PnL.")
    parser.add_argument("--normalize", action="store_true", help="Normalize PnL by max position.")
    args = parser.parse_args()

    args.config = args.config.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.result_root = (
        args.result_root.expanduser().resolve()
        if args.result_root is not None
        else _default_result_root(args.config)
    )
    args.max_plot_points = None if args.max_plot_points == 0 else args.max_plot_points
    if args.every <= 0:
        raise ValueError("--every must be > 0")
    if args.dpi <= 0:
        raise ValueError("--dpi must be > 0")
    if args.progress_every <= 0:
        raise ValueError("--progress-every must be > 0")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be > 0")
    if args.date:
        args.date = [_date_dash(day) for day in args.date]
    return args


def main() -> None:
    args = parse_args()
    symbols = {symbol.upper() for symbol in args.symbol} if args.symbol else None
    folders = _discover_folders(args.result_root, symbols=symbols, keyword=args.keyword)
    if not folders:
        raise SystemExit(f"no strat__all folders found under {args.result_root}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"result_root={args.result_root}", flush=True)
    print(f"output_dir={args.output_dir}", flush=True)
    print(f"parameter_folders={len(folders)}", flush=True)

    all_rows: list[dict[str, Any]] = []
    for folder in folders:
        all_rows.extend(_run_folder(folder, args))

    pd.DataFrame(all_rows).to_csv(args.output_dir / "summary.csv", index=False)
    print(f"summary={args.output_dir / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
