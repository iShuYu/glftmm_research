from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from sampler import intensity, resample, volatility  # noqa: E402


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
        return [item.strip() for item in value.split(",") if item.strip()]
    return [value]


def require_non_empty(cfg: dict[str, Any], key: str) -> Any:
    value = cfg.get(key)
    if value in (None, "", []):
        raise ValueError(f"config requires {key}")
    return value


def normalize_category(value: Any, default: str) -> str:
    category = str(value if value is not None else default).strip().upper()
    if not category:
        raise ValueError("category must not be empty")
    return category


def unique_paths(paths: list[Path]) -> tuple[Path, ...]:
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return tuple(unique)


@dataclass(frozen=True)
class InputPaths:
    bookticker_roots: tuple[Path, ...]
    trade_roots: tuple[Path, ...]
    ticker_category: str
    trade_category: str


def resolve_category_roots(
    cfg: dict[str, Any],
    input_root: Path,
    backup_root: Path | None,
    category: str,
    explicit_path_key: str,
    explicit_backup_path_key: str,
) -> tuple[Path, ...]:
    explicit_primary = cfg.get(explicit_path_key)
    primary = (
        input_root / category
        if explicit_primary in (None, "", [])
        else Path(explicit_primary)
    )
    roots = [primary]

    explicit_backup = cfg.get(explicit_backup_path_key)
    if explicit_backup not in (None, "", []):
        roots.append(Path(explicit_backup))
    elif backup_root is not None:
        roots.append(backup_root / category)

    return unique_paths(roots)


def resolve_input_paths(cfg: dict[str, Any]) -> InputPaths:
    input_root = Path(require_non_empty(cfg, "input_path"))
    raw_backup_root = cfg.get("input_backup_path")
    backup_root = None if raw_backup_root in (None, "", []) else Path(raw_backup_root)

    ticker_category = normalize_category(cfg.get("ticker_category"), "BOOKTICKER")
    trade_category = normalize_category(cfg.get("trade_category"), "TRADE")

    bookticker_roots = resolve_category_roots(
        cfg=cfg,
        input_root=input_root,
        backup_root=backup_root,
        category=ticker_category,
        explicit_path_key="bookticker_path",
        explicit_backup_path_key="bookticker_backup_path",
    )
    trade_roots = resolve_category_roots(
        cfg=cfg,
        input_root=input_root,
        backup_root=backup_root,
        category=trade_category,
        explicit_path_key="trade_path",
        explicit_backup_path_key="trade_backup_path",
    )
    return InputPaths(
        bookticker_roots=bookticker_roots,
        trade_roots=trade_roots,
        ticker_category=ticker_category,
        trade_category=trade_category,
    )


def parallel_config(cfg: dict[str, Any]) -> dict[str, Any]:
    num_workers = int(cfg.get("num_worker", cfg.get("num_workers", 1)))
    return {
        "num_workers": max(1, num_workers),
        "start_method": str(cfg.get("start_method", "spawn")),
        "maxtasksperchild": cfg.get("maxtasksperchild"),
    }


def common_dates_and_symbols(cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "symbols": [str(symbol).upper() for symbol in ensure_list(require_non_empty(cfg, "symbols"))],
        "date_start": require_non_empty(cfg, "date_start"),
        "date_end": cfg.get("date_end", cfg["date_start"]),
    }


def build_stage_configs(cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    common = common_dates_and_symbols(cfg)
    input_paths = resolve_input_paths(cfg)

    output_root = Path(require_non_empty(cfg, "output_path"))

    freqs = [int(freq) for freq in ensure_list(require_non_empty(cfg, "freq_ms"))]
    if not freqs:
        raise ValueError("config requires freq_ms")
    lookbacks = [int(lookback) for lookback in ensure_list(require_non_empty(cfg, "lookback"))]
    if not lookbacks:
        raise ValueError("config requires lookback")

    name_intensity = [
        str(name).strip().lower()
        for name in ensure_list(cfg.get("name_intensity", []))
        if str(name).strip()
    ]
    name_volatility = [
        str(name).strip().lower()
        for name in ensure_list(cfg.get("name_volatility", []))
        if str(name).strip()
    ]

    overwrite = bool(cfg.get("overwrite", False))
    strict_validate = bool(cfg.get("strict_validate", True))
    compression = str(cfg.get("compression", "snappy"))
    parallel = parallel_config(cfg)

    resample_cfg = {
        **common,
        "paths": {
            "bookticker_roots": [str(path) for path in input_paths.bookticker_roots],
            "output_root": str(output_root),
        },
        "sampler": {
            "freq": freqs,
            "ticker_category": input_paths.ticker_category,
            "overwrite": overwrite,
            "strict_validate": strict_validate,
            "compression": compression,
        },
        "parallel": parallel,
    }

    intensity_cfg = {
        **common,
        "paths": {
            "ticker_cache_root": str(output_root),
            "bookticker_roots": [str(path) for path in input_paths.bookticker_roots],
            "trade_roots": [str(path) for path in input_paths.trade_roots],
            "output_root": str(output_root),
        },
        "intensity": {
            "freq": freqs,
            "indicator": name_intensity,
            "lookback": lookbacks,
            "ticker_category": input_paths.ticker_category,
            "trade_category": input_paths.trade_category,
            "auto_resample": False,
            "overwrite": overwrite,
            "strict_validate": strict_validate,
            "compression": compression,
        },
        "parallel": parallel,
    }

    volatility_cfg = {
        **common,
        "paths": {
            "ticker_cache_root": str(output_root),
            "bookticker_roots": [str(path) for path in input_paths.bookticker_roots],
            "trade_roots": [str(path) for path in input_paths.trade_roots],
            "output_root": str(output_root),
        },
        "volatility": {
            "freq": freqs,
            "indicator": name_volatility,
            "lookback": lookbacks,
            "ticker_category": input_paths.ticker_category,
            "trade_category": input_paths.trade_category,
            "auto_resample": False,
            "overwrite": overwrite,
            "strict_validate": strict_validate,
            "annualize": bool(cfg.get("annualize", False)),
            "trading_minutes_per_year": int(
                cfg.get("trading_minutes_per_year", 365 * 24 * 60)
            ),
            "min_periods": int(cfg.get("min_periods", 1)),
            "compression": compression,
        },
        "parallel": parallel,
    }

    return resample_cfg, intensity_cfg, volatility_cfg


def run_pipeline(cfg: dict[str, Any]) -> None:
    resample_cfg, intensity_cfg, volatility_cfg = build_stage_configs(cfg)

    print("=== stage 1/3: resample ===", flush=True)
    resample.run_all(resample_cfg)

    if intensity_cfg["intensity"]["indicator"]:
        print("=== stage 2/3: intensity ===", flush=True)
        intensity.run_all(intensity_cfg)
    else:
        print("=== stage 2/3: intensity skipped ===", flush=True)

    if volatility_cfg["volatility"]["indicator"]:
        print("=== stage 3/3: volatility ===", flush=True)
        volatility.run_all(volatility_cfg)
    else:
        print("=== stage 3/3: volatility skipped ===", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run resample, intensity, and volatility stages in order."
    )
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()

    run_pipeline(load_config(args.config))


if __name__ == "__main__":
    main()
