from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime
from multiprocessing import TimeoutError as MpTimeoutError
from multiprocessing import get_context
from time import perf_counter
from typing import Any

import pandas as pd

from sim.loader import BinanceEventLoader
from sim.strategy import (
    SimulationConfig,
    SimpleMakerStrategy,
)


DATE_FMT_DASH = "%Y-%m-%d"
DATE_FMT_COMPACT = "%Y%m%d"
MAX_DIR_SEGMENT_LEN = 240
MIN_DAYS_FOR_NEG_STOP = 60
MIN_DAYS_FOR_LOW_ANNUALIZED_STOP = 305
MIN_ANNUALIZED_RETURN_RATIO = 0.2
MAX_DRAWDOWN_LIMIT_RATIO = 0.2

SIM_REQUIRED_KEYS = (
    "freq",
    "latency",
    "price_precision",
    "qty_precision",
    "mode",
    "taker_fee",
    "maker_fee",
    "max_position",
)
INTENSITY_PARAM_KEYS = ("name_intensity", "lookback_intensity")
INSTRUCTOR_PARAM_KEYS = ("name_instructor", "lookback_instructor")
VOLATILITY_PARAM_KEYS = ("name_volatility", "lookback_volatility")

PARAM_KEY_ALIAS = {
    "latency": "lat",
    "price_precision": "pp",
    "qty_precision": "qp",
    "mode": "md",
    "taker_fee": "tf",
    "maker_fee": "mf",
    "name_volatility": "vn",
    "name_instructor": "ir",
    "name_intensity": "in",
    "lookback_volatility": "vlb",
    "lookback_instructor": "ilb",
    "lookback_intensity": "lbi",
    "max_position": "mp",
    "max_position_usdt": "mp",
    "max_open_inventory_utilization": "moiu",
    "tier_jump_bps": "tjb",
    "boost_tier": "bt",
    "max_holding_time": "mht",
    "freq": "fr",
    "adj_spread_intensity": "asi",
    "adj_spread_instructor": "asir",
    "passive_only": "po",
    "adj_spread_volatility": "asv",
    "min_quote_distance_bps": "mqdb",
    "inventory_skew": "isk",
    "stoploss": "sl",
    "takeprofit": "tp",
    "open_curve": "oc",
    "close_curve": "cc",
    "boost_underwater": "bu",
    "boost_profitzone": "bp",
    "toxic_lock": "tl",
    "strict_mode": "stm",
    "min_order_qty": "moq",
    "min_order_notional": "mon",
    "simple_mode": "sm",
}

SIM_OPTIONAL_KEYS = (
    "name_intensity",
    "name_instructor",
    "name_volatility",
    "lookback_intensity",
    "lookback_instructor",
    "lookback_volatility",
    "max_holding_time",
    "max_open_inventory_utilization",
    "tier_jump_bps",
    "boost_tier",
    "adj_spread_intensity",
    "adj_spread_instructor",
    "passive_only",
    "adj_spread_volatility",
    "min_quote_distance_bps",
    "inventory_skew",
    "min_order_qty",
    "min_order_notional",
    "stoploss",
    "takeprofit",
    "open_curve",
    "close_curve",
    "boost_underwater",
    "boost_profitzone",
    "toxic_lock",
    "strict_mode",
    "simple_mode",
)
SIM_ALLOWED_KEYS = set(SIM_REQUIRED_KEYS) | set(SIM_OPTIONAL_KEYS)


@dataclass(frozen=True)
class Task:
    symbol: str
    dates: list[str]
    sim_params: dict[str, Any]
    cfg: dict[str, Any]


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_list_map(d: dict[str, Any]) -> dict[str, list[Any]]:
    return {k: v if isinstance(v, list) else [v] for k, v in d.items()}


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _normalize_non_negative_scalar(value: Any, key: str) -> float:
    if not _is_number(value):
        raise ValueError(f"simulation.{key} must be a number")
    value_float = float(value)
    if not math.isfinite(value_float) or value_float < 0.0:
        raise ValueError(f"simulation.{key} must be finite and >= 0")
    return value_float


def _normalize_ratio(value: Any, key: str) -> float:
    if not _is_number(value):
        raise ValueError(f"simulation.{key} must be a number")
    value_float = float(value)
    if not math.isfinite(value_float) or value_float < 0.0 or value_float > 1.0:
        raise ValueError(f"simulation.{key} must be finite and between 0 and 1")
    return value_float


def _normalize_tier_values(value: Any, key: str) -> list[float]:
    if _is_number(value):
        values = [float(value)]
    elif isinstance(value, (list, tuple)) and value:
        values = [float(item) for item in value]
    else:
        raise ValueError(f"simulation.{key} must be a number or a non-empty tier list")
    if any((not math.isfinite(item)) or item < 0.0 for item in values):
        raise ValueError(f"simulation.{key} tiers must be finite and >= 0")
    if len(values) > 1:
        if values[0] <= 0.0:
            raise ValueError(f"simulation.{key} tiers must be > 0 when tiered")
        for prev, current in zip(values, values[1:]):
            if current <= prev:
                raise ValueError(f"simulation.{key} tiers must be strictly increasing")
    return values


def _normalize_max_position_rows(
    value: Any,
    key: str,
    *,
    scalar_grid: bool,
) -> list[float | list[float]]:
    if _is_number(value):
        return [_normalize_non_negative_scalar(value, key)]
    if not isinstance(value, list) or not value:
        raise ValueError(f"simulation.{key} must be a number or list of tiers")
    if all(_is_number(item) for item in value):
        if scalar_grid:
            return [_normalize_non_negative_scalar(item, key) for item in value]
        return [_normalize_tier_values(value, key)]
    rows: list[float | list[float]] = []
    for row in value:
        tiers = _normalize_tier_values(row, key)
        rows.append(tiers[0] if len(tiers) == 1 else tiers)
    return rows


def _normalize_tier_control_rows(value: Any, key: str) -> list[float | list[float]]:
    if _is_number(value):
        value_float = _normalize_non_negative_scalar(value, key)
        return [value_float]
    if not isinstance(value, list) or not value:
        raise ValueError(f"simulation.{key} must be a number or list of tier rows")
    if all(_is_number(item) for item in value):
        values = [float(item) for item in value]
        if any((not math.isfinite(item)) or item < 0.0 for item in values):
            raise ValueError(f"simulation.{key} values must be finite and >= 0")
        return [values[0] if len(values) == 1 else values]
    rows: list[float | list[float]] = []
    for row in value:
        if _is_number(row):
            rows.append(_normalize_non_negative_scalar(row, key))
            continue
        if not isinstance(row, (list, tuple)) or not row:
            raise ValueError(f"simulation.{key} rows must be non-empty")
        values = [float(item) for item in row]
        if any((not math.isfinite(item)) or item < 0.0 for item in values):
            raise ValueError(f"simulation.{key} values must be finite and >= 0")
        rows.append(values[0] if len(values) == 1 else values)
    return rows


def _normalize_tier_jump_rows(value: Any) -> list[float | list[float]]:
    return _normalize_tier_control_rows(value, "tier_jump_bps")


def _normalize_boost_tier_rows(value: Any) -> list[float | list[float]]:
    return _normalize_tier_control_rows(value, "boost_tier")


def _normalize_open_close_pair(value: Any, key: str, *, positive: bool) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"simulation.{key} must be [open, close]")
    open_value = float(value[0])
    close_value = float(value[1])
    if not math.isfinite(open_value) or not math.isfinite(close_value):
        raise ValueError(f"simulation.{key} open and close must be finite")
    if positive:
        if open_value < 0.0 or close_value < 0.0:
            raise ValueError(f"simulation.{key} open and close must be >= 0")
    elif open_value < 0.0 or close_value < 0.0:
        raise ValueError(f"simulation.{key} open and close must be >= 0")
    return [open_value, close_value]


def _normalize_open_close_rows(value: Any, key: str, *, positive: bool) -> list[list[float]]:
    if (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(_is_number(item) for item in value)
    ):
        return [_normalize_open_close_pair(value, key, positive=positive)]
    rows = value if isinstance(value, list) else [value]
    return [
        _normalize_open_close_pair(row, key, positive=positive)
        for row in rows
    ]


def _normalize_toxic_lock_pair(value: Any) -> list[int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("simulation.toxic_lock must be [lookback_bar, open_side_slip]")
    if not all(_is_number(item) for item in value):
        raise ValueError("simulation.toxic_lock values must be numbers")

    lookback_bar_float = float(value[0])
    open_side_slip_float = float(value[1])
    if (
        not math.isfinite(lookback_bar_float)
        or not math.isfinite(open_side_slip_float)
        or lookback_bar_float < 0.0
        or open_side_slip_float < 0.0
    ):
        raise ValueError("simulation.toxic_lock values must be finite and >= 0")
    if (
        not lookback_bar_float.is_integer()
        or not open_side_slip_float.is_integer()
    ):
        raise ValueError("simulation.toxic_lock values must be integers")

    lookback_bar = int(lookback_bar_float)
    open_side_slip = int(open_side_slip_float)
    if lookback_bar > 0 and open_side_slip > lookback_bar:
        raise ValueError(
            "simulation.toxic_lock open_side_slip must be <= lookback_bar"
        )
    return [lookback_bar, open_side_slip]


def _normalize_toxic_lock_rows(value: Any) -> list[list[int]]:
    if not isinstance(value, list) or (
        len(value) == 2 and all(_is_number(item) for item in value)
    ):
        raise ValueError(
            "simulation.toxic_lock must be list of [lookback_bar, open_side_slip] pairs"
        )
    return [_normalize_toxic_lock_pair(row) for row in value]


def _max_position_total(value: Any) -> float | None:
    try:
        tiers = _normalize_tier_values(value, "max_position")
    except (TypeError, ValueError):
        return None
    max_position_usdt = float(tiers[-1])
    return max_position_usdt if max_position_usdt > 0.0 else None


def parse_date_input(s: str) -> datetime:
    for fmt in (DATE_FMT_DASH, DATE_FMT_COMPACT):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    raise ValueError(f"unsupported date format: {s}, expected YYYY-MM-DD or YYYYMMDD")


def generate_dates(start: str, end: str) -> list[str]:
    start_dt = parse_date_input(start)
    end_dt = parse_date_input(end)
    if end_dt < start_dt:
        raise ValueError(f"date_end must be >= date_start, got {end} < {start}")
    return (
        pd.date_range(start=start_dt, end=end_dt, freq="D")
        .strftime(DATE_FMT_DASH)
        .tolist()
    )


def _short_param_key(key: str) -> str:
    if key in PARAM_KEY_ALIAS:
        return PARAM_KEY_ALIAS[key]
    parts = [p for p in key.lower().split("_") if p]
    if not parts:
        return key[:8]
    initials = "".join(p[0] for p in parts)
    return initials[:8] if len(initials) >= 2 else parts[0][:8]


def _short_param_val(val: Any) -> str:
    if isinstance(val, (list, tuple)):
        text = "x".join(_short_param_val(item) for item in val)
    elif isinstance(val, bool):
        text = "1" if val else "0"
    else:
        text = f"{val:.10g}" if isinstance(val, float) else str(val)
    return text.replace("/", "_").replace("-", "m").replace(".", "p").replace("+", "")


def _parse_bool(raw: Any, key: str) -> bool:
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, int) and raw in (0, 1):
        return bool(raw)
    if isinstance(raw, str):
        value = raw.strip().lower()
        if value in {"true", "1", "yes", "y"}:
            return True
        if value in {"false", "0", "no", "n"}:
            return False
    raise ValueError(f"simulation.{key} must be boolean")


def _safe_dir_segment(prefix: str, body: str) -> str:
    segment = f"{prefix}{body}" if body else f"{prefix}none"
    if len(segment) <= MAX_DIR_SEGMENT_LEN:
        return segment
    digest = hashlib.sha1(segment.encode("utf-8")).hexdigest()[:10]
    keep = MAX_DIR_SEGMENT_LEN - len("__h") - len(digest)
    return f"{segment[:keep]}__h{digest}"


def _build_param_body(params: dict[str, Any]) -> str:
    parts = [
        f"{_short_param_key(k)}{_short_param_val(params[k])}"
        for k in sorted(params.keys())
    ]
    return "__".join(parts) if parts else "none"


def build_param_path_parts(sim_params: dict[str, Any]) -> tuple[str, str]:
    sim_body = _build_param_body(sim_params)
    sim_dir = _safe_dir_segment(prefix="sim__", body=sim_body)
    return sim_dir, "strat__all"


def _is_effectively_zero(raw: Any) -> bool:
    if isinstance(raw, (list, tuple)):
        return bool(raw) and all(_is_effectively_zero(item) for item in raw)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return False
    return math.isfinite(value) and abs(value) <= 1e-12


def _effective_sim_params(sim_params: dict[str, Any]) -> dict[str, Any]:
    effective = dict(sim_params)
    if _is_effectively_zero(effective.get("adj_spread_intensity", [1.0, 1.0])):
        for key in INTENSITY_PARAM_KEYS:
            effective.pop(key, None)
    if _is_effectively_zero(effective.get("adj_spread_instructor", 0.0)):
        for key in INSTRUCTOR_PARAM_KEYS:
            effective.pop(key, None)
    if _is_effectively_zero(effective.get("adj_spread_volatility", [0.0, 0.0])):
        for key in VOLATILITY_PARAM_KEYS:
            effective.pop(key, None)
    return effective


def _sim_params_dedupe_key(sim_params: dict[str, Any]) -> str:
    return json.dumps(sim_params, sort_keys=True, separators=(",", ":"))


def _normalize_sim_param_map(cfg: dict[str, Any]) -> dict[str, list[Any]]:
    sim_raw = cfg.get("simulation")
    if sim_raw is None or not isinstance(sim_raw, dict):
        raise ValueError("config requires simulation section(object)")
    if "strategy" in cfg:
        raise ValueError("strategy section is not supported, use simulation only")
    if "cooldown_time" in sim_raw:
        raise ValueError(
            "simulation.cooldown_time has been replaced by simulation.toxic_lock"
        )

    sim_map = ensure_list_map(sim_raw)
    if "max_position" in sim_raw and "max_position_usdt" in sim_raw:
        raise ValueError("simulation.max_position_usdt has been replaced by simulation.max_position")
    if "max_position" in sim_raw:
        sim_map["max_position"] = _normalize_max_position_rows(
            sim_raw["max_position"],
            "max_position",
            scalar_grid=False,
        )
    elif "max_position_usdt" in sim_raw:
        sim_map.pop("max_position_usdt", None)
        sim_map["max_position"] = _normalize_max_position_rows(
            sim_raw["max_position_usdt"],
            "max_position_usdt",
            scalar_grid=True,
        )

    if "inventory_skew" in sim_raw:
        skew_rows = ensure_list_map({"inventory_skew": sim_raw["inventory_skew"]})[
            "inventory_skew"
        ]
        normalized_rows: list[list[float]] = []
        for row in skew_rows:
            if not isinstance(row, (list, tuple)) or len(row) != 2:
                raise ValueError(
                    "simulation.inventory_skew must be list of [max_tick, skew_power] pairs"
                )
            normalized_rows.append([float(row[0]), float(row[1])])
        sim_map["inventory_skew"] = normalized_rows

    if "max_open_inventory_utilization" in sim_map:
        sim_map["max_open_inventory_utilization"] = [
            _normalize_ratio(row, "max_open_inventory_utilization")
            for row in sim_map["max_open_inventory_utilization"]
        ]

    if "tier_jump_bps" in sim_raw:
        sim_map["tier_jump_bps"] = _normalize_tier_jump_rows(sim_raw["tier_jump_bps"])
    if "boost_tier" in sim_raw:
        sim_map["boost_tier"] = _normalize_boost_tier_rows(sim_raw["boost_tier"])

    if "stoploss" in sim_map:
        sim_map["stoploss"] = [
            _normalize_non_negative_scalar(row, "stoploss")
            for row in sim_map["stoploss"]
        ]

    if "takeprofit" in sim_map:
        sim_map["takeprofit"] = [
            _normalize_non_negative_scalar(row, "takeprofit")
            for row in sim_map["takeprofit"]
        ]

    if "toxic_lock" in sim_raw:
        sim_map["toxic_lock"] = _normalize_toxic_lock_rows(sim_raw["toxic_lock"])

    if "min_quote_distance_bps" in sim_map:
        sim_map["min_quote_distance_bps"] = [
            _normalize_non_negative_scalar(row, "min_quote_distance_bps")
            for row in sim_map["min_quote_distance_bps"]
        ]

    if "open_curve" in sim_map:
        open_curve_rows = ensure_list_map({"open_curve": sim_map["open_curve"]})[
            "open_curve"
        ]
        normalized_rows: list[list[float]] = []
        for row in open_curve_rows:
            if not isinstance(row, (list, tuple)) or len(row) != 3:
                raise ValueError(
                    "simulation.open_curve must be list of (min, max, order) triples"
                )
            normalized_rows.append([float(row[0]), float(row[1]), float(row[2])])
        sim_map["open_curve"] = normalized_rows

    if "close_curve" in sim_map:
        close_curve_rows = ensure_list_map({"close_curve": sim_map["close_curve"]})[
            "close_curve"
        ]
        normalized_rows = []
        for row in close_curve_rows:
            if not isinstance(row, (list, tuple)) or len(row) != 3:
                raise ValueError(
                    "simulation.close_curve must be list of (min, max, order) triples"
                )
            normalized_rows.append([float(row[0]), float(row[1]), float(row[2])])
        sim_map["close_curve"] = normalized_rows

    for key, positive in (
        ("adj_spread_intensity", True),
        ("adj_spread_volatility", False),
        ("boost_underwater", False),
        ("boost_profitzone", False),
    ):
        if key in sim_raw:
            sim_map[key] = _normalize_open_close_rows(
                sim_raw[key],
                key,
                positive=positive,
            )

    unknown = sorted(set(sim_map) - SIM_ALLOWED_KEYS)
    if unknown:
        raise ValueError(f"unknown simulation keys: {unknown}")

    missing = [k for k in SIM_REQUIRED_KEYS if k not in sim_map]
    if missing:
        raise ValueError(f"missing simulation keys: {missing}")
    return sim_map


def _build_simulation_config(raw: dict[str, Any]) -> SimulationConfig:
    if "cooldown_time" in raw:
        raise ValueError(
            "simulation.cooldown_time has been replaced by simulation.toxic_lock"
        )
    if "min_quote_distance_ticks" in raw:
        raise ValueError(
            "simulation.min_quote_distance_ticks has been replaced by "
            "simulation.min_quote_distance_bps"
        )
    if "max_position" in raw and "max_position_usdt" in raw:
        raise ValueError("simulation.max_position_usdt has been replaced by simulation.max_position")
    raw_allowed_keys = SIM_ALLOWED_KEYS | {"max_position_usdt"}
    unknown = sorted(set(raw) - raw_allowed_keys)
    if unknown:
        raise ValueError(f"unknown simulation keys: {unknown}")

    adj_spread_intensity = raw.get("adj_spread_intensity", [1.0, 1.0])
    lookback_intensity_raw = raw.get("lookback_intensity")
    if lookback_intensity_raw is None:
        if not _is_effectively_zero(adj_spread_intensity):
            raise ValueError(
                "simulation.lookback_intensity is required when "
                "adj_spread_intensity is non-zero"
            )
        lookback_intensity = 100
    else:
        lookback_intensity = int(lookback_intensity_raw)

    name_instructor = raw.get("name_instructor")
    if name_instructor is not None:
        name_instructor = str(name_instructor).strip() or None
    lookback_instructor = raw.get("lookback_instructor")
    if lookback_instructor is not None:
        lookback_instructor = int(lookback_instructor)
    if name_instructor is not None and lookback_instructor is None:
        raise ValueError("simulation.lookback_instructor is required when name_instructor is set")

    name_volatility = raw.get("name_volatility")
    if name_volatility is not None:
        name_volatility = str(name_volatility).strip() or None
    lookback_volatility = raw.get("lookback_volatility")
    if lookback_volatility is not None:
        lookback_volatility = int(lookback_volatility)
    if name_volatility is not None and lookback_volatility is None:
        raise ValueError("simulation.lookback_volatility is required when name_volatility is set")

    inventory_skew_raw = raw.get("inventory_skew")
    inventory_skew: tuple[float, float] | None = None
    if inventory_skew_raw is not None:
        if not isinstance(inventory_skew_raw, (list, tuple)) or len(inventory_skew_raw) != 2:
            raise ValueError("simulation.inventory_skew must be [max_tick, skew_power]")
        inventory_skew = (float(inventory_skew_raw[0]), float(inventory_skew_raw[1]))

    open_curve_raw = raw.get("open_curve")
    open_curve: tuple[float, float, float] | None
    if open_curve_raw is None:
        open_curve = None
    else:
        if not isinstance(open_curve_raw, (list, tuple)) or len(open_curve_raw) != 3:
            raise ValueError("simulation.open_curve must be [min, max, order]")
        open_curve = tuple(float(v) for v in open_curve_raw)

    close_curve_raw = raw.get("close_curve")
    close_curve: tuple[float, float, float] | None
    if close_curve_raw is None:
        close_curve = None
    else:
        if not isinstance(close_curve_raw, (list, tuple)) or len(close_curve_raw) != 3:
            raise ValueError("simulation.close_curve must be [min, max, order]")
        close_curve = tuple(float(v) for v in close_curve_raw)

    toxic_lock_raw = raw.get("toxic_lock", [0, 0])
    if (
        isinstance(toxic_lock_raw, list)
        and len(toxic_lock_raw) == 1
        and isinstance(toxic_lock_raw[0], (list, tuple))
    ):
        toxic_lock_raw = toxic_lock_raw[0]
    elif (
        isinstance(toxic_lock_raw, list)
        and toxic_lock_raw
        and all(isinstance(row, (list, tuple)) for row in toxic_lock_raw)
    ):
        raise ValueError(
            "simulation.toxic_lock must be a single [lookback_bar, open_side_slip] pair after parameter normalization"
        )

    return SimulationConfig(
        freq=int(raw["freq"]),
        latency=int(raw["latency"]),
        price_precision=int(raw["price_precision"]),
        qty_precision=int(raw["qty_precision"]),
        mode=int(raw["mode"]),
        taker_fee=float(raw["taker_fee"]),
        maker_fee=float(raw["maker_fee"]),
        name_instructor=(
            None if name_instructor is None else str(name_instructor).strip().lower()
        ),
        lookback_instructor=lookback_instructor,
        name_intensity=str(raw.get("name_intensity", "k")).strip().lower(),
        lookback_intensity=lookback_intensity,
        name_volatility=(
            None if name_volatility is None else str(name_volatility).strip().lower()
        ),
        lookback_volatility=lookback_volatility,
        max_position_usdt=raw.get("max_position", raw.get("max_position_usdt")),
        max_open_inventory_utilization=_normalize_ratio(
            raw.get("max_open_inventory_utilization", 1.0),
            "max_open_inventory_utilization",
        ),
        tier_jump_bps=raw.get("tier_jump_bps", 0.0),
        boost_tier=raw.get("boost_tier", 1.0),
        max_holding_time=int(raw.get("max_holding_time", 0)),
        adj_spread_intensity=adj_spread_intensity,
        adj_spread_instructor=float(raw.get("adj_spread_instructor", 0.0)),
        passive_only=_parse_bool(raw.get("passive_only", False), "passive_only"),
        adj_spread_volatility=raw.get("adj_spread_volatility", [0.0, 0.0]),
        min_quote_distance_bps=_normalize_non_negative_scalar(
            raw.get("min_quote_distance_bps", 0.0),
            "min_quote_distance_bps",
        ),
        inventory_skew=inventory_skew,
        min_order_qty=float(raw.get("min_order_qty", 0.0)),
        min_order_notional=float(raw.get("min_order_notional", 0.0)),
        stoploss=_normalize_non_negative_scalar(raw.get("stoploss", 0.0), "stoploss"),
        takeprofit=_normalize_non_negative_scalar(
            raw.get("takeprofit", 0.0),
            "takeprofit",
        ),
        open_curve=open_curve,
        close_curve=close_curve,
        boost_underwater=raw.get("boost_underwater", [1.0, 1.0]),
        boost_profitzone=raw.get("boost_profitzone", [1.0, 1.0]),
        toxic_lock=toxic_lock_raw,
        strict_mode=_parse_bool(raw.get("strict_mode", True), "strict_mode"),
        simple_mode=_parse_bool(raw.get("simple_mode", True), "simple_mode"),
    )


def _path_value(paths: dict[str, Any], key: str, required: bool = False) -> str | None:
    value = paths.get(key)
    if value not in (None, "", []):
        return str(value)
    if required:
        raise ValueError(f"paths.{key} is required")
    return None


def _config_paths(cfg: dict[str, Any]) -> dict[str, Any]:
    paths = cfg.get("paths")
    if not isinstance(paths, dict):
        raise ValueError("config requires paths object")
    return dict(paths)


def _normalize_category(value: Any, default: str) -> str:
    category = str(value if value not in (None, "", []) else default).strip().upper()
    if not category:
        raise ValueError("category must not be empty")
    return category


def _resolve_category_roots(
    paths: dict[str, Any],
    *,
    category: str,
    require_input_path: bool = True,
) -> list[str]:
    input_root = _path_value(paths, "input_path", required=require_input_path)
    if input_root is None:
        return []
    roots = [os.path.join(input_root, category)]
    input_backup_root = _path_value(paths, "input_backup_path")
    if input_backup_root:
        roots.append(os.path.join(input_backup_root, category))
    return roots


def _output_dir_from_cfg(cfg: dict[str, Any]) -> str:
    paths = _config_paths(cfg)
    out_dir = _path_value(paths, "output_dir", required=True)
    assert out_dir is not None
    return out_dir


def _build_loader(
    cfg: dict[str, Any],
    simulation: SimulationConfig | None = None,
    require_volatility_cache: bool = False,
) -> BinanceEventLoader:
    del simulation, require_volatility_cache
    paths = _config_paths(cfg)
    sampler_output_root = _path_value(paths, "output_path", required=True)
    assert sampler_output_root is not None
    trade_category = _normalize_category(paths.get("trade_category"), "TRADE")
    trade_roots = _resolve_category_roots(
        paths,
        category=trade_category,
    )
    return BinanceEventLoader(
        trade_roots=trade_roots,
        cache_root=sampler_output_root,
        trade_category=trade_category,
        scheme_shift=int(paths.get("scheme_shift", 0)),
    )


def _atomic_write_parquet(df: pd.DataFrame, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp_path = f"{out_path}.tmp.{os.getpid()}"
    df.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, out_path)


def _atomic_write_json(data: dict[str, Any], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp_path = f"{out_path}.tmp.{os.getpid()}"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    os.replace(tmp_path, out_path)


def _state_path(out_dir: str, day: str) -> str:
    return os.path.join(out_dir, "_state", f"{day}.json")


def _load_state(out_dir: str, day: str) -> dict[str, Any] | None:
    path = _state_path(out_dir, day)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_state(state: dict[str, Any], out_dir: str, day: str) -> None:
    _atomic_write_json(state, _state_path(out_dir, day))


def _day_completed(out_dir: str, day: str) -> bool:
    return os.path.exists(os.path.join(out_dir, f"{day}.parquet")) and os.path.exists(
        _state_path(out_dir, day)
    )


def _state_total_pnl(state: dict[str, Any]) -> float | None:
    try:
        total = float(state["total_pnl"])
    except (KeyError, TypeError, ValueError):
        return None
    return total if math.isfinite(total) else None


def _state_realized_pnl(state: dict[str, Any]) -> float | None:
    try:
        realized = float(state["realized_pnl"])
    except (KeyError, TypeError, ValueError):
        return None
    return realized if math.isfinite(realized) else None


def _state_day_n(state: dict[str, Any]) -> int | None:
    day_raw = state.get("day_n")
    try:
        day_n = int(day_raw)
    except (TypeError, ValueError):
        return None
    return day_n if day_n > 0 else None


def _state_max_pnl_ever(state: dict[str, Any]) -> float | None:
    max_raw = state.get("max_pnl_ever")
    try:
        max_pnl = float(max_raw)
    except (TypeError, ValueError):
        return None
    return max_pnl if math.isfinite(max_pnl) else None


def _state_max_position_usdt(state: dict[str, Any]) -> float | None:
    max_pos_raw = state.get("max_position_usdt")
    try:
        max_pos = float(max_pos_raw)
    except (TypeError, ValueError):
        return None
    return max_pos if math.isfinite(max_pos) and max_pos > 0.0 else None


def _state_cost_notional_usdt(state: dict[str, Any]) -> float | None:
    pos = state.get("position")
    if not isinstance(pos, dict):
        return None
    try:
        cost_notional = float(pos["cost_notional_usdt"])
    except (KeyError, TypeError, ValueError):
        return None
    return cost_notional if math.isfinite(cost_notional) else None


def _state_drawdown_ratio(state: dict[str, Any]) -> float | None:
    total_pnl = _state_total_pnl(state)
    max_pnl_ever = _state_max_pnl_ever(state)
    max_position_usdt = _state_max_position_usdt(state)
    if total_pnl is None or max_pnl_ever is None or max_position_usdt is None:
        return None
    return (total_pnl - max_pnl_ever) / max_position_usdt


def _state_annualized_return_ratio(state: dict[str, Any]) -> float | None:
    realized_pnl = _state_realized_pnl(state)
    day_n = _state_day_n(state)
    max_position_usdt = _state_max_position_usdt(state)
    if realized_pnl is None or day_n is None or max_position_usdt is None:
        return None
    return (realized_pnl / max_position_usdt) * 365.0 / day_n


def _should_stop_by_dayn_pnl(
    state: dict[str, Any], min_days: int = MIN_DAYS_FOR_NEG_STOP
) -> bool:
    if bool(state.get("stop_due_to_dayn_pnl", False)):
        return True
    day_n = _state_day_n(state)
    total_pnl = _state_total_pnl(state)
    return bool(
        day_n is not None
        and day_n > min_days
        and total_pnl is not None
        and total_pnl < 0.0
    )


def _should_stop_by_low_annualized_return(
    state: dict[str, Any],
    min_days: int = MIN_DAYS_FOR_LOW_ANNUALIZED_STOP,
    min_annualized_return_ratio: float = MIN_ANNUALIZED_RETURN_RATIO,
) -> bool:
    if bool(state.get("stop_due_to_low_annualized_return", False)):
        return True
    day_n = _state_day_n(state)
    annualized_return_ratio = _state_annualized_return_ratio(state)
    return bool(
        day_n is not None
        and day_n > min_days
        and annualized_return_ratio is not None
        and annualized_return_ratio < min_annualized_return_ratio
    )


def _should_stop_by_max_drawdown(
    state: dict[str, Any], drawdown_limit_ratio: float = MAX_DRAWDOWN_LIMIT_RATIO
) -> bool:
    if bool(state.get("stop_due_to_max_drawdown", False)):
        return True
    drawdown_ratio = _state_drawdown_ratio(state)
    return bool(drawdown_ratio is not None and drawdown_ratio <= (-drawdown_limit_ratio))


def _annotate_state_for_stops(
    state: dict[str, Any],
    *,
    prev_max_pnl_ever: float | None = None,
    max_position_usdt: float | None = None,
    min_days_for_neg_stop: int = MIN_DAYS_FOR_NEG_STOP,
    min_days_for_low_annualized_stop: int = MIN_DAYS_FOR_LOW_ANNUALIZED_STOP,
    min_annualized_return_ratio: float = MIN_ANNUALIZED_RETURN_RATIO,
    max_drawdown_limit_ratio: float = MAX_DRAWDOWN_LIMIT_RATIO,
) -> dict[str, Any]:
    total_pnl = _state_total_pnl(state)
    if total_pnl is not None:
        state["total_pnl"] = float(total_pnl)

    current_max = _state_max_pnl_ever(state)
    candidates = [
        v
        for v in (prev_max_pnl_ever, current_max, total_pnl)
        if v is not None and math.isfinite(v)
    ]
    max_pnl_ever = max(candidates) if candidates else None
    if max_pnl_ever is not None:
        state["max_pnl_ever"] = float(max_pnl_ever)

    resolved_max_position_usdt = max_position_usdt or _state_max_position_usdt(state)
    if (
        resolved_max_position_usdt is not None
        and math.isfinite(resolved_max_position_usdt)
        and resolved_max_position_usdt > 0.0
    ):
        state["max_position_usdt"] = float(resolved_max_position_usdt)

    day_n = _state_day_n(state)
    annualized_return_ratio = _state_annualized_return_ratio(state)
    state["stop_due_to_dayn_pnl"] = bool(
        day_n is not None
        and total_pnl is not None
        and day_n > min_days_for_neg_stop
        and total_pnl < 0.0
    )
    state["stop_due_to_low_annualized_return"] = bool(
        day_n is not None
        and annualized_return_ratio is not None
        and day_n > min_days_for_low_annualized_stop
        and annualized_return_ratio < min_annualized_return_ratio
    )
    state["stop_due_to_max_drawdown"] = bool(
        total_pnl is not None
        and max_pnl_ever is not None
        and resolved_max_position_usdt is not None
        and ((total_pnl - max_pnl_ever) / resolved_max_position_usdt)
        <= (-max_drawdown_limit_ratio)
    )
    return state


def _metrics_from_state(state: dict[str, Any]) -> dict[str, float]:
    pos = state.get("position") if isinstance(state.get("position"), dict) else {}
    realized = float(pos.get("realized_pnl", 0.0))
    unrealized = float(pos.get("unrealized_pnl", 0.0))
    total = realized + unrealized
    volume = float(state.get("traded_volume", 0.0))
    pnl_bps = float(total / volume * 1e4) if volume > 0 else 0.0
    return {
        "strategy_total_pnl": total,
        "strategy_realized_pnl": realized,
        "strategy_unrealized_pnl": unrealized,
        "final_position": float(pos.get("qty", 0.0)),
        "traded_volume": volume,
        "pnl_bps_on_volume": pnl_bps,
    }


def _metrics_from_saved(out_dir: str, dates: list[str]) -> dict[str, float]:
    state = _load_state(out_dir=out_dir, day=dates[-1])
    if state is not None:
        return _metrics_from_state(state)

    first_df = pd.read_parquet(os.path.join(out_dir, f"{dates[0]}.parquet"))
    last_df = pd.read_parquet(os.path.join(out_dir, f"{dates[-1]}.parquet"))
    if first_df.empty or last_df.empty:
        return _metrics_from_state({})
    first = first_df.iloc[0]
    last = last_df.iloc[-1]
    volume = float(last["traded_volume"] - first["traded_volume"])
    total = float(last["realized_pnl"] + last["unrealized_pnl"])
    pnl_bps = float(total / volume * 1e4) if volume > 0 else 0.0
    return {
        "strategy_total_pnl": total,
        "strategy_realized_pnl": float(last["realized_pnl"]),
        "strategy_unrealized_pnl": float(last["unrealized_pnl"]),
        "final_position": float(last["position"]),
        "traded_volume": volume,
        "pnl_bps_on_volume": pnl_bps,
    }


def _run_daily_incremental(
    out_dir: str,
    symbol: str,
    dates: list[str],
    simulation: SimulationConfig,
    loader: BinanceEventLoader,
    overwrite: bool,
) -> dict[str, float]:
    os.makedirs(out_dir, exist_ok=True)

    def _fmt_pnl(value: float | None) -> str:
        return f"{value:.6f}" if value is not None else "None"

    if overwrite:
        resume_idx = 0
    else:
        resume_idx = len(dates)
        for i, day in enumerate(dates):
            if not _day_completed(out_dir=out_dir, day=day):
                resume_idx = i
                break
    if resume_idx >= len(dates):
        print(f"[{symbol}] skip existing strategy: {out_dir}")
        return _metrics_from_saved(out_dir=out_dir, dates=dates)

    prev_day = dates[resume_idx - 1] if resume_idx > 0 else None
    prev_state = _load_state(out_dir=out_dir, day=prev_day) if prev_day is not None else None

    if (not overwrite) and resume_idx > 0:
        if prev_state is None:
            raise FileNotFoundError(f"[{symbol}] missing resume state: {prev_day}")
        if _should_stop_by_dayn_pnl(prev_state):
            total_pnl = _state_total_pnl(prev_state)
            print(
                f"[{symbol}] skip resume strategy: prev total_pnl={_fmt_pnl(total_pnl)} < 0 at {prev_day}",
                flush=True,
            )
            return _metrics_from_state(prev_state)
        if _should_stop_by_low_annualized_return(prev_state):
            annualized_return_ratio = _state_annualized_return_ratio(prev_state)
            print(
                f"[{symbol}] skip resume strategy: annualized_return_ratio={_fmt_pnl(annualized_return_ratio)} "
                f"< {MIN_ANNUALIZED_RETURN_RATIO:.2f} at {prev_day}",
                flush=True,
            )
            return _metrics_from_state(prev_state)
        if _should_stop_by_max_drawdown(prev_state):
            drawdown_ratio = _state_drawdown_ratio(prev_state)
            print(
                f"[{symbol}] skip resume strategy: drawdown_ratio={_fmt_pnl(drawdown_ratio)} "
                f"<= -{MAX_DRAWDOWN_LIMIT_RATIO:.2f} at {prev_day}",
                flush=True,
            )
            return _metrics_from_state(prev_state)

    print(f"[{symbol}] run strategy daily: {dates[resume_idx]} -> {dates[-1]}")
    engine = SimpleMakerStrategy(
        simulation=simulation,
        loader=loader,
    )
    if resume_idx > 0:
        if prev_state is None:
            raise FileNotFoundError(f"[{symbol}] missing resume state: {prev_day}")
        engine.restore_state(prev_state)

    stopped_state: dict[str, Any] | None = None
    prev_max_pnl_ever = _state_max_pnl_ever(prev_state) if prev_state is not None else None
    for day_idx, day in enumerate(dates[resume_idx:], start=resume_idx):
        day_t0 = perf_counter()
        day_df = engine.run_day(symbol=symbol, date=day)
        _atomic_write_parquet(day_df, os.path.join(out_dir, f"{day}.parquet"))
        day_state = engine.snapshot_state()
        day_state["day_n"] = int(day_idx + 1)
        _annotate_state_for_stops(
            day_state,
            prev_max_pnl_ever=prev_max_pnl_ever,
            max_position_usdt=simulation.total_max_position_usdt,
        )
        day_total_pnl = _state_total_pnl(day_state)
        day_annualized_return_ratio = _state_annualized_return_ratio(day_state)
        day_drawdown_ratio = _state_drawdown_ratio(day_state)
        day_cost_notional_usdt = _state_cost_notional_usdt(day_state)
        _save_state(day_state, out_dir=out_dir, day=day)
        prev_max_pnl_ever = _state_max_pnl_ever(day_state)
        print(
            f"[{symbol}] strategy day done: {day} rows={len(day_df)} "
            f"total_pnl={_fmt_pnl(day_total_pnl)} "
            f"annualized_return_ratio={_fmt_pnl(day_annualized_return_ratio)} "
            f"drawdown_ratio={_fmt_pnl(day_drawdown_ratio)} "
            f"cost_notional_usdt={_fmt_pnl(day_cost_notional_usdt)} "
            f"sec={perf_counter() - day_t0:.1f}",
            flush=True,
        )

        if bool(day_state.get("stop_due_to_dayn_pnl", False)):
            print(
                f"[{symbol}] stop strategy: day_n={day_state['day_n']} > {MIN_DAYS_FOR_NEG_STOP} "
                f"and total_pnl={_fmt_pnl(day_total_pnl)} < 0 at {day}",
                flush=True,
            )
            stopped_state = day_state
            break
        if bool(day_state.get("stop_due_to_low_annualized_return", False)):
            annualized_return_ratio = _state_annualized_return_ratio(day_state)
            print(
                f"[{symbol}] stop strategy: day_n={day_state['day_n']} > {MIN_DAYS_FOR_LOW_ANNUALIZED_STOP} "
                f"and annualized_return_ratio={_fmt_pnl(annualized_return_ratio)} < "
                f"{MIN_ANNUALIZED_RETURN_RATIO:.2f} at {day}",
                flush=True,
            )
            stopped_state = day_state
            break
        if bool(day_state.get("stop_due_to_max_drawdown", False)):
            drawdown_ratio = _state_drawdown_ratio(day_state)
            print(
                f"[{symbol}] stop strategy: drawdown_ratio={_fmt_pnl(drawdown_ratio)} "
                f"<= -{MAX_DRAWDOWN_LIMIT_RATIO:.2f} at {day}",
                flush=True,
            )
            stopped_state = day_state
            break

    if stopped_state is not None:
        return _metrics_from_state(stopped_state)
    return _metrics_from_saved(out_dir=out_dir, dates=dates)


def _build_tier_param_rows(sim_map: dict[str, list[Any]]) -> list[dict[str, Any]]:
    max_position_rows = sim_map.get("max_position")
    if max_position_rows is None:
        return [{}]

    paired_keys = ("tier_jump_bps", "boost_tier")
    paired_rows = {
        key: sim_map[key]
        for key in paired_keys
        if key in sim_map
    }
    for key, rows in paired_rows.items():
        if len(max_position_rows) != len(rows):
            raise ValueError(
                "simulation.max_position, simulation.tier_jump_bps, and "
                "simulation.boost_tier must have the same number of rows"
            )

    result: list[dict[str, Any]] = []
    for idx, max_position in enumerate(max_position_rows):
        row = {"max_position": max_position}
        for key, rows in paired_rows.items():
            row[key] = rows[idx]
        result.append(row)
    return result


def _build_tasks(cfg: dict[str, Any]) -> list[Task]:
    symbols = cfg.get("symbols")
    if not isinstance(symbols, list) or not symbols:
        raise ValueError("symbols must be a non-empty list")

    dates = generate_dates(cfg["date_start"], cfg["date_end"])
    sim_map = _normalize_sim_param_map(cfg)

    tier_rows = _build_tier_param_rows(sim_map)
    tier_paired_keys = {"max_position", "tier_jump_bps", "boost_tier"}
    outer_sim_map = {
        key: value
        for key, value in sim_map.items()
        if key not in tier_paired_keys
    }
    sim_keys = sorted(outer_sim_map.keys())
    sim_lists = [outer_sim_map[k] for k in sim_keys]
    tasks: list[Task] = []
    for symbol in symbols:
        seen_params: set[str] = set()
        for tier_row in tier_rows:
            combos = itertools.product(*sim_lists) if sim_lists else [()]
            for combo in combos:
                sim_params = {k: combo[i] for i, k in enumerate(sim_keys)}
                sim_params.update(tier_row)
                sim_params = _effective_sim_params(sim_params)
                dedupe_key = _sim_params_dedupe_key(sim_params)
                if dedupe_key in seen_params:
                    continue
                seen_params.add(dedupe_key)
                tasks.append(
                    Task(
                        symbol=str(symbol),
                        dates=dates,
                        sim_params=sim_params,
                        cfg=cfg,
                    )
                )
    return tasks


def _run_strategy_task(task: Task) -> dict[str, Any]:
    cfg = task.cfg
    out_root = _output_dir_from_cfg(cfg)

    simulation = _build_simulation_config(task.sim_params)
    loader = _build_loader(
        cfg=cfg,
        simulation=simulation,
        require_volatility_cache=simulation.name_volatility is not None,
    )

    sim_dir, strat_dir = build_param_path_parts(task.sim_params)
    out_dir = os.path.join(out_root, task.symbol, sim_dir, strat_dir)

    output_cfg = cfg.get("output", {})
    overwrite = bool(output_cfg.get("overwrite", False))
    metrics = _run_daily_incremental(
        out_dir=out_dir,
        symbol=task.symbol,
        dates=task.dates,
        simulation=simulation,
        loader=loader,
        overwrite=overwrite,
    )

    row = {
        "symbol": task.symbol,
        "sim_dir": sim_dir,
        "strat_dir": strat_dir,
        "output_dir": out_dir,
    }
    row.update(task.sim_params)
    row.update(metrics)
    return row


def run_all_strategy(cfg: dict[str, Any]) -> None:
    tasks = _build_tasks(cfg)
    debug_mode = bool(cfg.get("__debug_mode__", False))
    if debug_mode and tasks:
        first = tasks[0]
        tasks = [
            Task(
                symbol=first.symbol,
                dates=[first.dates[0]],
                sim_params=first.sim_params,
                cfg=first.cfg,
            )
        ]
        print(
            f"debug mode: run first configuration for first day only "
            f"({tasks[0].symbol} {tasks[0].dates[0]})"
        )

    print(f"total strategy tasks: {len(tasks)}")
    if not tasks:
        return

    parallel_cfg = cfg.get("parallel", {})
    workers = max(1, int(parallel_cfg.get("num_workers", 1)))
    start_method = str(parallel_cfg.get("start_method", "spawn")).strip().lower()
    allowed = {"fork", "spawn", "forkserver"}
    if start_method not in allowed:
        raise ValueError(
            f"parallel.start_method must be one of {sorted(allowed)}, got: {start_method}"
        )
    raw_maxtasksperchild = parallel_cfg.get("maxtasksperchild")
    maxtasksperchild = (
        None if raw_maxtasksperchild in (None, 0) else max(1, int(raw_maxtasksperchild))
    )

    rows: list[dict[str, Any]] = []
    if workers > 1:
        print(
            "parallel config: "
            f"workers={workers}, start_method={start_method}, "
            f"maxtasksperchild={maxtasksperchild}"
        )
        ctx = get_context(start_method)
        progress_every = max(1, len(tasks) // 50)
        with ctx.Pool(processes=workers, maxtasksperchild=maxtasksperchild) as pool:
            it = pool.imap_unordered(_run_strategy_task, tasks, chunksize=1)
            idx = 0
            while idx < len(tasks):
                try:
                    row = it.next(timeout=60)
                    rows.append(row)
                    idx += 1
                    if idx == 1 or idx == len(tasks) or (idx % progress_every == 0):
                        print(f"[{idx}/{len(tasks)}] job completed", flush=True)
                except MpTimeoutError:
                    print(
                        f"[{idx}/{len(tasks)}] waiting... no job finished in last 60s",
                        flush=True,
                    )
    else:
        for task in tasks:
            rows.append(_run_strategy_task(task))

    compare_df = pd.DataFrame(rows).sort_values(
        by=["strategy_total_pnl"],
        ascending=[False],
    )
    output_root = _output_dir_from_cfg(cfg)
    compare_name = str(
        cfg.get("strategy_test", {}).get("summary_csv", "strategy_summary.csv")
    )
    compare_path = os.path.join(output_root, compare_name)
    os.makedirs(output_root, exist_ok=True)
    compare_df.to_csv(compare_path, index=False)
    print(f"strategy summary saved: {compare_path}")


def run_all(cfg: dict[str, Any]) -> None:
    print("running strategy only...")
    run_all_strategy(cfg)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg["__debug_mode__"] = bool(args.debug)
    run_all(cfg)


if __name__ == "__main__":
    main()
