from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import pandas as pd

from core.manager import Manager as ValidatedManager
from core.position import Position
from sim.loader import BinanceEventLoader, DateLike


MakerLevel = tuple[float, float]
PendingMakerQuote = tuple[int, list[MakerLevel], list[MakerLevel], float, float, bool]
SimplePendingMakerQuote = tuple[
    int,
    Optional[int],
    int,
    Optional[int],
    int,
    int,
    int,
    bool,
]
QTY_EPS = 1e-12


def _normalize_open_close_pair(
    value: float | Sequence[float],
    key: str,
    *,
    positive: bool,
) -> tuple[float, float]:
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError(f"{key} must be a number or an (open, close) pair")
        open_value = float(value[0])
        close_value = float(value[1])
    else:
        open_value = float(value)
        close_value = open_value

    if not math.isfinite(open_value) or not math.isfinite(close_value):
        raise ValueError(f"{key} must be finite")
    if positive:
        if open_value < 0.0 or close_value < 0.0:
            raise ValueError(f"{key} open and close must be >= 0")
    elif open_value < 0.0 or close_value < 0.0:
        raise ValueError(f"{key} open and close must be >= 0")
    return open_value, close_value


def _round_to_precision(value: float, precision: int) -> float:
    return float(f"{value:.{precision}f}")


def get_minimum_size(
    mid: float,
    qty_precision: int,
    min_order_notional: float,
    min_order_qty: float = 0.0,
) -> float:
    qty_precision = int(qty_precision)
    if qty_precision < 0:
        raise ValueError("qty_precision must be >= 0")

    mid = float(mid)
    if not math.isfinite(mid) or mid <= 0.0:
        return 0.0

    min_order_notional = float(min_order_notional)
    if not math.isfinite(min_order_notional) or min_order_notional < 0.0:
        raise ValueError("min_order_notional must be finite and >= 0")
    min_order_qty = float(min_order_qty)
    if not math.isfinite(min_order_qty) or min_order_qty < 0.0:
        raise ValueError("min_order_qty must be finite and >= 0")

    step_size = 10.0 ** (-qty_precision)
    target_qty = max(step_size, min_order_qty)
    if min_order_notional > 0.0:
        target_qty = max(target_qty, min_order_notional / mid)

    steps = math.ceil((target_qty / step_size) - QTY_EPS)
    return _round_to_precision(steps * step_size, qty_precision)


@dataclass(frozen=True)
class SimulationConfig:
    freq: int
    latency: int
    price_precision: int
    qty_precision: int
    mode: int
    taker_fee: float
    maker_fee: float
    name_instructor: Optional[str] = None
    lookback_instructor: Optional[int] = None
    name_intensity: str = "k"
    lookback_intensity: int = 100
    name_volatility: Optional[str] = None
    lookback_volatility: Optional[int] = None
    max_position_usdt: float | Sequence[float] = 0.0
    max_open_inventory_utilization: float = 1.0
    tier_jump_bps: float | Sequence[float] = 0.0
    boost_tier: float | Sequence[float] = 1.0
    max_holding_time: int = 0
    adj_spread_intensity: float | tuple[float, float] = 1.0
    adj_spread_instructor: float = 0.0
    passive_only: bool = False
    adj_spread_volatility: float | tuple[float, float] = 0.0
    min_quote_distance_bps: float = 0.0
    inventory_skew: Optional[tuple[float, float]] = None
    min_order_qty: float = 0.0
    min_order_notional: float = 0.0
    stoploss: float = 0.0
    takeprofit: float = 0.0
    open_curve: Optional[tuple[float, float, float]] = None
    close_curve: Optional[tuple[float, float, float]] = None
    boost_underwater: float | tuple[float, float] = (1.0, 1.0)
    boost_profitzone: float | tuple[float, float] = (1.0, 1.0)
    toxic_lock: tuple[int, int] | Sequence[int] = (0, 0)
    strict_mode: bool = True
    simple_mode: bool = True
    max_position_tiers: tuple[float, ...] = field(init=False, default=())
    tier_jump_bps_tuple: tuple[float, ...] = field(init=False, default=())
    boost_tier_tuple: tuple[float, ...] = field(init=False, default=())

    def __post_init__(self) -> None:
        max_position_tiers = self._normalize_tiers(
            self.max_position_usdt,
            "max_position",
        )
        stoploss = self._normalize_scalar(self.stoploss, "stoploss")
        takeprofit = self._normalize_scalar(self.takeprofit, "takeprofit")
        max_open_inventory_utilization = self._normalize_scalar(
            self.max_open_inventory_utilization,
            "max_open_inventory_utilization",
        )
        tier_jump_bps = self._normalize_tier_jump_bps(
            self.tier_jump_bps,
            tier_count=len(max_position_tiers),
        )
        boost_tier = self._normalize_boost_tier(
            self.boost_tier,
            tier_count=len(max_position_tiers),
        )
        object.__setattr__(
            self,
            "max_position_usdt",
            float(max_position_tiers[-1]) if max_position_tiers else 0.0,
        )
        object.__setattr__(self, "max_position_tiers", max_position_tiers)
        object.__setattr__(self, "tier_jump_bps_tuple", tier_jump_bps)
        object.__setattr__(self, "tier_jump_bps", tier_jump_bps)
        object.__setattr__(self, "boost_tier_tuple", boost_tier)
        object.__setattr__(self, "boost_tier", boost_tier)
        object.__setattr__(self, "stoploss", stoploss)
        object.__setattr__(self, "takeprofit", takeprofit)
        object.__setattr__(
            self,
            "max_open_inventory_utilization",
            max_open_inventory_utilization,
        )
        if self.freq <= 0:
            raise ValueError("freq must be > 0")
        if self.latency < 0:
            raise ValueError("latency must be >= 0")
        if self.price_precision < 0:
            raise ValueError("price_precision must be >= 0")
        if self.qty_precision < 0:
            raise ValueError("qty_precision must be >= 0")
        if self.mode not in (0, 1):
            raise ValueError("mode must be 0 or 1")
        if self.name_instructor is not None:
            instructor_name = str(self.name_instructor).strip().lower()
            if self.lookback_instructor is None:
                raise ValueError(
                    "lookback_instructor must be provided when name_instructor is set"
                )
            if instructor_name == "bbo_imbalance":
                if int(self.lookback_instructor) != 0:
                    raise ValueError("bbo_imbalance only supports lookback_instructor 0")
            elif int(self.lookback_instructor) <= 0:
                raise ValueError(
                    "lookback_instructor must be > 0 when name_instructor is provided"
                )
        if self.lookback_intensity <= 0:
            raise ValueError("lookback_intensity must be > 0")
        if self.name_volatility is not None:
            if self.lookback_volatility is None or int(self.lookback_volatility) <= 0:
                raise ValueError("lookback_volatility must be > 0 when name_volatility is provided")
        if self.max_position_usdt < 0:
            raise ValueError("max_position must be >= 0")
        if (
            self.max_open_inventory_utilization < 0
            or self.max_open_inventory_utilization > 1
        ):
            raise ValueError("max_open_inventory_utilization must be between 0 and 1")
        object.__setattr__(
            self,
            "adj_spread_intensity",
            _normalize_open_close_pair(
                self.adj_spread_intensity,
                "adj_spread_intensity",
                positive=True,
            ),
        )
        if not math.isfinite(float(self.adj_spread_instructor)):
            raise ValueError("adj_spread_instructor must be finite")
        if not isinstance(self.passive_only, bool):
            raise ValueError("passive_only must be boolean")
        object.__setattr__(
            self,
            "adj_spread_volatility",
            _normalize_open_close_pair(
                self.adj_spread_volatility,
                "adj_spread_volatility",
                positive=False,
            ),
        )
        try:
            min_quote_distance_bps = float(self.min_quote_distance_bps)
        except (TypeError, ValueError):
            raise ValueError("min_quote_distance_bps must be finite and >= 0") from None
        if not math.isfinite(min_quote_distance_bps) or min_quote_distance_bps < 0.0:
            raise ValueError("min_quote_distance_bps must be finite and >= 0")
        object.__setattr__(
            self,
            "min_quote_distance_bps",
            min_quote_distance_bps,
        )
        if self.inventory_skew is not None:
            if not isinstance(self.inventory_skew, (tuple, list)) or len(self.inventory_skew) != 2:
                raise ValueError("inventory_skew must be a (max_tick, skew_power) pair when provided")
            max_tick = float(self.inventory_skew[0])
            skew_power = float(self.inventory_skew[1])
            if max_tick < 0:
                raise ValueError("inventory_skew max_tick must be >= 0")
            if skew_power <= 0:
                raise ValueError("inventory_skew skew_power must be > 0")
        if self.min_order_qty < 0:
            raise ValueError("min_order_qty must be >= 0")
        if self.min_order_notional < 0:
            raise ValueError("min_order_notional must be >= 0")
        if self.stoploss < 0:
            raise ValueError("stoploss must be >= 0")
        if self.takeprofit < 0:
            raise ValueError("takeprofit must be >= 0")
        if not isinstance(self.simple_mode, bool):
            raise ValueError("simple_mode must be boolean")
        if self.open_curve is not None:
            if not isinstance(self.open_curve, (tuple, list)) or len(self.open_curve) != 3:
                raise ValueError("open_curve must be a (min, max, order) triple when provided")
            float(self.open_curve[0])
            max_scale = float(self.open_curve[1])
            order = float(self.open_curve[2])
            if max_scale < 0:
                raise ValueError("open_curve max must be >= 0")
            if order < 0:
                raise ValueError("open_curve order must be >= 0")
        if self.close_curve is not None:
            if not isinstance(self.close_curve, (tuple, list)) or len(self.close_curve) != 3:
                raise ValueError("close_curve must be a (min, max, order) triple when provided")
            float(self.close_curve[0])
            max_scale = float(self.close_curve[1])
            order = float(self.close_curve[2])
            if max_scale < 0:
                raise ValueError("close_curve max must be >= 0")
            if order < 0:
                raise ValueError("close_curve order must be >= 0")
        object.__setattr__(
            self,
            "boost_underwater",
            _normalize_open_close_pair(
                self.boost_underwater,
                "boost_underwater",
                positive=False,
            ),
        )
        object.__setattr__(
            self,
            "boost_profitzone",
            _normalize_open_close_pair(
                self.boost_profitzone,
                "boost_profitzone",
                positive=False,
            ),
        )
        if not isinstance(self.toxic_lock, (tuple, list)) or len(self.toxic_lock) != 2:
            raise ValueError("toxic_lock must be a (lookback_bar, open_side_slip) pair")
        lookback_bar = self._normalize_count(
            self.toxic_lock[0],
            "toxic_lock lookback_bar",
        )
        open_side_slip = self._normalize_count(
            self.toxic_lock[1],
            "toxic_lock open_side_slip",
        )
        if lookback_bar > 0 and open_side_slip > lookback_bar:
            raise ValueError("toxic_lock open_side_slip must be <= lookback_bar")
        object.__setattr__(
            self,
            "toxic_lock",
            (lookback_bar, open_side_slip),
        )
        if not isinstance(self.strict_mode, bool):
            raise ValueError("strict_mode must be boolean")
    @property
    def tick_size(self) -> float:
        return 10.0 ** (-self.price_precision)

    @property
    def step_size(self) -> float:
        return 10.0 ** (-self.qty_precision)

    @staticmethod
    def _normalize_scalar(value: object, name: str) -> float:
        if isinstance(value, (list, tuple)):
            raise ValueError(f"{name} must be a single number")
        value_float = float(value)
        if not math.isfinite(value_float):
            raise ValueError(f"{name} must be finite")
        return value_float

    @staticmethod
    def _normalize_tiers(value: object, name: str) -> tuple[float, ...]:
        if isinstance(value, (list, tuple)):
            if not value:
                raise ValueError(f"{name} must contain at least one tier")
            tiers = tuple(float(v) for v in value)
        else:
            tiers = (float(value),)
        if any((not math.isfinite(v)) or v < 0.0 for v in tiers):
            raise ValueError(f"{name} tiers must be finite and >= 0")
        if len(tiers) > 1:
            prev = tiers[0]
            if prev <= 0.0:
                raise ValueError(f"{name} tiers must be > 0 when tiered")
            for current in tiers[1:]:
                if current <= prev:
                    raise ValueError(f"{name} tiers must be strictly increasing")
                prev = current
        return tiers

    @staticmethod
    def _normalize_tier_jump_bps(
        value: object,
        *,
        tier_count: int,
    ) -> tuple[float, ...]:
        if tier_count <= 0:
            return ()
        if isinstance(value, (list, tuple)):
            jumps = tuple(float(v) for v in value)
        else:
            jumps = (float(value),)
        if len(jumps) == 1 and tier_count > 1:
            jumps = tuple([jumps[0]] * tier_count)
        if len(jumps) != tier_count:
            raise ValueError("tier_jump_bps must have the same length as max_position")
        if any((not math.isfinite(v)) or v < 0.0 for v in jumps):
            raise ValueError("tier_jump_bps values must be finite and >= 0")
        return jumps

    @staticmethod
    def _normalize_boost_tier(
        value: object,
        *,
        tier_count: int,
    ) -> tuple[float, ...]:
        if tier_count <= 0:
            return ()
        if isinstance(value, (list, tuple)):
            boosts = tuple(float(v) for v in value)
        else:
            boosts = (float(value),)
        if len(boosts) == 1 and tier_count > 1:
            boosts = tuple([boosts[0]] * tier_count)
        if len(boosts) != tier_count:
            raise ValueError("boost_tier must have the same length as max_position")
        if any((not math.isfinite(v)) or v < 0.0 for v in boosts):
            raise ValueError("boost_tier values must be finite and >= 0")
        return boosts

    @staticmethod
    def _normalize_count(value: object, name: str) -> int:
        if isinstance(value, bool):
            raise ValueError(f"{name} must be an integer")
        value_float = float(value)
        if (
            not math.isfinite(value_float)
            or value_float < 0.0
            or not value_float.is_integer()
        ):
            raise ValueError(f"{name} must be an integer >= 0")
        return int(value_float)

    @property
    def total_max_position_usdt(self) -> float:
        return float(self.max_position_usdt)


class SimpleMakerStrategy:
    EPS = 1e-12

    def __init__(
        self,
        simulation: SimulationConfig,
        strategy: Optional[SimulationConfig] = None,
        loader: Optional[BinanceEventLoader] = None,
        position: Optional[Position] = None,
    ):
        self.sim = simulation
        self.cfg = strategy or simulation
        self.loader = loader or BinanceEventLoader()
        self.manager = ValidatedManager(
            position=position or Position(),
            mode=self.sim.mode,
            symbol_rules={
                "tick_size": self.sim.tick_size,
                "step_size": self.sim.step_size,
                "min_qty": self.cfg.min_order_qty,
                "min_notional": self.cfg.min_order_notional,
            },
        )

        self._latest_best_ask: Optional[float] = None
        self._latest_best_bid: Optional[float] = None
        self._traded_volume: float = 0.0
        self._records: list[tuple[int, float, float, float, float, float, float, float, float, float]] = []
        self._reach_and_release_active: bool = False
        self._max_holding_start_ts: Optional[int] = None
        self._max_holding_was_at_limit: bool = False
        self._pending_maker_quotes: deque[PendingMakerQuote] = deque()
        self._pending_simple_maker_quotes: deque[SimplePendingMakerQuote] = deque()
        self._toxic_lock_prev_mid: Optional[float] = None
        self._toxic_lock_mid_moves: deque[int] = self._new_toxic_lock_mid_moves()
        self._tier_anchor_side: int = 0
        self._tier_price_anchor: Optional[float] = None

    def snapshot_state(self) -> dict:
        pos = self.manager.position
        realized_pnl = float(pos.realized_pnl)
        unrealized_pnl = float(pos.unrealized_pnl)
        return {
            "position": {
                "qty": float(pos.qty),
                "cost": None if math.isnan(float(pos.cost)) else float(pos.cost),
                "mark_notional_usdt": float(pos.mark_notional_usdt),
                "cost_notional_usdt": float(pos.cost_notional_usdt),
                "gross_cost_notional_usdt": float(pos.gross_cost_notional_usdt),
                "realized_pnl": realized_pnl,
                "unrealized_pnl": unrealized_pnl,
                "mid": float(pos.mid),
            },
            "realized_pnl": realized_pnl,
            "unrealized_pnl": unrealized_pnl,
            "total_pnl": float(realized_pnl + unrealized_pnl),
            "max_position_usdt": float(self.cfg.max_position_usdt),
            "max_position_tiers": [float(v) for v in self.cfg.max_position_tiers],
            "tier_jump_bps": [float(v) for v in self.cfg.tier_jump_bps_tuple],
            "boost_tier": [float(v) for v in self.cfg.boost_tier_tuple],
            "stoploss_usdt": float(self.cfg.stoploss),
            "takeprofit_usdt": float(self.cfg.takeprofit),
            "traded_volume": float(self._traded_volume),
            "latest_best_ask": self._latest_best_ask,
            "latest_best_bid": self._latest_best_bid,
            "reach_and_release_active": bool(self._reach_and_release_active),
            "max_holding_start_ts": self._max_holding_start_ts,
            "max_holding_was_at_limit": bool(self._max_holding_was_at_limit),
            "toxic_lock_prev_mid": self._toxic_lock_prev_mid,
            "toxic_lock_mid_moves": list(self._toxic_lock_mid_moves),
            "toxic_lock_open_side_slip": self._toxic_lock_open_side_slip_count(),
            "toxic_lock_active": self._toxic_lock_active(),
            "tier_anchor_side": int(self._tier_anchor_side),
            "tier_price_anchor": (
                None if self._tier_price_anchor is None else float(self._tier_price_anchor)
            ),
            "pending_maker_quotes": [
                {
                    "active_ts": int(active_ts),
                    "ask_levels": [
                        {"price": float(price), "qty": float(qty)}
                        for price, qty in ask_levels
                    ],
                    "bid_levels": [
                        {"price": float(price), "qty": float(qty)}
                        for price, qty in bid_levels
                    ],
                    "best_ask": float(best_ask),
                    "best_bid": float(best_bid),
                    "close_only": bool(close_only),
                }
                for (
                    active_ts,
                    ask_levels,
                    bid_levels,
                    best_ask,
                    best_bid,
                    close_only,
                ) in self._pending_maker_quotes
            ],
            "pending_simple_maker_quotes": [
                {
                    "active_ts": int(active_ts),
                    "ask_price_ticks": ask_price_ticks,
                    "ask_qty_steps": int(ask_qty_steps),
                    "bid_price_ticks": bid_price_ticks,
                    "bid_qty_steps": int(bid_qty_steps),
                    "best_ask_ticks": int(best_ask_ticks),
                    "best_bid_ticks": int(best_bid_ticks),
                    "close_only": bool(close_only),
                }
                for (
                    active_ts,
                    ask_price_ticks,
                    ask_qty_steps,
                    bid_price_ticks,
                    bid_qty_steps,
                    best_ask_ticks,
                    best_bid_ticks,
                    close_only,
                ) in self._pending_simple_maker_quotes
            ],
            "books": self.manager.snapshot_ticks(),
        }

    def restore_state(self, state: dict) -> None:
        pos_state = state.get("position", {}) if isinstance(state, dict) else {}
        pos = Position()
        pos.qty = float(pos_state.get("qty", 0.0))
        cost_raw = pos_state.get("cost")
        pos.cost = math.nan if cost_raw is None else float(cost_raw)
        pos.realized_pnl = float(pos_state.get("realized_pnl", 0.0))

        mid_raw = pos_state.get("mid")
        if mid_raw is None:
            unrealized = float(pos_state.get("unrealized_pnl", 0.0))
            if abs(pos.qty) > self.EPS and math.isfinite(pos.cost):
                pos.mid = pos.cost + unrealized / pos.qty
            else:
                pos.mid = 0.0
        else:
            pos.mid = float(mid_raw)

        self.manager.position = pos
        if hasattr(self.manager, "_sync_position_steps"):
            self.manager._sync_position_steps()
        self._traded_volume = float(state.get("traded_volume", 0.0))
        self._latest_best_ask = state.get("latest_best_ask")
        self._latest_best_bid = state.get("latest_best_bid")
        self._reach_and_release_active = bool(state.get("reach_and_release_active", False))
        start_ts_raw = state.get("max_holding_start_ts")
        self._max_holding_start_ts = None if start_ts_raw is None else int(start_ts_raw)
        self._max_holding_was_at_limit = bool(state.get("max_holding_was_at_limit", False))
        prev_mid_raw = state.get("toxic_lock_prev_mid")
        self._toxic_lock_prev_mid = (
            None if prev_mid_raw is None else float(prev_mid_raw)
        )
        self._toxic_lock_mid_moves = self._new_toxic_lock_mid_moves(
            state.get("toxic_lock_mid_moves", [])
        )
        self._tier_anchor_side = int(
            state.get("tier_anchor_side", state.get("tier_crit_side", 0))
        )
        tier_price_anchor_raw = state.get(
            "tier_price_anchor",
            state.get("tier_cost_crit"),
        )
        if tier_price_anchor_raw is None:
            self._tier_price_anchor = None
        else:
            tier_price_anchor = float(tier_price_anchor_raw)
            self._tier_price_anchor = (
                tier_price_anchor
                if math.isfinite(tier_price_anchor) and tier_price_anchor > 0.0
                else None
            )
        self._normalize_tier_price_anchor_state()
        self._pending_maker_quotes = deque()
        self._pending_simple_maker_quotes = deque()
        pending_quotes = state.get("pending_maker_quotes", [])
        if isinstance(pending_quotes, list):
            for row in pending_quotes:
                if not isinstance(row, dict):
                    continue
                try:
                    ask_levels_raw = row.get("ask_levels")
                    bid_levels_raw = row.get("bid_levels")
                    ask_levels = self._parse_pending_levels(ask_levels_raw)
                    bid_levels = self._parse_pending_levels(bid_levels_raw)
                    self._pending_maker_quotes.append(
                        (
                            int(row["active_ts"]),
                            ask_levels,
                            bid_levels,
                            float(row["best_ask"]),
                            float(row["best_bid"]),
                            bool(row.get("close_only", False)),
                        )
                    )
                except (TypeError, ValueError, KeyError):
                    continue
        pending_simple_quotes = state.get("pending_simple_maker_quotes", [])
        if isinstance(pending_simple_quotes, list):
            for row in pending_simple_quotes:
                if not isinstance(row, dict):
                    continue
                try:
                    ask_price_raw = row.get("ask_price_ticks")
                    bid_price_raw = row.get("bid_price_ticks")
                    self._pending_simple_maker_quotes.append(
                        (
                            int(row["active_ts"]),
                            None if ask_price_raw is None else int(ask_price_raw),
                            int(row.get("ask_qty_steps", 0)),
                            None if bid_price_raw is None else int(bid_price_raw),
                            int(row.get("bid_qty_steps", 0)),
                            int(row["best_ask_ticks"]),
                            int(row["best_bid_ticks"]),
                            bool(row.get("close_only", False)),
                        )
                    )
                except (TypeError, ValueError, KeyError):
                    continue
        books = state.get("books", {})
        if isinstance(books, dict):
            self.manager.restore_snapshot_ticks(books)

    @staticmethod
    def _parse_pending_levels(levels: object) -> list[MakerLevel]:
        parsed: list[MakerLevel] = []
        if not isinstance(levels, list):
            return parsed
        for row in levels:
            try:
                if isinstance(row, dict):
                    price = float(row["price"])
                    qty = float(row["qty"])
                else:
                    price = float(row[0])
                    qty = float(row[1])
            except (TypeError, ValueError, KeyError, IndexError):
                continue
            if price > 0.0 and qty > 0.0:
                parsed.append((price, qty))
        return parsed

    def run_day(self, symbol: str, date: DateLike) -> pd.DataFrame:
        self._records = []
        vol_specs = self._selected_volatility_specs()
        ti_spec = self._selected_intensity_spec()
        instructor_spec = self._selected_instructor_spec()

        for event in self.loader.iter_merged_alpha_trade_tuples(
            symbol=symbol,
            date=date,
            freq=self.sim.freq,
            trade_intensity_spec=ti_spec,
            volatility_specs=vol_specs,
            instructor_spec=instructor_spec,
        ):
            kind = event[0]
            ts = int(event[1])
            self._activate_due_maker_quotes(current_timestamp=ts)
            if kind == "trade":
                self._on_trade_event(
                    trade_time=ts,
                    is_buyer_maker=bool(event[2]),
                    trade_price=float(event[3]),
                    trade_qty=float(event[4]),
                )
            else:
                if hasattr(event, "intensity_positive"):
                    intensity_positive = event.intensity_positive
                    intensity_negative = event.intensity_negative
                    volatility_scalar = float(event.volatility_scalar)
                elif len(event) >= 12:
                    intensity_positive = event[5]
                    intensity_negative = event[6]
                    volatility_scalar = float(event[7])
                else:
                    intensity_positive = event[5]
                    intensity_negative = event[5]
                    volatility_scalar = float(event[6])
                self._on_ticker_event(
                    timestamp=ts,
                    best_bid=float(event[2]),
                    best_ask=float(event[3]),
                    instructor_value=event[4],
                    intensity_positive=intensity_positive,
                    intensity_negative=intensity_negative,
                    volatility_scalar=volatility_scalar,
                )
        return pd.DataFrame(
            self._records,
            columns=[
                "timestamp",
                "price",
                "position",
                "mark_notional_usdt",
                "cost_notional_usdt",
                "gross_cost_notional_usdt",
                "realized_pnl",
                "unrealized_pnl",
                "total_pnl",
                "traded_volume",
            ],
        )

    def run_dates(self, symbol: str, dates: Iterable[DateLike]) -> pd.DataFrame:
        frames = [self.run_day(symbol=symbol, date=date) for date in dates]
        if not frames:
            return pd.DataFrame(
                columns=[
                    "timestamp",
                    "price",
                    "position",
                    "mark_notional_usdt",
                    "cost_notional_usdt",
                    "gross_cost_notional_usdt",
                    "realized_pnl",
                    "unrealized_pnl",
                    "total_pnl",
                    "traded_volume",
                ]
            )
        return pd.concat(frames, ignore_index=True)

    def _on_trade_event(
        self,
        trade_time: int,
        is_buyer_maker: bool,
        trade_price: float,
        trade_qty: float,
    ) -> float:
        trade_price = self._round_to_precision(float(trade_price), self.sim.price_precision)
        trade_qty = self._round_to_precision(float(trade_qty), self.sim.qty_precision)
        if trade_price <= 0.0 or trade_qty <= 0.0:
            return 0.0

        prev_abs_pos_qty = abs(float(self.manager.position.qty))
        prev_pos_qty = float(self.manager.position.qty)
        # is_buyer_maker=True means public sell flow hits passive bids.
        maker_fills = self.manager.match_maker_fills(
            trade_time=trade_time,
            trade_side=not bool(is_buyer_maker),
            trade_price=trade_price,
            trade_qty=trade_qty,
            fee_rate=self.sim.maker_fee,
        )
        maker_filled_qty = self._filled_qty_from_fills(maker_fills)
        remaining_qty = max(0.0, trade_qty - maker_filled_qty)
        taker_fills: list[MakerLevel] = []
        taker_filled_qty = 0.0
        if remaining_qty > self.EPS:
            taker_fills = self.manager.match_taker_fills(
                trade_time=trade_time,
                trade_side=bool(is_buyer_maker),
                trade_price=trade_price,
                trade_qty=remaining_qty,
                fee_rate=self.sim.taker_fee,
            )
            taker_filled_qty = self._filled_qty_from_fills(taker_fills)
        filled_qty = maker_filled_qty + taker_filled_qty
        if filled_qty > 0.0:
            self._traded_volume += float(filled_qty) * float(trade_price)
            maker_sign = 1.0 if bool(is_buyer_maker) else -1.0
            taker_sign = -maker_sign
            fill_events = [
                (maker_sign * float(qty), float(price))
                for price, qty in maker_fills
            ]
            fill_events.extend(
                (taker_sign * float(qty), float(price))
                for price, qty in taker_fills
            )
            self._update_tier_price_anchor_after_trade(
                prev_qty=prev_pos_qty,
                fills=fill_events,
            )
        new_abs_pos_qty = abs(float(self.manager.position.qty))
        position_reduced = new_abs_pos_qty + self.EPS < prev_abs_pos_qty
        self._update_max_holding_tracking(
            timestamp=int(trade_time),
            position_reduced=position_reduced,
        )
        return float(filled_qty)

    def _clear_open_maker_books(self) -> None:
        pos_qty = float(self.manager.position.qty)
        if pos_qty > self.EPS:
            self.manager.books.bid_maker.clear()
        elif pos_qty < -self.EPS:
            self.manager.books.ask_maker.clear()
        else:
            self.manager.books.ask_maker.clear()
            self.manager.books.bid_maker.clear()

    def _toxic_lock_params(self) -> tuple[int, int]:
        lookback_bar, open_side_slip = self.cfg.toxic_lock
        return int(lookback_bar), int(open_side_slip)

    def _new_toxic_lock_mid_moves(
        self,
        moves: Iterable[int] = (),
    ) -> deque[int]:
        lookback_bar, _open_side_slip = self._toxic_lock_params()
        maxlen = max(0, int(lookback_bar))
        normalized: list[int] = []
        for move in moves:
            move_int = int(move)
            if move_int < 0:
                normalized.append(-1)
            elif move_int > 0:
                normalized.append(1)
            else:
                normalized.append(0)
        return deque(normalized[-maxlen:] if maxlen > 0 else [], maxlen=maxlen)

    def _update_toxic_lock_mid_moves(self, mid: float) -> None:
        lookback_bar, open_side_slip = self._toxic_lock_params()
        mid = float(mid)
        prev_mid = self._toxic_lock_prev_mid
        self._toxic_lock_prev_mid = mid

        if lookback_bar <= 0 or open_side_slip <= 0:
            self._toxic_lock_mid_moves.clear()
            return
        if prev_mid is None or not math.isfinite(prev_mid) or not math.isfinite(mid):
            return

        if mid < prev_mid - self.EPS:
            self._toxic_lock_mid_moves.append(-1)
        elif mid > prev_mid + self.EPS:
            self._toxic_lock_mid_moves.append(1)
        else:
            self._toxic_lock_mid_moves.append(0)

    def _toxic_lock_open_side_slip_count(self) -> int:
        pos_qty = float(self.manager.position.qty)
        if pos_qty > self.EPS:
            return sum(1 for move in self._toxic_lock_mid_moves if move < 0)
        if pos_qty < -self.EPS:
            return sum(1 for move in self._toxic_lock_mid_moves if move > 0)
        return 0

    def _toxic_lock_active(self) -> bool:
        lookback_bar, open_side_slip = self._toxic_lock_params()
        if lookback_bar <= 0 or open_side_slip <= 0:
            return False
        if len(self._toxic_lock_mid_moves) < lookback_bar:
            return False
        if abs(float(self.manager.position.qty)) <= self.EPS:
            return False
        return self._toxic_lock_open_side_slip_count() >= open_side_slip

    def _on_ticker_event(
        self,
        timestamp: int,
        best_bid: float,
        best_ask: float,
        intensity_value: Optional[float] = None,
        volatility_scalar: float = 0.0,
        instructor_value: Optional[float] = None,
        intensity_positive: Optional[float] = None,
        intensity_negative: Optional[float] = None,
    ) -> None:
        rounded_best_bid = self._round_to_precision(float(best_bid), self.sim.price_precision)
        rounded_best_ask = self._round_to_precision(float(best_ask), self.sim.price_precision)
        if (
            (not math.isfinite(rounded_best_bid))
            or (not math.isfinite(rounded_best_ask))
            or rounded_best_bid <= 0.0
            or rounded_best_ask <= 0.0
            or rounded_best_bid >= rounded_best_ask
        ):
            return
        mid = 0.5 * (rounded_best_bid + rounded_best_ask)

        if intensity_positive is None:
            intensity_positive = intensity_value
        if intensity_negative is None:
            intensity_negative = intensity_value
        positive_intensity_base = self._safe_non_negative(intensity_positive)
        negative_intensity_base = self._safe_non_negative(intensity_negative)
        instructor_scalar = self._safe_finite(instructor_value)
        vol_scalar = self._safe_non_negative(volatility_scalar)
        open_ask_distance = self._quote_distance(
            intensity_base=positive_intensity_base,
            vol_scalar=vol_scalar,
            close=False,
        )
        open_bid_distance = self._quote_distance(
            intensity_base=negative_intensity_base,
            vol_scalar=vol_scalar,
            close=False,
        )
        close_ask_distance = self._quote_distance(
            intensity_base=positive_intensity_base,
            vol_scalar=vol_scalar,
            close=True,
        )
        close_bid_distance = self._quote_distance(
            intensity_base=negative_intensity_base,
            vol_scalar=vol_scalar,
            close=True,
        )
        # Instructor values are directional alpha signals, so convert them to
        # a common price shift applied to both sides of the quote.
        instructor_shift = instructor_scalar * self.cfg.adj_spread_instructor * mid
        skew_shift = self._inventory_skew_price_shift(mid=mid)
        ask_instructor_shift = instructor_shift
        bid_instructor_shift = instructor_shift
        if self.cfg.passive_only:
            ask_instructor_shift = max(0.0, ask_instructor_shift)
            bid_instructor_shift = min(0.0, bid_instructor_shift)

        raw_open_ask_price = (
            rounded_best_ask + open_ask_distance + ask_instructor_shift + skew_shift
        )
        raw_open_bid_price = (
            rounded_best_bid - open_bid_distance + bid_instructor_shift + skew_shift
        )
        raw_close_ask_price = (
            rounded_best_ask + close_ask_distance + ask_instructor_shift + skew_shift
        )
        raw_close_bid_price = (
            rounded_best_bid - close_bid_distance + bid_instructor_shift + skew_shift
        )
        open_ask_price = self._price_ceil(raw_open_ask_price)
        open_bid_price = self._price_floor(raw_open_bid_price)
        close_ask_price = self._price_ceil(raw_close_ask_price)
        close_bid_price = self._price_floor(raw_close_bid_price)

        if (
            (not math.isfinite(open_ask_price))
            or (not math.isfinite(open_bid_price))
            or (not math.isfinite(close_ask_price))
            or (not math.isfinite(close_bid_price))
            or open_ask_price <= 0.0
            or open_bid_price <= 0.0
            or close_ask_price <= 0.0
            or close_bid_price <= 0.0
        ):
            return

        self._latest_best_bid = rounded_best_bid
        self._latest_best_ask = rounded_best_ask
        self._update_toxic_lock_mid_moves(mid)
        self._update_max_holding_tracking(timestamp=timestamp)

        if self._should_activate_stoploss(mid=mid):
            self._reach_and_release_active = True
        if self._should_activate_takeprofit(mid=mid):
            self._reach_and_release_active = True
        if self._should_activate_max_holding_timeout(timestamp=timestamp):
            self._reach_and_release_active = True

        if self._reach_and_release_active and self._place_reach_and_release_taker(mid=mid):
            self._append_record(timestamp=timestamp, mid=mid)
            return

        self._clear_taker_books()
        toxic_lock_active = self._toxic_lock_active()
        if toxic_lock_active:
            self._clear_open_maker_books()
        open_allowed = not toxic_lock_active
        if self.sim.simple_mode:
            best_ask_ticks = self._price_round_ticks(rounded_best_ask)
            best_bid_ticks = self._price_round_ticks(rounded_best_bid)
            open_ask_price_ticks = self._price_ceil_ticks(raw_open_ask_price)
            open_bid_price_ticks = self._price_floor_ticks(raw_open_bid_price)
            close_ask_price_ticks = self._price_ceil_ticks(raw_close_ask_price)
            close_bid_price_ticks = self._price_floor_ticks(raw_close_bid_price)
            if (
                best_ask_ticks <= 0
                or best_bid_ticks <= 0
                or open_ask_price_ticks <= 0
                or open_bid_price_ticks <= 0
                or close_ask_price_ticks <= 0
                or close_bid_price_ticks <= 0
            ):
                return

            (
                quote_ask_price_ticks,
                ask_qty_steps,
                quote_bid_price_ticks,
                bid_qty_steps,
                quote_close_only,
            ) = self._simple_maker_steps_for_ticker(
                mid=mid,
                open_allowed=open_allowed,
                open_ask_price_ticks=open_ask_price_ticks,
                open_bid_price_ticks=open_bid_price_ticks,
                close_ask_price_ticks=close_ask_price_ticks,
                close_bid_price_ticks=close_bid_price_ticks,
            )
            (
                quote_ask_price_ticks,
                ask_qty_steps,
                quote_bid_price_ticks,
                bid_qty_steps,
            ) = self._apply_tier_open_gate_steps(
                ask_price_ticks=quote_ask_price_ticks,
                ask_qty_steps=ask_qty_steps,
                bid_price_ticks=quote_bid_price_ticks,
                bid_qty_steps=bid_qty_steps,
                close_only=quote_close_only,
            )
            (
                quote_ask_price_ticks,
                ask_qty_steps,
                quote_bid_price_ticks,
                bid_qty_steps,
            ) = self._clip_simple_quote_steps_to_bbo_distance(
                ask_price_ticks=quote_ask_price_ticks,
                ask_qty_steps=ask_qty_steps,
                bid_price_ticks=quote_bid_price_ticks,
                bid_qty_steps=bid_qty_steps,
                best_ask_ticks=best_ask_ticks,
                best_bid_ticks=best_bid_ticks,
            )
            self._pending_maker_quotes.clear()
            self._pending_simple_maker_quotes.clear()
            self._enqueue_simple_maker_quote(
                base_timestamp=timestamp,
                ask_price_ticks=quote_ask_price_ticks,
                ask_qty_steps=ask_qty_steps,
                bid_price_ticks=quote_bid_price_ticks,
                bid_qty_steps=bid_qty_steps,
                best_ask_ticks=best_ask_ticks,
                best_bid_ticks=best_bid_ticks,
                close_only=quote_close_only,
            )
            self._activate_pending_simple_maker_quotes(current_timestamp=timestamp)
        else:
            ask_qty, bid_qty = self._base_symmetric_qty(mid=mid)
            ask_qty, bid_qty = self._apply_inventory_limit(
                mid=mid,
                ask_qty=ask_qty,
                bid_qty=bid_qty,
            )
            ask_levels, bid_levels, quote_close_only = self._maker_levels_for_ticker(
                mid=mid,
                open_allowed=open_allowed,
                open_ask_price=open_ask_price,
                ask_qty=ask_qty,
                open_bid_price=open_bid_price,
                bid_qty=bid_qty,
                close_ask_price=close_ask_price,
                close_bid_price=close_bid_price,
                best_ask=rounded_best_ask,
                best_bid=rounded_best_bid,
            )
            ask_levels, bid_levels = self._apply_tier_open_gate(
                ask_levels=ask_levels,
                bid_levels=bid_levels,
                close_only=quote_close_only,
            )
            ask_levels, bid_levels = self._clip_quote_levels_to_bbo_distance(
                ask_levels=ask_levels,
                bid_levels=bid_levels,
                best_ask=rounded_best_ask,
                best_bid=rounded_best_bid,
            )
            self._pending_simple_maker_quotes.clear()
            self._pending_maker_quotes.clear()
            self._enqueue_maker_levels(
                base_timestamp=timestamp,
                ask_levels=ask_levels,
                bid_levels=bid_levels,
                best_ask=float(self._latest_best_ask),
                best_bid=float(self._latest_best_bid),
                close_only=quote_close_only,
            )
            self._activate_pending_maker_quotes(current_timestamp=timestamp)

        self._append_record(timestamp=timestamp, mid=mid)

    def _append_record(self, timestamp: int, mid: float) -> None:
        self.manager.position.mark(mid)
        self._records.append(
            (
                int(timestamp),
                float(mid),
                float(self.manager.position.qty),
                float(self.manager.position.mark_notional_usdt),
                float(self.manager.position.cost_notional_usdt),
                float(self.manager.position.gross_cost_notional_usdt),
                float(self.manager.position.realized_pnl),
                float(self.manager.position.unrealized_pnl),
                float(self.manager.position.realized_pnl + self.manager.position.unrealized_pnl),
                float(self._traded_volume),
            )
        )

    def _should_activate_stoploss(self, mid: float) -> bool:
        stoploss_usdt = float(self.cfg.stoploss)
        if stoploss_usdt <= self.EPS:
            return False

        pos = self.manager.position
        pos_qty = float(pos.qty)
        if abs(pos_qty) <= self.EPS:
            return False
        cost = float(pos.cost)
        if not math.isfinite(cost):
            return False

        floating_unrealized = pos_qty * (float(mid) - cost)
        floating_loss = max(0.0, -floating_unrealized)
        return floating_loss + self.EPS >= stoploss_usdt

    def _should_activate_takeprofit(self, mid: float) -> bool:
        takeprofit_usdt = float(self.cfg.takeprofit)
        if takeprofit_usdt <= self.EPS:
            return False

        pos = self.manager.position
        pos_qty = float(pos.qty)
        if abs(pos_qty) <= self.EPS:
            return False
        cost = float(pos.cost)
        if not math.isfinite(cost):
            return False

        floating_unrealized = pos_qty * (float(mid) - cost)
        floating_profit = max(0.0, floating_unrealized)
        return floating_profit + self.EPS >= takeprofit_usdt

    def _should_activate_max_holding_timeout(self, timestamp: int) -> bool:
        max_holding_time = int(self.cfg.max_holding_time)
        if max_holding_time < 0:
            return False
        if self._max_holding_start_ts is None:
            return False
        if max_holding_time == 0:
            return True
        return int(timestamp) - int(self._max_holding_start_ts) > max_holding_time

    def _update_max_holding_tracking(
        self,
        timestamp: int,
        position_reduced: bool = False,
    ) -> None:
        at_limit = self._is_at_max_position()
        if position_reduced:
            self._max_holding_start_ts = None
            self._max_holding_was_at_limit = at_limit
            return
        if at_limit and not self._max_holding_was_at_limit:
            self._max_holding_start_ts = int(timestamp)
        elif not at_limit:
            self._max_holding_start_ts = None
        self._max_holding_was_at_limit = at_limit

    def _is_at_max_position(self) -> bool:
        max_pos_usdt = float(self.cfg.max_position_usdt)
        if max_pos_usdt <= self.EPS:
            return False
        return (
            float(self.manager.position.gross_cost_notional_usdt) + self.EPS
            >= max_pos_usdt
        )

    def _position_side(self) -> int:
        pos_qty = float(self.manager.position.qty)
        return self._side_from_qty(pos_qty)

    def _side_from_qty(self, qty: float) -> int:
        qty = float(qty)
        if qty > self.EPS:
            return 1
        if qty < -self.EPS:
            return -1
        return 0

    def _gross_position_usdt(self) -> float:
        return float(self.manager.position.gross_cost_notional_usdt)

    def _tier_index_for_gross(self, gross_cost_notional_usdt: float) -> int:
        tiers = self.cfg.max_position_tiers
        if not tiers:
            return 0
        gross = max(0.0, float(gross_cost_notional_usdt))
        for idx, cap in enumerate(tiers):
            if gross < float(cap) - self.EPS:
                return idx
        return len(tiers) - 1

    def _current_open_tier_index(self) -> int:
        return self._tier_index_for_gross(self._gross_position_usdt())

    def _current_close_tier_index(self) -> int:
        return self._tier_index_for_gross(self._gross_position_usdt())

    def _normalize_tier_price_anchor_state(self) -> None:
        current_side = self._position_side()
        if current_side == 0:
            self._tier_anchor_side = 0
            self._tier_price_anchor = None
            return
        if self._tier_anchor_side != current_side:
            self._tier_anchor_side = current_side
            self._tier_price_anchor = None
        if self._tier_price_anchor is not None:
            anchor = float(self._tier_price_anchor)
            if not math.isfinite(anchor) or anchor <= 0.0:
                self._tier_price_anchor = None

    def _update_tier_price_anchor_after_trade(
        self,
        *,
        prev_qty: float,
        fills: Sequence[tuple[float, float]],
    ) -> None:
        running_qty = float(prev_qty)
        for signed_qty, price in fills:
            signed_qty = float(signed_qty)
            price = float(price)
            if abs(signed_qty) <= self.EPS:
                continue

            prev_side = self._side_from_qty(running_qty)
            running_qty += signed_qty
            current_side = self._side_from_qty(running_qty)
            if current_side == 0:
                self._tier_anchor_side = 0
                self._tier_price_anchor = None
                continue
            if prev_side != current_side and math.isfinite(price) and price > 0.0:
                self._tier_anchor_side = current_side
                self._tier_price_anchor = price

        current_side = self._position_side()
        if current_side == 0:
            self._tier_anchor_side = 0
            self._tier_price_anchor = None
            return
        if self._tier_anchor_side != current_side:
            self._tier_anchor_side = current_side
            self._tier_price_anchor = None
        self._normalize_tier_price_anchor_state()

    def _tier_gate_price_anchor(self, tier_index: int) -> Optional[float]:
        if tier_index <= 0:
            return None
        self._normalize_tier_price_anchor_state()
        if self._tier_price_anchor is not None:
            return float(self._tier_price_anchor)
        cost = float(self.manager.position.cost)
        if math.isfinite(cost) and cost > 0.0:
            return cost
        return None

    def _tier_jump_bps_for_open(self, tier_index: int) -> float:
        if tier_index <= 0 or tier_index >= len(self.cfg.tier_jump_bps_tuple):
            return 0.0
        return float(self.cfg.tier_jump_bps_tuple[tier_index])

    def _apply_tier_open_gate_steps(
        self,
        *,
        ask_price_ticks: Optional[int],
        ask_qty_steps: int,
        bid_price_ticks: Optional[int],
        bid_qty_steps: int,
        close_only: bool,
    ) -> tuple[Optional[int], int, Optional[int], int]:
        if close_only:
            return ask_price_ticks, ask_qty_steps, bid_price_ticks, bid_qty_steps

        pos_qty = float(self.manager.position.qty)
        if pos_qty > self.EPS:
            bid_price_ticks = self._tier_gate_open_price_ticks(
                price_ticks=bid_price_ticks,
                is_ask=False,
            )
        elif pos_qty < -self.EPS:
            ask_price_ticks = self._tier_gate_open_price_ticks(
                price_ticks=ask_price_ticks,
                is_ask=True,
            )
        return ask_price_ticks, ask_qty_steps, bid_price_ticks, bid_qty_steps

    def _tier_gate_open_price_ticks(
        self,
        *,
        price_ticks: Optional[int],
        is_ask: bool,
    ) -> Optional[int]:
        if price_ticks is None:
            return price_ticks
        tier_index = self._current_open_tier_index()
        jump_bps = self._tier_jump_bps_for_open(tier_index)
        if jump_bps <= self.EPS:
            return price_ticks
        price_anchor = self._tier_gate_price_anchor(tier_index)
        if price_anchor is None:
            return price_ticks

        gate_multiplier = 1.0 + jump_bps / 1e4 if is_ask else 1.0 - jump_bps / 1e4
        gate_price = price_anchor * gate_multiplier
        if not math.isfinite(gate_price) or gate_price <= 0.0:
            return price_ticks
        gate_ticks = (
            self._price_ceil_ticks(gate_price)
            if is_ask
            else self._price_floor_ticks(gate_price)
        )
        if gate_ticks <= 0:
            return price_ticks
        return (
            max(int(price_ticks), gate_ticks)
            if is_ask
            else min(int(price_ticks), gate_ticks)
        )

    def _apply_tier_open_gate(
        self,
        *,
        ask_levels: list[MakerLevel],
        bid_levels: list[MakerLevel],
        close_only: bool,
    ) -> tuple[list[MakerLevel], list[MakerLevel]]:
        if close_only:
            return ask_levels, bid_levels

        pos_qty = float(self.manager.position.qty)
        if pos_qty > self.EPS:
            bid_levels = self._tier_gate_open_levels(
                levels=bid_levels,
                is_ask=False,
            )
            return ask_levels, bid_levels
        if pos_qty < -self.EPS:
            ask_levels = self._tier_gate_open_levels(
                levels=ask_levels,
                is_ask=True,
            )
            return ask_levels, bid_levels
        return ask_levels, bid_levels

    def _tier_gate_open_levels(
        self,
        *,
        levels: Sequence[MakerLevel],
        is_ask: bool,
    ) -> list[MakerLevel]:
        if not levels:
            return list(levels)
        tier_index = self._current_open_tier_index()
        jump_bps = self._tier_jump_bps_for_open(tier_index)
        if jump_bps <= self.EPS:
            return list(levels)
        price_anchor = self._tier_gate_price_anchor(tier_index)
        if price_anchor is None:
            return list(levels)

        gate_multiplier = 1.0 + jump_bps / 1e4 if is_ask else 1.0 - jump_bps / 1e4
        gate_price = price_anchor * gate_multiplier
        if not math.isfinite(gate_price) or gate_price <= 0.0:
            return list(levels)
        gate_price = (
            self._price_ceil(gate_price)
            if is_ask
            else self._price_floor(gate_price)
        )
        adjusted: list[MakerLevel] = []
        for price, qty in levels:
            price = (
                max(float(price), gate_price)
                if is_ask
                else min(float(price), gate_price)
            )
            adjusted.append((price, float(qty)))
        return adjusted

    @staticmethod
    def _filled_qty_from_fills(fills: Sequence[MakerLevel]) -> float:
        return sum(float(qty) for _, qty in fills)

    def _place_reach_and_release_taker(self, mid: float) -> bool:
        del mid
        pos_qty = float(self.manager.position.qty)
        if abs(pos_qty) <= self.EPS:
            self._reach_and_release_active = False
            self._max_holding_start_ts = None
            self._max_holding_was_at_limit = False
            self._clear_taker_books()
            return False

        if self._latest_best_ask is None or self._latest_best_bid is None:
            return False

        close_qty = self._close_qty_from_position(pos_qty)
        if close_qty <= 0.0:
            self._reach_and_release_active = False
            self._max_holding_start_ts = None
            self._max_holding_was_at_limit = False
            self._clear_taker_books()
            return False

        self._pending_maker_quotes.clear()
        self._pending_simple_maker_quotes.clear()
        self._clear_maker_books()

        if pos_qty > 0.0:
            ask_qty = close_qty
            bid_qty = 0.0
            ask_price = float(self._latest_best_bid)
            bid_price = float(self._latest_best_ask)
        else:
            ask_qty = 0.0
            bid_qty = close_qty
            ask_price = float(self._latest_best_bid)
            bid_price = float(self._latest_best_ask)

        self._clear_taker_books()
        self.manager.place_taker(
            ask_price=ask_price,
            ask_qty=ask_qty,
            bid_price=bid_price,
            bid_qty=bid_qty,
            best_ask=float(self._latest_best_ask),
            best_bid=float(self._latest_best_bid),
            close_only=True,
        )
        return True

    def _activate_due_maker_quotes(self, current_timestamp: int) -> None:
        self._activate_pending_maker_quotes(current_timestamp=current_timestamp)
        self._activate_pending_simple_maker_quotes(current_timestamp=current_timestamp)

    def _enqueue_maker_levels(
        self,
        base_timestamp: int,
        ask_levels: Sequence[MakerLevel],
        bid_levels: Sequence[MakerLevel],
        best_ask: float,
        best_bid: float,
        close_only: bool = False,
    ) -> None:
        active_ts = int(base_timestamp) + int(self.sim.latency)
        self._pending_maker_quotes.append(
            (
                active_ts,
                [(float(price), float(qty)) for price, qty in ask_levels if qty > self.EPS],
                [(float(price), float(qty)) for price, qty in bid_levels if qty > self.EPS],
                float(best_ask),
                float(best_bid),
                bool(close_only),
            )
        )

    def _enqueue_simple_maker_quote(
        self,
        base_timestamp: int,
        ask_price_ticks: Optional[int],
        ask_qty_steps: int,
        bid_price_ticks: Optional[int],
        bid_qty_steps: int,
        best_ask_ticks: int,
        best_bid_ticks: int,
        close_only: bool = False,
    ) -> None:
        active_ts = int(base_timestamp) + int(self.sim.latency)
        ask_qty_steps = max(0, int(ask_qty_steps))
        bid_qty_steps = max(0, int(bid_qty_steps))
        self._pending_simple_maker_quotes.append(
            (
                active_ts,
                None if ask_qty_steps <= 0 else ask_price_ticks,
                ask_qty_steps,
                None if bid_qty_steps <= 0 else bid_price_ticks,
                bid_qty_steps,
                int(best_ask_ticks),
                int(best_bid_ticks),
                bool(close_only and (ask_qty_steps > 0 or bid_qty_steps > 0)),
            )
        )

    def _activate_pending_maker_quotes(self, current_timestamp: int) -> None:
        ts = int(current_timestamp)
        last_due_quote: Optional[PendingMakerQuote] = None
        while self._pending_maker_quotes and self._pending_maker_quotes[0][0] <= ts:
            last_due_quote = self._pending_maker_quotes.popleft()

        if last_due_quote is None:
            return

        (
            _active_ts,
            ask_levels,
            bid_levels,
            best_ask,
            best_bid,
            close_only,
        ) = last_due_quote
        self._clear_maker_books()
        if len(ask_levels) <= 1 and len(bid_levels) <= 1:
            self.manager.place_maker_single_levels(
                ask_levels=ask_levels,
                bid_levels=bid_levels,
                best_ask=best_ask,
                best_bid=best_bid,
                close_only=close_only,
            )
        else:
            self.manager.place_maker_levels(
                ask_levels=ask_levels,
                bid_levels=bid_levels,
                best_ask=best_ask,
                best_bid=best_bid,
                close_only=close_only,
            )

    def _activate_pending_simple_maker_quotes(self, current_timestamp: int) -> None:
        ts = int(current_timestamp)
        last_due_quote: Optional[SimplePendingMakerQuote] = None
        while (
            self._pending_simple_maker_quotes
            and self._pending_simple_maker_quotes[0][0] <= ts
        ):
            last_due_quote = self._pending_simple_maker_quotes.popleft()

        if last_due_quote is None:
            return

        (
            _active_ts,
            ask_price_ticks,
            ask_qty_steps,
            bid_price_ticks,
            bid_qty_steps,
            best_ask_ticks,
            best_bid_ticks,
            close_only,
        ) = last_due_quote
        self.manager.place_maker_steps(
            ask_price_ticks=ask_price_ticks,
            ask_qty_steps=ask_qty_steps,
            bid_price_ticks=bid_price_ticks,
            bid_qty_steps=bid_qty_steps,
            best_ask_ticks=best_ask_ticks,
            best_bid_ticks=best_bid_ticks,
            close_only=close_only,
        )

    def _simple_maker_steps_for_ticker(
        self,
        *,
        mid: float,
        open_allowed: bool,
        open_ask_price_ticks: int,
        open_bid_price_ticks: int,
        close_ask_price_ticks: int,
        close_bid_price_ticks: int,
    ) -> tuple[Optional[int], int, Optional[int], int, bool]:
        pos_qty = float(self.manager.position.qty)
        cost = float(self.manager.position.cost)
        if abs(pos_qty) <= self.EPS or cost <= 0.0 or math.isnan(cost):
            if not open_allowed:
                return None, 0, None, 0, False
            ask_steps = self._simple_open_steps(price_ticks=open_ask_price_ticks, mid=mid)
            bid_steps = self._simple_open_steps(price_ticks=open_bid_price_ticks, mid=mid)
            ask_steps = self._boost_open_steps(
                ask_steps,
                profitzone=False,
                include_zone_boost=False,
            )
            bid_steps = self._boost_open_steps(
                bid_steps,
                profitzone=False,
                include_zone_boost=False,
            )
            ask_steps, bid_steps = self._apply_inventory_limit_steps(
                ask_steps=ask_steps,
                bid_steps=bid_steps,
            )
            return (
                open_ask_price_ticks if ask_steps > 0 else None,
                ask_steps,
                open_bid_price_ticks if bid_steps > 0 else None,
                bid_steps,
                False,
            )

        if not self.cfg.strict_mode:
            if pos_qty > 0.0:
                profitzone = mid > cost + self.EPS
                ask_steps = min(
                    self._boost_close_steps(
                        self._simple_close_steps(
                            price_ticks=close_ask_price_ticks,
                            mid=mid,
                            pos_qty=pos_qty,
                        ),
                        profitzone=profitzone,
                    ),
                    self._close_steps_from_position(pos_qty),
                )
                bid_steps = self._boost_open_steps(
                    self._simple_open_steps(price_ticks=open_bid_price_ticks, mid=mid),
                    profitzone=profitzone,
                ) if open_allowed else 0
                _ask_open_steps, bid_steps = self._apply_inventory_limit_steps(
                    ask_steps=0,
                    bid_steps=bid_steps,
                )
                return (
                    close_ask_price_ticks if ask_steps > 0 else None,
                    ask_steps,
                    open_bid_price_ticks if bid_steps > 0 else None,
                    bid_steps,
                    False,
                )

            profitzone = mid < cost - self.EPS
            ask_steps = self._boost_open_steps(
                self._simple_open_steps(price_ticks=open_ask_price_ticks, mid=mid),
                profitzone=profitzone,
            ) if open_allowed else 0
            bid_steps = min(
                self._boost_close_steps(
                    self._simple_close_steps(
                        price_ticks=close_bid_price_ticks,
                        mid=mid,
                        pos_qty=pos_qty,
                    ),
                    profitzone=profitzone,
                ),
                self._close_steps_from_position(pos_qty),
            )
            ask_steps, _bid_open_steps = self._apply_inventory_limit_steps(
                ask_steps=ask_steps,
                bid_steps=0,
            )
            return (
                open_ask_price_ticks if ask_steps > 0 else None,
                ask_steps,
                close_bid_price_ticks if bid_steps > 0 else None,
                bid_steps,
                False,
            )

        if pos_qty > 0.0:
            if mid <= cost + self.EPS:
                bid_steps = self._boost_open_steps(
                    self._simple_open_steps(price_ticks=open_bid_price_ticks, mid=mid),
                    profitzone=False,
                ) if open_allowed else 0
                _ask_steps, bid_steps = self._apply_inventory_limit_steps(
                    ask_steps=0,
                    bid_steps=bid_steps,
                )
                return (
                    None,
                    0,
                    open_bid_price_ticks if bid_steps > 0 else None,
                    bid_steps,
                    False,
                )
            ask_steps = self._boost_close_steps(
                self._simple_close_steps(
                    price_ticks=close_ask_price_ticks,
                    mid=mid,
                    pos_qty=pos_qty,
                ),
                profitzone=True,
            )
            return (
                close_ask_price_ticks if ask_steps > 0 else None,
                ask_steps,
                None,
                0,
                ask_steps > 0,
            )

        if mid >= cost - self.EPS:
            ask_steps = self._boost_open_steps(
                self._simple_open_steps(price_ticks=open_ask_price_ticks, mid=mid),
                profitzone=False,
            ) if open_allowed else 0
            ask_steps, _bid_steps = self._apply_inventory_limit_steps(
                ask_steps=ask_steps,
                bid_steps=0,
            )
            return (
                open_ask_price_ticks if ask_steps > 0 else None,
                ask_steps,
                None,
                0,
                False,
            )
        bid_steps = self._boost_close_steps(
            self._simple_close_steps(
                price_ticks=close_bid_price_ticks,
                mid=mid,
                pos_qty=pos_qty,
            ),
            profitzone=True,
        )
        return (
            None,
            0,
            close_bid_price_ticks if bid_steps > 0 else None,
            bid_steps,
            bid_steps > 0,
        )

    def _quote_distance(
        self,
        *,
        intensity_base: float,
        vol_scalar: float,
        close: bool,
    ) -> float:
        pair_index = 1 if close else 0
        return (
            intensity_base * float(self.cfg.adj_spread_intensity[pair_index])
            + vol_scalar * float(self.cfg.adj_spread_volatility[pair_index])
        )

    def _clip_simple_quote_steps_to_bbo_distance(
        self,
        *,
        ask_price_ticks: Optional[int],
        ask_qty_steps: int,
        bid_price_ticks: Optional[int],
        bid_qty_steps: int,
        best_ask_ticks: int,
        best_bid_ticks: int,
    ) -> tuple[Optional[int], int, Optional[int], int]:
        ask_min_ticks = self._min_quote_distance_ticks(
            reference_price=self.manager.converter.from_ticks(int(best_ask_ticks))
        )
        bid_min_ticks = self._min_quote_distance_ticks(
            reference_price=self.manager.converter.from_ticks(int(best_bid_ticks))
        )
        if ask_min_ticks <= 0 and bid_min_ticks <= 0:
            return ask_price_ticks, ask_qty_steps, bid_price_ticks, bid_qty_steps

        ask_qty_steps = max(0, int(ask_qty_steps))
        bid_qty_steps = max(0, int(bid_qty_steps))
        if ask_price_ticks is None or ask_qty_steps <= 0:
            ask_price_ticks = None
            ask_qty_steps = 0
        elif ask_min_ticks > 0:
            ask_price_ticks = max(
                int(ask_price_ticks),
                int(best_ask_ticks) + int(ask_min_ticks),
            )

        if bid_price_ticks is None or bid_qty_steps <= 0:
            bid_price_ticks = None
            bid_qty_steps = 0
        elif bid_min_ticks > 0:
            bid_price_ticks = min(
                int(bid_price_ticks),
                int(best_bid_ticks) - int(bid_min_ticks),
            )
            if bid_price_ticks <= 0:
                bid_price_ticks = None
                bid_qty_steps = 0
        return ask_price_ticks, ask_qty_steps, bid_price_ticks, bid_qty_steps

    def _clip_quote_levels_to_bbo_distance(
        self,
        *,
        ask_levels: Sequence[MakerLevel],
        bid_levels: Sequence[MakerLevel],
        best_ask: float,
        best_bid: float,
    ) -> tuple[list[MakerLevel], list[MakerLevel]]:
        ask_min_ticks = self._min_quote_distance_ticks(reference_price=best_ask)
        bid_min_ticks = self._min_quote_distance_ticks(reference_price=best_bid)
        if ask_min_ticks <= 0 and bid_min_ticks <= 0:
            return list(ask_levels), list(bid_levels)

        ask_min_distance = float(ask_min_ticks) * self.sim.tick_size
        bid_min_distance = float(bid_min_ticks) * self.sim.tick_size
        ask_floor = self._price_ceil(float(best_ask) + ask_min_distance)
        bid_ceiling = self._price_floor(float(best_bid) - bid_min_distance)

        clipped_ask: list[MakerLevel] = []
        clipped_bid: list[MakerLevel] = []
        for price, qty in ask_levels:
            price = max(float(price), ask_floor) if ask_min_ticks > 0 else float(price)
            qty = float(qty)
            if price > self.EPS and qty > self.EPS:
                clipped_ask.append((price, qty))
        for price, qty in bid_levels:
            price = min(float(price), bid_ceiling) if bid_min_ticks > 0 else float(price)
            qty = float(qty)
            if price > self.EPS and qty > self.EPS:
                clipped_bid.append((price, qty))
        return clipped_ask, clipped_bid

    def _min_quote_distance_ticks(self, *, reference_price: float) -> int:
        min_bps = float(self.cfg.min_quote_distance_bps)
        if min_bps <= 0.0 or reference_price <= 0.0:
            return 0
        distance = float(reference_price) * min_bps / 1e4
        return max(0, math.floor((distance / self.sim.tick_size) + self.EPS))

    def _maker_levels_for_ticker(
        self,
        *,
        mid: float,
        open_allowed: bool,
        open_ask_price: float,
        ask_qty: float,
        open_bid_price: float,
        bid_qty: float,
        close_ask_price: float,
        close_bid_price: float,
        best_ask: float,
        best_bid: float,
    ) -> tuple[list[MakerLevel], list[MakerLevel], bool]:
        del best_ask, best_bid

        pos_qty = float(self.manager.position.qty)
        cost = float(self.manager.position.cost)
        if abs(pos_qty) <= self.EPS or cost <= 0.0 or math.isnan(cost):
            if not open_allowed:
                return [], [], False
            ask_qty = self._curve_open_qty(mid=mid, base_open_qty=ask_qty)
            bid_qty = self._curve_open_qty(mid=mid, base_open_qty=bid_qty)
            ask_qty = self._boost_open_qty(
                ask_qty,
                profitzone=False,
                include_zone_boost=False,
            )
            bid_qty = self._boost_open_qty(
                bid_qty,
                profitzone=False,
                include_zone_boost=False,
            )
            return self._single_quote_levels(open_ask_price, ask_qty, open_bid_price, bid_qty, False)

        if not self.cfg.strict_mode:
            if pos_qty > 0.0:
                profitzone = mid > cost + self.EPS
                close_qty = min(
                    self._boost_close_qty(
                        self._curve_close_qty(mid=mid, pos_qty=pos_qty),
                        profitzone=profitzone,
                    ),
                    self._close_qty_from_position(pos_qty),
                )
                bid_qty = self._boost_open_qty(
                    self._grid_open_qty(mid=mid, base_open_qty=bid_qty),
                    profitzone=profitzone,
                ) if open_allowed else 0.0
                _ask_open_qty, bid_qty = self._apply_inventory_limit(
                    mid=mid,
                    ask_qty=0.0,
                    bid_qty=bid_qty,
                )
                return self._single_quote_levels(
                    close_ask_price,
                    close_qty,
                    open_bid_price,
                    bid_qty,
                    False,
                )

            profitzone = mid < cost - self.EPS
            ask_qty = self._boost_open_qty(
                self._grid_open_qty(mid=mid, base_open_qty=ask_qty),
                profitzone=profitzone,
            ) if open_allowed else 0.0
            close_qty = min(
                self._boost_close_qty(
                    self._curve_close_qty(mid=mid, pos_qty=pos_qty),
                    profitzone=profitzone,
                ),
                self._close_qty_from_position(pos_qty),
            )
            ask_qty, _bid_open_qty = self._apply_inventory_limit(
                mid=mid,
                ask_qty=ask_qty,
                bid_qty=0.0,
            )
            return self._single_quote_levels(
                open_ask_price,
                ask_qty,
                close_bid_price,
                close_qty,
                False,
            )

        if pos_qty > 0.0:
            if mid <= cost + self.EPS:
                bid_qty = self._boost_open_qty(
                    self._grid_open_qty(mid=mid, base_open_qty=bid_qty),
                    profitzone=False,
                ) if open_allowed else 0.0
                return self._single_quote_levels(open_ask_price, 0.0, open_bid_price, bid_qty, False)
            close_qty = self._boost_close_qty(
                self._curve_close_qty(mid=mid, pos_qty=pos_qty),
                profitzone=True,
            )
            return self._single_quote_levels(close_ask_price, close_qty, close_bid_price, 0.0, True)

        if mid >= cost - self.EPS:
            ask_qty = self._boost_open_qty(
                self._grid_open_qty(mid=mid, base_open_qty=ask_qty),
                profitzone=False,
            ) if open_allowed else 0.0
            return self._single_quote_levels(open_ask_price, ask_qty, open_bid_price, 0.0, False)
        close_qty = self._boost_close_qty(
            self._curve_close_qty(mid=mid, pos_qty=pos_qty),
            profitzone=True,
        )
        return self._single_quote_levels(close_ask_price, 0.0, close_bid_price, close_qty, True)

    def _single_quote_levels(
        self,
        ask_price: float,
        ask_qty: float,
        bid_price: float,
        bid_qty: float,
        close_only: bool,
    ) -> tuple[list[MakerLevel], list[MakerLevel], bool]:
        ask_levels = [(float(ask_price), float(ask_qty))] if ask_qty > self.EPS else []
        bid_levels = [(float(bid_price), float(bid_qty))] if bid_qty > self.EPS else []
        return ask_levels, bid_levels, bool(close_only and (ask_levels or bid_levels))

    def _grid_open_qty(self, mid: float, base_open_qty: float) -> float:
        if base_open_qty <= 0.0:
            return 0.0
        return self._curve_open_qty(mid=mid, base_open_qty=base_open_qty)

    def _boost_pair(self, *, profitzone: bool) -> tuple[float, float]:
        return self.cfg.boost_profitzone if profitzone else self.cfg.boost_underwater

    def _tier_boost_for_open(self) -> float:
        boosts = self.cfg.boost_tier_tuple
        if not boosts:
            return 1.0
        tier_index = self._current_open_tier_index()
        tier_index = min(max(0, tier_index), len(boosts) - 1)
        return float(boosts[tier_index])

    def _tier_boost_for_close(self) -> float:
        boosts = self.cfg.boost_tier_tuple
        if not boosts:
            return 1.0
        tier_index = self._current_close_tier_index()
        tier_index = min(max(0, tier_index), len(boosts) - 1)
        return float(boosts[tier_index])

    def _boost_open_qty(
        self,
        qty: float,
        *,
        profitzone: bool,
        price: Optional[float] = None,
        include_zone_boost: bool = True,
    ) -> float:
        if qty <= 0.0:
            return 0.0
        multiplier = self._tier_boost_for_open()
        if include_zone_boost:
            multiplier *= float(self._boost_pair(profitzone=profitzone)[0])
        return self._qty_floor(qty * multiplier)

    def _boost_close_qty(self, qty: float, *, profitzone: bool) -> float:
        if qty <= 0.0:
            return 0.0
        multiplier = float(self._boost_pair(profitzone=profitzone)[1])
        multiplier *= self._tier_boost_for_close()
        return self._qty_floor(qty * multiplier)

    def _boost_open_steps(
        self,
        steps: int,
        *,
        profitzone: bool,
        price_ticks: Optional[int] = None,
        include_zone_boost: bool = True,
    ) -> int:
        if steps <= 0:
            return 0
        multiplier = self._tier_boost_for_open()
        if include_zone_boost:
            multiplier *= float(self._boost_pair(profitzone=profitzone)[0])
        return max(0, math.floor(float(steps) * multiplier + self.EPS))

    def _boost_close_steps(self, steps: int, *, profitzone: bool) -> int:
        if steps <= 0:
            return 0
        multiplier = float(self._boost_pair(profitzone=profitzone)[1])
        multiplier *= self._tier_boost_for_close()
        return max(0, math.floor(float(steps) * multiplier + self.EPS))

    def _curve_close_qty(self, mid: float, pos_qty: float) -> float:
        base_close_qty = self._close_qty_from_position(pos_qty)
        if base_close_qty <= 0.0:
            return 0.0

        min_bet = self._minimum_bet_qty(mid=mid)
        if min_bet <= self.EPS:
            return 0.0
        if base_close_qty + self.EPS < min_bet:
            return base_close_qty

        units = self._inventory_close_curve_units(mid=mid)
        if units <= 0:
            return 0.0

        return self._qty_floor(float(units) * min_bet)

    def _base_symmetric_qty(self, mid: float) -> tuple[float, float]:
        q = self._minimum_bet_qty(mid=mid)
        return q, q

    def _apply_inventory_limit(self, mid: float, ask_qty: float, bid_qty: float) -> tuple[float, float]:
        if ask_qty <= 0.0 and bid_qty <= 0.0:
            return 0.0, 0.0

        max_open_cost_notional_usdt = self._max_open_cost_notional_usdt()
        if max_open_cost_notional_usdt <= self.EPS:
            return 0.0, 0.0

        cost_notional_usdt = float(self.manager.position.cost_notional_usdt)
        if cost_notional_usdt + self.EPS >= max_open_cost_notional_usdt:
            bid_qty = 0.0
        if cost_notional_usdt - self.EPS <= -max_open_cost_notional_usdt:
            ask_qty = 0.0
        return ask_qty, bid_qty

    def _simple_open_steps(self, price_ticks: int, mid: float) -> int:
        if price_ticks <= 0:
            return 0
        units = self._inventory_open_curve_units(mid=mid)
        if units <= 0:
            return 0
        return int(units) * self._minimum_bet_steps(price_ticks=price_ticks)

    def _simple_close_steps(self, price_ticks: int, mid: float, pos_qty: float) -> int:
        if price_ticks <= 0:
            return 0
        base_close_steps = self._close_steps_from_position(pos_qty)
        if base_close_steps <= 0:
            return 0

        min_bet_steps = self._minimum_bet_steps(price_ticks=price_ticks)
        if min_bet_steps <= 0:
            return 0
        if base_close_steps < min_bet_steps:
            return base_close_steps

        units = self._inventory_close_curve_units(mid=mid)
        if units <= 0:
            return 0
        return int(units) * min_bet_steps

    def _apply_inventory_limit_steps(
        self,
        ask_steps: int,
        bid_steps: int,
    ) -> tuple[int, int]:
        ask_steps = max(0, int(ask_steps))
        bid_steps = max(0, int(bid_steps))
        if ask_steps <= 0 and bid_steps <= 0:
            return 0, 0

        max_open_cost_notional_usdt = self._max_open_cost_notional_usdt()
        if max_open_cost_notional_usdt <= self.EPS:
            return 0, 0

        cost_notional_usdt = float(self.manager.position.cost_notional_usdt)
        if cost_notional_usdt + self.EPS >= max_open_cost_notional_usdt:
            bid_steps = 0
        if cost_notional_usdt - self.EPS <= -max_open_cost_notional_usdt:
            ask_steps = 0
        return ask_steps, bid_steps

    def _max_open_cost_notional_usdt(self) -> float:
        max_position_usdt = float(self.cfg.max_position_usdt)
        if max_position_usdt <= self.EPS:
            return 0.0
        return max_position_usdt * float(self.cfg.max_open_inventory_utilization)

    def _inventory_skew_price_shift(self, mid: float) -> float:
        skew_cfg = self.cfg.inventory_skew
        if not skew_cfg:
            return 0.0

        max_skew_ticks = float(skew_cfg[0])
        skew_power = float(skew_cfg[1])
        if max_skew_ticks <= self.EPS or mid <= self.EPS:
            return 0.0

        max_position_usdt = float(self.cfg.max_position_usdt)
        if max_position_usdt <= self.EPS:
            return 0.0

        cost_notional_usdt = float(self.manager.position.cost_notional_usdt)
        inventory_util = max(-1.0, min(1.0, cost_notional_usdt / max_position_usdt))
        if abs(inventory_util) <= self.EPS:
            return 0.0

        skew_ticks_abs = math.floor(((abs(inventory_util) ** skew_power) * max_skew_ticks) + self.EPS)
        if skew_ticks_abs <= 0:
            return 0.0

        signed_tick_shift = -skew_ticks_abs if inventory_util > 0.0 else skew_ticks_abs
        return float(signed_tick_shift) * self.sim.tick_size

    def _curve_open_qty(self, mid: float, base_open_qty: float) -> float:
        if base_open_qty <= 0.0:
            return 0.0
        units = self._inventory_open_curve_units(mid=mid)
        if units <= 0:
            return 0.0
        min_bet = self._minimum_bet_qty(mid=mid)
        if min_bet <= self.EPS:
            return 0.0
        return self._qty_floor(float(units) * min_bet)

    def _inventory_open_curve_units(self, mid: float) -> int:
        scale = self.cfg.open_curve
        if not scale or mid <= self.EPS:
            return 1

        min_scale = float(scale[0])
        max_scale = float(scale[1])
        order = float(scale[2])
        if order <= self.EPS:
            return 1

        max_position_usdt = float(self.cfg.max_position_usdt)
        if max_position_usdt <= self.EPS:
            return 1

        gross_cost_notional_usdt = float(self.manager.position.gross_cost_notional_usdt)
        util = min(1.0, max(0.0, gross_cost_notional_usdt / max_position_usdt))
        decay = (1.0 - util) ** order
        return max(0, math.floor(min_scale + (max_scale - min_scale) * decay + self.EPS))

    def _inventory_close_curve_units(self, mid: float) -> int:
        scale = self.cfg.close_curve
        if not scale or mid <= self.EPS:
            return 1

        min_scale = float(scale[0])
        max_scale = float(scale[1])
        order = float(scale[2])
        if order <= self.EPS:
            return 1

        max_position_usdt = float(self.cfg.max_position_usdt)
        if max_position_usdt <= self.EPS:
            return 1

        gross_cost_notional_usdt = float(self.manager.position.gross_cost_notional_usdt)
        util = min(1.0, max(0.0, gross_cost_notional_usdt / max_position_usdt))
        growth = util ** order
        return max(0, math.floor(min_scale + (max_scale - min_scale) * growth + self.EPS))

    def _minimum_bet_qty(self, mid: float) -> float:
        return get_minimum_size(
            mid=mid,
            qty_precision=self.sim.qty_precision,
            min_order_notional=self.cfg.min_order_notional,
            min_order_qty=self.cfg.min_order_qty,
        )

    def _minimum_bet_steps(self, price_ticks: int) -> int:
        price = self.manager.converter.from_ticks(int(price_ticks))
        if price <= self.EPS:
            return 0

        min_steps = max(1, int(self.manager.validator.min_qty_steps))
        min_notional = float(self.cfg.min_order_notional)
        if min_notional > 0.0:
            notional_steps = math.ceil(
                (min_notional / (price * self.sim.step_size)) - self.EPS
            )
            min_steps = max(min_steps, notional_steps)
        return min_steps

    def _price_floor(self, price: float) -> float:
        if price <= 0.0:
            return 0.0
        ticks = math.floor((price / self.sim.tick_size) + self.EPS)
        return self._round_to_precision(ticks * self.sim.tick_size, self.sim.price_precision)

    def _price_floor_ticks(self, price: float) -> int:
        if price <= 0.0:
            return 0
        return math.floor((price / self.sim.tick_size) + self.EPS)

    def _price_ceil(self, price: float) -> float:
        if price <= 0.0:
            return 0.0
        ticks = math.ceil((price / self.sim.tick_size) - self.EPS)
        return self._round_to_precision(ticks * self.sim.tick_size, self.sim.price_precision)

    def _price_ceil_ticks(self, price: float) -> int:
        if price <= 0.0:
            return 0
        return math.ceil((price / self.sim.tick_size) - self.EPS)

    def _price_round_ticks(self, price: float) -> int:
        if price <= 0.0:
            return 0
        return int(round(price / self.sim.tick_size))

    def _qty_floor(self, qty: float) -> float:
        if qty <= 0.0:
            return 0.0
        steps = math.floor((qty / self.sim.step_size) + self.EPS)
        return self._round_to_precision(steps * self.sim.step_size, self.sim.qty_precision)

    def _close_qty_from_position(self, pos_qty: float) -> float:
        if abs(pos_qty) <= self.EPS:
            return 0.0
        steps = self._close_steps_from_position(pos_qty)
        if steps <= 0:
            return 0.0
        return self._round_to_precision(
            self.manager.converter.from_steps(steps),
            self.sim.qty_precision,
        )

    def _close_steps_from_position(self, pos_qty: float) -> int:
        if abs(pos_qty) <= self.EPS:
            return 0
        position_steps = abs(int(getattr(self.manager, "position_steps", 0)))
        if position_steps > 0:
            return position_steps
        return abs(self.manager.converter.to_steps(pos_qty, strict=False, rounding="round"))

    def _clear_maker_books(self) -> None:
        self.manager.books.ask_maker.clear()
        self.manager.books.bid_maker.clear()

    def _clear_taker_books(self) -> None:
        self.manager.books.ask_taker.clear()
        self.manager.books.bid_taker.clear()

    @staticmethod
    def _round_to_precision(value: float, precision: int) -> float:
        return float(f"{value:.{precision}f}")

    @staticmethod
    def _safe_non_negative(value: Optional[float]) -> float:
        try:
            v = 0.0 if value is None else float(value)
        except (TypeError, ValueError):
            return 0.0
        if (not math.isfinite(v)) or v < 0.0:
            return 0.0
        return v

    @staticmethod
    def _safe_finite(value: Optional[float]) -> float:
        try:
            v = 0.0 if value is None else float(value)
        except (TypeError, ValueError):
            return 0.0
        return v if math.isfinite(v) else 0.0

    def _selected_intensity_spec(self) -> dict[str, int | str] | None:
        if all(float(value) <= self.EPS for value in self.sim.adj_spread_intensity):
            return None
        return {
            "name": str(self.sim.name_intensity).strip().lower(),
            "lookback": int(self.sim.lookback_intensity),
        }

    def _selected_instructor_spec(self) -> dict[str, int | str] | None:
        if abs(float(self.sim.adj_spread_instructor)) <= self.EPS:
            return None
        if self.sim.name_instructor is None:
            return None
        if self.sim.lookback_instructor is None:
            raise ValueError("lookback_instructor must be provided when name_instructor is set")
        return {
            "name": str(self.sim.name_instructor).strip().lower(),
            "lookback": int(self.sim.lookback_instructor),
        }

    def _selected_volatility_specs(self) -> list[dict[str, int | str]]:
        if all(float(value) <= self.EPS for value in self.sim.adj_spread_volatility):
            return []
        if self.sim.name_volatility is None:
            return []
        if self.sim.lookback_volatility is None:
            raise ValueError("lookback_volatility must be provided when name_volatility is set")
        return [
            {
                "name": str(self.sim.name_volatility).strip().lower(),
                "lookback": int(self.sim.lookback_volatility),
            }
        ]
