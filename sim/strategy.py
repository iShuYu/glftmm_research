from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
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
    max_position_usdt: float = 0.0
    max_open_inventory_utilization: float = 1.0
    max_holding_time: int = 0
    adj_spread_intensity: float | tuple[float, float] = 1.0
    adj_spread_instructor: float = 0.0
    passive_only: bool = False
    optimize_by_orderbook: float = -1.0
    adj_spread_volatility: float | tuple[float, float] = 0.0
    min_quote_distance_bps: float = 0.0
    inventory_skew: Optional[tuple[float, float]] = None
    min_order_qty: float = 0.0
    min_order_notional: float = 0.0
    stoploss: float = 0.0
    open_curve: Optional[tuple[float, float, float]] = None
    close_curve: Optional[tuple[float, float, float]] = None
    boost_underwater: float | tuple[float, float] = (1.0, 1.0)
    boost_profitzone: float | tuple[float, float] = (1.0, 1.0)
    cooldown_time: int = 0
    strict_mode: bool = True
    simple_mode: bool = True

    def __post_init__(self) -> None:
        max_position_usdt = self._normalize_scalar(
            self.max_position_usdt,
            "max_position_usdt",
        )
        stoploss = self._normalize_scalar(self.stoploss, "stoploss")
        max_open_inventory_utilization = self._normalize_scalar(
            self.max_open_inventory_utilization,
            "max_open_inventory_utilization",
        )
        object.__setattr__(self, "max_position_usdt", max_position_usdt)
        object.__setattr__(self, "stoploss", stoploss)
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
            raise ValueError("max_position_usdt must be >= 0")
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
        try:
            optimize_by_orderbook = float(self.optimize_by_orderbook)
        except (TypeError, ValueError):
            raise ValueError("optimize_by_orderbook must be -1, 0, or a notional threshold") from None
        if (
            not math.isfinite(optimize_by_orderbook)
            or (optimize_by_orderbook < 0.0 and abs(optimize_by_orderbook + 1.0) > 1e-12)
        ):
            raise ValueError("optimize_by_orderbook must be -1, 0, or a notional threshold")
        object.__setattr__(self, "optimize_by_orderbook", optimize_by_orderbook)
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
        cooldown_time = int(self.cooldown_time)
        if cooldown_time < 0:
            raise ValueError("cooldown_time must be >= 0")
        object.__setattr__(
            self,
            "cooldown_time",
            cooldown_time,
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
        self._open_cooldown_until: Optional[int] = None
        self._open_liquidity_snap_adjusted: int = 0
        self._open_liquidity_snap_cancelled: int = 0
        self._open_liquidity_snap_moved_ticks: int = 0
        self._open_liquidity_snap_max_move_ticks: int = 0

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
            "stoploss_usdt": float(self.cfg.stoploss),
            "traded_volume": float(self._traded_volume),
            "latest_best_ask": self._latest_best_ask,
            "latest_best_bid": self._latest_best_bid,
            "reach_and_release_active": bool(self._reach_and_release_active),
            "max_holding_start_ts": self._max_holding_start_ts,
            "max_holding_was_at_limit": bool(self._max_holding_was_at_limit),
            "open_cooldown_until": self._open_cooldown_until,
            "open_liquidity_snap_adjusted": int(self._open_liquidity_snap_adjusted),
            "open_liquidity_snap_cancelled": int(self._open_liquidity_snap_cancelled),
            "open_liquidity_snap_moved_ticks": int(self._open_liquidity_snap_moved_ticks),
            "open_liquidity_snap_max_move_ticks": int(self._open_liquidity_snap_max_move_ticks),
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
        cooldown_until_raw = state.get("open_cooldown_until")
        self._open_cooldown_until = (
            None if cooldown_until_raw is None else int(cooldown_until_raw)
        )
        self._open_liquidity_snap_adjusted = int(
            state.get("open_liquidity_snap_adjusted", 0)
        )
        self._open_liquidity_snap_cancelled = int(
            state.get("open_liquidity_snap_cancelled", 0)
        )
        self._open_liquidity_snap_moved_ticks = int(
            state.get("open_liquidity_snap_moved_ticks", 0)
        )
        self._open_liquidity_snap_max_move_ticks = int(
            state.get("open_liquidity_snap_max_move_ticks", 0)
        )
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
        self._reset_open_liquidity_snap_stats()
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
                self._on_ticker_event(
                    timestamp=ts,
                    best_bid=float(event[2]),
                    best_ask=float(event[3]),
                    instructor_value=event[4],
                    intensity_value=event[5],
                    volatility_scalar=float(event[6]),
                    replay_bid_ticks=event[7] if len(event) > 7 else None,
                    replay_ask_ticks=event[8] if len(event) > 8 else None,
                    replay_bid_notional=event[9] if len(event) > 9 else None,
                    replay_ask_notional=event[10] if len(event) > 10 else None,
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
        new_abs_pos_qty = abs(float(self.manager.position.qty))
        maker_position_opened = new_abs_pos_qty > prev_abs_pos_qty + self.EPS
        if maker_position_opened:
            self._record_open_maker_fill(trade_time=int(trade_time))
        position_reduced = new_abs_pos_qty + self.EPS < prev_abs_pos_qty
        self._update_max_holding_tracking(
            timestamp=int(trade_time),
            position_reduced=position_reduced,
        )
        return float(filled_qty)

    def _open_cooldown_active(self, timestamp: int) -> bool:
        return (
            self._open_cooldown_until is not None
            and int(timestamp) < int(self._open_cooldown_until)
        )

    def _record_open_maker_fill(self, *, trade_time: int) -> None:
        cooldown_time = int(self.cfg.cooldown_time)
        if cooldown_time <= 0:
            return
        self._open_cooldown_until = int(trade_time) + cooldown_time
        self._clear_open_maker_books()
        self._pending_maker_quotes.clear()
        self._pending_simple_maker_quotes.clear()

    def _clear_open_maker_books(self) -> None:
        pos_qty = float(self.manager.position.qty)
        if pos_qty > self.EPS:
            self.manager.books.bid_maker.clear()
        elif pos_qty < -self.EPS:
            self.manager.books.ask_maker.clear()
        else:
            self.manager.books.ask_maker.clear()
            self.manager.books.bid_maker.clear()

    def _on_ticker_event(
        self,
        timestamp: int,
        best_bid: float,
        best_ask: float,
        intensity_value: Optional[float],
        volatility_scalar: float,
        instructor_value: Optional[float] = None,
        replay_bid_ticks: object = None,
        replay_ask_ticks: object = None,
        replay_bid_notional: object = None,
        replay_ask_notional: object = None,
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

        intensity_base = self._safe_non_negative(intensity_value)
        instructor_scalar = self._safe_finite(instructor_value)
        vol_scalar = self._safe_non_negative(volatility_scalar)
        open_distance = self._quote_distance(
            intensity_base=intensity_base,
            vol_scalar=vol_scalar,
            close=False,
        )
        close_distance = self._quote_distance(
            intensity_base=intensity_base,
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

        raw_open_ask_price = rounded_best_ask + open_distance + ask_instructor_shift + skew_shift
        raw_open_bid_price = rounded_best_bid - open_distance + bid_instructor_shift + skew_shift
        raw_close_ask_price = rounded_best_ask + close_distance + ask_instructor_shift + skew_shift
        raw_close_bid_price = rounded_best_bid - close_distance + bid_instructor_shift + skew_shift
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
        self._update_max_holding_tracking(timestamp=timestamp)

        if self._should_activate_stoploss(mid=mid):
            self._reach_and_release_active = True
        if self._should_activate_max_holding_timeout(timestamp=timestamp):
            self._reach_and_release_active = True

        if self._reach_and_release_active and self._place_reach_and_release_taker(mid=mid):
            self._append_record(timestamp=timestamp, mid=mid)
            return

        self._clear_taker_books()
        open_allowed = not self._open_cooldown_active(timestamp)
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
            ) = self._apply_open_liquidity_snap_steps(
                ask_price_ticks=quote_ask_price_ticks,
                ask_qty_steps=ask_qty_steps,
                bid_price_ticks=quote_bid_price_ticks,
                bid_qty_steps=bid_qty_steps,
                close_only=quote_close_only,
                replay_ask_ticks=replay_ask_ticks,
                replay_bid_ticks=replay_bid_ticks,
                replay_ask_notional=replay_ask_notional,
                replay_bid_notional=replay_bid_notional,
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
            ask_levels, bid_levels = self._apply_open_liquidity_snap(
                ask_levels=ask_levels,
                bid_levels=bid_levels,
                close_only=quote_close_only,
                replay_ask_ticks=replay_ask_ticks,
                replay_bid_ticks=replay_bid_ticks,
                replay_ask_notional=replay_ask_notional,
                replay_bid_notional=replay_bid_notional,
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

    def _reset_open_liquidity_snap_stats(self) -> None:
        self._open_liquidity_snap_adjusted = 0
        self._open_liquidity_snap_cancelled = 0
        self._open_liquidity_snap_moved_ticks = 0
        self._open_liquidity_snap_max_move_ticks = 0

    def _apply_open_liquidity_snap_steps(
        self,
        *,
        ask_price_ticks: Optional[int],
        ask_qty_steps: int,
        bid_price_ticks: Optional[int],
        bid_qty_steps: int,
        close_only: bool,
        replay_ask_ticks: object,
        replay_bid_ticks: object,
        replay_ask_notional: object,
        replay_bid_notional: object,
    ) -> tuple[Optional[int], int, Optional[int], int]:
        notional_threshold = float(self.cfg.optimize_by_orderbook)
        if notional_threshold < 0.0 or close_only:
            return ask_price_ticks, ask_qty_steps, bid_price_ticks, bid_qty_steps

        pos_qty = float(self.manager.position.qty)
        if pos_qty <= self.EPS:
            ask_price_ticks, ask_qty_steps = self._snap_open_side_steps(
                price_ticks=ask_price_ticks,
                qty_steps=ask_qty_steps,
                is_ask=True,
                replay_ticks=replay_ask_ticks,
                replay_notional=replay_ask_notional,
                notional_threshold=notional_threshold,
            )
        if pos_qty >= -self.EPS:
            bid_price_ticks, bid_qty_steps = self._snap_open_side_steps(
                price_ticks=bid_price_ticks,
                qty_steps=bid_qty_steps,
                is_ask=False,
                replay_ticks=replay_bid_ticks,
                replay_notional=replay_bid_notional,
                notional_threshold=notional_threshold,
            )
        return ask_price_ticks, ask_qty_steps, bid_price_ticks, bid_qty_steps

    def _snap_open_side_steps(
        self,
        *,
        price_ticks: Optional[int],
        qty_steps: int,
        is_ask: bool,
        replay_ticks: object,
        replay_notional: object,
        notional_threshold: float,
    ) -> tuple[Optional[int], int]:
        if price_ticks is None or qty_steps <= 0:
            return None, 0
        if replay_ticks is None:
            return price_ticks, qty_steps

        target_tick = int(price_ticks)
        snapped_tick = self._snap_open_price_tick(
            target_tick=target_tick,
            is_ask=is_ask,
            replay_ticks=replay_ticks,
            replay_notional=replay_notional,
            notional_threshold=notional_threshold,
        )
        if snapped_tick is None:
            self._open_liquidity_snap_cancelled += 1
            return None, 0
        if snapped_tick != target_tick:
            self._record_open_liquidity_snap_move(target_tick=target_tick, snapped_tick=snapped_tick)
        return int(snapped_tick), int(qty_steps)

    def _apply_open_liquidity_snap(
        self,
        *,
        ask_levels: Sequence[MakerLevel],
        bid_levels: Sequence[MakerLevel],
        close_only: bool,
        replay_ask_ticks: object,
        replay_bid_ticks: object,
        replay_ask_notional: object,
        replay_bid_notional: object,
    ) -> tuple[list[MakerLevel], list[MakerLevel]]:
        notional_threshold = float(self.cfg.optimize_by_orderbook)
        if notional_threshold < 0.0 or close_only:
            return list(ask_levels), list(bid_levels)

        pos_qty = float(self.manager.position.qty)
        adjusted_ask = (
            self._snap_open_side_levels(
                levels=ask_levels,
                is_ask=True,
                replay_ticks=replay_ask_ticks,
                replay_notional=replay_ask_notional,
                notional_threshold=notional_threshold,
            )
            if pos_qty <= self.EPS
            else list(ask_levels)
        )
        adjusted_bid = (
            self._snap_open_side_levels(
                levels=bid_levels,
                is_ask=False,
                replay_ticks=replay_bid_ticks,
                replay_notional=replay_bid_notional,
                notional_threshold=notional_threshold,
            )
            if pos_qty >= -self.EPS
            else list(bid_levels)
        )
        return adjusted_ask, adjusted_bid

    def _snap_open_side_levels(
        self,
        *,
        levels: Sequence[MakerLevel],
        is_ask: bool,
        replay_ticks: object,
        replay_notional: object,
        notional_threshold: float,
    ) -> list[MakerLevel]:
        if not levels:
            return []
        if replay_ticks is None:
            return list(levels)

        adjusted: list[MakerLevel] = []
        for price, qty in levels:
            price = float(price)
            qty = float(qty)
            if qty <= self.EPS:
                continue
            target_tick = self.manager.converter.to_ticks(price)
            snapped_tick = self._snap_open_price_tick(
                target_tick=target_tick,
                is_ask=is_ask,
                replay_ticks=replay_ticks,
                replay_notional=replay_notional,
                notional_threshold=notional_threshold,
            )
            if snapped_tick is None:
                self._open_liquidity_snap_cancelled += 1
                continue
            if snapped_tick != target_tick:
                self._record_open_liquidity_snap_move(
                    target_tick=target_tick,
                    snapped_tick=snapped_tick,
                )
                price = self._round_to_precision(
                    self.manager.converter.from_ticks(int(snapped_tick)),
                    self.sim.price_precision,
                )
            adjusted.append((price, qty))
        return adjusted

    def _snap_open_price_tick(
        self,
        *,
        target_tick: int,
        is_ask: bool,
        replay_ticks: object,
        replay_notional: object,
        notional_threshold: float,
    ) -> int | None:
        ticks = np.asarray(replay_ticks)
        if ticks.ndim != 1:
            ticks = ticks.reshape(-1)
        valid = ticks > 0
        if notional_threshold > 0.0:
            if replay_notional is None:
                return None
            notionals = np.asarray(replay_notional)
            if notionals.ndim != 1:
                notionals = notionals.reshape(-1)
            if notionals.shape[0] != ticks.shape[0]:
                return None
            valid = valid & (notionals > notional_threshold)
        ticks = ticks[valid]
        if ticks.size == 0:
            return None
        if bool(np.any(ticks == int(target_tick))):
            return int(target_tick)

        if is_ask:
            candidates = ticks[ticks > int(target_tick)]
            if candidates.size == 0:
                return None
            snapped = int(candidates[0]) - 1
            return snapped if snapped > 0 else None

        candidates = ticks[ticks < int(target_tick)]
        if candidates.size == 0:
            return None
        snapped = int(candidates[0]) + 1
        return snapped if snapped > 0 else None

    def _record_open_liquidity_snap_move(self, *, target_tick: int, snapped_tick: int) -> None:
        moved_ticks = abs(int(snapped_tick) - int(target_tick))
        self._open_liquidity_snap_adjusted += 1
        self._open_liquidity_snap_moved_ticks += moved_ticks
        self._open_liquidity_snap_max_move_ticks = max(
            self._open_liquidity_snap_max_move_ticks,
            moved_ticks,
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

    def _boost_open_qty(self, qty: float, *, profitzone: bool) -> float:
        if qty <= 0.0:
            return 0.0
        return self._qty_floor(qty * float(self._boost_pair(profitzone=profitzone)[0]))

    def _boost_close_qty(self, qty: float, *, profitzone: bool) -> float:
        if qty <= 0.0:
            return 0.0
        return self._qty_floor(qty * float(self._boost_pair(profitzone=profitzone)[1]))

    def _boost_open_steps(self, steps: int, *, profitzone: bool) -> int:
        if steps <= 0:
            return 0
        return max(0, math.floor(float(steps) * float(self._boost_pair(profitzone=profitzone)[0]) + self.EPS))

    def _boost_close_steps(self, steps: int, *, profitzone: bool) -> int:
        if steps <= 0:
            return 0
        return max(0, math.floor(float(steps) * float(self._boost_pair(profitzone=profitzone)[1]) + self.EPS))

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
