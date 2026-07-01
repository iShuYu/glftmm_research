from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Iterable, Literal, Optional, Sequence

import pandas as pd

from core.manager import Manager as ValidatedManager
from core.position import Position
from sim.loader import BinanceEventLoader, DateLike


MakerLevel = tuple[float, float]
QuoteSide = Literal["ask", "bid"]
PendingMakerQuote = tuple[
    int,
    QuoteSide | None,
    list[MakerLevel],
    list[MakerLevel],
    float,
    float,
    bool,
]
SimplePendingMakerQuote = tuple[
    int,
    QuoteSide | None,
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
    latency: int
    price_precision: int
    qty_precision: int
    mode: int
    taker_fee: float
    maker_fee: float
    max_position_usdt: float = 0.0
    max_open_inventory_utilization: float = 1.0
    max_holding_time: int = 0
    adj_spread_intensity: float | tuple[float, float] = 1.0
    ewma_intensity: float = 1.0
    consequtive_sameside: int = 0
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
    daily_parallel: bool = False

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
        if self.latency < 0:
            raise ValueError("latency must be >= 0")
        if self.price_precision < 0:
            raise ValueError("price_precision must be >= 0")
        if self.qty_precision < 0:
            raise ValueError("qty_precision must be >= 0")
        if self.mode not in (0, 1):
            raise ValueError("mode must be 0 or 1")
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
        ewma_intensity = self._normalize_scalar(self.ewma_intensity, "ewma_intensity")
        if ewma_intensity <= 0.0 or ewma_intensity > 1.0:
            raise ValueError("ewma_intensity must be > 0 and <= 1")
        object.__setattr__(self, "ewma_intensity", ewma_intensity)
        object.__setattr__(
            self,
            "consequtive_sameside",
            self._normalize_non_negative_int(
                self.consequtive_sameside,
                "consequtive_sameside",
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
        if not isinstance(self.daily_parallel, bool):
            raise ValueError("daily_parallel must be boolean")
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

    @staticmethod
    def _normalize_non_negative_int(value: object, name: str) -> int:
        if isinstance(value, bool) or isinstance(value, (list, tuple)):
            raise ValueError(f"{name} must be a non-negative integer")
        value_float = float(value)
        if (
            not math.isfinite(value_float)
            or value_float < 0.0
            or not value_float.is_integer()
        ):
            raise ValueError(f"{name} must be a non-negative integer")
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
        self._latest_sell_intensity: float = 0.0
        self._latest_buy_intensity: float = 0.0
        self._traded_volume: float = 0.0
        self._records: list[tuple[int, float, float, float, float, float, float, float, float, float]] = []
        self._reach_and_release_active: bool = False
        self._max_holding_start_ts: Optional[int] = None
        self._max_holding_was_at_limit: bool = False
        self._pending_maker_quotes: deque[PendingMakerQuote] = deque()
        self._pending_simple_maker_quotes: deque[SimplePendingMakerQuote] = deque()
        self._open_cooldown_until: Optional[int] = None
        self._last_simple_quote_key: Optional[tuple[object, ...]] = None
        self._last_level_quote_key: Optional[tuple[object, ...]] = None
        self._last_simple_quote_side_keys: dict[QuoteSide, Optional[tuple[object, ...]]] = {
            "ask": None,
            "bid": None,
        }
        self._last_level_quote_side_keys: dict[QuoteSide, Optional[tuple[object, ...]]] = {
            "ask": None,
            "bid": None,
        }
        self._last_nonzero_agg_side: Optional[QuoteSide] = None
        self._same_side_nonzero_count: int = 0
        self._blocked_quote_sides: dict[QuoteSide, bool] = {"ask": False, "bid": False}
        self._daily_stop_trading: bool = False
        self._daily_stop_date_key: Optional[str] = None
        self._current_date_key: Optional[str] = None

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
            "latest_sell_intensity": self._latest_sell_intensity,
            "latest_buy_intensity": self._latest_buy_intensity,
            "reach_and_release_active": bool(self._reach_and_release_active),
            "max_holding_start_ts": self._max_holding_start_ts,
            "max_holding_was_at_limit": bool(self._max_holding_was_at_limit),
            "open_cooldown_until": self._open_cooldown_until,
            "last_nonzero_agg_side": self._last_nonzero_agg_side,
            "same_side_nonzero_count": int(self._same_side_nonzero_count),
            "blocked_quote_sides": {
                side: bool(blocked)
                for side, blocked in self._blocked_quote_sides.items()
            },
            "daily_parallel": bool(self.cfg.daily_parallel),
            "daily_stop_trading": bool(self._daily_stop_trading),
            "daily_stop_date": self._daily_stop_date_key,
            "pending_maker_quotes": [
                {
                    "active_ts": int(active_ts),
                    "side": quote_side,
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
                    quote_side,
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
                    "side": quote_side,
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
                    quote_side,
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
        self._latest_sell_intensity = float(state.get("latest_sell_intensity", 0.0))
        self._latest_buy_intensity = float(state.get("latest_buy_intensity", 0.0))
        self._reach_and_release_active = bool(state.get("reach_and_release_active", False))
        start_ts_raw = state.get("max_holding_start_ts")
        self._max_holding_start_ts = None if start_ts_raw is None else int(start_ts_raw)
        self._max_holding_was_at_limit = bool(state.get("max_holding_was_at_limit", False))
        cooldown_until_raw = state.get("open_cooldown_until")
        self._open_cooldown_until = (
            None if cooldown_until_raw is None else int(cooldown_until_raw)
        )
        self._last_simple_quote_key = None
        self._last_level_quote_key = None
        self._last_simple_quote_side_keys = {"ask": None, "bid": None}
        self._last_level_quote_side_keys = {"ask": None, "bid": None}
        side_raw = state.get("last_nonzero_agg_side")
        self._last_nonzero_agg_side = side_raw if side_raw in ("ask", "bid") else None
        self._same_side_nonzero_count = max(
            0,
            int(state.get("same_side_nonzero_count", 0)),
        )
        blocked_raw = state.get("blocked_quote_sides")
        if not isinstance(blocked_raw, dict):
            blocked_raw = state.get("blocked_open_sides")
        if isinstance(blocked_raw, dict):
            self._blocked_quote_sides = {
                "ask": bool(blocked_raw.get("ask", False)),
                "bid": bool(blocked_raw.get("bid", False)),
            }
        else:
            self._blocked_quote_sides = {"ask": False, "bid": False}
        self._daily_stop_trading = bool(state.get("daily_stop_trading", False))
        stop_date_raw = state.get("daily_stop_date")
        self._daily_stop_date_key = None if stop_date_raw is None else str(stop_date_raw)
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
                            self._parse_quote_side(row.get("side")),
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
                            self._parse_quote_side(row.get("side")),
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
    def _parse_quote_side(side: object) -> QuoteSide | None:
        if side is None:
            return None
        normalized = str(side).strip().lower()
        if normalized in {"ask", "sell"}:
            return "ask"
        if normalized in {"bid", "buy"}:
            return "bid"
        raise ValueError(f"unsupported quote side: {side}")

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
        self._current_date_key = str(date)
        if self._daily_stop_date_key != self._current_date_key:
            self._daily_stop_trading = False
            self._daily_stop_date_key = None

        for event in self.loader.iter_merged_trade_intensity_tuples(
            symbol=symbol,
            date=date,
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
            elif kind == "bookticker":
                self._on_bookticker_event(
                    timestamp=ts,
                    best_bid=float(event[2]),
                    best_ask=float(event[3]),
                )
            elif kind == "aggtrade":
                self._on_aggtrade_event(
                    timestamp=ts,
                    is_buyer_maker=bool(event[2]),
                    impact=float(event[3]),
                    intensity_value=float(event[4]),
                    volume=float(event[5]),
                    first_price=float(event[6]),
                    last_price=float(event[7]),
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

    def _on_bookticker_event(
        self,
        *,
        timestamp: int,
        best_bid: float,
        best_ask: float,
    ) -> None:
        if not self._update_latest_bbo(best_bid=best_bid, best_ask=best_ask):
            return
        self._update_max_holding_tracking(timestamp=timestamp)
        self._enforce_passive_bbo_guard(timestamp=timestamp)

    def _on_aggtrade_event(
        self,
        *,
        timestamp: int,
        is_buyer_maker: bool,
        impact: float,
        intensity_value: float,
        volume: float,
        first_price: float,
        last_price: float,
    ) -> None:
        del impact, volume, first_price, last_price
        if self._latest_best_bid is None or self._latest_best_ask is None:
            return
        intensity = self._safe_non_negative(intensity_value)
        if intensity <= self.EPS:
            return
        if is_buyer_maker:
            self._latest_sell_intensity = self._ewma_intensity_update(
                previous=self._latest_sell_intensity,
                current=intensity,
            )
            quote_side: QuoteSide = "bid"
        else:
            self._latest_buy_intensity = self._ewma_intensity_update(
                previous=self._latest_buy_intensity,
                current=intensity,
            )
            quote_side = "ask"
        self._record_nonzero_agg_side(quote_side)
        self._on_ticker_event(
            timestamp=timestamp,
            best_bid=float(self._latest_best_bid),
            best_ask=float(self._latest_best_ask),
            apply_quote_gate=True,
            quote_side=quote_side,
        )

    def _ewma_intensity_update(self, *, previous: float, current: float) -> float:
        current = self._safe_non_negative(current)
        if current <= self.EPS:
            return self._safe_non_negative(previous)
        previous = self._safe_non_negative(previous)
        if previous <= self.EPS:
            return current
        alpha = float(self.cfg.ewma_intensity)
        return alpha * current + (1.0 - alpha) * previous

    def _record_nonzero_agg_side(self, side: QuoteSide) -> None:
        threshold = int(self.cfg.consequtive_sameside)
        if threshold <= 0:
            return
        opposite = self._opposite_quote_side(side)
        if self._last_nonzero_agg_side == side:
            self._same_side_nonzero_count += 1
        else:
            self._same_side_nonzero_count = 1
            self._blocked_quote_sides[opposite] = False
        self._last_nonzero_agg_side = side
        if self._same_side_nonzero_count >= threshold:
            self._blocked_quote_sides[side] = True

    @staticmethod
    def _opposite_quote_side(side: QuoteSide) -> QuoteSide:
        return "bid" if side == "ask" else "ask"

    def _blocked_quote_side_active(self, side: QuoteSide) -> bool:
        return (
            int(self.cfg.consequtive_sameside) > 0
            and bool(self._blocked_quote_sides.get(side, False))
        )

    def _update_latest_bbo(self, *, best_bid: float, best_ask: float) -> bool:
        rounded_best_bid = self._round_to_precision(float(best_bid), self.sim.price_precision)
        rounded_best_ask = self._round_to_precision(float(best_ask), self.sim.price_precision)
        if (
            (not math.isfinite(rounded_best_bid))
            or (not math.isfinite(rounded_best_ask))
            or rounded_best_bid <= 0.0
            or rounded_best_ask <= 0.0
            or rounded_best_bid >= rounded_best_ask
        ):
            return False
        self._latest_best_bid = rounded_best_bid
        self._latest_best_ask = rounded_best_ask
        return True

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
        self._forget_quote_gate_state()

    def _clear_open_maker_books(self) -> None:
        pos_qty = float(self.manager.position.qty)
        if pos_qty > self.EPS:
            self.manager.books.bid_maker.clear()
        elif pos_qty < -self.EPS:
            self.manager.books.ask_maker.clear()
        else:
            self.manager.books.ask_maker.clear()
            self.manager.books.bid_maker.clear()
        self._forget_quote_gate_state()

    def _on_ticker_event(
        self,
        timestamp: int,
        best_bid: float,
        best_ask: float,
        apply_quote_gate: bool = False,
        quote_side: QuoteSide | None = None,
    ) -> None:
        if not self._update_latest_bbo(best_bid=best_bid, best_ask=best_ask):
            return
        rounded_best_bid = float(self._latest_best_bid)
        rounded_best_ask = float(self._latest_best_ask)
        mid = 0.5 * (rounded_best_bid + rounded_best_ask)

        ask_intensity = self._latest_buy_intensity
        bid_intensity = self._latest_sell_intensity
        if self._daily_stop_trading_active():
            self._handle_daily_stop_ticker(timestamp=timestamp, mid=mid)
            return

        ask_open_distance = self._quote_distance(
            intensity_base=ask_intensity,
            close=False,
        )
        ask_close_distance = self._quote_distance(
            intensity_base=ask_intensity,
            close=True,
        )
        bid_open_distance = self._quote_distance(
            intensity_base=bid_intensity,
            close=False,
        )
        bid_close_distance = self._quote_distance(
            intensity_base=bid_intensity,
            close=True,
        )
        skew_shift = self._inventory_skew_price_shift(mid=mid)

        raw_open_ask_price = rounded_best_ask + ask_open_distance + skew_shift
        raw_open_bid_price = rounded_best_bid - bid_open_distance + skew_shift
        raw_close_ask_price = rounded_best_ask + ask_close_distance + skew_shift
        raw_close_bid_price = rounded_best_bid - bid_close_distance + skew_shift
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

        self._update_max_holding_tracking(timestamp=timestamp)

        if self._should_activate_stoploss(mid=mid):
            self._reach_and_release_active = True
            self._activate_daily_stop_trading()
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
            ) = self._clip_simple_quote_steps_to_bbo_distance(
                ask_price_ticks=quote_ask_price_ticks,
                ask_qty_steps=ask_qty_steps,
                bid_price_ticks=quote_bid_price_ticks,
                bid_qty_steps=bid_qty_steps,
                best_ask_ticks=best_ask_ticks,
                best_bid_ticks=best_bid_ticks,
            )
            if self._blocked_quote_side_active("ask"):
                quote_ask_price_ticks = None
                ask_qty_steps = 0
            if self._blocked_quote_side_active("bid"):
                quote_bid_price_ticks = None
                bid_qty_steps = 0
            if quote_side is not None:
                if quote_side == "ask":
                    quote_bid_price_ticks = None
                    bid_qty_steps = 0
                else:
                    quote_ask_price_ticks = None
                    ask_qty_steps = 0
                if apply_quote_gate and self._should_skip_simple_quote_side(
                    side=quote_side,
                    ask_price_ticks=quote_ask_price_ticks,
                    ask_qty_steps=ask_qty_steps,
                    bid_price_ticks=quote_bid_price_ticks,
                    bid_qty_steps=bid_qty_steps,
                    close_only=quote_close_only,
                ):
                    self._append_record(timestamp=timestamp, mid=mid)
                    return
                self._clear_pending_maker_side(quote_side)
            else:
                if apply_quote_gate and self._should_skip_simple_quote(
                    ask_price_ticks=quote_ask_price_ticks,
                    ask_qty_steps=ask_qty_steps,
                    bid_price_ticks=quote_bid_price_ticks,
                    bid_qty_steps=bid_qty_steps,
                    close_only=quote_close_only,
                ):
                    self._append_record(timestamp=timestamp, mid=mid)
                    return
                self._pending_maker_quotes.clear()
                self._pending_simple_maker_quotes.clear()
            self._enqueue_simple_maker_quote(
                base_timestamp=timestamp,
                quote_side=quote_side,
                ask_price_ticks=quote_ask_price_ticks,
                ask_qty_steps=ask_qty_steps,
                bid_price_ticks=quote_bid_price_ticks,
                bid_qty_steps=bid_qty_steps,
                best_ask_ticks=best_ask_ticks,
                best_bid_ticks=best_bid_ticks,
                close_only=quote_close_only,
            )
            if quote_side is not None:
                self._record_simple_quote_side_amend(
                    side=quote_side,
                    ask_price_ticks=quote_ask_price_ticks,
                    ask_qty_steps=ask_qty_steps,
                    bid_price_ticks=quote_bid_price_ticks,
                    bid_qty_steps=bid_qty_steps,
                    close_only=quote_close_only,
                )
            else:
                self._record_simple_quote_amend(
                    ask_price_ticks=quote_ask_price_ticks,
                    ask_qty_steps=ask_qty_steps,
                    bid_price_ticks=quote_bid_price_ticks,
                    bid_qty_steps=bid_qty_steps,
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
            ask_levels, bid_levels = self._clip_quote_levels_to_bbo_distance(
                ask_levels=ask_levels,
                bid_levels=bid_levels,
                best_ask=rounded_best_ask,
                best_bid=rounded_best_bid,
            )
            if self._blocked_quote_side_active("ask"):
                ask_levels = []
            if self._blocked_quote_side_active("bid"):
                bid_levels = []
            if quote_side is not None:
                if quote_side == "ask":
                    bid_levels = []
                else:
                    ask_levels = []
                if apply_quote_gate and self._should_skip_level_quote_side(
                    side=quote_side,
                    ask_levels=ask_levels,
                    bid_levels=bid_levels,
                    close_only=quote_close_only,
                ):
                    self._append_record(timestamp=timestamp, mid=mid)
                    return
                self._clear_pending_maker_side(quote_side)
            else:
                if apply_quote_gate and self._should_skip_level_quote(
                    ask_levels=ask_levels,
                    bid_levels=bid_levels,
                    close_only=quote_close_only,
                ):
                    self._append_record(timestamp=timestamp, mid=mid)
                    return
                self._pending_simple_maker_quotes.clear()
                self._pending_maker_quotes.clear()
            self._enqueue_maker_levels(
                base_timestamp=timestamp,
                quote_side=quote_side,
                ask_levels=ask_levels,
                bid_levels=bid_levels,
                best_ask=float(self._latest_best_ask),
                best_bid=float(self._latest_best_bid),
                close_only=quote_close_only,
            )
            if quote_side is not None:
                self._record_level_quote_side_amend(
                    side=quote_side,
                    ask_levels=ask_levels,
                    bid_levels=bid_levels,
                    close_only=quote_close_only,
                )
            else:
                self._record_level_quote_amend(
                    ask_levels=ask_levels,
                    bid_levels=bid_levels,
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

    def _daily_stop_trading_active(self) -> bool:
        return bool(self.cfg.daily_parallel and self._daily_stop_trading)

    def _activate_daily_stop_trading(self) -> None:
        if not self.cfg.daily_parallel:
            return
        self._daily_stop_trading = True
        self._daily_stop_date_key = self._current_date_key
        self._clear_strategy_maker_state()

    def _clear_strategy_maker_state(self) -> None:
        self._pending_maker_quotes.clear()
        self._pending_simple_maker_quotes.clear()
        self._clear_maker_books()
        self._forget_quote_gate_state()

    def _handle_daily_stop_ticker(self, *, timestamp: int, mid: float) -> None:
        self._clear_strategy_maker_state()
        if abs(float(self.manager.position.qty)) <= self.EPS:
            self._reach_and_release_active = False
            self._max_holding_start_ts = None
            self._max_holding_was_at_limit = False
            self._clear_taker_books()
            self._append_record(timestamp=timestamp, mid=mid)
            return

        self._reach_and_release_active = True
        if self._place_reach_and_release_taker(mid=mid):
            self._append_record(timestamp=timestamp, mid=mid)

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
        self._forget_quote_gate_state()

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

    def _forget_quote_gate_state(self) -> None:
        self._last_simple_quote_key = None
        self._last_level_quote_key = None
        self._last_simple_quote_side_keys = {"ask": None, "bid": None}
        self._last_level_quote_side_keys = {"ask": None, "bid": None}

    def _forget_quote_gate_side(self, side: QuoteSide) -> None:
        self._last_simple_quote_key = None
        self._last_level_quote_key = None
        self._last_simple_quote_side_keys[side] = None
        self._last_level_quote_side_keys[side] = None

    def _has_live_or_pending_maker_quote(self) -> bool:
        return bool(
            self.manager.books.ask_maker
            or self.manager.books.bid_maker
            or self._pending_maker_quotes
            or self._pending_simple_maker_quotes
        )

    @staticmethod
    def _pending_quote_applies_to_side(
        pending_side: QuoteSide | None,
        side: QuoteSide,
    ) -> bool:
        return pending_side is None or pending_side == side

    def _has_live_or_pending_maker_quote_side(self, side: QuoteSide) -> bool:
        if side == "ask":
            if self.manager.books.ask_maker:
                return True
        elif self.manager.books.bid_maker:
            return True

        for pending in self._pending_maker_quotes:
            (
                _active_ts,
                pending_side,
                ask_levels,
                bid_levels,
                _best_ask,
                _best_bid,
                _close_only,
            ) = pending
            if not self._pending_quote_applies_to_side(pending_side, side):
                continue
            if side == "ask" and ask_levels:
                return True
            if side == "bid" and bid_levels:
                return True

        for pending in self._pending_simple_maker_quotes:
            (
                _active_ts,
                pending_side,
                _ask_price_ticks,
                ask_qty_steps,
                _bid_price_ticks,
                bid_qty_steps,
                _best_ask_ticks,
                _best_bid_ticks,
                _close_only,
            ) = pending
            if not self._pending_quote_applies_to_side(pending_side, side):
                continue
            if side == "ask" and int(ask_qty_steps) > 0:
                return True
            if side == "bid" and int(bid_qty_steps) > 0:
                return True

        return False

    def _simple_quote_key(
        self,
        *,
        ask_price_ticks: Optional[int],
        ask_qty_steps: int,
        bid_price_ticks: Optional[int],
        bid_qty_steps: int,
        close_only: bool,
    ) -> tuple[object, ...]:
        ask_qty_steps = max(0, int(ask_qty_steps))
        bid_qty_steps = max(0, int(bid_qty_steps))
        return (
            None if ask_price_ticks is None or ask_qty_steps <= 0 else int(ask_price_ticks),
            ask_qty_steps,
            None if bid_price_ticks is None or bid_qty_steps <= 0 else int(bid_price_ticks),
            bid_qty_steps,
            bool(close_only and (ask_qty_steps > 0 or bid_qty_steps > 0)),
        )

    def _simple_quote_side_key(
        self,
        *,
        side: QuoteSide,
        ask_price_ticks: Optional[int],
        ask_qty_steps: int,
        bid_price_ticks: Optional[int],
        bid_qty_steps: int,
        close_only: bool,
    ) -> tuple[object, ...]:
        if side == "ask":
            price_ticks = ask_price_ticks
            qty_steps = max(0, int(ask_qty_steps))
        else:
            price_ticks = bid_price_ticks
            qty_steps = max(0, int(bid_qty_steps))
        return (
            side,
            None if price_ticks is None or qty_steps <= 0 else int(price_ticks),
            qty_steps,
            bool(close_only and qty_steps > 0),
        )

    def _should_skip_simple_quote(
        self,
        *,
        ask_price_ticks: Optional[int],
        ask_qty_steps: int,
        bid_price_ticks: Optional[int],
        bid_qty_steps: int,
        close_only: bool,
    ) -> bool:
        key = self._simple_quote_key(
            ask_price_ticks=ask_price_ticks,
            ask_qty_steps=ask_qty_steps,
            bid_price_ticks=bid_price_ticks,
            bid_qty_steps=bid_qty_steps,
            close_only=close_only,
        )
        target_empty = int(key[1]) <= 0 and int(key[3]) <= 0
        has_live = self._has_live_or_pending_maker_quote()
        if target_empty:
            return not has_live
        if not has_live or self._last_simple_quote_key is None:
            return False
        if key == self._last_simple_quote_key:
            return True
        return False

    def _should_skip_simple_quote_side(
        self,
        *,
        side: QuoteSide,
        ask_price_ticks: Optional[int],
        ask_qty_steps: int,
        bid_price_ticks: Optional[int],
        bid_qty_steps: int,
        close_only: bool,
    ) -> bool:
        key = self._simple_quote_side_key(
            side=side,
            ask_price_ticks=ask_price_ticks,
            ask_qty_steps=ask_qty_steps,
            bid_price_ticks=bid_price_ticks,
            bid_qty_steps=bid_qty_steps,
            close_only=close_only,
        )
        target_empty = int(key[2]) <= 0
        has_live = self._has_live_or_pending_maker_quote_side(side)
        if target_empty:
            return not has_live
        last_key = self._last_simple_quote_side_keys.get(side)
        if not has_live or last_key is None:
            return False
        return key == last_key

    def _record_simple_quote_amend(
        self,
        *,
        ask_price_ticks: Optional[int],
        ask_qty_steps: int,
        bid_price_ticks: Optional[int],
        bid_qty_steps: int,
        close_only: bool,
    ) -> None:
        self._last_simple_quote_key = self._simple_quote_key(
            ask_price_ticks=ask_price_ticks,
            ask_qty_steps=ask_qty_steps,
            bid_price_ticks=bid_price_ticks,
            bid_qty_steps=bid_qty_steps,
            close_only=close_only,
        )
        self._last_simple_quote_side_keys["ask"] = self._simple_quote_side_key(
            side="ask",
            ask_price_ticks=ask_price_ticks,
            ask_qty_steps=ask_qty_steps,
            bid_price_ticks=bid_price_ticks,
            bid_qty_steps=bid_qty_steps,
            close_only=close_only,
        )
        self._last_simple_quote_side_keys["bid"] = self._simple_quote_side_key(
            side="bid",
            ask_price_ticks=ask_price_ticks,
            ask_qty_steps=ask_qty_steps,
            bid_price_ticks=bid_price_ticks,
            bid_qty_steps=bid_qty_steps,
            close_only=close_only,
        )
        self._last_level_quote_key = None
        self._last_level_quote_side_keys = {"ask": None, "bid": None}

    def _record_simple_quote_side_amend(
        self,
        *,
        side: QuoteSide,
        ask_price_ticks: Optional[int],
        ask_qty_steps: int,
        bid_price_ticks: Optional[int],
        bid_qty_steps: int,
        close_only: bool,
    ) -> None:
        self._last_simple_quote_key = None
        self._last_simple_quote_side_keys[side] = self._simple_quote_side_key(
            side=side,
            ask_price_ticks=ask_price_ticks,
            ask_qty_steps=ask_qty_steps,
            bid_price_ticks=bid_price_ticks,
            bid_qty_steps=bid_qty_steps,
            close_only=close_only,
        )
        self._last_level_quote_key = None
        self._last_level_quote_side_keys = {"ask": None, "bid": None}

    def _level_quote_key(
        self,
        *,
        ask_levels: Sequence[MakerLevel],
        bid_levels: Sequence[MakerLevel],
        close_only: bool,
    ) -> tuple[object, ...]:
        def encode(levels: Sequence[MakerLevel]) -> tuple[tuple[int, int], ...]:
            encoded: list[tuple[int, int]] = []
            for price, qty in levels:
                if qty <= self.EPS:
                    continue
                encoded.append(
                    (
                        self.manager.converter.to_ticks(float(price), strict=False),
                        self.manager.converter.to_steps(float(qty), strict=False),
                    )
                )
            return tuple(encoded)

        return (encode(ask_levels), encode(bid_levels), bool(close_only))

    def _level_quote_side_key(
        self,
        *,
        side: QuoteSide,
        ask_levels: Sequence[MakerLevel],
        bid_levels: Sequence[MakerLevel],
        close_only: bool,
    ) -> tuple[object, ...]:
        def encode(levels: Sequence[MakerLevel]) -> tuple[tuple[int, int], ...]:
            encoded: list[tuple[int, int]] = []
            for price, qty in levels:
                if qty <= self.EPS:
                    continue
                encoded.append(
                    (
                        self.manager.converter.to_ticks(float(price), strict=False),
                        self.manager.converter.to_steps(float(qty), strict=False),
                    )
                )
            return tuple(encoded)

        encoded_levels = encode(ask_levels if side == "ask" else bid_levels)
        return (side, encoded_levels, bool(close_only and encoded_levels))

    def _should_skip_level_quote(
        self,
        *,
        ask_levels: Sequence[MakerLevel],
        bid_levels: Sequence[MakerLevel],
        close_only: bool,
    ) -> bool:
        key = self._level_quote_key(
            ask_levels=ask_levels,
            bid_levels=bid_levels,
            close_only=close_only,
        )
        target_empty = not key[0] and not key[1]
        has_live = self._has_live_or_pending_maker_quote()
        if target_empty:
            return not has_live
        if not has_live or self._last_level_quote_key is None:
            return False
        if key == self._last_level_quote_key:
            return True
        return False

    def _should_skip_level_quote_side(
        self,
        *,
        side: QuoteSide,
        ask_levels: Sequence[MakerLevel],
        bid_levels: Sequence[MakerLevel],
        close_only: bool,
    ) -> bool:
        key = self._level_quote_side_key(
            side=side,
            ask_levels=ask_levels,
            bid_levels=bid_levels,
            close_only=close_only,
        )
        target_empty = not key[1]
        has_live = self._has_live_or_pending_maker_quote_side(side)
        if target_empty:
            return not has_live
        last_key = self._last_level_quote_side_keys.get(side)
        if not has_live or last_key is None:
            return False
        return key == last_key

    def _record_level_quote_amend(
        self,
        *,
        ask_levels: Sequence[MakerLevel],
        bid_levels: Sequence[MakerLevel],
        close_only: bool,
    ) -> None:
        self._last_level_quote_key = self._level_quote_key(
            ask_levels=ask_levels,
            bid_levels=bid_levels,
            close_only=close_only,
        )
        self._last_level_quote_side_keys["ask"] = self._level_quote_side_key(
            side="ask",
            ask_levels=ask_levels,
            bid_levels=bid_levels,
            close_only=close_only,
        )
        self._last_level_quote_side_keys["bid"] = self._level_quote_side_key(
            side="bid",
            ask_levels=ask_levels,
            bid_levels=bid_levels,
            close_only=close_only,
        )
        self._last_simple_quote_key = None
        self._last_simple_quote_side_keys = {"ask": None, "bid": None}

    def _record_level_quote_side_amend(
        self,
        *,
        side: QuoteSide,
        ask_levels: Sequence[MakerLevel],
        bid_levels: Sequence[MakerLevel],
        close_only: bool,
    ) -> None:
        self._last_level_quote_key = None
        self._last_level_quote_side_keys[side] = self._level_quote_side_key(
            side=side,
            ask_levels=ask_levels,
            bid_levels=bid_levels,
            close_only=close_only,
        )
        self._last_simple_quote_key = None
        self._last_simple_quote_side_keys = {"ask": None, "bid": None}


    def _enforce_passive_bbo_guard(self, *, timestamp: int) -> None:
        del timestamp
        if self._latest_best_bid is None or self._latest_best_ask is None:
            return
        try:
            best_bid_ticks = self.manager.converter.to_ticks(float(self._latest_best_bid))
            best_ask_ticks = self.manager.converter.to_ticks(float(self._latest_best_ask))
        except ValueError:
            return

        ask_min_ticks = self._min_quote_distance_ticks(reference_price=float(self._latest_best_ask))
        bid_min_ticks = self._min_quote_distance_ticks(reference_price=float(self._latest_best_bid))
        ask_floor = int(best_ask_ticks) + int(ask_min_ticks) if ask_min_ticks > 0 else None
        bid_ceiling = int(best_bid_ticks) - int(bid_min_ticks) if bid_min_ticks > 0 else None

        changed_sides: list[QuoteSide] = []
        ask_best = self.manager.books.ask_maker.best_level()
        if ask_best is not None:
            ask_tick = int(ask_best[0])
            if ask_tick <= int(best_bid_ticks) or (ask_floor is not None and ask_tick < ask_floor):
                self.manager.books.ask_maker.clear()
                changed_sides.append("ask")
        bid_best = self.manager.books.bid_maker.best_level()
        if bid_best is not None:
            bid_tick = int(bid_best[0])
            if bid_tick >= int(best_ask_ticks) or (bid_ceiling is not None and bid_tick > bid_ceiling):
                self.manager.books.bid_maker.clear()
                changed_sides.append("bid")
        for side in changed_sides:
            self._clear_pending_maker_side(side)
            self._forget_quote_gate_side(side)

    def _activate_due_maker_quotes(self, current_timestamp: int) -> None:
        self._activate_pending_maker_quotes(current_timestamp=current_timestamp)
        self._activate_pending_simple_maker_quotes(current_timestamp=current_timestamp)

    def _clear_pending_maker_side(self, side: QuoteSide) -> None:
        self._pending_maker_quotes = deque(
            pending
            for pending in self._pending_maker_quotes
            if not self._pending_quote_applies_to_side(pending[1], side)
        )
        self._pending_simple_maker_quotes = deque(
            pending
            for pending in self._pending_simple_maker_quotes
            if not self._pending_quote_applies_to_side(pending[1], side)
        )

    def _enqueue_maker_levels(
        self,
        base_timestamp: int,
        quote_side: QuoteSide | None,
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
                quote_side,
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
        quote_side: QuoteSide | None,
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
                quote_side,
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
        due_quotes: list[PendingMakerQuote] = []
        while self._pending_maker_quotes and self._pending_maker_quotes[0][0] <= ts:
            due_quotes.append(self._pending_maker_quotes.popleft())

        if not due_quotes:
            return

        for due_quote in due_quotes:
            self._activate_maker_quote(due_quote)

    def _activate_maker_quote(self, quote: PendingMakerQuote) -> None:
        (
            _active_ts,
            quote_side,
            ask_levels,
            bid_levels,
            best_ask,
            best_bid,
            close_only,
        ) = quote
        if quote_side is not None:
            levels = ask_levels if quote_side == "ask" else bid_levels
            self.manager.place_maker_levels_side(
                side="sell" if quote_side == "ask" else "buy",
                levels=levels,
                best_ask=best_ask,
                best_bid=best_bid,
                close_only=(
                    bool(close_only)
                    or self._side_is_close_at_placement(
                        side=quote_side,
                        has_order=bool(levels),
                    )
                ),
            )
            return

        self.manager.place_maker_levels_side(
            side="sell",
            levels=ask_levels,
            best_ask=best_ask,
            best_bid=best_bid,
            close_only=(
                bool(close_only)
                or self._side_is_close_at_placement(
                    side="ask",
                    has_order=bool(ask_levels),
                )
            ),
        )
        self.manager.place_maker_levels_side(
            side="buy",
            levels=bid_levels,
            best_ask=best_ask,
            best_bid=best_bid,
            close_only=(
                bool(close_only)
                or self._side_is_close_at_placement(
                    side="bid",
                    has_order=bool(bid_levels),
                )
            ),
        )

    def _side_is_close_at_placement(self, *, side: QuoteSide, has_order: bool) -> bool:
        if not has_order:
            return False
        position_steps = int(getattr(self.manager, "position_steps", 0))
        return (side == "ask" and position_steps > 0) or (
            side == "bid" and position_steps < 0
        )

    def _activate_pending_simple_maker_quotes(self, current_timestamp: int) -> None:
        ts = int(current_timestamp)
        due_quotes: list[SimplePendingMakerQuote] = []
        while (
            self._pending_simple_maker_quotes
            and self._pending_simple_maker_quotes[0][0] <= ts
        ):
            due_quotes.append(self._pending_simple_maker_quotes.popleft())

        if not due_quotes:
            return

        for due_quote in due_quotes:
            self._activate_simple_maker_quote(due_quote)

    def _activate_simple_maker_quote(self, quote: SimplePendingMakerQuote) -> None:
        (
            _active_ts,
            quote_side,
            ask_price_ticks,
            ask_qty_steps,
            bid_price_ticks,
            bid_qty_steps,
            best_ask_ticks,
            best_bid_ticks,
            close_only,
        ) = quote
        if quote_side == "ask":
            self.manager.place_maker_steps_side(
                side="sell",
                price_ticks=ask_price_ticks,
                qty_steps=ask_qty_steps,
                best_ask_ticks=best_ask_ticks,
                best_bid_ticks=best_bid_ticks,
                close_only=(
                    bool(close_only)
                    or self._side_is_close_at_placement(
                        side="ask",
                        has_order=int(ask_qty_steps) > 0,
                    )
                ),
            )
        elif quote_side == "bid":
            self.manager.place_maker_steps_side(
                side="buy",
                price_ticks=bid_price_ticks,
                qty_steps=bid_qty_steps,
                best_ask_ticks=best_ask_ticks,
                best_bid_ticks=best_bid_ticks,
                close_only=(
                    bool(close_only)
                    or self._side_is_close_at_placement(
                        side="bid",
                        has_order=int(bid_qty_steps) > 0,
                    )
                ),
            )
        else:
            self.manager.place_maker_steps_side(
                side="sell",
                price_ticks=ask_price_ticks,
                qty_steps=ask_qty_steps,
                best_ask_ticks=best_ask_ticks,
                best_bid_ticks=best_bid_ticks,
                close_only=(
                    bool(close_only)
                    or self._side_is_close_at_placement(
                        side="ask",
                        has_order=int(ask_qty_steps) > 0,
                    )
                ),
            )
            self.manager.place_maker_steps_side(
                side="buy",
                price_ticks=bid_price_ticks,
                qty_steps=bid_qty_steps,
                best_ask_ticks=best_ask_ticks,
                best_bid_ticks=best_bid_ticks,
                close_only=(
                    bool(close_only)
                    or self._side_is_close_at_placement(
                        side="bid",
                        has_order=int(bid_qty_steps) > 0,
                    )
                ),
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

    def _quote_distance(
        self,
        *,
        intensity_base: float,
        close: bool,
    ) -> float:
        pair_index = 1 if close else 0
        return intensity_base * float(self.cfg.adj_spread_intensity[pair_index])

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
