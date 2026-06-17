from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import pandas as pd

from core.manager import Manager as ValidatedManager
from core.position import Position
from sim.loader import BinanceEventLoader, DateLike


MakerLevel = tuple[float, float]
PendingMakerQuote = tuple[int, list[MakerLevel], list[MakerLevel], float, float, bool]


def _round_to_precision(value: float, precision: int) -> float:
    return float(f"{value:.{precision}f}")


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
    order_amt: float = 0.0
    max_position_usdt: float = 0.0
    max_holding_time: int = 0
    adj_spread_intensity: float = 1.0
    adj_spread_instructor: float = 0.0
    passive_only: bool = False
    adj_spread_volatility: float = 0.0
    inventory_skew: Optional[tuple[float, float]] = None
    min_order_qty: float = 0.0
    min_order_notional: float = 0.0
    stoploss: float = 0.0
    open_curve: Optional[tuple[float, float, float]] = None
    close_curve: Optional[tuple[float, float, float]] = None

    def __post_init__(self) -> None:
        max_position_usdt = self._normalize_scalar(
            self.max_position_usdt,
            "max_position_usdt",
        )
        stoploss = self._normalize_scalar(self.stoploss, "stoploss")
        object.__setattr__(self, "max_position_usdt", max_position_usdt)
        object.__setattr__(self, "stoploss", stoploss)
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
        if self.order_amt < 0:
            raise ValueError("order_amt must be >= 0")
        if self.max_position_usdt < 0:
            raise ValueError("max_position_usdt must be >= 0")
        if self.adj_spread_intensity <= 0:
            raise ValueError("adj_spread_intensity must be > 0")
        if not math.isfinite(float(self.adj_spread_instructor)):
            raise ValueError("adj_spread_instructor must be finite")
        if not isinstance(self.passive_only, bool):
            raise ValueError("passive_only must be boolean")
        if self.adj_spread_volatility < 0:
            raise ValueError("adj_spread_volatility must be >= 0")
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
        self._records: list[tuple[int, float, float, float, float, float, float, float, float]] = []
        self._reach_and_release_active: bool = False
        self._max_holding_start_ts: Optional[int] = None
        self._max_holding_was_at_limit: bool = False
        self._pending_maker_quotes: deque[PendingMakerQuote] = deque()

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
        self._pending_maker_quotes = deque()
        pending_quotes = state.get("pending_maker_quotes", [])
        if isinstance(pending_quotes, list):
            for row in pending_quotes:
                if not isinstance(row, dict):
                    continue
                try:
                    ask_levels_raw = row.get("ask_levels")
                    bid_levels_raw = row.get("bid_levels")
                    if ask_levels_raw is None:
                        ask_levels = [
                            (float(row["ask_price"]), float(row["ask_qty"]))
                        ]
                    else:
                        ask_levels = self._parse_pending_levels(ask_levels_raw)
                    if bid_levels_raw is None:
                        bid_levels = [
                            (float(row["bid_price"]), float(row["bid_qty"]))
                        ]
                    else:
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
            self._activate_pending_maker_quotes(current_timestamp=ts)
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
        position_reduced = new_abs_pos_qty + self.EPS < prev_abs_pos_qty
        self._update_max_holding_tracking(
            timestamp=int(trade_time),
            position_reduced=position_reduced,
        )
        return float(filled_qty)

    def _on_ticker_event(
        self,
        timestamp: int,
        best_bid: float,
        best_ask: float,
        intensity_value: Optional[float],
        volatility_scalar: float,
        instructor_value: Optional[float] = None,
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
        distance = (
            intensity_base * self.cfg.adj_spread_intensity
            + vol_scalar * self.cfg.adj_spread_volatility
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

        close_ask_price = self._price_ceil(
            rounded_best_ask + distance + ask_instructor_shift + skew_shift
        )
        close_bid_price = self._price_floor(
            rounded_best_bid - distance + bid_instructor_shift + skew_shift
        )

        open_ask_price = self._price_ceil(
            rounded_best_ask + distance + ask_instructor_shift + skew_shift
        )
        open_bid_price = self._price_floor(
            rounded_best_bid - distance + bid_instructor_shift + skew_shift
        )

        ask_qty, bid_qty = self._base_symmetric_qty(mid=mid)
        ask_qty, bid_qty = self._apply_inventory_limit(mid=mid, ask_qty=ask_qty, bid_qty=bid_qty)

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
        ask_levels, bid_levels, quote_close_only = self._maker_levels_for_ticker(
            mid=mid,
            open_ask_price=open_ask_price,
            ask_qty=ask_qty,
            open_bid_price=open_bid_price,
            bid_qty=bid_qty,
            close_ask_price=close_ask_price,
            close_bid_price=close_bid_price,
            best_ask=rounded_best_ask,
            best_bid=rounded_best_bid,
        )
        self._update_max_holding_tracking(timestamp=timestamp)

        if self._should_activate_stoploss(mid=mid):
            self._reach_and_release_active = True
        if self._should_activate_max_holding_timeout(timestamp=timestamp):
            self._reach_and_release_active = True

        if self._reach_and_release_active and self._place_reach_and_release_taker(mid=mid):
            self._append_record(timestamp=timestamp, mid=mid)
            return

        self._clear_taker_books()
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

    def _maker_levels_for_ticker(
        self,
        *,
        mid: float,
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
            ask_qty = self._boosted_open_qty(mid=mid, base_open_qty=ask_qty, multiplier=1.0)
            bid_qty = self._boosted_open_qty(mid=mid, base_open_qty=bid_qty, multiplier=1.0)
            return self._single_quote_levels(open_ask_price, ask_qty, open_bid_price, bid_qty, False)

        if pos_qty > 0.0:
            if mid <= cost + self.EPS:
                bid_qty = self._grid_open_qty(mid=mid, base_open_qty=bid_qty)
                return self._single_quote_levels(open_ask_price, 0.0, open_bid_price, bid_qty, False)
            close_qty = self._curve_close_qty(mid=mid, pos_qty=pos_qty)
            return self._single_quote_levels(close_ask_price, close_qty, close_bid_price, 0.0, True)

        if mid >= cost - self.EPS:
            ask_qty = self._grid_open_qty(mid=mid, base_open_qty=ask_qty)
            return self._single_quote_levels(open_ask_price, ask_qty, open_bid_price, 0.0, False)
        close_qty = self._curve_close_qty(mid=mid, pos_qty=pos_qty)
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
        multiplier = self._inventory_open_curve_multiplier(mid=mid)
        return self._boosted_open_qty(mid=mid, base_open_qty=base_open_qty, multiplier=multiplier)

    def _curve_close_qty(self, mid: float, pos_qty: float) -> float:
        base_close_qty = self._close_qty_from_position(pos_qty)
        if base_close_qty <= 0.0:
            return 0.0

        min_close_qty = self._min_lawful_qty(mid=mid)
        if min_close_qty > self.EPS and base_close_qty + self.EPS < min_close_qty:
            return base_close_qty

        multiplier = self._inventory_close_curve_multiplier(mid=mid)
        if multiplier <= self.EPS:
            return 0.0

        close_qty = self._qty_floor(base_close_qty * multiplier)
        if close_qty <= self.EPS:
            close_qty = min(
                base_close_qty,
                self._round_to_precision(self.sim.step_size, self.sim.qty_precision),
            )
        if min_close_qty > self.EPS and close_qty + self.EPS < min_close_qty:
            close_qty = min(base_close_qty, min_close_qty)
        return close_qty

    def _base_symmetric_qty(self, mid: float) -> tuple[float, float]:
        q = self._lawful_qty_from_qty(mid=mid, qty=self._qty_floor(float(self.cfg.order_amt) / mid))
        return q, q

    def _apply_inventory_limit(self, mid: float, ask_qty: float, bid_qty: float) -> tuple[float, float]:
        if ask_qty <= 0.0 and bid_qty <= 0.0:
            return 0.0, 0.0

        max_position_usdt = float(self.cfg.max_position_usdt)
        if max_position_usdt <= self.EPS:
            return 0.0, 0.0

        cost_notional_usdt = float(self.manager.position.cost_notional_usdt)
        if cost_notional_usdt + self.EPS >= max_position_usdt:
            bid_qty = 0.0
        if cost_notional_usdt - self.EPS <= -max_position_usdt:
            ask_qty = 0.0
        return ask_qty, bid_qty

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

    def _boosted_open_qty(self, mid: float, base_open_qty: float, multiplier: float) -> float:
        if base_open_qty <= 0.0:
            return 0.0
        boosted_qty = self._qty_floor(base_open_qty * multiplier)
        return self._lawful_qty_from_qty(mid=mid, qty=boosted_qty)

    def _inventory_open_curve_multiplier(self, mid: float) -> float:
        scale = self.cfg.open_curve
        if not scale or mid <= self.EPS:
            return 1.0

        min_scale = float(scale[0])
        max_scale = float(scale[1])
        order = float(scale[2])
        if order <= self.EPS:
            return 1.0

        max_position_usdt = float(self.cfg.max_position_usdt)
        if max_position_usdt <= self.EPS:
            return 1.0

        gross_cost_notional_usdt = float(self.manager.position.gross_cost_notional_usdt)
        util = min(1.0, max(0.0, gross_cost_notional_usdt / max_position_usdt))
        decay = (1.0 - util) ** order
        return min_scale + (max_scale - min_scale) * decay

    def _inventory_close_curve_multiplier(self, mid: float) -> float:
        scale = self.cfg.close_curve
        if not scale or mid <= self.EPS:
            return 1.0

        min_scale = float(scale[0])
        max_scale = float(scale[1])
        order = float(scale[2])
        if order <= self.EPS:
            return 1.0

        max_position_usdt = float(self.cfg.max_position_usdt)
        if max_position_usdt <= self.EPS:
            return 1.0

        gross_cost_notional_usdt = float(self.manager.position.gross_cost_notional_usdt)
        util = min(1.0, max(0.0, gross_cost_notional_usdt / max_position_usdt))
        growth = util ** order
        return min_scale + (max_scale - min_scale) * growth

    def _lawful_qty_from_qty(self, mid: float, qty: float) -> float:
        if qty <= 0.0 or mid <= self.EPS:
            return 0.0

        qty = self._qty_floor(qty)
        if qty <= 0.0:
            return 0.0

        min_qty = float(self.cfg.min_order_qty)
        min_notional = float(self.cfg.min_order_notional)
        if min_notional > 0.0:
            min_qty = max(min_qty, min_notional / mid)
        if qty + self.EPS < min_qty:
            return 0.0
        return qty

    def _min_lawful_qty(self, mid: float) -> float:
        min_qty = float(self.cfg.min_order_qty)
        min_notional = float(self.cfg.min_order_notional)
        if min_notional > 0.0 and mid > self.EPS:
            min_qty = max(min_qty, min_notional / mid)
        if min_qty <= self.EPS:
            return 0.0
        return self._qty_ceil(min_qty)

    def _price_floor(self, price: float) -> float:
        if price <= 0.0:
            return 0.0
        ticks = math.floor((price / self.sim.tick_size) + self.EPS)
        return self._round_to_precision(ticks * self.sim.tick_size, self.sim.price_precision)

    def _price_ceil(self, price: float) -> float:
        if price <= 0.0:
            return 0.0
        ticks = math.ceil((price / self.sim.tick_size) - self.EPS)
        return self._round_to_precision(ticks * self.sim.tick_size, self.sim.price_precision)

    def _qty_floor(self, qty: float) -> float:
        if qty <= 0.0:
            return 0.0
        steps = math.floor((qty / self.sim.step_size) + self.EPS)
        return self._round_to_precision(steps * self.sim.step_size, self.sim.qty_precision)

    def _qty_ceil(self, qty: float) -> float:
        if qty <= 0.0:
            return 0.0
        steps = math.ceil((qty / self.sim.step_size) - self.EPS)
        return self._round_to_precision(steps * self.sim.step_size, self.sim.qty_precision)

    def _close_qty_from_position(self, pos_qty: float) -> float:
        if abs(pos_qty) <= self.EPS:
            return 0.0
        steps = abs(self.manager.converter.to_steps(pos_qty, strict=False, rounding="round"))
        if steps <= 0:
            return 0.0
        return self._round_to_precision(
            self.manager.converter.from_steps(steps),
            self.sim.qty_precision,
        )

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

    def _selected_intensity_spec(self) -> dict[str, int | str]:
        return {
            "name": str(self.sim.name_intensity).strip().lower(),
            "lookback": int(self.sim.lookback_intensity),
        }

    def _selected_instructor_spec(self) -> dict[str, int | str] | None:
        if self.sim.name_instructor is None:
            return None
        if self.sim.lookback_instructor is None:
            raise ValueError("lookback_instructor must be provided when name_instructor is set")
        return {
            "name": str(self.sim.name_instructor).strip().lower(),
            "lookback": int(self.sim.lookback_instructor),
        }

    def _selected_volatility_specs(self) -> list[dict[str, int | str]]:
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
