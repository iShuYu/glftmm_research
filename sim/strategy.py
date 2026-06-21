from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, replace
from typing import Iterable, Optional, Sequence

import pandas as pd

from core.manager import Manager as ValidatedManager
from core.position import Position
from sim.loader import BinanceEventLoader, DateLike


MakerLevel = tuple[float, float]
PendingMakerQuote = tuple[int, list[MakerLevel], list[MakerLevel], float, float, bool]
PROFIT_GRID_QTY_DISTRIBUTIONS = {"equal", "power", "exponential"}


def _round_to_precision(value: float, precision: int) -> float:
    return float(f"{value:.{precision}f}")


def _floor_to_step(value: float, step_size: float, precision: int) -> float:
    if value <= 0.0 or step_size <= 0.0:
        return 0.0
    steps = math.floor((value / step_size) + 1e-12)
    return _round_to_precision(steps * step_size, precision)


def _ceil_to_step(value: float, step_size: float, precision: int) -> float:
    if value <= 0.0 or step_size <= 0.0:
        return 0.0
    steps = math.ceil((value / step_size) - 1e-12)
    return _round_to_precision(steps * step_size, precision)


def _ceil_to_tick(value: float, tick_size: float, precision: int) -> float:
    if value <= 0.0 or tick_size <= 0.0:
        return 0.0
    ticks = math.ceil((value / tick_size) - 1e-12)
    return _round_to_precision(ticks * tick_size, precision)


def _floor_to_tick(value: float, tick_size: float, precision: int) -> float:
    if value <= 0.0 or tick_size <= 0.0:
        return 0.0
    ticks = math.floor((value / tick_size) + 1e-12)
    return _round_to_precision(ticks * tick_size, precision)


@dataclass(frozen=True)
class ProfitGridQtyDistribution:
    kind: str = "equal"
    param: float = 1.0

    def __post_init__(self) -> None:
        kind = str(self.kind).strip().lower()
        if kind in ("uniform", "flat"):
            kind = "equal"
        elif kind in ("quadratic", "power2"):
            kind = "power"
            if float(self.param) == 1.0:
                object.__setattr__(self, "param", 2.0)
        elif kind in ("exp", "exponential_decay"):
            kind = "exponential"

        if kind not in PROFIT_GRID_QTY_DISTRIBUTIONS:
            raise ValueError(
                "profit_grid_qty_distribution kind must be one of "
                f"{sorted(PROFIT_GRID_QTY_DISTRIBUTIONS)}"
            )

        param = float(self.param)
        if not math.isfinite(param):
            raise ValueError("profit_grid_qty_distribution param must be finite")
        if kind == "equal":
            param = 1.0
        elif kind == "power" and param <= 0.0:
            raise ValueError("profit_grid_qty_distribution power param must be > 0")
        elif kind == "exponential" and param < 0.0:
            raise ValueError("profit_grid_qty_distribution exponential param must be >= 0")

        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "param", param)


def _profit_grid_qty_default_param(kind: str) -> float:
    kind = str(kind).strip().lower()
    if kind in ("power", "quadratic", "power2"):
        return 2.0
    return 1.0


def normalize_profit_grid_qty_distribution(raw: object | None) -> ProfitGridQtyDistribution:
    if raw is None:
        return ProfitGridQtyDistribution()
    if isinstance(raw, ProfitGridQtyDistribution):
        return raw
    if isinstance(raw, str):
        return ProfitGridQtyDistribution(
            kind=raw,
            param=_profit_grid_qty_default_param(raw),
        )
    if isinstance(raw, dict):
        kind = (
            raw.get("kind")
            or raw.get("type")
            or raw.get("name")
            or raw.get("function")
        )
        if kind is None:
            raise ValueError(
                "profit_grid_qty_distribution dict requires kind/type/name/function"
            )
        param = raw.get("param")
        if param is None:
            param = (
                raw.get("exponent")
                if "exponent" in raw
                else raw.get("power")
                if "power" in raw
                else raw.get("rate")
                if "rate" in raw
                else raw.get("decay")
                if "decay" in raw
                else _profit_grid_qty_default_param(str(kind))
            )
        return ProfitGridQtyDistribution(kind=str(kind), param=float(param))
    if isinstance(raw, (list, tuple)):
        if not raw:
            raise ValueError("profit_grid_qty_distribution cannot be empty")
        kind = raw[0]
        if not isinstance(kind, str):
            raise ValueError("profit_grid_qty_distribution first item must be a kind string")
        if len(raw) == 1:
            param = _profit_grid_qty_default_param(kind)
        elif len(raw) == 2:
            param = raw[1]
        else:
            raise ValueError("profit_grid_qty_distribution must be [kind] or [kind, param]")
        return ProfitGridQtyDistribution(kind=kind, param=float(param))
    raise ValueError("unsupported profit_grid_qty_distribution value")


def normalize_profit_grid_spec(
    raw: object | None,
    qty_distribution: object | None = None,
) -> tuple[tuple[float, float, int] | None, ProfitGridQtyDistribution]:
    distribution_raw = qty_distribution
    if raw is None:
        return None, normalize_profit_grid_qty_distribution(distribution_raw)
    if not isinstance(raw, (list, tuple)) or len(raw) not in (3, 4, 5):
        raise ValueError(
            "profit_grid must be [lower_bps, upper_bps, num_grid], "
            "[lower_bps, upper_bps, num_grid, qty_kind], or "
            "[lower_bps, upper_bps, num_grid, qty_kind, qty_param]"
        )

    if len(raw) > 3:
        if distribution_raw is not None:
            raise ValueError(
                "profit_grid_qty_distribution cannot be set when profit_grid "
                "already includes qty distribution"
            )
        distribution_raw = raw[3] if len(raw) == 4 else [raw[3], raw[4]]

    profit_grid = (float(raw[0]), float(raw[1]), int(raw[2]))
    return profit_grid, normalize_profit_grid_qty_distribution(distribution_raw)


def _profit_grid_qty_weights(
    num_grid: int,
    distribution: ProfitGridQtyDistribution,
) -> list[float]:
    if distribution.kind == "equal":
        return [1.0 for _ in range(num_grid)]
    if distribution.kind == "power":
        return [
            float(num_grid - idx) ** float(distribution.param)
            for idx in range(num_grid)
        ]
    if distribution.kind == "exponential":
        return [
            math.exp(-float(distribution.param) * float(idx))
            for idx in range(num_grid)
        ]
    raise ValueError(f"unsupported profit grid qty distribution: {distribution.kind}")


def build_profit_grid_levels(
    *,
    lower_bps: float,
    upper_bps: float,
    num_grid: int,
    position_qty: float,
    cost: float,
    is_long: bool,
    tick_size: float,
    step_size: float,
    price_precision: int,
    qty_precision: int,
    min_order_qty: float,
    min_order_notional: float,
    qty_distribution: object | None = None,
) -> list[MakerLevel]:
    if num_grid <= 0 or position_qty <= 0.0 or cost <= 0.0:
        return []
    if lower_bps < 0.0 or upper_bps <= lower_bps:
        return []

    total_qty = _floor_to_step(abs(float(position_qty)), step_size, qty_precision)
    if total_qty <= 0.0:
        return []

    lower = float(lower_bps) / 1e4
    upper = float(upper_bps) / 1e4
    prices: list[float] = []
    for idx in range(int(num_grid)):
        ratio = 1.0 if num_grid == 1 else idx / float(num_grid - 1)
        bps_ratio = lower + (upper - lower) * ratio
        raw_price = cost * (1.0 + bps_ratio) if is_long else cost * (1.0 - bps_ratio)
        price = (
            _ceil_to_tick(raw_price, tick_size, price_precision)
            if is_long
            else _floor_to_tick(raw_price, tick_size, price_precision)
        )
        if price > 0.0 and math.isfinite(price):
            prices.append(price)

    if not prices:
        return []

    distribution = normalize_profit_grid_qty_distribution(qty_distribution)
    weights = _profit_grid_qty_weights(len(prices), distribution)
    weight_sum = sum(weights)
    if weight_sum <= 0.0 or not math.isfinite(weight_sum):
        return []

    def min_lawful_qty(price: float) -> float:
        raw_min_qty = max(float(min_order_qty), 0.0)
        if min_order_notional > 0.0 and price > 0.0:
            raw_min_qty = max(raw_min_qty, float(min_order_notional) / price)
        if raw_min_qty <= 0.0:
            return _round_to_precision(step_size, qty_precision)
        return _ceil_to_step(raw_min_qty, step_size, qty_precision)

    allocations = [0.0 for _ in prices]
    remaining = total_qty
    for idx, (price, weight) in enumerate(zip(prices, weights)):
        if remaining <= 0.0:
            break
        target_qty = total_qty * float(weight) / weight_sum
        qty = _floor_to_step(target_qty, step_size, qty_precision)
        if qty <= 0.0 or qty + 1e-12 < min_lawful_qty(price):
            break
        allocations[idx] = qty
        remaining = _round_to_precision(remaining - qty, qty_precision)

    if remaining > 0.0:
        allocations[0] = _round_to_precision(
            allocations[0] + remaining,
            qty_precision,
        )

    return [
        (price, qty)
        for price, qty in zip(prices, allocations)
        if qty > 0.0 and math.isfinite(qty)
    ]


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
    max_position_usdt: float | tuple[float, ...] = 0.0
    new_open_lot_crit: float = 0.0
    max_holding_time: int = 0
    adj_spread_intensity: float = 1.0
    adj_spread_instructor: float = 0.0
    open_passive_only: bool = False
    adj_spread_volatility: float = 0.0
    inventory_skew: Optional[tuple[float, float]] = None
    min_order_qty: float = 0.0
    min_order_notional: float = 0.0
    stoploss: float | tuple[float, ...] = 0.0
    open_curve_underwater: Optional[tuple[float, float, float]] = None
    profit_grid: Optional[tuple[float, float, int]] = None
    profit_grid_qty_distribution: object | None = None

    def __post_init__(self) -> None:
        profit_grid, profit_grid_qty_distribution = normalize_profit_grid_spec(
            self.profit_grid,
            self.profit_grid_qty_distribution,
        )
        object.__setattr__(self, "profit_grid", profit_grid)
        object.__setattr__(
            self,
            "profit_grid_qty_distribution",
            profit_grid_qty_distribution,
        )
        max_position_lots = self._normalize_max_position_lots(self.max_position_usdt)
        stoploss_lots = self._normalize_stoploss_lots(self.stoploss)
        new_open_lot_crit = float(self.new_open_lot_crit)
        if len(stoploss_lots) != len(max_position_lots):
            raise ValueError(
                "stoploss and max_position_usdt must have the same number of lots"
            )
        object.__setattr__(self, "new_open_lot_crit", new_open_lot_crit)
        object.__setattr__(
            self,
            "max_position_usdt",
            max_position_lots[0] if len(max_position_lots) == 1 else max_position_lots,
        )
        object.__setattr__(
            self,
            "stoploss",
            stoploss_lots[0] if len(stoploss_lots) == 1 else stoploss_lots,
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
        if self.order_amt < 0:
            raise ValueError("order_amt must be >= 0")
        if any(value < 0 for value in max_position_lots):
            raise ValueError("max_position_usdt lots must be >= 0")
        if not math.isfinite(new_open_lot_crit):
            raise ValueError("new_open_lot_crit must be finite")
        if new_open_lot_crit < 0.0 or new_open_lot_crit >= 1.0:
            raise ValueError("new_open_lot_crit must be in [0, 1)")
        if self.adj_spread_intensity <= 0:
            raise ValueError("adj_spread_intensity must be > 0")
        if not math.isfinite(float(self.adj_spread_instructor)):
            raise ValueError("adj_spread_instructor must be finite")
        if not isinstance(self.open_passive_only, bool):
            raise ValueError("open_passive_only must be boolean")
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
        if any(value < 0 for value in stoploss_lots):
            raise ValueError("stoploss lots must be >= 0")
        if self.profit_grid is not None:
            lower_bps = float(self.profit_grid[0])
            upper_bps = float(self.profit_grid[1])
            num_grid = int(self.profit_grid[2])
            if lower_bps < 0:
                raise ValueError("profit_grid lower_bps must be >= 0")
            if upper_bps <= lower_bps:
                raise ValueError("profit_grid upper_bps must be greater than lower_bps")
            if num_grid <= 0:
                raise ValueError("profit_grid num_grid must be > 0")
        if self.open_curve_underwater is not None:
            if not isinstance(self.open_curve_underwater, (tuple, list)) or len(self.open_curve_underwater) != 3:
                raise ValueError("open_curve_underwater must be a (min, max, order) triple when provided")
            min_scale, max_scale, order = (
                float(self.open_curve_underwater[0]),
                float(self.open_curve_underwater[1]),
                float(self.open_curve_underwater[2]),
            )
            if abs(min_scale) > 1e-12:
                raise ValueError("open_curve_underwater min must be 0")
            if max_scale < 0:
                raise ValueError("open_curve_underwater max must be >= 0")
            if order < 0:
                raise ValueError("open_curve_underwater order must be >= 0")
    @property
    def tick_size(self) -> float:
        return 10.0 ** (-self.price_precision)

    @property
    def step_size(self) -> float:
        return 10.0 ** (-self.qty_precision)

    @staticmethod
    def _normalize_max_position_lots(value: object) -> tuple[float, ...]:
        if isinstance(value, (list, tuple)):
            if not value:
                raise ValueError("max_position_usdt must not be empty")
            lots = tuple(float(item) for item in value)
        else:
            lots = (float(value),)
        if not all(math.isfinite(item) for item in lots):
            raise ValueError("max_position_usdt lots must be finite")
        return lots

    @property
    def max_position_lots(self) -> tuple[float, ...]:
        return self._normalize_max_position_lots(self.max_position_usdt)

    @property
    def total_max_position_usdt(self) -> float:
        return float(sum(self.max_position_lots))

    @staticmethod
    def _normalize_stoploss_lots(value: object) -> tuple[float, ...]:
        if isinstance(value, (list, tuple)):
            if not value:
                raise ValueError("stoploss must not be empty")
            lots = tuple(float(item) for item in value)
        else:
            lots = (float(value),)
        if not all(math.isfinite(item) for item in lots):
            raise ValueError("stoploss lots must be finite")
        return lots

    @property
    def stoploss_lots(self) -> tuple[float, ...]:
        return self._normalize_stoploss_lots(self.stoploss)


class _SingleLotMakerStrategy:
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
        self._active_profit_grid_key: tuple | None = None

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
            "new_open_lot_crit": float(self.cfg.new_open_lot_crit),
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
        self._active_profit_grid_key = None
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
        open_allowed: bool = True,
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
        ask_instructor_shift = instructor_shift
        bid_instructor_shift = instructor_shift
        if self.cfg.open_passive_only:
            ask_instructor_shift = max(0.0, ask_instructor_shift)
            bid_instructor_shift = min(0.0, bid_instructor_shift)
        skew_shift = self._inventory_skew_price_shift(mid=mid)

        ask_price = self._price_ceil(
            rounded_best_ask + distance + ask_instructor_shift + skew_shift
        )
        bid_price = self._price_floor(
            rounded_best_bid - distance + bid_instructor_shift + skew_shift
        )

        ask_qty, bid_qty = self._base_symmetric_qty(mid=mid)
        ask_qty, bid_qty = self._apply_inventory_limit(mid=mid, ask_qty=ask_qty, bid_qty=bid_qty)

        if (
            (not math.isfinite(ask_price))
            or (not math.isfinite(bid_price))
            or ask_price <= 0.0
            or bid_price <= 0.0
        ):
            return

        self._latest_best_bid = rounded_best_bid
        self._latest_best_ask = rounded_best_ask
        ask_levels, bid_levels, quote_close_only = self._maker_levels_for_ticker(
            mid=mid,
            ask_price=ask_price,
            ask_qty=ask_qty,
            bid_price=bid_price,
            bid_qty=bid_qty,
            best_ask=rounded_best_ask,
            best_bid=rounded_best_bid,
        )
        if not open_allowed:
            ask_levels, bid_levels = self._close_only_maker_levels(
                ask_levels=ask_levels,
                bid_levels=bid_levels,
            )
            quote_close_only = bool(ask_levels or bid_levels)
        self._update_max_holding_tracking(timestamp=timestamp)

        if self._should_activate_stoploss(mid=mid):
            self._reach_and_release_active = True
        if self._should_activate_max_holding_timeout(timestamp=timestamp):
            self._reach_and_release_active = True

        if self._reach_and_release_active and self._place_reach_and_release_taker(mid=mid):
            self._append_record(timestamp=timestamp, mid=mid)
            return

        self._clear_taker_books()
        profit_grid_key = (
            self._profit_grid_quote_key(ask_levels=ask_levels, bid_levels=bid_levels)
            if quote_close_only
            else None
        )
        if quote_close_only and profit_grid_key == self._active_profit_grid_key:
            self._append_record(timestamp=timestamp, mid=mid)
            return
        if quote_close_only:
            self._active_profit_grid_key = profit_grid_key
        else:
            self._active_profit_grid_key = None
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
            self._active_profit_grid_key = None
            self._clear_taker_books()
            return False

        if self._latest_best_ask is None or self._latest_best_bid is None:
            return False

        close_qty = self._close_qty_from_position(pos_qty)
        if close_qty <= 0.0:
            self._reach_and_release_active = False
            self._max_holding_start_ts = None
            self._max_holding_was_at_limit = False
            self._active_profit_grid_key = None
            self._clear_taker_books()
            return False

        self._pending_maker_quotes.clear()
        self._active_profit_grid_key = None
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
        ask_price: float,
        ask_qty: float,
        bid_price: float,
        bid_qty: float,
        best_ask: float,
        best_bid: float,
    ) -> tuple[list[MakerLevel], list[MakerLevel], bool]:
        if not self._profit_grid_enabled():
            ask_qty = self._boosted_open_qty(mid=mid, base_open_qty=ask_qty, multiplier=1.0)
            bid_qty = self._boosted_open_qty(mid=mid, base_open_qty=bid_qty, multiplier=1.0)
            return self._single_quote_levels(ask_price, ask_qty, bid_price, bid_qty, False)

        pos_qty = float(self.manager.position.qty)
        cost = float(self.manager.position.cost)
        if abs(pos_qty) <= self.EPS or cost <= 0.0 or math.isnan(cost):
            ask_qty = self._boosted_open_qty(mid=mid, base_open_qty=ask_qty, multiplier=1.0)
            bid_qty = self._boosted_open_qty(mid=mid, base_open_qty=bid_qty, multiplier=1.0)
            return self._single_quote_levels(ask_price, ask_qty, bid_price, bid_qty, False)

        if pos_qty > 0.0:
            if mid <= cost + self.EPS:
                bid_qty = self._grid_open_qty(mid=mid, base_open_qty=bid_qty)
                return self._single_quote_levels(ask_price, 0.0, bid_price, bid_qty, False)
            close_levels = self._profit_close_levels(
                mid=mid,
                best_ask=best_ask,
                best_bid=best_bid,
                pos_qty=pos_qty,
                cost=cost,
            )
            return close_levels, [], True

        if mid >= cost - self.EPS:
            ask_qty = self._grid_open_qty(mid=mid, base_open_qty=ask_qty)
            return self._single_quote_levels(ask_price, ask_qty, bid_price, 0.0, False)
        close_levels = self._profit_close_levels(
            mid=mid,
            best_ask=best_ask,
            best_bid=best_bid,
            pos_qty=pos_qty,
            cost=cost,
        )
        return [], close_levels, True

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
        return ask_levels, bid_levels, close_only

    def _profit_grid_quote_key(
        self,
        *,
        ask_levels: Sequence[MakerLevel],
        bid_levels: Sequence[MakerLevel],
    ) -> tuple:
        ask_key = tuple(float(price) for price, _ in ask_levels)
        bid_key = tuple(float(price) for price, _ in bid_levels)
        cost = float(self.manager.position.cost)
        cost_key = None if math.isnan(cost) else self._round_to_precision(cost, self.sim.price_precision)
        return ask_key, bid_key, cost_key

    def _close_only_maker_levels(
        self,
        *,
        ask_levels: Sequence[MakerLevel],
        bid_levels: Sequence[MakerLevel],
    ) -> tuple[list[MakerLevel], list[MakerLevel]]:
        pos_qty = float(self.manager.position.qty)
        if pos_qty > self.EPS:
            return list(ask_levels), []
        if pos_qty < -self.EPS:
            return [], list(bid_levels)
        return [], []

    def _profit_grid_enabled(self) -> bool:
        return self.cfg.profit_grid is not None

    def _grid_open_qty(self, mid: float, base_open_qty: float) -> float:
        if base_open_qty <= 0.0:
            return 0.0
        multiplier = self._inventory_open_curve_underwater_multiplier(mid=mid)
        return self._boosted_open_qty(mid=mid, base_open_qty=base_open_qty, multiplier=multiplier)

    def _profit_close_levels(
        self,
        *,
        mid: float,
        best_ask: float,
        best_bid: float,
        pos_qty: float,
        cost: float,
    ) -> list[MakerLevel]:
        if self.cfg.profit_grid is None:
            return []
        lower_bps, upper_bps, num_grid = self.cfg.profit_grid
        close_qty = self._close_qty_from_position(pos_qty)
        if close_qty <= 0.0:
            return []
        if self._is_past_profit_upper(mid=mid, pos_qty=pos_qty, cost=cost):
            price = self._profit_upper_release_price(
                pos_qty=pos_qty,
                best_ask=best_ask,
                best_bid=best_bid,
            )
            return [(price, close_qty)] if price > 0.0 else []

        levels = build_profit_grid_levels(
            lower_bps=float(lower_bps),
            upper_bps=float(upper_bps),
            num_grid=int(num_grid),
            position_qty=close_qty,
            cost=cost,
            is_long=pos_qty > 0.0,
            tick_size=self.sim.tick_size,
            step_size=self.sim.step_size,
            price_precision=self.sim.price_precision,
            qty_precision=self.sim.qty_precision,
            min_order_qty=float(self.cfg.min_order_qty),
            min_order_notional=float(self.cfg.min_order_notional),
            qty_distribution=self.cfg.profit_grid_qty_distribution,
        )
        return self._clip_profit_close_levels_to_bbo(
            levels=levels,
            is_long=pos_qty > 0.0,
            best_ask=best_ask,
            best_bid=best_bid,
        )

    def _clip_profit_close_levels_to_bbo(
        self,
        *,
        levels: Sequence[MakerLevel],
        is_long: bool,
        best_ask: float,
        best_bid: float,
    ) -> list[MakerLevel]:
        clipped: list[MakerLevel] = []
        for price, qty in levels:
            price = float(price)
            qty = float(qty)
            if qty <= self.EPS:
                continue
            if is_long and price < best_ask:
                price = float(best_ask)
            elif (not is_long) and price > best_bid:
                price = float(best_bid)
            clipped.append((price, qty))
        return clipped

    def _is_past_profit_upper(self, mid: float, pos_qty: float, cost: float) -> bool:
        if self.cfg.profit_grid is None:
            return False
        upper = float(self.cfg.profit_grid[1]) / 1e4
        if pos_qty > 0.0:
            return mid + self.EPS >= cost * (1.0 + upper)
        return mid <= cost * (1.0 - upper) + self.EPS

    def _profit_upper_release_price(
        self,
        *,
        pos_qty: float,
        best_ask: float,
        best_bid: float,
    ) -> float:
        if pos_qty > 0.0:
            return min(self._price_ceil(best_bid + self.sim.tick_size), float(best_ask))
        return max(self._price_floor(best_ask - self.sim.tick_size), float(best_bid))

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

    def _inventory_open_curve_underwater_multiplier(self, mid: float) -> float:
        scale = self.cfg.open_curve_underwater
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


class SimpleMakerStrategy(_SingleLotMakerStrategy):
    def __init__(
        self,
        simulation: SimulationConfig,
        strategy: Optional[SimulationConfig] = None,
        loader: Optional[BinanceEventLoader] = None,
        position: Optional[Position] = None,
    ):
        self._lot_strategies: list[_SingleLotMakerStrategy] = []
        cfg = strategy or simulation
        lots = cfg.max_position_lots
        if len(lots) <= 1:
            super().__init__(
                simulation=simulation,
                strategy=strategy,
                loader=loader,
                position=position,
            )
            return

        if position is not None:
            raise ValueError("multi-lot strategy does not support injecting a shared position")
        self.sim = simulation
        self.cfg = cfg
        self.loader = loader or BinanceEventLoader()
        self._records: list[
            tuple[int, float, float, float, float, float, float, float, float]
        ] = []
        self._lot_strategies = [
            _SingleLotMakerStrategy(
                simulation=replace(
                    simulation,
                    max_position_usdt=float(max_position_usdt),
                    stoploss=float(stoploss_usdt),
                ),
                strategy=replace(
                    cfg,
                    max_position_usdt=float(max_position_usdt),
                    stoploss=float(stoploss_usdt),
                ),
                loader=self.loader,
            )
            for max_position_usdt, stoploss_usdt in zip(lots, cfg.stoploss_lots)
        ]
        self._active_lot_count: int = 1
        self._active_lot_indices: list[int] = [0]
        self._frozen_lot_indices: list[int] = list(range(1, len(self._lot_strategies)))

    @property
    def _is_multi_lot(self) -> bool:
        return bool(self._lot_strategies)

    def run_day(self, symbol: str, date: DateLike) -> pd.DataFrame:
        if not self._is_multi_lot:
            return super().run_day(symbol=symbol, date=date)

        self._records = []
        for lot in self._lot_strategies:
            lot._records = []

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
            for lot in self._lot_strategies:
                lot._activate_pending_maker_quotes(current_timestamp=ts)
            if kind == "trade":
                self._on_trade_event(
                    trade_time=ts,
                    is_buyer_maker=bool(event[2]),
                    trade_price=float(event[3]),
                    trade_qty=float(event[4]),
                )
            else:
                best_bid = float(event[2])
                best_ask = float(event[3])
                mid = 0.5 * (best_bid + best_ask)
                self._sync_lot_stack()
                self._activate_backup_lot_if_needed(mid=mid)
                openable_idx = self._openable_lot_index(mid=mid)
                for idx, lot in enumerate(self._lot_strategies):
                    if idx >= self._active_lot_count:
                        self._cancel_inactive_flat_lot(lot)
                        continue
                    if idx != openable_idx and not self._lot_has_position(lot):
                        self._cancel_inactive_flat_lot(lot)
                        continue
                    lot._on_ticker_event(
                        timestamp=ts,
                        best_bid=best_bid,
                        best_ask=best_ask,
                        instructor_value=event[4],
                        intensity_value=event[5],
                        volatility_scalar=float(event[6]),
                        open_allowed=idx == openable_idx,
                    )
                self._sync_lot_stack()
                self._append_aggregate_record(timestamp=ts, mid=mid)

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
        if not self._is_multi_lot:
            return super().run_dates(symbol=symbol, dates=dates)
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

    def snapshot_state(self) -> dict:
        if not self._is_multi_lot:
            return super().snapshot_state()
        lots = [lot.snapshot_state() for lot in self._lot_strategies]
        state = self._aggregate_state_from_lots(lots)
        state["lots"] = lots
        state["max_position_usdt"] = float(
            sum(float(lot.cfg.max_position_usdt) for lot in self._lot_strategies)
        )
        state["max_position_lots"] = [
            float(lot.cfg.max_position_usdt) for lot in self._lot_strategies
        ]
        state["stoploss_lots"] = [
            float(lot.cfg.stoploss) for lot in self._lot_strategies
        ]
        state["new_open_lot_crit"] = float(self.cfg.new_open_lot_crit)
        state["active_lot_count"] = int(self._active_lot_count)
        state["active_lot_indices"] = list(self._active_lot_indices)
        state["frozen_lot_indices"] = list(self._frozen_lot_indices)
        return state

    def restore_state(self, state: dict) -> None:
        if not self._is_multi_lot:
            super().restore_state(state)
            return
        lot_states = state.get("lots") if isinstance(state, dict) else None
        if not isinstance(lot_states, list):
            if self._lot_strategies:
                self._lot_strategies[0].restore_state(state)
            return
        for lot, lot_state in zip(self._lot_strategies, lot_states):
            if isinstance(lot_state, dict):
                lot.restore_state(lot_state)
                self._restore_lot_config(lot, lot_state)
        self._restore_lot_stack(state)
        self._sync_lot_stack()

    def _append_aggregate_record(self, timestamp: int, mid: float) -> None:
        state = self._aggregate_state_from_lots(
            [lot.snapshot_state() for lot in self._lot_strategies]
        )
        pos = state["position"]
        self._records.append(
            (
                int(timestamp),
                float(mid),
                float(pos["qty"]),
                float(pos["mark_notional_usdt"]),
                float(pos["cost_notional_usdt"]),
                float(pos["gross_cost_notional_usdt"]),
                float(pos["realized_pnl"]),
                float(pos["unrealized_pnl"]),
                float(state["traded_volume"]),
            )
        )

    def _on_trade_event(
        self,
        trade_time: int,
        is_buyer_maker: bool,
        trade_price: float,
        trade_qty: float,
    ) -> float:
        if not self._is_multi_lot:
            return super()._on_trade_event(
                trade_time=trade_time,
                is_buyer_maker=is_buyer_maker,
                trade_price=trade_price,
                trade_qty=trade_qty,
            )

        trade_price = self._round_to_precision(float(trade_price), self.sim.price_precision)
        trade_qty = self._round_to_precision(float(trade_qty), self.sim.qty_precision)
        if trade_price <= 0.0 or trade_qty <= 0.0:
            return 0.0

        self._sync_lot_stack()
        remaining_qty = trade_qty
        filled_qty = 0.0
        for lot in self._lot_strategies[: self._active_lot_count]:
            if remaining_qty <= self.EPS:
                break
            lot_filled = lot._on_trade_event(
                trade_time=trade_time,
                is_buyer_maker=is_buyer_maker,
                trade_price=trade_price,
                trade_qty=remaining_qty,
            )
            lot_filled = min(remaining_qty, max(0.0, float(lot_filled)))
            if lot_filled <= self.EPS:
                continue
            filled_qty = self._round_to_precision(
                filled_qty + lot_filled,
                self.sim.qty_precision,
            )
            remaining_qty = self._round_to_precision(
                remaining_qty - lot_filled,
                self.sim.qty_precision,
            )

        self._sync_lot_stack()
        return float(filled_qty)

    def _lot_should_run_on_ticker(self, *, idx: int, mid: float) -> bool:
        del mid
        self._sync_lot_stack()
        if idx < 0 or idx >= self._active_lot_count:
            return False
        if idx == self._openable_lot_index(mid=mid):
            return True
        return self._lot_has_position(self._lot_strategies[idx])

    def _restore_lot_config(self, lot: _SingleLotMakerStrategy, state: dict) -> None:
        max_position_raw = state.get("max_position_usdt")
        stoploss_raw = state.get("stoploss_usdt")
        updates: dict[str, float] = {}
        try:
            if max_position_raw is not None:
                max_position_usdt = float(max_position_raw)
                if math.isfinite(max_position_usdt):
                    updates["max_position_usdt"] = max_position_usdt
            if stoploss_raw is not None:
                stoploss = float(stoploss_raw)
                if math.isfinite(stoploss):
                    updates["stoploss"] = stoploss
        except (TypeError, ValueError):
            return
        if not updates:
            return
        lot.sim = replace(lot.sim, **updates)
        lot.cfg = replace(lot.cfg, **updates)

    def _restore_lot_stack(self, state: dict) -> None:
        count_raw = state.get("active_lot_count")
        if count_raw is None:
            active_raw = state.get("active_lot_indices")
            count_raw = len(active_raw) if isinstance(active_raw, list) else 1
        try:
            active_count = int(count_raw)
        except (TypeError, ValueError):
            active_count = 1
        self._active_lot_count = active_count
        self._refresh_lot_index_state()

    def _sync_lot_pools(self) -> None:
        self._sync_lot_stack()

    def _sync_lot_stack(self) -> None:
        if not self._lot_strategies:
            self._active_lot_count = 0
            self._active_lot_indices = []
            self._frozen_lot_indices = []
            return

        self._active_lot_count = max(
            1,
            min(int(self._active_lot_count), len(self._lot_strategies)),
        )
        self._include_positioned_lots_in_active_prefix()
        self._rotate_flat_inner_lots()

        if self._outermost_nonempty_lot_index() is None:
            self._active_lot_count = 1

        for idx, lot in enumerate(self._lot_strategies):
            if idx >= self._active_lot_count:
                self._cancel_inactive_flat_lot(lot)
        self._refresh_lot_index_state()

    def _activate_backup_lot_if_needed(self, *, mid: float) -> None:
        self._sync_lot_stack()
        if self._active_lot_count >= len(self._lot_strategies):
            return
        previous_lot = self._lot_strategies[self._active_lot_count - 1]
        if not self._lot_has_position(previous_lot):
            return
        if not previous_lot._is_at_max_position():
            return
        if not self._new_open_lot_crit_reached(previous_lot, mid=mid):
            return
        self._active_lot_count += 1
        self._refresh_lot_index_state()

    def _include_positioned_lots_in_active_prefix(self) -> None:
        positioned = [
            idx
            for idx, lot in enumerate(self._lot_strategies)
            if self._lot_has_position(lot)
        ]
        if positioned:
            self._active_lot_count = max(self._active_lot_count, max(positioned) + 1)

    def _rotate_flat_inner_lots(self) -> None:
        while True:
            outer_idx = self._outermost_nonempty_lot_index()
            if outer_idx is None or outer_idx <= 0:
                return
            moved = False
            for idx in range(min(outer_idx, self._active_lot_count)):
                lot = self._lot_strategies[idx]
                if self._lot_has_position(lot):
                    continue
                self._cancel_inactive_flat_lot(lot)
                moved_lot = self._lot_strategies.pop(idx)
                self._lot_strategies.append(moved_lot)
                moved = True
                break
            if not moved:
                return

    def _refresh_lot_index_state(self) -> None:
        self._active_lot_count = max(
            0,
            min(int(self._active_lot_count), len(self._lot_strategies)),
        )
        self._active_lot_indices = list(range(self._active_lot_count))
        self._frozen_lot_indices = list(
            range(self._active_lot_count, len(self._lot_strategies))
        )

    def _openable_lot_index(self, *, mid: Optional[float] = None) -> Optional[int]:
        if not self._lot_strategies:
            return None
        outer_idx = self._outermost_nonempty_lot_index()
        if outer_idx is None:
            return 0
        backup_idx = outer_idx + 1
        if (
            backup_idx < self._active_lot_count
            and not self._lot_has_position(self._lot_strategies[backup_idx])
            and self._lot_strategies[outer_idx]._is_at_max_position()
            and self._new_open_lot_crit_reached(
                self._lot_strategies[outer_idx],
                mid=mid,
            )
        ):
            return backup_idx
        return outer_idx

    def _new_open_lot_crit_reached(
        self,
        previous_lot: _SingleLotMakerStrategy,
        *,
        mid: Optional[float],
    ) -> bool:
        crit = float(self.cfg.new_open_lot_crit)
        if crit <= self.EPS:
            return True
        if mid is None:
            return False
        mid = float(mid)
        if not math.isfinite(mid) or mid <= self.EPS:
            return False

        pos = previous_lot.manager.position
        qty = float(pos.qty)
        cost = float(pos.cost)
        if abs(qty) <= previous_lot.EPS:
            return False
        if not math.isfinite(cost) or cost <= previous_lot.EPS:
            return False

        if qty > 0.0:
            return mid <= cost * (1.0 - crit) + self.EPS
        return mid >= cost * (1.0 + crit) - self.EPS

    def _outermost_nonempty_lot_index(self) -> Optional[int]:
        upper = min(self._active_lot_count, len(self._lot_strategies))
        for idx in range(upper - 1, -1, -1):
            if self._lot_has_position(self._lot_strategies[idx]):
                return idx
        return None

    def _cancel_inactive_flat_lot(self, lot: _SingleLotMakerStrategy) -> None:
        if self._lot_has_position(lot):
            return
        lot._pending_maker_quotes.clear()
        lot._active_profit_grid_key = None
        lot._clear_maker_books()
        lot._clear_taker_books()

    @staticmethod
    def _lot_has_position(lot: _SingleLotMakerStrategy) -> bool:
        return abs(float(lot.manager.position.qty)) > lot.EPS

    @staticmethod
    def _lot_needs_next_layer(lot: _SingleLotMakerStrategy, *, mid: float) -> bool:
        if not SimpleMakerStrategy._lot_has_position(lot) or not lot._is_at_max_position():
            return False
        crit = float(lot.cfg.new_open_lot_crit)
        if crit <= lot.EPS:
            return True
        mid = float(mid)
        if not math.isfinite(mid) or mid <= lot.EPS:
            return False
        qty = float(lot.manager.position.qty)
        cost = float(lot.manager.position.cost)
        if not math.isfinite(cost) or cost <= lot.EPS:
            return False
        if qty > 0.0:
            return mid <= cost * (1.0 - crit) + lot.EPS
        return mid >= cost * (1.0 + crit) - lot.EPS

    @staticmethod
    def _aggregate_state_from_lots(lots: Sequence[dict]) -> dict:
        qty = 0.0
        mark_notional = 0.0
        cost_notional = 0.0
        gross_cost_notional = 0.0
        realized = 0.0
        unrealized = 0.0
        volume = 0.0
        latest_best_ask = None
        latest_best_bid = None
        for lot_state in lots:
            pos = lot_state.get("position") if isinstance(lot_state, dict) else {}
            if not isinstance(pos, dict):
                pos = {}
            qty += float(pos.get("qty", 0.0))
            mark_notional += float(pos.get("mark_notional_usdt", 0.0))
            cost_notional += float(pos.get("cost_notional_usdt", 0.0))
            gross_cost_notional += float(pos.get("gross_cost_notional_usdt", 0.0))
            realized += float(pos.get("realized_pnl", 0.0))
            unrealized += float(pos.get("unrealized_pnl", 0.0))
            volume += float(lot_state.get("traded_volume", 0.0))
            latest_best_ask = lot_state.get("latest_best_ask", latest_best_ask)
            latest_best_bid = lot_state.get("latest_best_bid", latest_best_bid)

        cost = None if abs(qty) <= _SingleLotMakerStrategy.EPS else cost_notional / qty
        mid = 0.0 if abs(qty) <= _SingleLotMakerStrategy.EPS else mark_notional / qty
        return {
            "position": {
                "qty": float(qty),
                "cost": cost,
                "mark_notional_usdt": float(mark_notional),
                "cost_notional_usdt": float(cost_notional),
                "gross_cost_notional_usdt": float(gross_cost_notional),
                "realized_pnl": float(realized),
                "unrealized_pnl": float(unrealized),
                "mid": float(mid),
            },
            "realized_pnl": float(realized),
            "unrealized_pnl": float(unrealized),
            "total_pnl": float(realized + unrealized),
            "traded_volume": float(volume),
            "latest_best_ask": latest_best_ask,
            "latest_best_bid": latest_best_bid,
        }
