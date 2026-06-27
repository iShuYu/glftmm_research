from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

from .orderbook import OrderBook, OrderSide, PriceLevel
from .position import EPS, Position


@dataclass(frozen=True, slots=True)
class SymbolRules:
    tick_size: float = 1.0
    step_size: float = 1.0
    min_qty: float = 0.0
    min_notional: float = 0.0

    def __post_init__(self) -> None:
        if self.tick_size <= 0.0 or self.step_size <= 0.0:
            raise ValueError("tick_size and step_size must be > 0")
        if self.min_qty < 0.0 or self.min_notional < 0.0:
            raise ValueError("min_qty and min_notional must be >= 0")


class PriceConverter:
    """Price/quantity conversion between floats and integer ticks/steps."""

    def __init__(self, rules: SymbolRules, eps: float = EPS) -> None:
        self.rules = rules
        self.eps = eps

    def to_ticks(self, price: float, strict: bool = True) -> int:
        if math.isnan(price) or price <= 0.0:
            raise ValueError(f"invalid price: {price}")

        ticks = price / self.rules.tick_size
        result = int(round(ticks))
        if strict and abs(ticks - result) > self.eps:
            raise ValueError(f"price {price} is not a multiple of tick_size {self.rules.tick_size}")
        return result

    def to_steps(self, qty: float, strict: bool = True, rounding: str = "floor") -> int:
        if math.isnan(qty):
            raise ValueError("qty is NaN")

        steps = abs(qty) / self.rules.step_size
        if strict:
            result = int(round(steps))
            if abs(steps - result) > self.eps:
                raise ValueError(f"qty {qty} is not a multiple of step_size {self.rules.step_size}")
            return int(math.copysign(result, qty))

        if rounding == "floor":
            result = int(math.floor(steps + self.eps))
        elif rounding == "ceil":
            result = int(math.ceil(steps - self.eps))
        elif rounding == "round":
            result = int(round(steps))
        else:
            raise ValueError(f"unsupported rounding mode: {rounding}")

        return int(math.copysign(result, qty))

    def from_ticks(self, ticks: int) -> float:
        return ticks * self.rules.tick_size

    def from_steps(self, steps: int) -> float:
        return steps * self.rules.step_size

    def notional(self, price_ticks: int, qty_steps: int) -> float:
        return self.from_ticks(price_ticks) * abs(self.from_steps(qty_steps))


class OrderBookManager:
    """Own maker/taker books for one symbol."""

    def __init__(self, converter: PriceConverter, mode: int = OrderBook.STRICT) -> None:
        self.converter = converter
        self.ask_maker = OrderBook(increasing=True, mode=mode)
        self.bid_maker = OrderBook(increasing=False, mode=mode)
        self.ask_taker = OrderBook(increasing=True, mode=mode)
        self.bid_taker = OrderBook(increasing=False, mode=mode)

    def clear_all(self) -> None:
        self.ask_maker.clear()
        self.bid_maker.clear()
        self.ask_taker.clear()
        self.bid_taker.clear()

    def _select_book(self, is_ask: bool, is_taker: bool) -> OrderBook:
        if is_taker:
            return self.ask_taker if is_ask else self.bid_taker
        return self.ask_maker if is_ask else self.bid_maker

    def add_order(self, is_ask: bool, is_taker: bool, price_ticks: int, qty_steps: int) -> None:
        if qty_steps <= 0:
            return
        self._select_book(is_ask, is_taker).merge([(price_ticks, qty_steps)])

    def replace_order(
        self,
        is_ask: bool,
        is_taker: bool,
        price_ticks: int | None,
        qty_steps: int,
    ) -> None:
        book = self._select_book(is_ask, is_taker)
        if price_ticks is None or qty_steps <= 0:
            book.replace_single(None)
            return
        book.replace_single((price_ticks, qty_steps))

    def clear_side(self, is_ask: bool) -> None:
        self._select_book(is_ask, is_taker=False).clear()
        self._select_book(is_ask, is_taker=True).clear()

    def match_order(
        self,
        is_ask: bool,
        is_taker: bool,
        price_ticks: int,
        qty_steps: int,
    ) -> list[PriceLevel]:
        if qty_steps <= 0:
            return []
        return self._select_book(is_ask, is_taker).delete_head(price_ticks, qty_steps)

    def snapshot(self) -> dict[str, list[PriceLevel]]:
        return {
            "ask_maker": self.ask_maker.snapshot(),
            "bid_maker": self.bid_maker.snapshot(),
            "ask_taker": self.ask_taker.snapshot(),
            "bid_taker": self.bid_taker.snapshot(),
        }

    def restore_snapshot(self, snapshot: dict[str, list[PriceLevel]]) -> None:
        self.clear_all()
        books = {
            "ask_maker": self.ask_maker,
            "bid_maker": self.bid_maker,
            "ask_taker": self.ask_taker,
            "bid_taker": self.bid_taker,
        }
        for name, book in books.items():
            levels = snapshot.get(name, [])
            valid_levels: list[PriceLevel] = []
            for price, qty in levels:
                if qty > 0:
                    valid_levels.append((int(price), int(qty)))
            if valid_levels:
                book.merge(valid_levels)


class OrderValidator:
    """Exchange filters plus close-only position filtering."""

    def __init__(self, converter: PriceConverter, rules: SymbolRules) -> None:
        self.converter = converter
        self.rules = rules
        self.min_qty_steps = converter.to_steps(rules.min_qty, strict=False, rounding="ceil")

    def validate_order(
        self,
        is_ask: bool,
        price_ticks: int,
        qty_steps: int,
        position_steps: int,
        close_only: bool,
    ) -> int:
        if qty_steps <= 0:
            return 0

        if is_ask and position_steps > 0:
            closable = position_steps
        elif not is_ask and position_steps < 0:
            closable = -position_steps
        else:
            closable = 0

        close_steps = min(qty_steps, closable)
        open_steps = qty_steps - close_steps
        if close_only:
            return close_steps

        if open_steps > 0 and self.min_qty_steps > 0 and open_steps < self.min_qty_steps:
            open_steps = 0

        if open_steps > 0 and self.rules.min_notional > 0.0:
            notional = self.converter.notional(price_ticks, open_steps)
            if notional + self.converter.eps < self.rules.min_notional:
                open_steps = 0

        return close_steps + open_steps

    def is_aggressive_price(
        self,
        is_ask: bool,
        price_ticks: int,
        best_ask_ticks: int,
        best_bid_ticks: int,
    ) -> bool:
        if is_ask:
            return price_ticks <= best_bid_ticks
        return price_ticks >= best_ask_ticks


class Manager:
    """
    Matching state for one symbol.

    It tracks own maker/taker limit books, validates order size/notional, and
    applies matched fills to Position.
    """

    def __init__(
        self,
        position: Position | None = None,
        mode: int = OrderBook.STRICT,
        symbol_rules: dict[str, Any] | SymbolRules | None = None,
    ) -> None:
        if isinstance(symbol_rules, SymbolRules):
            self.rules = symbol_rules
        elif symbol_rules is None:
            self.rules = SymbolRules()
        else:
            self.rules = SymbolRules(**symbol_rules)

        self.converter = PriceConverter(self.rules)
        self.validator = OrderValidator(self.converter, self.rules)
        self.books = OrderBookManager(self.converter, mode)
        self.position = position or Position()
        self.position_steps = 0
        self._sync_position_steps()

    def _sync_position_steps(self) -> None:
        self.position_steps = self.converter.to_steps(
            self.position.qty,
            strict=False,
            rounding="round",
        )
        if self.position_steps == 0:
            self.position.qty = 0.0

    def _emit_fill(
        self,
        trade_time: int,
        delta_steps: int,
        price_ticks: int,
        fee_rate: float,
    ) -> None:
        if delta_steps == 0:
            return

        _ = trade_time
        delta_qty = self.converter.from_steps(delta_steps)
        price = self.converter.from_ticks(price_ticks)
        self.position.execute(delta_qty, price, fee_rate=fee_rate)
        self._sync_position_steps()

    def place_limit_order(
        self,
        ask_price: float | None,
        ask_qty: float,
        bid_price: float | None,
        bid_qty: float,
        best_ask: float,
        best_bid: float,
        is_taker: bool = False,
        close_only: bool = False,
    ) -> None:
        best_ask_ticks = self.converter.to_ticks(best_ask)
        best_bid_ticks = self.converter.to_ticks(best_bid)

        self._place_one_side(
            is_ask=True,
            price=ask_price,
            qty=ask_qty,
            best_ask_ticks=best_ask_ticks,
            best_bid_ticks=best_bid_ticks,
            is_taker=is_taker,
            close_only=close_only,
        )
        self._place_one_side(
            is_ask=False,
            price=bid_price,
            qty=bid_qty,
            best_ask_ticks=best_ask_ticks,
            best_bid_ticks=best_bid_ticks,
            is_taker=is_taker,
            close_only=close_only,
        )

    def place_limit_levels(
        self,
        ask_levels: Sequence[tuple[float, float]],
        bid_levels: Sequence[tuple[float, float]],
        best_ask: float,
        best_bid: float,
        is_taker: bool = False,
        close_only: bool = False,
    ) -> None:
        best_ask_ticks = self.converter.to_ticks(best_ask)
        best_bid_ticks = self.converter.to_ticks(best_bid)

        for ask_price, ask_qty in ask_levels:
            self._place_one_side(
                is_ask=True,
                price=ask_price,
                qty=ask_qty,
                best_ask_ticks=best_ask_ticks,
                best_bid_ticks=best_bid_ticks,
                is_taker=is_taker,
                close_only=close_only,
            )
        for bid_price, bid_qty in bid_levels:
            self._place_one_side(
                is_ask=False,
                price=bid_price,
                qty=bid_qty,
                best_ask_ticks=best_ask_ticks,
                best_bid_ticks=best_bid_ticks,
                is_taker=is_taker,
                close_only=close_only,
            )

    def place_limit_single_levels(
        self,
        ask_levels: Sequence[tuple[float, float]],
        bid_levels: Sequence[tuple[float, float]],
        best_ask: float,
        best_bid: float,
        is_taker: bool = False,
        close_only: bool = False,
    ) -> None:
        ask_levels = list(ask_levels)
        bid_levels = list(bid_levels)
        if len(ask_levels) > 1 or len(bid_levels) > 1:
            raise ValueError("single-level placement supports at most one level per side")

        best_ask_ticks = self.converter.to_ticks(best_ask)
        best_bid_ticks = self.converter.to_ticks(best_bid)
        ask_price, ask_qty = ask_levels[0] if ask_levels else (None, 0.0)
        bid_price, bid_qty = bid_levels[0] if bid_levels else (None, 0.0)

        self._place_one_side(
            is_ask=True,
            price=ask_price,
            qty=ask_qty,
            best_ask_ticks=best_ask_ticks,
            best_bid_ticks=best_bid_ticks,
            is_taker=is_taker,
            close_only=close_only,
            replace=True,
        )
        self._place_one_side(
            is_ask=False,
            price=bid_price,
            qty=bid_qty,
            best_ask_ticks=best_ask_ticks,
            best_bid_ticks=best_bid_ticks,
            is_taker=is_taker,
            close_only=close_only,
            replace=True,
        )

    def _place_one_side(
        self,
        is_ask: bool,
        price: float | None,
        qty: float,
        best_ask_ticks: int,
        best_bid_ticks: int,
        is_taker: bool,
        close_only: bool,
        replace: bool = False,
    ) -> None:
        if price is None or qty <= EPS:
            if replace:
                self.books.clear_side(is_ask)
            return

        price_ticks = self.converter.to_ticks(price)
        qty_steps = self.converter.to_steps(qty)
        qty_steps = self.validator.validate_order(
            is_ask=is_ask,
            price_ticks=price_ticks,
            qty_steps=qty_steps,
            position_steps=self.position_steps,
            close_only=close_only,
        )
        if replace:
            self.books.clear_side(is_ask)
        if qty_steps <= 0:
            return

        is_aggressive = self.validator.is_aggressive_price(
            is_ask,
            price_ticks,
            best_ask_ticks,
            best_bid_ticks,
        )
        if replace:
            self.books.replace_order(
                is_ask=is_ask,
                is_taker=is_taker or is_aggressive,
                price_ticks=price_ticks,
                qty_steps=qty_steps,
            )
        else:
            self.books.add_order(
                is_ask=is_ask,
                is_taker=is_taker or is_aggressive,
                price_ticks=price_ticks,
                qty_steps=qty_steps,
            )

    def _place_one_side_steps(
        self,
        is_ask: bool,
        price_ticks: int | None,
        qty_steps: int,
        best_ask_ticks: int,
        best_bid_ticks: int,
        is_taker: bool,
        close_only: bool,
        replace: bool,
    ) -> None:
        if price_ticks is None or qty_steps <= 0:
            if replace:
                self.books.clear_side(is_ask)
            return

        qty_steps = int(qty_steps)
        if close_only:
            if is_ask and self.position_steps > 0:
                qty_steps = min(qty_steps, self.position_steps)
            elif (not is_ask) and self.position_steps < 0:
                qty_steps = min(qty_steps, -self.position_steps)
            else:
                qty_steps = 0

        if replace:
            self.books.clear_side(is_ask)
        if qty_steps <= 0:
            return

        is_aggressive = self.validator.is_aggressive_price(
            is_ask,
            int(price_ticks),
            best_ask_ticks,
            best_bid_ticks,
        )
        self.books.replace_order(
            is_ask=is_ask,
            is_taker=is_taker or is_aggressive,
            price_ticks=int(price_ticks),
            qty_steps=qty_steps,
        )

    def place_limit_order_steps(
        self,
        ask_price_ticks: int | None,
        ask_qty_steps: int,
        bid_price_ticks: int | None,
        bid_qty_steps: int,
        best_ask_ticks: int,
        best_bid_ticks: int,
        is_taker: bool = False,
        close_only: bool = False,
        replace: bool = True,
    ) -> None:
        self._place_one_side_steps(
            is_ask=True,
            price_ticks=ask_price_ticks,
            qty_steps=ask_qty_steps,
            best_ask_ticks=best_ask_ticks,
            best_bid_ticks=best_bid_ticks,
            is_taker=is_taker,
            close_only=close_only,
            replace=replace,
        )
        self._place_one_side_steps(
            is_ask=False,
            price_ticks=bid_price_ticks,
            qty_steps=bid_qty_steps,
            best_ask_ticks=best_ask_ticks,
            best_bid_ticks=best_bid_ticks,
            is_taker=is_taker,
            close_only=close_only,
            replace=replace,
        )

    def place_maker(
        self,
        ask_price: float | None,
        ask_qty: float,
        bid_price: float | None,
        bid_qty: float,
        best_ask: float,
        best_bid: float,
        close_only: bool = False,
    ) -> None:
        self.place_limit_order(
            ask_price,
            ask_qty,
            bid_price,
            bid_qty,
            best_ask,
            best_bid,
            is_taker=False,
            close_only=close_only,
        )

    def place_maker_steps(
        self,
        ask_price_ticks: int | None,
        ask_qty_steps: int,
        bid_price_ticks: int | None,
        bid_qty_steps: int,
        best_ask_ticks: int,
        best_bid_ticks: int,
        close_only: bool = False,
    ) -> None:
        self.place_limit_order_steps(
            ask_price_ticks=ask_price_ticks,
            ask_qty_steps=ask_qty_steps,
            bid_price_ticks=bid_price_ticks,
            bid_qty_steps=bid_qty_steps,
            best_ask_ticks=best_ask_ticks,
            best_bid_ticks=best_bid_ticks,
            is_taker=False,
            close_only=close_only,
        )

    def place_maker_levels(
        self,
        ask_levels: Sequence[tuple[float, float]],
        bid_levels: Sequence[tuple[float, float]],
        best_ask: float,
        best_bid: float,
        close_only: bool = False,
    ) -> None:
        self.place_limit_levels(
            ask_levels=ask_levels,
            bid_levels=bid_levels,
            best_ask=best_ask,
            best_bid=best_bid,
            is_taker=False,
            close_only=close_only,
        )

    def place_maker_single_levels(
        self,
        ask_levels: Sequence[tuple[float, float]],
        bid_levels: Sequence[tuple[float, float]],
        best_ask: float,
        best_bid: float,
        close_only: bool = False,
    ) -> None:
        self.place_limit_single_levels(
            ask_levels=ask_levels,
            bid_levels=bid_levels,
            best_ask=best_ask,
            best_bid=best_bid,
            is_taker=False,
            close_only=close_only,
        )

    def place_taker(
        self,
        ask_price: float | None,
        ask_qty: float,
        bid_price: float | None,
        bid_qty: float,
        best_ask: float,
        best_bid: float,
        close_only: bool = False,
    ) -> None:
        self.place_limit_order(
            ask_price,
            ask_qty,
            bid_price,
            bid_qty,
            best_ask,
            best_bid,
            is_taker=True,
            close_only=close_only,
        )

    def match_order(
        self,
        trade_time: int,
        side: OrderSide | str,
        trade_price: float,
        trade_qty: float,
        fee_rate: float,
        is_taker: bool = True,
    ) -> float:
        fills = self.match_order_fills(
            trade_time=trade_time,
            side=side,
            trade_price=trade_price,
            trade_qty=trade_qty,
            fee_rate=fee_rate,
            is_taker=is_taker,
        )
        return sum(qty for _, qty in fills)

    def match_order_fills(
        self,
        trade_time: int,
        side: OrderSide | str,
        trade_price: float,
        trade_qty: float,
        fee_rate: float,
        is_taker: bool = True,
    ) -> list[tuple[float, float]]:
        side = OrderSide(side)
        price_ticks = self.converter.to_ticks(trade_price)
        qty_steps = self.converter.to_steps(trade_qty)
        if qty_steps <= 0:
            return []

        is_ask = side == OrderSide.SELL
        fills = self.books.match_order(
            is_ask=is_ask,
            is_taker=is_taker,
            price_ticks=price_ticks,
            qty_steps=qty_steps,
        )

        result: list[tuple[float, float]] = []
        for fill_price_ticks, fill_qty_steps in fills:
            self._emit_fill(
                trade_time=trade_time,
                delta_steps=side.position_sign * fill_qty_steps,
                price_ticks=fill_price_ticks,
                fee_rate=fee_rate,
            )
            result.append(
                (
                    self.converter.from_ticks(fill_price_ticks),
                    self.converter.from_steps(fill_qty_steps),
                )
            )

        return result

    def match_taker(
        self,
        trade_time: int,
        trade_side: bool,
        trade_price: float,
        trade_qty: float,
        fee_rate: float,
    ) -> float:
        side = OrderSide.SELL if trade_side else OrderSide.BUY
        return self.match_order(trade_time, side, trade_price, trade_qty, fee_rate, is_taker=True)

    def match_taker_fills(
        self,
        trade_time: int,
        trade_side: bool,
        trade_price: float,
        trade_qty: float,
        fee_rate: float,
    ) -> list[tuple[float, float]]:
        side = OrderSide.SELL if trade_side else OrderSide.BUY
        return self.match_order_fills(
            trade_time,
            side,
            trade_price,
            trade_qty,
            fee_rate,
            is_taker=True,
        )

    def match_maker(
        self,
        trade_time: int,
        trade_side: bool,
        trade_price: float,
        trade_qty: float,
        fee_rate: float,
    ) -> float:
        side = OrderSide.SELL if trade_side else OrderSide.BUY
        return self.match_order(trade_time, side, trade_price, trade_qty, fee_rate, is_taker=False)

    def match_maker_fills(
        self,
        trade_time: int,
        trade_side: bool,
        trade_price: float,
        trade_qty: float,
        fee_rate: float,
    ) -> list[tuple[float, float]]:
        side = OrderSide.SELL if trade_side else OrderSide.BUY
        return self.match_order_fills(
            trade_time,
            side,
            trade_price,
            trade_qty,
            fee_rate,
            is_taker=False,
        )

    def snapshot(self) -> dict[str, list[tuple[float, float]]]:
        return {
            key: [
                (self.converter.from_ticks(price), self.converter.from_steps(qty))
                for price, qty in levels
            ]
            for key, levels in self.books.snapshot().items()
        }

    def snapshot_ticks(self) -> dict[str, list[PriceLevel]]:
        return self.books.snapshot()

    def restore_snapshot_ticks(self, snapshot: dict[str, list[PriceLevel]]) -> None:
        self.books.restore_snapshot(snapshot)
