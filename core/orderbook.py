from __future__ import annotations

from collections import deque
from enum import StrEnum
from typing import Iterable, Iterator


PriceLevel = tuple[int, int]


class OrderSide(StrEnum):
    BUY = "buy"
    SELL = "sell"

    @property
    def is_aggressor_buy(self) -> bool:
        return self == OrderSide.BUY

    @property
    def position_sign(self) -> int:
        return 1 if self == OrderSide.BUY else -1


class OrderBook:
    """
    Single-side price-level book backed by a deque.

    - increasing=True: ask-side ordering, best level at head, low -> high
    - increasing=False: bid-side ordering, best level at head, high -> low
    - mode=0: inclusive threshold matching, <= / >=
    - mode=1: strict threshold matching, < / >
    """

    INCLUSIVE = 0
    STRICT = 1

    def __init__(
        self,
        levels: Iterable[PriceLevel] | None = None,
        mode: int = STRICT,
        increasing: bool = True,
    ) -> None:
        if mode not in (self.INCLUSIVE, self.STRICT):
            raise ValueError(f"mode must be {self.INCLUSIVE} or {self.STRICT}, got {mode}")

        self.deque: deque[PriceLevel] = deque()
        self.increasing = increasing
        self.mode = mode
        if levels is not None:
            self.merge(levels)

    def __bool__(self) -> bool:
        return bool(self.deque)

    def __len__(self) -> int:
        return len(self.deque)

    def __iter__(self) -> Iterator[PriceLevel]:
        return iter(self.deque)

    @staticmethod
    def _as_int(value: int, field: str) -> int:
        if isinstance(value, bool):
            raise ValueError(f"{field} must be integer, got {value}")

        ivalue = int(value)
        if ivalue != value:
            raise ValueError(f"{field} must be integer, got {value}")
        return ivalue

    def _sort_key(self, price: int) -> int:
        return price if self.increasing else -price

    def _is_head_matchable(self, price: int, threshold: int) -> bool:
        if self.increasing:
            return price <= threshold if self.mode == self.INCLUSIVE else price < threshold
        return price >= threshold if self.mode == self.INCLUSIVE else price > threshold

    def _is_tail_removable(self, price: int, threshold: int) -> bool:
        if self.increasing:
            return price >= threshold if self.mode == self.INCLUSIVE else price > threshold
        return price <= threshold if self.mode == self.INCLUSIVE else price < threshold

    def clear(self) -> None:
        self.deque.clear()

    def best_level(self) -> PriceLevel | None:
        if not self.deque:
            return None
        return self.deque[0]

    def snapshot(self) -> list[PriceLevel]:
        return list(self.deque)

    def merge(self, levels: Iterable[PriceLevel]) -> OrderBook:
        """
        Merge price levels into the book.

        Positive qty is added to the existing quantity at that price.
        Qty <= 0 removes the whole price level.
        """
        by_price = dict(self.deque)

        for raw_price, raw_qty in levels:
            price = self._as_int(raw_price, "price_tick")
            qty = self._as_int(raw_qty, "qty_steps")
            if qty <= 0:
                by_price.pop(price, None)
            else:
                by_price[price] = by_price.get(price, 0) + qty

        self.deque = deque(sorted(by_price.items(), key=lambda item: self._sort_key(item[0])))
        return self

    def delete_head(self, threshold: int, trade_qty: int) -> list[PriceLevel]:
        """
        Match from the best level while it crosses threshold.

        Returns filled (price_tick, qty_steps) levels.
        """
        fills: list[PriceLevel] = []
        threshold = self._as_int(threshold, "threshold_tick")
        remaining = max(self._as_int(trade_qty, "trade_qty_steps"), 0)

        while self.deque and remaining > 0:
            price, qty = self.deque[0]
            if not self._is_head_matchable(price, threshold):
                break

            matched = min(remaining, qty)
            fills.append((price, matched))
            remaining -= matched

            left = qty - matched
            if left <= 0:
                self.deque.popleft()
            else:
                self.deque[0] = (price, left)

        return fills

    def delete_tail(self, threshold: int, count: int) -> list[PriceLevel]:
        """
        Remove at most count levels from the worst side while they cross threshold.

        Returns removed full price levels.
        """
        removed: list[PriceLevel] = []
        threshold = self._as_int(threshold, "threshold_tick")
        remaining = max(self._as_int(count, "count"), 0)

        while self.deque and remaining > 0:
            price, qty = self.deque[-1]
            if not self._is_tail_removable(price, threshold):
                break

            removed.append((price, qty))
            self.deque.pop()
            remaining -= 1

        return removed
