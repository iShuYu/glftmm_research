from __future__ import annotations

import math


EPS = 1e-8
NAN = math.nan


class Position:
    """
    Fast position tracker with NaN-as-flat cost semantics.

    Positive qty is long, negative qty is short. The execute qty uses the same
    sign convention: positive buys, negative sells.
    """

    __slots__ = ("qty", "cost", "realized_pnl", "mid")

    def __init__(self) -> None:
        self.qty: float = 0.0
        self.cost: float = NAN
        self.realized_pnl: float = 0.0
        self.mid: float = 0.0

    @property
    def is_flat(self) -> bool:
        return self.qty == 0.0

    @property
    def unrealized_pnl(self) -> float:
        return self.mark_notional_usdt - self.cost_notional_usdt

    @property
    def mark_notional_usdt(self) -> float:
        return 0.0 if self.qty == 0.0 else self.qty * self.mid

    @property
    def cost_notional_usdt(self) -> float:
        if self.qty == 0.0 or not math.isfinite(self.cost):
            return 0.0
        return self.qty * self.cost

    @property
    def gross_cost_notional_usdt(self) -> float:
        return abs(self.cost_notional_usdt)

    def mark(self, mid_price: float) -> None:
        self.mid = mid_price

    def execute(self, qty: float, price: float, fee_rate: float = 0.0) -> float:
        """
        Execute a signed trade and return realized PnL from the closed quantity.

        Assumes abs(qty) > EPS. Fee is deducted from cumulative realized_pnl,
        but the returned fill PnL excludes fee.
        """
        fee = fee_rate * abs(qty) * price
        self.realized_pnl -= fee

        old_qty = self.qty
        new_qty = old_qty + qty

        if old_qty == 0.0:
            self.qty = qty
            self.cost = price
            return 0.0

        if old_qty * qty > 0.0:
            old_abs = old_qty if old_qty > 0.0 else -old_qty
            add_abs = qty if qty > 0.0 else -qty
            self.cost = (old_abs * self.cost + add_abs * price) / (old_abs + add_abs)
            self.qty = new_qty
            return 0.0

        close_qty = min(abs(old_qty), abs(qty))
        realized = close_qty * (price - self.cost)
        if old_qty < 0.0:
            realized = -realized

        self.realized_pnl += realized

        if abs(new_qty) <= EPS:
            self.qty = 0.0
            self.cost = NAN
            return realized

        self.qty = new_qty
        if old_qty * new_qty < 0.0:
            self.cost = price

        return realized
