from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


Side = Literal["buy", "sell"]


@dataclass(frozen=True, slots=True)
class BaseEvent:
    timestamp: int


@dataclass(frozen=True, slots=True)
class TickerEvent(BaseEvent):
    symbol: str
    bid_price: float
    bid_size: float
    ask_price: float
    ask_size: float

    @property
    def mid_price(self) -> float:
        return (self.bid_price + self.ask_price) / 2.0


@dataclass(frozen=True, slots=True)
class TradeEvent(BaseEvent):
    symbol: str
    price: float
    size: float
    side: Side
    trade_id: str | None = None


@dataclass(frozen=True, slots=True)
class IntensityEvent(BaseEvent):
    intensity: float


@dataclass(frozen=True, slots=True)
class InstructorEvent(BaseEvent):
    instructor: float


@dataclass(frozen=True, slots=True)
class VolatilityEvent(BaseEvent):
    volatility: float


@dataclass(frozen=True, slots=True)
class AlphaEvent(BaseEvent):
    ticker_event: TickerEvent
    instructor_event: InstructorEvent
    intensity_event: IntensityEvent
    volatility_event: VolatilityEvent
    prediction: float = 0.0
