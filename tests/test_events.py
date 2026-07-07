import unittest

from core.events import (
    AlphaEvent,
    InstructorEvent,
    IntensityEvent,
    TickerEvent,
    VolatilityEvent,
)


class AlphaEventTest(unittest.TestCase):
    def test_alpha_event_carries_prediction_value(self):
        event = AlphaEvent(
            timestamp=1000,
            ticker_event=TickerEvent(
                timestamp=1000,
                symbol="BTCUSDT",
                bid_price=99.0,
                bid_size=1.0,
                ask_price=101.0,
                ask_size=2.0,
            ),
            instructor_event=InstructorEvent(timestamp=1000, instructor=0.1),
            intensity_event=IntensityEvent(timestamp=1000, intensity=0.2),
            volatility_event=VolatilityEvent(timestamp=1000, volatility=0.3),
            prediction=0.4,
        )

        self.assertEqual(event.prediction, 0.4)

    def test_alpha_event_prediction_defaults_to_zero(self):
        event = AlphaEvent(
            timestamp=1000,
            ticker_event=TickerEvent(
                timestamp=1000,
                symbol="BTCUSDT",
                bid_price=99.0,
                bid_size=1.0,
                ask_price=101.0,
                ask_size=2.0,
            ),
            instructor_event=InstructorEvent(timestamp=1000, instructor=0.1),
            intensity_event=IntensityEvent(timestamp=1000, intensity=0.2),
            volatility_event=VolatilityEvent(timestamp=1000, volatility=0.3),
        )

        self.assertEqual(event.prediction, 0.0)
