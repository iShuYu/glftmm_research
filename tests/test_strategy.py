import unittest

from core.position import Position
from sim.report import _normalize_report_columns
from sim.strategy import (
    SimpleMakerStrategy,
    SimulationConfig,
)
from run_strategy import (
    _build_tasks,
    _build_simulation_config,
    _normalize_sim_param_map,
    _state_cost_notional_usdt,
    _state_total_pnl,
)


def make_config(**overrides):
    values = {
        "freq": 1000,
        "latency": 0,
        "price_precision": 1,
        "qty_precision": 3,
        "mode": 1,
        "taker_fee": 0.00025,
        "maker_fee": -0.00003,
        "name_intensity": "k",
        "lookback_intensity": 300,
        "name_volatility": "sigma",
        "lookback_volatility": 300,
        "order_amt": 100.0,
        "max_position_usdt": 100000.0,
        "max_holding_time": -1,
        "adj_spread_intensity": 1.0,
        "adj_spread_instructor": 0.0,
        "passive_only": False,
        "adj_spread_volatility": 0.0,
        "inventory_skew": (0.0, 1.0),
        "min_order_qty": 0.0,
        "min_order_notional": 0.0,
        "stoploss": 0.0,
        "open_curve": (0.0, 1.0, 0.0),
        "close_curve": (0.0, 1.0, 0.0),
    }
    values.update(overrides)
    return SimulationConfig(**values)


def raw_config(**overrides):
    values = {
        "freq": 1000,
        "latency": 0,
        "price_precision": 1,
        "qty_precision": 3,
        "mode": 1,
        "taker_fee": 0.00025,
        "maker_fee": -0.00003,
        "name_intensity": "k",
        "lookback_intensity": 300,
        "name_volatility": "sigma",
        "lookback_volatility": 300,
        "order_amt": 100.0,
        "max_position_usdt": 100000.0,
        "max_holding_time": -1,
        "adj_spread_intensity": 1.0,
        "adj_spread_instructor": 0.0,
        "passive_only": False,
        "adj_spread_volatility": 0.0,
        "min_order_qty": 0.0,
        "min_order_notional": 0.0,
        "stoploss": 0.0,
    }
    values.update(overrides)
    return {"simulation": values}


def raw_task_config(**overrides):
    cfg = raw_config(**overrides)
    cfg.update(
        {
            "symbols": ["BTCUSDT"],
            "date_start": "2025-01-01",
            "date_end": "2025-01-01",
        }
    )
    return cfg


def set_position(engine, qty, cost):
    engine.manager.position.qty = qty
    engine.manager.position.cost = cost
    engine.manager._sync_position_steps()


def price_levels(engine, book):
    return [
        (
            round(engine.manager.converter.from_ticks(price), engine.sim.price_precision),
            round(engine.manager.converter.from_steps(qty), engine.sim.qty_precision),
        )
        for price, qty in book.snapshot()
    ]


class StaticEventLoader:
    def __init__(self, events):
        self.events = list(events)

    def iter_merged_alpha_trade_tuples(self, **kwargs):
        del kwargs
        yield from self.events


class PositionNotionalTest(unittest.TestCase):
    def test_position_tracks_mark_and_cost_notional_separately(self):
        pos = Position()
        pos.execute(qty=2.0, price=100.0)
        pos.mark(110.0)

        self.assertEqual(pos.mark_notional_usdt, 220.0)
        self.assertEqual(pos.cost_notional_usdt, 200.0)
        self.assertEqual(pos.gross_cost_notional_usdt, 200.0)
        self.assertEqual(pos.unrealized_pnl, 20.0)

        pos.mark(80.0)
        self.assertEqual(pos.mark_notional_usdt, 160.0)
        self.assertEqual(pos.cost_notional_usdt, 200.0)
        self.assertEqual(pos.unrealized_pnl, -40.0)

    def test_short_notional_preserves_sign(self):
        pos = Position()
        pos.execute(qty=-2.0, price=100.0)
        pos.mark(90.0)

        self.assertEqual(pos.mark_notional_usdt, -180.0)
        self.assertEqual(pos.cost_notional_usdt, -200.0)
        self.assertEqual(pos.gross_cost_notional_usdt, 200.0)
        self.assertEqual(pos.unrealized_pnl, 20.0)


class StrategyConfigTest(unittest.TestCase):
    def test_build_config_accepts_single_max_position_and_stoploss(self):
        cfg = _build_simulation_config(
            raw_config(
                max_position_usdt=250.0,
                stoploss=25.0,
            )["simulation"]
        )

        self.assertEqual(cfg.max_position_usdt, 250.0)
        self.assertEqual(cfg.total_max_position_usdt, 250.0)
        self.assertEqual(cfg.stoploss, 25.0)

    def test_rejects_lot_shaped_max_position_and_stoploss(self):
        with self.assertRaisesRegex(ValueError, "max_position_usdt must be a number"):
            _build_simulation_config(
                raw_config(
                    max_position_usdt=[250.0, 750.0],
                    stoploss=25.0,
                )["simulation"]
            )

        with self.assertRaisesRegex(ValueError, "stoploss must be a number"):
            _build_simulation_config(
                raw_config(
                    max_position_usdt=250.0,
                    stoploss=[25.0, 75.0],
                )["simulation"]
            )

    def test_normalizes_max_position_and_stoploss_as_parameter_grids(self):
        sim_map = _normalize_sim_param_map(
            raw_config(
                max_position_usdt=[250.0, 750.0],
                stoploss=[25.0, 75.0],
            )
        )

        self.assertEqual(sim_map["max_position_usdt"], [250.0, 750.0])
        self.assertEqual(sim_map["stoploss"], [25.0, 75.0])

    def test_build_config_parses_close_curve(self):
        cfg = _build_simulation_config(
            raw_config(close_curve=[0.0, 2.0, 1.0])["simulation"]
        )

        self.assertEqual(cfg.close_curve, (0.0, 2.0, 1.0))

    def test_normalizes_close_curve_rows(self):
        sim_map = _normalize_sim_param_map(
            raw_config(close_curve=[[0.0, 1.0, 0.0], [0.0, 2.0, 1.0]])
        )

        self.assertEqual(
            sim_map["close_curve"],
            [[0.0, 1.0, 0.0], [0.0, 2.0, 1.0]],
        )

    def test_rejects_legacy_curve_names(self):
        with self.assertRaisesRegex(ValueError, "unknown simulation keys"):
            _normalize_sim_param_map(
                raw_config(open_curve_underwater=[[0.0, 1.0, 0.0]])
            )

        with self.assertRaisesRegex(ValueError, "unknown simulation keys"):
            _normalize_sim_param_map(
                raw_config(close_curve_above_water=[[0.0, 1.0, 0.0]])
            )

    def test_build_tasks_crosses_max_position_and_stoploss_grids(self):
        tasks = _build_tasks(
            raw_task_config(
                mode=[0, 1],
                max_position_usdt=[250.0, 500.0],
                stoploss=[50.0, 100.0],
            )
        )

        self.assertEqual(len(tasks), 8)
        pairs = {
            (
                task.sim_params["max_position_usdt"],
                task.sim_params["stoploss"],
            )
            for task in tasks
        }
        self.assertEqual(
            pairs,
            {
                (250.0, 50.0),
                (250.0, 100.0),
                (500.0, 50.0),
                (500.0, 100.0),
            },
        )
        self.assertEqual([task.sim_params["mode"] for task in tasks].count(0), 4)
        self.assertEqual([task.sim_params["mode"] for task in tasks].count(1), 4)

    def test_build_config_parses_instructor_alpha_adjustment(self):
        cfg = _build_simulation_config(
            raw_config(
                name_instructor="trade_imbalance",
                lookback_instructor=5,
                adj_spread_instructor=-1e-5,
                passive_only=True,
            )["simulation"]
        )

        self.assertEqual(cfg.name_instructor, "trade_imbalance")
        self.assertEqual(cfg.lookback_instructor, 5)
        self.assertEqual(cfg.adj_spread_instructor, -1e-5)
        self.assertTrue(cfg.passive_only)

    def test_rejects_legacy_passive_only_name(self):
        with self.assertRaisesRegex(ValueError, "unknown simulation keys"):
            _normalize_sim_param_map(raw_config(open_passive_only=[True]))

    def test_build_config_allows_bbo_imbalance_zero_lookback(self):
        cfg = _build_simulation_config(
            raw_config(
                name_instructor="bbo_imbalance",
                lookback_instructor=0,
            )["simulation"]
        )

        self.assertEqual(cfg.name_instructor, "bbo_imbalance")
        self.assertEqual(cfg.lookback_instructor, 0)

class StrategyQuoteTest(unittest.TestCase):
    def test_instructor_shifts_open_quotes_in_mid_price_units(self):
        engine = SimpleMakerStrategy(make_config(adj_spread_instructor=0.002))

        engine._on_ticker_event(
            timestamp=1,
            best_bid=99.8,
            best_ask=100.2,
            intensity_value=0.0,
            volatility_scalar=0.0,
            instructor_value=0.5,
        )

        self.assertEqual(
            price_levels(engine, engine.manager.books.ask_maker),
            [(100.3, 1.0)],
        )
        self.assertEqual(
            price_levels(engine, engine.manager.books.bid_maker),
            [(99.9, 1.0)],
        )

    def test_passive_only_clips_alpha_shift_crossing_ask(self):
        engine = SimpleMakerStrategy(
            make_config(adj_spread_instructor=-0.002, passive_only=True)
        )

        engine._on_ticker_event(
            timestamp=1,
            best_bid=99.8,
            best_ask=100.2,
            intensity_value=0.0,
            volatility_scalar=0.0,
            instructor_value=0.5,
        )

        self.assertEqual(
            price_levels(engine, engine.manager.books.ask_maker),
            [(100.2, 1.0)],
        )
        self.assertEqual(
            price_levels(engine, engine.manager.books.bid_maker),
            [(99.7, 1.0)],
        )

    def test_passive_only_clips_alpha_shift_crossing_bid(self):
        engine = SimpleMakerStrategy(
            make_config(adj_spread_instructor=0.002, passive_only=True)
        )

        engine._on_ticker_event(
            timestamp=1,
            best_bid=99.8,
            best_ask=100.2,
            intensity_value=0.0,
            volatility_scalar=0.0,
            instructor_value=0.5,
        )

        self.assertEqual(
            price_levels(engine, engine.manager.books.ask_maker),
            [(100.3, 1.0)],
        )
        self.assertEqual(
            price_levels(engine, engine.manager.books.bid_maker),
            [(99.8, 1.0)],
        )

    def test_passive_only_ignores_tightening_alpha_without_clipping_distance(self):
        engine = SimpleMakerStrategy(
            make_config(adj_spread_instructor=0.002, passive_only=True)
        )

        engine._on_ticker_event(
            timestamp=1,
            best_bid=99.8,
            best_ask=100.2,
            intensity_value=0.2,
            volatility_scalar=0.0,
            instructor_value=0.5,
        )

        self.assertEqual(
            price_levels(engine, engine.manager.books.ask_maker),
            [(100.5, 1.0)],
        )
        self.assertEqual(
            price_levels(engine, engine.manager.books.bid_maker),
            [(99.6, 1.0)],
        )

    def test_passive_only_ignores_tightening_ask_alpha_without_clipping_distance(self):
        engine = SimpleMakerStrategy(
            make_config(adj_spread_instructor=-0.002, passive_only=True)
        )

        engine._on_ticker_event(
            timestamp=1,
            best_bid=99.8,
            best_ask=100.2,
            intensity_value=0.2,
            volatility_scalar=0.0,
            instructor_value=0.5,
        )

        self.assertEqual(
            price_levels(engine, engine.manager.books.ask_maker),
            [(100.4, 1.0)],
        )
        self.assertEqual(
            price_levels(engine, engine.manager.books.bid_maker),
            [(99.5, 1.0)],
        )

    def test_instructor_shifts_close_quotes_in_mid_price_units(self):
        engine = SimpleMakerStrategy(make_config(adj_spread_instructor=0.002))
        set_position(engine, qty=1.0, cost=100.0)

        engine._on_ticker_event(
            timestamp=1,
            best_bid=100.4,
            best_ask=100.6,
            intensity_value=0.0,
            volatility_scalar=0.0,
            instructor_value=0.5,
        )

        self.assertEqual(price_levels(engine, engine.manager.books.bid_maker), [])
        self.assertEqual(
            price_levels(engine, engine.manager.books.ask_maker),
            [(100.8, 1.0)],
        )

    def test_passive_only_clips_close_ask_tightening_alpha(self):
        engine = SimpleMakerStrategy(
            make_config(adj_spread_instructor=-0.002, passive_only=True)
        )
        set_position(engine, qty=1.0, cost=100.0)

        engine._on_ticker_event(
            timestamp=1,
            best_bid=100.4,
            best_ask=100.6,
            intensity_value=0.0,
            volatility_scalar=0.0,
            instructor_value=0.5,
        )

        self.assertEqual(price_levels(engine, engine.manager.books.bid_maker), [])
        self.assertEqual(price_levels(engine, engine.manager.books.ask_maker), [(100.6, 1.0)])

    def test_passive_only_clips_close_bid_tightening_alpha(self):
        engine = SimpleMakerStrategy(
            make_config(adj_spread_instructor=0.004, passive_only=True)
        )
        set_position(engine, qty=-1.0, cost=100.0)

        engine._on_ticker_event(
            timestamp=1,
            best_bid=99.4,
            best_ask=99.6,
            intensity_value=0.0,
            volatility_scalar=0.0,
            instructor_value=0.5,
        )

        self.assertEqual(price_levels(engine, engine.manager.books.ask_maker), [])
        self.assertEqual(price_levels(engine, engine.manager.books.bid_maker), [(99.4, 1.0)])

    def test_inventory_limit_uses_cost_notional_not_mark_notional(self):
        engine = SimpleMakerStrategy(make_config(max_position_usdt=150.0))
        set_position(engine, qty=1.0, cost=100.0)

        ask_qty, bid_qty = engine._apply_inventory_limit(mid=200.0, ask_qty=1.0, bid_qty=1.0)
        self.assertEqual((ask_qty, bid_qty), (1.0, 1.0))

        engine = SimpleMakerStrategy(make_config(max_position_usdt=150.0))
        set_position(engine, qty=1.0, cost=200.0)

        ask_qty, bid_qty = engine._apply_inventory_limit(mid=100.0, ask_qty=1.0, bid_qty=1.0)
        self.assertEqual((ask_qty, bid_qty), (1.0, 0.0))

    def test_short_inventory_limit_uses_signed_cost_notional(self):
        engine = SimpleMakerStrategy(make_config(max_position_usdt=150.0))
        set_position(engine, qty=-1.0, cost=200.0)

        ask_qty, bid_qty = engine._apply_inventory_limit(mid=100.0, ask_qty=1.0, bid_qty=1.0)
        self.assertEqual((ask_qty, bid_qty), (0.0, 1.0))

    def test_inventory_skew_uses_cost_notional_not_mark_notional(self):
        engine = SimpleMakerStrategy(
            make_config(max_position_usdt=100.0, inventory_skew=(10.0, 1.0))
        )
        set_position(engine, qty=1.0, cost=50.0)

        self.assertEqual(engine._inventory_skew_price_shift(mid=100.0), -0.5)

    def test_open_curve_uses_gross_cost_notional(self):
        engine = SimpleMakerStrategy(
            make_config(
                max_position_usdt=100.0,
                open_curve=(0.0, 2.0, 1.0),
            )
        )
        set_position(engine, qty=1.0, cost=25.0)

        self.assertEqual(engine._inventory_open_curve_multiplier(mid=100.0), 1.5)

    def test_open_curve_allows_nonzero_min_scale(self):
        engine = SimpleMakerStrategy(
            make_config(
                max_position_usdt=100.0,
                open_curve=(0.25, 2.0, 1.0),
            )
        )
        set_position(engine, qty=1.0, cost=100.0)

        self.assertEqual(engine._inventory_open_curve_multiplier(mid=100.0), 0.25)

    def test_close_curve_uses_gross_cost_notional(self):
        engine = SimpleMakerStrategy(
            make_config(
                max_position_usdt=100.0,
                close_curve=(0.0, 2.0, 1.0),
            )
        )
        set_position(engine, qty=1.0, cost=25.0)

        self.assertEqual(engine._inventory_close_curve_multiplier(mid=100.0), 0.5)

    def test_close_curve_allows_nonzero_min_scale(self):
        engine = SimpleMakerStrategy(
            make_config(
                max_position_usdt=100.0,
                close_curve=(0.25, 2.0, 1.0),
            )
        )

        self.assertEqual(engine._inventory_close_curve_multiplier(mid=100.0), 0.25)

    def test_close_curve_qty_does_not_cap_to_position_qty(self):
        engine = SimpleMakerStrategy(
            make_config(
                max_position_usdt=100.0,
                close_curve=(0.0, 2.0, 1.0),
            )
        )
        set_position(engine, qty=1.0, cost=100.0)

        self.assertEqual(engine._curve_close_qty(mid=100.0, pos_qty=1.0), 2.0)

    def test_stoploss_is_absolute_usdt(self):
        engine = SimpleMakerStrategy(make_config(max_position_usdt=100.0, stoploss=25.0))
        set_position(engine, qty=1.0, cost=100.0)

        self.assertFalse(engine._should_activate_stoploss(mid=75.1))
        self.assertTrue(engine._should_activate_stoploss(mid=75.0))

    def test_long_under_cost_only_places_open_bid(self):
        engine = SimpleMakerStrategy(make_config())
        set_position(engine, qty=1.0, cost=100.0)

        engine._on_ticker_event(
            timestamp=1,
            best_bid=98.9,
            best_ask=99.1,
            intensity_value=0.0,
            volatility_scalar=0.0,
        )

        self.assertEqual(price_levels(engine, engine.manager.books.ask_maker), [])
        self.assertEqual(len(price_levels(engine, engine.manager.books.bid_maker)), 1)

    def test_long_above_cost_places_glftmm_close_ask_only(self):
        engine = SimpleMakerStrategy(make_config())
        set_position(engine, qty=1.0, cost=100.0)

        engine._on_ticker_event(
            timestamp=1,
            best_bid=100.4,
            best_ask=100.6,
            intensity_value=0.0,
            volatility_scalar=0.0,
        )

        self.assertEqual(price_levels(engine, engine.manager.books.bid_maker), [])
        self.assertEqual(price_levels(engine, engine.manager.books.ask_maker), [(100.6, 1.0)])

    def test_long_above_cost_close_curve_scales_normal_close_qty(self):
        engine = SimpleMakerStrategy(
            make_config(
                max_position_usdt=400.0,
                close_curve=(0.0, 1.0, 1.0),
            )
        )
        set_position(engine, qty=1.0, cost=100.0)

        engine._on_ticker_event(
            timestamp=1,
            best_bid=100.4,
            best_ask=100.6,
            intensity_value=0.0,
            volatility_scalar=0.0,
        )

        self.assertEqual(price_levels(engine, engine.manager.books.bid_maker), [])
        self.assertEqual(price_levels(engine, engine.manager.books.ask_maker), [(100.6, 0.25)])

    def test_close_curve_qty_is_at_least_min_order_qty(self):
        engine = SimpleMakerStrategy(
            make_config(
                max_position_usdt=1000.0,
                min_order_qty=0.2,
                close_curve=(0.0, 0.1, 1.0),
            )
        )
        set_position(engine, qty=1.0, cost=100.0)

        engine._on_ticker_event(
            timestamp=1,
            best_bid=100.4,
            best_ask=100.6,
            intensity_value=0.0,
            volatility_scalar=0.0,
        )

        self.assertEqual(price_levels(engine, engine.manager.books.bid_maker), [])
        self.assertEqual(price_levels(engine, engine.manager.books.ask_maker), [(100.6, 0.2)])

    def test_close_curve_allows_full_close_when_position_below_min_order_qty(self):
        engine = SimpleMakerStrategy(
            make_config(
                min_order_qty=0.2,
                close_curve=(0.0, 0.0, 1.0),
            )
        )
        set_position(engine, qty=0.05, cost=100.0)

        engine._on_ticker_event(
            timestamp=1,
            best_bid=100.4,
            best_ask=100.6,
            intensity_value=0.0,
            volatility_scalar=0.0,
        )

        self.assertEqual(price_levels(engine, engine.manager.books.bid_maker), [])
        self.assertEqual(price_levels(engine, engine.manager.books.ask_maker), [(100.6, 0.05)])

    def test_long_close_quote_uses_intensity_and_volatility_distance(self):
        engine = SimpleMakerStrategy(make_config(adj_spread_volatility=2.0))
        set_position(engine, qty=1.0, cost=100.0)

        engine._on_ticker_event(
            timestamp=1,
            best_bid=100.4,
            best_ask=100.6,
            intensity_value=0.2,
            volatility_scalar=0.1,
        )

        self.assertEqual(price_levels(engine, engine.manager.books.bid_maker), [])
        self.assertEqual(price_levels(engine, engine.manager.books.ask_maker), [(101.0, 1.0)])

    def test_long_far_above_cost_still_uses_glftmm_close_ask(self):
        engine = SimpleMakerStrategy(make_config())
        set_position(engine, qty=1.0, cost=100.0)

        engine._on_ticker_event(
            timestamp=1,
            best_bid=103.4,
            best_ask=103.5,
            intensity_value=0.0,
            volatility_scalar=0.0,
        )

        self.assertEqual(price_levels(engine, engine.manager.books.bid_maker), [])
        self.assertEqual(price_levels(engine, engine.manager.books.ask_maker), [(103.5, 1.0)])

    def test_short_min_step_residual_closes_in_profit_zone(self):
        engine = SimpleMakerStrategy(make_config())
        set_position(engine, qty=-0.0009999999999817438, cost=100.0)

        engine._on_ticker_event(
            timestamp=1,
            best_bid=99.7,
            best_ask=99.8,
            intensity_value=0.0,
            volatility_scalar=0.0,
        )

        self.assertEqual(price_levels(engine, engine.manager.books.ask_maker), [])
        self.assertEqual(price_levels(engine, engine.manager.books.bid_maker), [(99.7, 0.001)])

    def test_reach_and_release_closes_min_step_residual(self):
        engine = SimpleMakerStrategy(make_config())
        set_position(engine, qty=0.0009999999999817438, cost=100.0)
        engine._latest_best_bid = 99.9
        engine._latest_best_ask = 100.0

        placed = engine._place_reach_and_release_taker(mid=99.95)

        self.assertTrue(placed)
        self.assertEqual(price_levels(engine, engine.manager.books.ask_taker), [(99.9, 0.001)])
        self.assertEqual(price_levels(engine, engine.manager.books.bid_taker), [])

    def test_reach_and_release_ignores_close_curve(self):
        engine = SimpleMakerStrategy(
            make_config(close_curve=(0.0, 0.0, 1.0))
        )
        set_position(engine, qty=1.0, cost=100.0)
        engine._latest_best_bid = 99.9
        engine._latest_best_ask = 100.0

        placed = engine._place_reach_and_release_taker(mid=99.95)

        self.assertTrue(placed)
        self.assertEqual(price_levels(engine, engine.manager.books.ask_taker), [(99.9, 1.0)])
        self.assertEqual(price_levels(engine, engine.manager.books.bid_taker), [])

    def test_unchanged_close_quote_is_recomputed_each_ticker(self):
        engine = SimpleMakerStrategy(make_config())
        set_position(engine, qty=1.0, cost=100.0)

        engine._on_ticker_event(
            timestamp=1,
            best_bid=100.4,
            best_ask=100.6,
            intensity_value=0.0,
            volatility_scalar=0.0,
        )
        engine.manager.books.ask_maker.merge([(9999, 1)])
        engine._on_ticker_event(
            timestamp=2,
            best_bid=100.4,
            best_ask=100.6,
            intensity_value=0.0,
            volatility_scalar=0.0,
        )

        self.assertEqual(price_levels(engine, engine.manager.books.ask_maker), [(100.6, 1.0)])

    def test_partial_profit_fill_requotes_remaining_close_qty(self):
        engine = SimpleMakerStrategy(make_config())
        set_position(engine, qty=1.0, cost=100.0)

        engine._on_ticker_event(
            timestamp=1,
            best_bid=100.4,
            best_ask=100.6,
            intensity_value=0.0,
            volatility_scalar=0.0,
        )
        filled_qty = engine.manager.match_maker(
            trade_time=2,
            trade_side=True,
            trade_price=100.7,
            trade_qty=0.334,
            fee_rate=engine.sim.maker_fee,
        )
        self.assertAlmostEqual(filled_qty, 0.334)
        before = price_levels(engine, engine.manager.books.ask_maker)

        engine._on_ticker_event(
            timestamp=3,
            best_bid=100.4,
            best_ask=100.6,
            intensity_value=0.0,
            volatility_scalar=0.0,
        )

        self.assertEqual(price_levels(engine, engine.manager.books.ask_maker), before)

    def test_short_below_cost_places_glftmm_close_bid_only(self):
        engine = SimpleMakerStrategy(make_config())
        set_position(engine, qty=-1.0, cost=100.0)

        engine._on_ticker_event(
            timestamp=1,
            best_bid=99.4,
            best_ask=99.6,
            intensity_value=0.0,
            volatility_scalar=0.0,
        )

        self.assertEqual(price_levels(engine, engine.manager.books.ask_maker), [])
        self.assertEqual(price_levels(engine, engine.manager.books.bid_maker), [(99.4, 1.0)])

    def test_short_close_quote_uses_intensity_and_volatility_distance(self):
        engine = SimpleMakerStrategy(make_config(adj_spread_volatility=2.0))
        set_position(engine, qty=-1.0, cost=100.0)

        engine._on_ticker_event(
            timestamp=1,
            best_bid=99.4,
            best_ask=99.6,
            intensity_value=0.2,
            volatility_scalar=0.1,
        )

        self.assertEqual(price_levels(engine, engine.manager.books.ask_maker), [])
        self.assertEqual(price_levels(engine, engine.manager.books.bid_maker), [(99.0, 1.0)])

    def test_single_strategy_snapshot_has_no_lot_state(self):
        engine = SimpleMakerStrategy(make_config())

        engine._on_ticker_event(
            timestamp=1,
            best_bid=99.9,
            best_ask=100.1,
            intensity_value=0.0,
            volatility_scalar=0.0,
        )

        state = engine.snapshot_state()
        self.assertNotIn("lots", state)
        self.assertNotIn("max_position_lots", state)
        self.assertNotIn("stoploss_lots", state)
        self.assertFalse(hasattr(engine, "_lot_strategies"))
        self.assertLessEqual(len(engine.manager.books.ask_maker.snapshot()), 1)
        self.assertLessEqual(len(engine.manager.books.bid_maker.snapshot()), 1)

    def test_requotes_replace_single_level_per_side(self):
        engine = SimpleMakerStrategy(make_config())

        engine._on_ticker_event(
            timestamp=1,
            best_bid=99.9,
            best_ask=100.1,
            intensity_value=0.0,
            volatility_scalar=0.0,
        )
        first_ask = price_levels(engine, engine.manager.books.ask_maker)
        first_bid = price_levels(engine, engine.manager.books.bid_maker)

        engine._on_ticker_event(
            timestamp=2,
            best_bid=99.9,
            best_ask=100.1,
            intensity_value=0.2,
            volatility_scalar=0.0,
        )

        self.assertEqual(len(price_levels(engine, engine.manager.books.ask_maker)), 1)
        self.assertEqual(len(price_levels(engine, engine.manager.books.bid_maker)), 1)
        self.assertNotEqual(price_levels(engine, engine.manager.books.ask_maker), first_ask)
        self.assertNotEqual(price_levels(engine, engine.manager.books.bid_maker), first_bid)

    def test_rejects_multi_position_config_at_strategy_boundary(self):
        with self.assertRaisesRegex(ValueError, "max_position_usdt must be a single number"):
            SimpleMakerStrategy(
                make_config(max_position_usdt=[150.0, 300.0], stoploss=0.0),
            )

    def test_flat_strategy_run_uses_one_position_book(self):
        engine = SimpleMakerStrategy(
            make_config(max_position_usdt=150.0, stoploss=0.0),
            loader=StaticEventLoader(
                [
                    ("ticker", 1, 99.9, 100.1, 0.0, 0.0, 0.0),
                    ("trade", 2, True, 99.8, 1.0),
                    ("ticker", 3, 89.9, 90.1, 0.0, 0.0, 0.0),
                    ("trade", 4, True, 89.8, 2.0),
                    ("ticker", 5, 89.9, 90.1, 0.0, 0.0, 0.0),
                    ("trade", 6, True, 89.8, 1.0),
                    ("ticker", 7, 89.9, 90.1, 0.0, 0.0, 0.0),
                ]
            ),
        )

        df = engine.run_day(symbol="BTCUSDT", date="2025-01-01")
        state = engine.snapshot_state()

        self.assertEqual(state["max_position_usdt"], 150.0)
        self.assertGreaterEqual(state["position"]["gross_cost_notional_usdt"], 150.0)
        self.assertEqual(float(df.iloc[1]["position"]), 1.0)
        self.assertAlmostEqual(float(df.iloc[-1]["position"]), state["position"]["qty"])


class ReportNotionalTest(unittest.TestCase):
    def test_report_inventory_is_cost_notional_for_legacy_columns(self):
        import pandas as pd

        df = pd.DataFrame(
            {
                "timestamp": [1, 2],
                "price": [110.0, 90.0],
                "position": [2.0, -2.0],
                "realized_pnl": [0.0, 0.0],
                "unrealized_pnl": [20.0, 20.0],
            }
        )

        out = _normalize_report_columns(df)

        self.assertEqual(list(out["mark_notional_usdt"]), [220.0, -180.0])
        self.assertEqual(list(out["cost_notional_usdt"]), [200.0, -200.0])
        self.assertEqual(list(out["gross_cost_notional_usdt"]), [200.0, 200.0])
        self.assertEqual(list(out["inventory"]), [200.0, -200.0])

    def test_state_cost_notional_prefers_state_field_and_falls_back_to_cost(self):
        self.assertEqual(
            _state_cost_notional_usdt(
                {"position": {"qty": 2.0, "cost": 100.0, "cost_notional_usdt": 201.0}}
            ),
            201.0,
        )
        self.assertEqual(
            _state_cost_notional_usdt({"position": {"qty": -2.0, "cost": 100.0}}),
            -200.0,
        )


if __name__ == "__main__":
    unittest.main()
