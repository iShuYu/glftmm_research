import math
import unittest

from core.position import Position
from sim.report import _normalize_report_columns
from sim.strategy import (
    SimpleMakerStrategy,
    SimulationConfig,
    build_profit_grid_levels,
)
from run_strategy import (
    _build_tasks,
    _build_simulation_config,
    _normalize_sim_param_map,
    scheme_shift_path_component,
    _state_cost_notional_usdt,
    _state_realized_pnl,
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
        "new_open_lot_crit": 0.0,
        "max_holding_time": -1,
        "adj_spread_intensity": 1.0,
        "adj_spread_instructor": 0.0,
        "open_passive_only": False,
        "adj_spread_volatility": 0.0,
        "inventory_skew": (0.0, 1.0),
        "min_order_qty": 0.0,
        "min_order_notional": 0.0,
        "stoploss": 0.0,
        "open_curve_underwater": (0.0, 1.0, 0.0),
        "profit_grid": (100.0, 300.0, 3),
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
        "new_open_lot_crit": 0.0,
        "max_holding_time": -1,
        "adj_spread_intensity": 1.0,
        "adj_spread_instructor": 0.0,
        "open_passive_only": False,
        "adj_spread_volatility": 0.0,
        "min_order_qty": 0.0,
        "min_order_notional": 0.0,
        "stoploss": 0.0,
        "profit_grid": [100.0, 300.0, 3],
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


class ProfitGridAllocationTest(unittest.TestCase):
    def test_long_grid_equal_split_puts_remainder_on_nearest_level(self):
        levels = build_profit_grid_levels(
            lower_bps=100.0,
            upper_bps=300.0,
            num_grid=3,
            position_qty=1.003,
            cost=100.0,
            is_long=True,
            tick_size=0.1,
            step_size=0.001,
            price_precision=1,
            qty_precision=3,
            min_order_qty=0.0,
            min_order_notional=0.0,
        )

        self.assertEqual(levels, [(101.0, 0.335), (102.0, 0.334), (103.0, 0.334)])

    def test_small_position_goes_to_nearest_grid(self):
        levels = build_profit_grid_levels(
            lower_bps=100.0,
            upper_bps=300.0,
            num_grid=3,
            position_qty=0.025,
            cost=100.0,
            is_long=True,
            tick_size=0.1,
            step_size=0.001,
            price_precision=1,
            qty_precision=3,
            min_order_qty=0.0,
            min_order_notional=10.0,
        )

        self.assertEqual(levels, [(101.0, 0.025)])

    def test_power_distribution_sizes_near_levels_more_heavily(self):
        levels = build_profit_grid_levels(
            lower_bps=100.0,
            upper_bps=300.0,
            num_grid=3,
            position_qty=1.4,
            cost=100.0,
            is_long=True,
            tick_size=0.1,
            step_size=0.001,
            price_precision=1,
            qty_precision=3,
            min_order_qty=0.0,
            min_order_notional=0.0,
            qty_distribution=("power", 2.0),
        )

        self.assertEqual(levels, [(101.0, 0.9), (102.0, 0.4), (103.0, 0.1)])

    def test_exponential_distribution_decays_near_to_far(self):
        levels = build_profit_grid_levels(
            lower_bps=100.0,
            upper_bps=300.0,
            num_grid=3,
            position_qty=1.75,
            cost=100.0,
            is_long=True,
            tick_size=0.1,
            step_size=0.001,
            price_precision=1,
            qty_precision=3,
            min_order_qty=0.0,
            min_order_notional=0.0,
            qty_distribution=("exponential", math.log(2.0)),
        )

        self.assertEqual(levels, [(101.0, 1.0), (102.0, 0.5), (103.0, 0.25)])

    def test_invalid_later_level_rolls_remaining_to_nearest_grid(self):
        levels = build_profit_grid_levels(
            lower_bps=100.0,
            upper_bps=300.0,
            num_grid=3,
            position_qty=0.4,
            cost=100.0,
            is_long=True,
            tick_size=0.1,
            step_size=0.001,
            price_precision=1,
            qty_precision=3,
            min_order_qty=0.0,
            min_order_notional=10.0,
            qty_distribution=("power", 2.0),
        )

        self.assertEqual(levels, [(101.0, 0.286), (102.0, 0.114)])

    def test_short_grid_prices_move_down_from_cost(self):
        levels = build_profit_grid_levels(
            lower_bps=100.0,
            upper_bps=300.0,
            num_grid=3,
            position_qty=0.999,
            cost=100.0,
            is_long=False,
            tick_size=0.1,
            step_size=0.001,
            price_precision=1,
            qty_precision=3,
            min_order_qty=0.0,
            min_order_notional=0.0,
        )

        self.assertEqual(levels, [(99.0, 0.333), (98.0, 0.333), (97.0, 0.333)])


class ProfitGridConfigTest(unittest.TestCase):
    def test_normalizes_qty_distribution_search_values(self):
        sim_map = _normalize_sim_param_map(
            raw_config(
                profit_grid_qty_distribution=[
                    ["equal"],
                    ["quadratic"],
                    ["exponential", 0.5],
                ]
            )
        )

        self.assertEqual(
            sim_map["profit_grid_qty_distribution"],
            [["equal", 1.0], ["power", 2.0], ["exponential", 0.5]],
        )

    def test_build_config_parses_qty_distribution(self):
        cfg = _build_simulation_config(
            raw_config(profit_grid_qty_distribution=["power", 3.0])["simulation"]
        )

        self.assertEqual(cfg.profit_grid_qty_distribution.kind, "power")
        self.assertEqual(cfg.profit_grid_qty_distribution.param, 3.0)

    def test_normalizes_embedded_profit_grid_qty_distribution(self):
        sim_map = _normalize_sim_param_map(
            raw_config(
                profit_grid=[
                    [0.0, 50.0, 10, "equal"],
                    [0.0, 10.0, 10, "exponential", 0.1],
                ]
            )
        )

        self.assertEqual(
            sim_map["profit_grid"],
            [
                [0.0, 50.0, 10, "equal", 1.0],
                [0.0, 10.0, 10, "exponential", 0.1],
            ],
        )

    def test_build_config_parses_embedded_profit_grid_qty_distribution(self):
        cfg = _build_simulation_config(
            raw_config(profit_grid=[0.0, 10.0, 10, "exponential", 0.1])["simulation"]
        )

        self.assertEqual(cfg.profit_grid, (0.0, 10.0, 10))
        self.assertEqual(cfg.profit_grid_qty_distribution.kind, "exponential")
        self.assertEqual(cfg.profit_grid_qty_distribution.param, 0.1)

    def test_build_config_parses_max_position_lots(self):
        cfg = _build_simulation_config(
            raw_config(
                max_position_usdt=[250.0, 750.0],
                stoploss=[25.0, 75.0],
            )["simulation"]
        )

        self.assertEqual(cfg.max_position_lots, (250.0, 750.0))
        self.assertEqual(cfg.total_max_position_usdt, 1000.0)
        self.assertEqual(cfg.stoploss_lots, (25.0, 75.0))

    def test_build_config_parses_new_open_lot_crit(self):
        cfg = _build_simulation_config(
            raw_config(new_open_lot_crit=0.01)["simulation"]
        )

        self.assertEqual(cfg.new_open_lot_crit, 0.01)

    def test_normalizes_max_position_and_stoploss_lists_as_lot_groups(self):
        sim_map = _normalize_sim_param_map(
            raw_config(
                max_position_usdt=[250.0, 750.0],
                stoploss=[25.0, 75.0],
            )
        )

        self.assertEqual(sim_map["max_position_usdt"], [[250.0, 750.0]])
        self.assertEqual(sim_map["stoploss"], [[25.0, 75.0]])

    def test_normalizes_vectorized_lot_groups_as_rows(self):
        sim_map = _normalize_sim_param_map(
            raw_config(
                max_position_usdt=[[250.0, 250.0], [500.0]],
                stoploss=[[50.0, 50.0], [100.0]],
            )
        )

        self.assertEqual(sim_map["max_position_usdt"], [[250.0, 250.0], [500.0]])
        self.assertEqual(sim_map["stoploss"], [[50.0, 50.0], [100.0]])

    def test_build_tasks_pairs_lot_rows_before_outer_product(self):
        tasks = _build_tasks(
            raw_task_config(
                mode=[0, 1],
                max_position_usdt=[[250.0, 250.0], [500.0]],
                stoploss=[[50.0, 50.0], [100.0]],
            )
        )

        self.assertEqual(len(tasks), 4)
        pairs = {
            (
                tuple(task.sim_params["max_position_usdt"]),
                tuple(task.sim_params["stoploss"]),
            )
            for task in tasks
        }
        self.assertEqual(
            pairs,
            {
                ((250.0, 250.0), (50.0, 50.0)),
                ((500.0,), (100.0,)),
            },
        )
        self.assertEqual([task.sim_params["mode"] for task in tasks].count(0), 2)
        self.assertEqual([task.sim_params["mode"] for task in tasks].count(1), 2)
        self.assertEqual({task.scheme_shift for task in tasks}, {0})

        crit_tasks = _build_tasks(raw_task_config(new_open_lot_crit=[0.0, 0.01]))
        self.assertEqual(
            [task.sim_params["new_open_lot_crit"] for task in crit_tasks],
            [0.0, 0.01],
        )

        shifted_cfg = raw_task_config()
        shifted_cfg["scheme_shift"] = 250
        shifted_tasks = _build_tasks(shifted_cfg)
        self.assertEqual([task.scheme_shift for task in shifted_tasks], [250])
        self.assertEqual(scheme_shift_path_component(250), "scheme_shift_250ms")

    def test_vectorized_lot_row_counts_must_match(self):
        with self.assertRaisesRegex(ValueError, "same number of vectorized rows"):
            _build_tasks(
                raw_task_config(
                    max_position_usdt=[[250.0, 250.0], [500.0]],
                    stoploss=[[50.0, 50.0]],
                )
            )

    def test_vectorized_lot_lengths_must_match_per_row(self):
        with self.assertRaisesRegex(ValueError, "row 1"):
            _build_tasks(
                raw_task_config(
                    max_position_usdt=[[250.0, 250.0], [500.0]],
                    stoploss=[[50.0], [100.0]],
                )
            )

    def test_stoploss_and_max_position_lot_lengths_must_match(self):
        with self.assertRaises(ValueError):
            _build_simulation_config(
                raw_config(
                    max_position_usdt=[250.0, 750.0],
                    stoploss=[25.0],
                )["simulation"]
            )

    def test_build_config_parses_instructor_alpha_adjustment(self):
        cfg = _build_simulation_config(
            raw_config(
                name_instructor="trade_imbalance",
                lookback_instructor=5,
                adj_spread_instructor=-1e-5,
                open_passive_only=True,
            )["simulation"]
        )

        self.assertEqual(cfg.name_instructor, "trade_imbalance")
        self.assertEqual(cfg.lookback_instructor, 5)
        self.assertEqual(cfg.adj_spread_instructor, -1e-5)
        self.assertTrue(cfg.open_passive_only)

    def test_build_config_allows_bbo_imbalance_zero_lookback(self):
        cfg = _build_simulation_config(
            raw_config(
                name_instructor="bbo_imbalance",
                lookback_instructor=0,
            )["simulation"]
        )

        self.assertEqual(cfg.name_instructor, "bbo_imbalance")
        self.assertEqual(cfg.lookback_instructor, 0)

class ProfitGridStrategyTest(unittest.TestCase):
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

    def test_open_passive_only_clips_alpha_shift_crossing_ask(self):
        engine = SimpleMakerStrategy(
            make_config(adj_spread_instructor=-0.002, open_passive_only=True)
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

    def test_open_passive_only_clips_alpha_shift_crossing_bid(self):
        engine = SimpleMakerStrategy(
            make_config(adj_spread_instructor=0.002, open_passive_only=True)
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

    def test_open_passive_only_ignores_tightening_alpha_without_clipping_distance(self):
        engine = SimpleMakerStrategy(
            make_config(adj_spread_instructor=0.002, open_passive_only=True)
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

    def test_open_passive_only_ignores_tightening_ask_alpha_without_clipping_distance(self):
        engine = SimpleMakerStrategy(
            make_config(adj_spread_instructor=-0.002, open_passive_only=True)
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

    def test_instructor_does_not_move_profit_grid_close_levels(self):
        engine = SimpleMakerStrategy(make_config(adj_spread_instructor=-0.01))
        set_position(engine, qty=1.0, cost=100.0)

        engine._on_ticker_event(
            timestamp=1,
            best_bid=100.4,
            best_ask=100.6,
            intensity_value=0.0,
            volatility_scalar=0.0,
            instructor_value=1.0,
        )

        self.assertEqual(price_levels(engine, engine.manager.books.bid_maker), [])
        self.assertEqual(
            price_levels(engine, engine.manager.books.ask_maker),
            [(101.0, 0.334), (102.0, 0.333), (103.0, 0.333)],
        )

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
                open_curve_underwater=(0.0, 2.0, 1.0),
            )
        )
        set_position(engine, qty=1.0, cost=25.0)

        self.assertEqual(engine._inventory_open_curve_underwater_multiplier(mid=100.0), 1.5)

    def test_stoploss_is_absolute_usdt_per_lot(self):
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

    def test_long_profit_places_close_grid_only(self):
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
        self.assertEqual(
            price_levels(engine, engine.manager.books.ask_maker),
            [(101.0, 0.334), (102.0, 0.333), (103.0, 0.333)],
        )

    def test_long_profit_grid_clips_crossed_close_levels_to_best_ask(self):
        engine = SimpleMakerStrategy(make_config())
        set_position(engine, qty=1.0, cost=100.0)

        engine._on_ticker_event(
            timestamp=1,
            best_bid=101.4,
            best_ask=101.5,
            intensity_value=0.0,
            volatility_scalar=0.0,
        )

        self.assertEqual(price_levels(engine, engine.manager.books.bid_maker), [])
        self.assertEqual(
            price_levels(engine, engine.manager.books.ask_maker),
            [(101.5, 0.334), (102.0, 0.333), (103.0, 0.333)],
        )

    def test_strategy_uses_profit_grid_qty_distribution(self):
        engine = SimpleMakerStrategy(make_config(profit_grid=(100.0, 300.0, 3, "power", 2.0)))
        set_position(engine, qty=1.4, cost=100.0)

        engine._on_ticker_event(
            timestamp=1,
            best_bid=100.4,
            best_ask=100.6,
            intensity_value=0.0,
            volatility_scalar=0.0,
        )

        self.assertEqual(price_levels(engine, engine.manager.books.bid_maker), [])
        self.assertEqual(
            price_levels(engine, engine.manager.books.ask_maker),
            [(101.0, 0.9), (102.0, 0.4), (103.0, 0.1)],
        )

    def test_long_above_upper_releases_remaining_at_best_ask(self):
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

    def test_short_min_step_residual_releases_in_profit_zone(self):
        engine = SimpleMakerStrategy(make_config(profit_grid=(0.0, 20.0, 10)))
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

    def test_unchanged_profit_grid_is_not_reposted_every_second(self):
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

        self.assertIn((999.9, 0.001), price_levels(engine, engine.manager.books.ask_maker))

    def test_partial_profit_fill_keeps_remaining_grid_when_cost_unchanged(self):
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
            trade_price=101.1,
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

    def test_short_profit_places_close_grid_only(self):
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
        self.assertEqual(
            price_levels(engine, engine.manager.books.bid_maker),
            [(99.0, 0.334), (98.0, 0.333), (97.0, 0.333)],
        )

    def test_short_profit_grid_clips_crossed_close_levels_to_best_bid(self):
        engine = SimpleMakerStrategy(make_config())
        set_position(engine, qty=-1.0, cost=100.0)

        engine._on_ticker_event(
            timestamp=1,
            best_bid=98.5,
            best_ask=98.6,
            intensity_value=0.0,
            volatility_scalar=0.0,
        )

        self.assertEqual(price_levels(engine, engine.manager.books.ask_maker), [])
        self.assertEqual(
            price_levels(engine, engine.manager.books.bid_maker),
            [(98.5, 0.334), (98.0, 0.333), (97.0, 0.333)],
        )

    def test_max_position_lots_activate_sequentially(self):
        engine = SimpleMakerStrategy(
            make_config(max_position_usdt=[150.0, 300.0], stoploss=[0.0, 0.0]),
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
        lot0 = state["lots"][0]["position"]
        lot1 = state["lots"][1]["position"]

        self.assertEqual(len(state["lots"]), 2)
        self.assertEqual(state["max_position_lots"], [150.0, 300.0])
        self.assertEqual(state["max_position_usdt"], 450.0)
        self.assertGreaterEqual(lot0["gross_cost_notional_usdt"], 150.0)
        self.assertGreater(lot1["qty"], 0.0)
        self.assertAlmostEqual(state["position"]["qty"], lot0["qty"] + lot1["qty"])
        self.assertEqual(float(df.iloc[1]["position"]), 1.0)
        self.assertAlmostEqual(float(df.iloc[-1]["position"]), state["position"]["qty"])

    def test_multi_lot_top_level_pnl_is_total_book(self):
        engine = SimpleMakerStrategy(
            make_config(max_position_usdt=[150.0, 150.0], stoploss=[0.0, 0.0])
        )
        lot0, lot1 = engine._lot_strategies
        set_position(lot0, qty=1.0, cost=100.0)
        set_position(lot1, qty=2.0, cost=110.0)
        lot0.manager.position.realized_pnl = 3.0
        lot1.manager.position.realized_pnl = 5.0
        lot0.manager.position.mark(90.0)
        lot1.manager.position.mark(90.0)

        state = engine.snapshot_state()

        self.assertEqual(state["position"]["realized_pnl"], 8.0)
        self.assertEqual(state["realized_pnl"], 8.0)
        self.assertEqual(state["position"]["unrealized_pnl"], -50.0)
        self.assertEqual(state["unrealized_pnl"], -50.0)
        self.assertEqual(state["total_pnl"], -42.0)
        self.assertEqual(_state_realized_pnl(state), 8.0)
        self.assertEqual(_state_total_pnl(state), -42.0)

    def test_full_lot_activates_backup_even_when_stoploss_would_trigger(self):
        engine = SimpleMakerStrategy(
            make_config(max_position_usdt=[100.0, 100.0], stoploss=[5.0, 5.0])
        )
        lot0, _lot1 = engine._lot_strategies
        set_position(lot0, qty=1.0, cost=100.0)
        engine._sync_lot_pools()

        engine._activate_backup_lot_if_needed(mid=90.0)

        self.assertEqual(engine._active_lot_count, 2)
        self.assertEqual(engine._active_lot_indices, [0, 1])
        self.assertEqual(engine._frozen_lot_indices, [])

    def test_each_lot_uses_matching_stoploss(self):
        engine = SimpleMakerStrategy(
            make_config(max_position_usdt=[100.0, 200.0], stoploss=[5.0, 20.0])
        )

        self.assertEqual(engine._lot_strategies[0].cfg.stoploss, 5.0)
        self.assertEqual(engine._lot_strategies[1].cfg.stoploss, 20.0)

    def test_flat_inner_lot_rotates_to_outermost_slot(self):
        engine = SimpleMakerStrategy(
            make_config(
                max_position_usdt=[100.0, 200.0, 300.0, 400.0],
                stoploss=[10.0, 20.0, 30.0, 40.0],
            )
        )
        lots = engine._lot_strategies
        engine._active_lot_count = 3
        set_position(lots[1], qty=1.0, cost=200.0)
        set_position(lots[2], qty=1.0, cost=300.0)

        engine._sync_lot_pools()

        self.assertEqual(engine._active_lot_count, 3)
        self.assertEqual(engine._active_lot_indices, [0, 1, 2])
        self.assertEqual(engine._frozen_lot_indices, [3])
        self.assertEqual(
            [lot.cfg.max_position_usdt for lot in engine._lot_strategies],
            [200.0, 300.0, 400.0, 100.0],
        )
        self.assertEqual(
            [lot.cfg.stoploss for lot in engine._lot_strategies],
            [20.0, 30.0, 40.0, 10.0],
        )
        self.assertTrue(engine._lot_has_position(engine._lot_strategies[0]))
        self.assertTrue(engine._lot_has_position(engine._lot_strategies[1]))
        self.assertFalse(engine._lot_has_position(engine._lot_strategies[2]))
        self.assertFalse(engine._lot_has_position(engine._lot_strategies[3]))

    def test_active_backup_opens_only_after_outer_nonempty_lot_is_full(self):
        engine = SimpleMakerStrategy(
            make_config(max_position_usdt=[200.0, 200.0], stoploss=[0.0, 0.0])
        )
        lot0, lot1 = engine._lot_strategies
        engine._active_lot_count = 2
        set_position(lot0, qty=1.0, cost=100.0)
        engine._sync_lot_pools()

        lot0._on_ticker_event(
            timestamp=1,
            best_bid=98.9,
            best_ask=99.1,
            intensity_value=0.0,
            volatility_scalar=0.0,
            open_allowed=engine._openable_lot_index() == 0,
        )
        lot1._on_ticker_event(
            timestamp=1,
            best_bid=98.9,
            best_ask=99.1,
            intensity_value=0.0,
            volatility_scalar=0.0,
            open_allowed=engine._openable_lot_index() == 1,
        )

        self.assertEqual(engine._openable_lot_index(), 0)
        self.assertEqual(price_levels(lot0, lot0.manager.books.bid_maker), [(98.9, 1.01)])
        self.assertEqual(price_levels(lot1, lot1.manager.books.bid_maker), [])

        lot0.manager.books.bid_maker.clear()
        set_position(lot0, qty=2.0, cost=100.0)
        engine._sync_lot_pools()

        lot0._on_ticker_event(
            timestamp=2,
            best_bid=98.9,
            best_ask=99.1,
            intensity_value=0.0,
            volatility_scalar=0.0,
            open_allowed=engine._openable_lot_index() == 0,
        )
        lot1._on_ticker_event(
            timestamp=2,
            best_bid=98.9,
            best_ask=99.1,
            intensity_value=0.0,
            volatility_scalar=0.0,
            open_allowed=engine._openable_lot_index() == 1,
        )

        self.assertEqual(engine._openable_lot_index(), 1)
        self.assertEqual(price_levels(lot0, lot0.manager.books.bid_maker), [])
        self.assertEqual(price_levels(lot1, lot1.manager.books.bid_maker), [(98.9, 1.01)])

    def test_new_lot_requires_adverse_cost_deviation_for_long(self):
        engine = SimpleMakerStrategy(
            make_config(
                max_position_usdt=[100.0, 100.0],
                stoploss=[0.0, 0.0],
                new_open_lot_crit=0.01,
            )
        )
        lot0, _lot1 = engine._lot_strategies
        set_position(lot0, qty=1.0, cost=100.0)
        engine._sync_lot_pools()

        engine._activate_backup_lot_if_needed(mid=99.1)

        self.assertEqual(engine._active_lot_count, 1)

        engine._activate_backup_lot_if_needed(mid=99.0)

        self.assertEqual(engine._active_lot_count, 2)
        self.assertEqual(engine._openable_lot_index(mid=99.1), 0)
        self.assertEqual(engine._openable_lot_index(mid=99.0), 1)

    def test_new_lot_requires_adverse_cost_deviation_for_short(self):
        engine = SimpleMakerStrategy(
            make_config(
                max_position_usdt=[100.0, 100.0],
                stoploss=[0.0, 0.0],
                new_open_lot_crit=0.01,
            )
        )
        lot0, _lot1 = engine._lot_strategies
        set_position(lot0, qty=-1.0, cost=100.0)
        engine._sync_lot_pools()

        engine._activate_backup_lot_if_needed(mid=100.9)

        self.assertEqual(engine._active_lot_count, 1)

        engine._activate_backup_lot_if_needed(mid=101.0)

        self.assertEqual(engine._active_lot_count, 2)
        self.assertEqual(engine._openable_lot_index(mid=100.9), 0)
        self.assertEqual(engine._openable_lot_index(mid=101.0), 1)

    def test_inner_lot_is_close_only_when_outer_lot_is_nonempty(self):
        engine = SimpleMakerStrategy(
            make_config(max_position_usdt=[200.0, 200.0], stoploss=[0.0, 0.0])
        )
        lot0, lot1 = engine._lot_strategies
        engine._active_lot_count = 2
        set_position(lot0, qty=1.0, cost=100.0)
        set_position(lot1, qty=0.5, cost=100.0)
        engine._sync_lot_pools()

        self.assertEqual(engine._openable_lot_index(), 1)

        lot0._on_ticker_event(
            timestamp=1,
            best_bid=98.9,
            best_ask=99.1,
            intensity_value=0.0,
            volatility_scalar=0.0,
            open_allowed=engine._openable_lot_index() == 0,
        )
        lot1._on_ticker_event(
            timestamp=1,
            best_bid=98.9,
            best_ask=99.1,
            intensity_value=0.0,
            volatility_scalar=0.0,
            open_allowed=engine._openable_lot_index() == 1,
        )

        self.assertEqual(price_levels(lot0, lot0.manager.books.bid_maker), [])
        self.assertEqual(price_levels(lot1, lot1.manager.books.bid_maker), [(98.9, 1.01)])

    def test_close_only_ticker_without_profit_grid_keeps_reducing_side_only(self):
        engine = SimpleMakerStrategy(make_config(profit_grid=None))
        set_position(engine, qty=1.0, cost=100.0)

        engine._on_ticker_event(
            timestamp=1,
            best_bid=99.8,
            best_ask=100.2,
            intensity_value=0.0,
            volatility_scalar=0.0,
            open_allowed=False,
        )

        self.assertEqual(price_levels(engine, engine.manager.books.ask_maker), [(100.2, 1.0)])
        self.assertEqual(price_levels(engine, engine.manager.books.bid_maker), [])

    def test_trade_fills_inner_close_before_outer_open(self):
        engine = SimpleMakerStrategy(
            make_config(max_position_usdt=[100.0, 100.0], stoploss=[0.0, 0.0])
        )
        lot0, lot1 = engine._lot_strategies
        engine._active_lot_count = 2
        set_position(lot0, qty=1.0, cost=100.0)
        lot0.manager.place_maker_levels(
            ask_levels=[(101.0, 1.0)],
            bid_levels=[],
            best_ask=100.6,
            best_bid=100.4,
            close_only=True,
        )
        lot1.manager.place_maker_levels(
            ask_levels=[(101.0, 1.0)],
            bid_levels=[],
            best_ask=100.6,
            best_bid=100.4,
            close_only=False,
        )

        filled = engine._on_trade_event(
            trade_time=2,
            is_buyer_maker=False,
            trade_price=101.1,
            trade_qty=1.5,
        )

        self.assertEqual(filled, 1.5)
        self.assertEqual(lot0.manager.position.qty, 0.0)
        self.assertEqual(lot1.manager.position.qty, -0.5)
        self.assertEqual(price_levels(lot1, lot1.manager.books.ask_maker), [(101.0, 0.5)])


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
