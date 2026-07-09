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
    _build_loader,
    _build_simulation_config,
    _normalize_sim_param_map,
    _state_cost_notional_usdt,
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
        "phase_change_position": 0.0,
        "boost_phase_change": 1.0,
        "phase_mode": "market",
        "max_holding_time": -1,
        "adj_spread_intensity": 1.0,
        "adj_spread_instructor": 0.0,
        "open_passive_only": False,
        "optimize_by_orderbook": -1,
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
        "phase_change_position": 0.0,
        "boost_phase_change": 1.0,
        "phase_mode": "market",
        "max_holding_time": -1,
        "adj_spread_intensity": 1.0,
        "adj_spread_instructor": 0.0,
        "open_passive_only": False,
        "optimize_by_orderbook": -1,
        "adj_spread_volatility": 0.0,
        "min_order_qty": 0.0,
        "min_order_notional": 0.0,
        "stoploss": 0.0,
        "profit_grid": [100.0, 300.0, 3],
    }
    values.update(overrides)
    return {"simulation": values}


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

    def test_build_config_parses_phase_change_position(self):
        cfg = _build_simulation_config(
            raw_config(phase_change_position=250.0)["simulation"]
        )

        self.assertEqual(cfg.phase_change_position, 250.0)

    def test_build_config_parses_boost_phase_change(self):
        cfg = _build_simulation_config(
            raw_config(boost_phase_change=2.5)["simulation"]
        )

        self.assertEqual(cfg.boost_phase_change, 2.5)

    def test_build_config_parses_phase_mode(self):
        cfg = _build_simulation_config(
            raw_config(phase_mode="trade")["simulation"]
        )

        self.assertEqual(cfg.phase_mode, "trade")

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

    def test_build_config_parses_optimize_by_orderbook(self):
        cfg = _build_simulation_config(
            raw_config(optimize_by_orderbook=1000)["simulation"]
        )

        self.assertEqual(cfg.optimize_by_orderbook, 1000.0)

    def test_build_config_parses_min_quote_distance_bps(self):
        cfg = _build_simulation_config(
            raw_config(min_quote_distance_bps=20.5)["simulation"]
        )

        self.assertEqual(cfg.min_quote_distance_bps, 20.5)

    def test_build_config_rejects_boolean_optimize_by_orderbook(self):
        with self.assertRaises(ValueError):
            _build_simulation_config(raw_config(optimize_by_orderbook=True)["simulation"])

    def test_build_config_rejects_negative_min_quote_distance_bps(self):
        with self.assertRaises(ValueError):
            _build_simulation_config(
                raw_config(min_quote_distance_bps=-0.1)["simulation"]
            )

    def test_build_config_rejects_legacy_min_quote_distance_ticks(self):
        with self.assertRaisesRegex(ValueError, "min_quote_distance_bps"):
            _build_simulation_config(
                raw_config(min_quote_distance_ticks=20)["simulation"]
            )

    def test_build_config_allows_bbo_imbalance_zero_lookback(self):
        cfg = _build_simulation_config(
            raw_config(
                name_instructor="bbo_imbalance",
                lookback_instructor=0,
            )["simulation"]
        )

        self.assertEqual(cfg.name_instructor, "bbo_imbalance")
        self.assertEqual(cfg.lookback_instructor, 0)

    def test_zero_spread_adjustments_prune_inactive_task_axes(self):
        cfg = raw_config(
            name_intensity="k",
            lookback_intensity=[100, 300],
            name_instructor="trade_imbalance",
            lookback_instructor=[1, 5],
            freq_ms_intensity=[1000, 2000],
            freq_ms_instructor=[5000, 10000],
            freq_ms_volatility=[30000],
            adj_spread_intensity=0.0,
            adj_spread_instructor=0.0,
            adj_spread_volatility=0.0,
        )
        cfg["symbols"] = ["BTCUSDT"]
        cfg["date_start"] = "2025-01-02"
        cfg["date_end"] = "2025-01-02"

        tasks = _build_tasks(cfg)

        self.assertEqual(len(tasks), 1)
        self.assertNotIn("name_intensity", tasks[0].sim_params)
        self.assertNotIn("lookback_intensity", tasks[0].sim_params)
        self.assertNotIn("name_instructor", tasks[0].sim_params)
        self.assertNotIn("lookback_instructor", tasks[0].sim_params)
        self.assertNotIn("name_volatility", tasks[0].sim_params)
        self.assertNotIn("lookback_volatility", tasks[0].sim_params)
        self.assertNotIn("freq_ms_intensity", tasks[0].sim_params)
        self.assertNotIn("freq_ms_instructor", tasks[0].sim_params)
        self.assertNotIn("freq_ms_volatility", tasks[0].sim_params)

    def test_instructor_task_expansion_skips_invalid_indicator_lookbacks(self):
        cfg = raw_config(
            name_instructor=["bbo_imbalance", "trade_imbalance"],
            lookback_instructor=[0, 5],
            adj_spread_instructor=[-1000.0, 0.0, 1000.0],
            freq_ms_instructor=[60000],
        )
        cfg["symbols"] = ["BTCUSDT"]
        cfg["date_start"] = "2025-01-02"
        cfg["date_end"] = "2025-01-02"

        tasks = _build_tasks(cfg)
        active_pairs = sorted(
            {
                (
                    task.sim_params.get("adj_spread_instructor"),
                    task.sim_params.get("name_instructor"),
                    task.sim_params.get("lookback_instructor"),
                )
                for task in tasks
                if "name_instructor" in task.sim_params
            }
        )
        inactive = [task for task in tasks if "name_instructor" not in task.sim_params]

        self.assertEqual(len(tasks), 5)
        self.assertEqual(len(inactive), 1)
        self.assertEqual(
            active_pairs,
            [
                (-1000.0, "bbo_imbalance", 0),
                (-1000.0, "trade_imbalance", 5),
                (1000.0, "bbo_imbalance", 0),
                (1000.0, "trade_imbalance", 5),
            ],
        )

    def test_zero_spread_adjustments_skip_feature_specs(self):
        engine = SimpleMakerStrategy(
            make_config(
                name_instructor="trade_imbalance",
                lookback_instructor=1,
                name_volatility="sigma",
                lookback_volatility=300,
                adj_spread_intensity=0.0,
                adj_spread_instructor=0.0,
                adj_spread_volatility=0.0,
            )
        )

        self.assertIsNone(engine._selected_intensity_spec())
        self.assertIsNone(engine._selected_instructor_spec())
        self.assertEqual(engine._selected_volatility_specs(), [])

    def test_positive_intensity_adjustment_requires_lookback(self):
        cfg = raw_config(adj_spread_intensity=1.0)["simulation"]
        cfg.pop("lookback_intensity")

        with self.assertRaisesRegex(ValueError, "lookback_intensity"):
            _build_simulation_config(cfg)

    def test_split_alpha_frequencies_flow_into_feature_specs(self):
        cfg = _build_simulation_config(
            raw_config(
                name_instructor="trade_imbalance",
                lookback_instructor=5,
                name_volatility="sigma",
                lookback_volatility=300,
                adj_spread_instructor=0.001,
                adj_spread_volatility=0.5,
                freq_ms_intensity=1000,
                freq_ms_instructor=5000,
                freq_ms_volatility=60000,
            )["simulation"]
        )
        engine = SimpleMakerStrategy(cfg)

        self.assertEqual(cfg.alpha_freq, 1000)
        self.assertEqual(engine._selected_intensity_spec()["freq_ms"], 1000)
        self.assertEqual(engine._selected_instructor_spec()["freq_ms"], 5000)
        self.assertEqual(engine._selected_volatility_specs()[0]["freq_ms"], 60000)

    def test_invalid_phase_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            _build_simulation_config(raw_config(phase_mode="last_open")["simulation"])

    def test_invalid_boost_phase_change_is_rejected(self):
        with self.assertRaises(ValueError):
            _build_simulation_config(raw_config(boost_phase_change=-1.0)["simulation"])

    def test_loader_requires_orderbook_path_when_optimize_by_orderbook_enabled(self):
        cfg = raw_config(optimize_by_orderbook=0)
        cfg["output_path"] = "/tmp/cache"
        cfg["input_path"] = "/tmp/input"
        sim = _build_simulation_config(cfg["simulation"])

        with self.assertRaises(ValueError):
            _build_loader(cfg, simulation=sim)

    def test_loader_skips_orderbook_path_when_optimize_by_orderbook_disabled(self):
        cfg = raw_config(optimize_by_orderbook=-1)
        cfg["output_path"] = "/tmp/cache"
        cfg["input_path"] = "/tmp/input"
        sim = _build_simulation_config(cfg["simulation"])

        loader = _build_loader(cfg, simulation=sim)

        self.assertIsNone(loader.orderbook_replay_config)


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

    def test_min_quote_distance_bps_floors_flat_open_quotes_to_ticks(self):
        engine = SimpleMakerStrategy(make_config(min_quote_distance_bps=59.0))

        engine._on_ticker_event(
            timestamp=1,
            best_bid=99.8,
            best_ask=100.2,
            intensity_value=0.0,
            volatility_scalar=0.0,
        )

        self.assertEqual(
            price_levels(engine, engine.manager.books.ask_maker),
            [(100.7, 1.0)],
        )
        self.assertEqual(
            price_levels(engine, engine.manager.books.bid_maker),
            [(99.3, 1.0)],
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

    def test_min_quote_distance_bps_clips_profit_grid_close_levels(self):
        engine = SimpleMakerStrategy(make_config(min_quote_distance_bps=200.0))
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
            [(102.6, 0.667), (103.0, 0.333)],
        )

    def test_open_liquidity_snap_moves_flat_open_quotes_to_wall_front(self):
        engine = SimpleMakerStrategy(make_config(optimize_by_orderbook=0))

        engine._on_ticker_event(
            timestamp=1,
            best_bid=99.8,
            best_ask=100.2,
            intensity_value=0.3,
            volatility_scalar=0.0,
            replay_bid_ticks=[998, 993],
            replay_ask_ticks=[1002, 1007],
        )

        self.assertEqual(
            price_levels(engine, engine.manager.books.ask_maker),
            [(100.6, 1.0)],
        )
        self.assertEqual(
            price_levels(engine, engine.manager.books.bid_maker),
            [(99.4, 1.0)],
        )
        state = engine.snapshot_state()
        self.assertEqual(state["open_liquidity_snap_adjusted"], 2)
        self.assertEqual(state["open_liquidity_snap_moved_ticks"], 2)
        self.assertEqual(state["open_liquidity_snap_max_move_ticks"], 1)

    def test_open_liquidity_snap_leaves_wall_front_quotes_unchanged(self):
        engine = SimpleMakerStrategy(make_config(optimize_by_orderbook=0))

        engine._on_ticker_event(
            timestamp=1,
            best_bid=99.8,
            best_ask=100.2,
            intensity_value=0.3,
            volatility_scalar=0.0,
            replay_bid_ticks=[998, 994],
            replay_ask_ticks=[1002, 1006],
        )

        self.assertEqual(
            price_levels(engine, engine.manager.books.ask_maker),
            [(100.5, 1.0)],
        )
        self.assertEqual(
            price_levels(engine, engine.manager.books.bid_maker),
            [(99.5, 1.0)],
        )
        self.assertEqual(engine.snapshot_state()["open_liquidity_snap_adjusted"], 0)

    def test_open_liquidity_snap_ignores_thin_levels_by_notional_threshold(self):
        engine = SimpleMakerStrategy(make_config(optimize_by_orderbook=1000))

        engine._on_ticker_event(
            timestamp=1,
            best_bid=99.8,
            best_ask=100.2,
            intensity_value=0.3,
            volatility_scalar=0.0,
            replay_bid_ticks=[998, 994, 993],
            replay_ask_ticks=[1002, 1006, 1007],
            replay_bid_notional=[10_000.0, 500.0, 2_000.0],
            replay_ask_notional=[10_000.0, 500.0, 2_000.0],
        )

        self.assertEqual(
            price_levels(engine, engine.manager.books.ask_maker),
            [(100.6, 1.0)],
        )
        self.assertEqual(
            price_levels(engine, engine.manager.books.bid_maker),
            [(99.4, 1.0)],
        )
        state = engine.snapshot_state()
        self.assertEqual(state["open_liquidity_snap_adjusted"], 2)
        self.assertEqual(state["open_liquidity_snap_moved_ticks"], 2)

    def test_open_liquidity_snap_cancels_when_only_thin_levels_exist(self):
        engine = SimpleMakerStrategy(make_config(optimize_by_orderbook=1000))

        engine._on_ticker_event(
            timestamp=1,
            best_bid=99.8,
            best_ask=100.2,
            intensity_value=0.3,
            volatility_scalar=0.0,
            replay_bid_ticks=[998, 994],
            replay_ask_ticks=[1002, 1006],
            replay_bid_notional=[10_000.0, 500.0],
            replay_ask_notional=[10_000.0, 500.0],
        )

        self.assertEqual(price_levels(engine, engine.manager.books.ask_maker), [])
        self.assertEqual(price_levels(engine, engine.manager.books.bid_maker), [])
        self.assertEqual(engine.snapshot_state()["open_liquidity_snap_cancelled"], 2)

    def test_open_liquidity_snap_cancels_open_side_when_no_wall_exists(self):
        engine = SimpleMakerStrategy(make_config(optimize_by_orderbook=0))

        engine._on_ticker_event(
            timestamp=1,
            best_bid=99.8,
            best_ask=100.2,
            intensity_value=0.3,
            volatility_scalar=0.0,
            replay_bid_ticks=[998, 997],
            replay_ask_ticks=[1002, 1003],
        )

        self.assertEqual(price_levels(engine, engine.manager.books.ask_maker), [])
        self.assertEqual(price_levels(engine, engine.manager.books.bid_maker), [])
        self.assertEqual(engine.snapshot_state()["open_liquidity_snap_cancelled"], 2)

    def test_open_liquidity_snap_does_not_move_profit_grid_closes(self):
        engine = SimpleMakerStrategy(make_config(optimize_by_orderbook=0))
        set_position(engine, qty=1.0, cost=100.0)

        engine._on_ticker_event(
            timestamp=1,
            best_bid=100.4,
            best_ask=100.6,
            intensity_value=0.0,
            volatility_scalar=0.0,
            replay_bid_ticks=[1004, 900],
            replay_ask_ticks=[1006, 1100],
        )

        self.assertEqual(price_levels(engine, engine.manager.books.bid_maker), [])
        self.assertEqual(
            price_levels(engine, engine.manager.books.ask_maker),
            [(101.0, 0.334), (102.0, 0.333), (103.0, 0.333)],
        )
        self.assertEqual(engine.snapshot_state()["open_liquidity_snap_adjusted"], 0)

    def test_open_liquidity_snap_only_moves_long_underwater_bid_add(self):
        engine = SimpleMakerStrategy(make_config(optimize_by_orderbook=0))
        set_position(engine, qty=1.0, cost=100.0)

        engine._on_ticker_event(
            timestamp=1,
            best_bid=98.9,
            best_ask=99.1,
            intensity_value=0.2,
            volatility_scalar=0.0,
            replay_bid_ticks=[989, 984],
            replay_ask_ticks=[991, 995],
        )

        self.assertEqual(price_levels(engine, engine.manager.books.ask_maker), [])
        self.assertEqual(
            price_levels(engine, engine.manager.books.bid_maker),
            [(98.5, 1.01)],
        )

    def test_open_liquidity_snap_only_moves_short_underwater_ask_add(self):
        engine = SimpleMakerStrategy(make_config(optimize_by_orderbook=0))
        set_position(engine, qty=-1.0, cost=100.0)

        engine._on_ticker_event(
            timestamp=1,
            best_bid=100.9,
            best_ask=101.1,
            intensity_value=0.2,
            volatility_scalar=0.0,
            replay_bid_ticks=[1009, 1005],
            replay_ask_ticks=[1011, 1016],
        )

        self.assertEqual(
            price_levels(engine, engine.manager.books.ask_maker),
            [(101.5, 0.99)],
        )
        self.assertEqual(price_levels(engine, engine.manager.books.bid_maker), [])

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

    def test_phase_change_anchors_long_adds_to_best_mid(self):
        engine = SimpleMakerStrategy(make_config(phase_change_position=50.0))
        set_position(engine, qty=1.0, cost=100.0)
        engine._phase_side = 1
        engine._phase_best_mid = 99.0

        engine._on_ticker_event(
            timestamp=1,
            best_bid=99.9,
            best_ask=100.1,
            intensity_value=0.0,
            volatility_scalar=0.0,
        )

        self.assertEqual(
            price_levels(engine, engine.manager.books.bid_maker),
            [(99.0, 1.0)],
        )

        engine._on_ticker_event(
            timestamp=2,
            best_bid=98.4,
            best_ask=98.6,
            intensity_value=0.0,
            volatility_scalar=0.0,
        )

        self.assertEqual(
            price_levels(engine, engine.manager.books.bid_maker),
            [(98.4, 1.015)],
        )
        self.assertEqual(engine._phase_best_mid, 98.5)

    def test_phase_change_boosts_allowed_long_add_only(self):
        engine = SimpleMakerStrategy(
            make_config(
                phase_change_position=50.0,
                boost_phase_change=2.0,
                profit_grid=None,
            )
        )
        set_position(engine, qty=1.0, cost=100.0)
        engine._phase_side = 1
        engine._phase_best_mid = 99.0

        engine._on_ticker_event(
            timestamp=1,
            best_bid=98.4,
            best_ask=98.6,
            intensity_value=0.0,
            volatility_scalar=0.0,
        )

        self.assertEqual(
            price_levels(engine, engine.manager.books.ask_maker),
            [(98.6, 1.015)],
        )
        self.assertEqual(
            price_levels(engine, engine.manager.books.bid_maker),
            [(98.4, 2.03)],
        )

    def test_phase_change_anchors_short_adds_to_best_mid(self):
        engine = SimpleMakerStrategy(make_config(phase_change_position=50.0))
        set_position(engine, qty=-1.0, cost=100.0)
        engine._phase_side = -1
        engine._phase_best_mid = 101.0

        engine._on_ticker_event(
            timestamp=1,
            best_bid=99.9,
            best_ask=100.1,
            intensity_value=0.0,
            volatility_scalar=0.0,
        )

        self.assertEqual(
            price_levels(engine, engine.manager.books.ask_maker),
            [(101.0, 1.0)],
        )

        engine._on_ticker_event(
            timestamp=2,
            best_bid=101.9,
            best_ask=102.1,
            intensity_value=0.0,
            volatility_scalar=0.0,
        )

        self.assertEqual(
            price_levels(engine, engine.manager.books.ask_maker),
            [(102.1, 0.98)],
        )
        self.assertEqual(engine._phase_best_mid, 102.0)

    def test_phase_change_resets_after_flat(self):
        engine = SimpleMakerStrategy(make_config(phase_change_position=50.0))
        engine._phase_side = 1
        engine._phase_best_mid = 99.0
        set_position(engine, qty=0.0, cost=math.nan)

        engine._on_ticker_event(
            timestamp=1,
            best_bid=99.9,
            best_ask=100.1,
            intensity_value=0.0,
            volatility_scalar=0.0,
        )

        self.assertEqual(engine._phase_side, 0)
        self.assertIsNone(engine._phase_best_mid)

    def test_phase_change_long_window_restarts_after_partial_close(self):
        engine = SimpleMakerStrategy(make_config(phase_change_position=50.0))
        set_position(engine, qty=1.0, cost=100.0)
        engine._phase_side = 1
        engine._phase_best_mid = 95.0
        engine.manager.books.ask_maker.merge([(1000, 200)])

        engine._on_trade_event(
            trade_time=1,
            is_buyer_maker=False,
            trade_price=100.1,
            trade_qty=0.2,
        )

        self.assertAlmostEqual(engine.manager.position.qty, 0.8)
        self.assertEqual(engine._phase_side, 1)
        self.assertEqual(engine._phase_best_mid, 100.1)

        engine._on_ticker_event(
            timestamp=2,
            best_bid=99.9,
            best_ask=100.1,
            intensity_value=0.0,
            volatility_scalar=0.0,
        )

        self.assertEqual(len(price_levels(engine, engine.manager.books.bid_maker)), 1)
        self.assertEqual(engine._phase_best_mid, 100.0)

    def test_phase_change_short_window_restarts_after_partial_close(self):
        engine = SimpleMakerStrategy(make_config(phase_change_position=50.0))
        set_position(engine, qty=-1.0, cost=100.0)
        engine._phase_side = -1
        engine._phase_best_mid = 105.0
        engine.manager.books.bid_maker.merge([(1000, 200)])

        engine._on_trade_event(
            trade_time=1,
            is_buyer_maker=True,
            trade_price=99.9,
            trade_qty=0.2,
        )

        self.assertAlmostEqual(engine.manager.position.qty, -0.8)
        self.assertEqual(engine._phase_side, -1)
        self.assertEqual(engine._phase_best_mid, 99.9)

        engine._on_ticker_event(
            timestamp=2,
            best_bid=99.9,
            best_ask=100.1,
            intensity_value=0.0,
            volatility_scalar=0.0,
        )

        self.assertEqual(len(price_levels(engine, engine.manager.books.ask_maker)), 1)
        self.assertEqual(engine._phase_best_mid, 100.0)

    def test_phase_change_trade_mode_ignores_market_best_without_own_fill(self):
        engine = SimpleMakerStrategy(
            make_config(phase_change_position=50.0, phase_mode="trade")
        )
        set_position(engine, qty=1.0, cost=100.0)
        engine._phase_side = 1
        engine._phase_best_mid = 99.0

        engine._on_ticker_event(
            timestamp=1,
            best_bid=98.4,
            best_ask=98.6,
            intensity_value=0.0,
            volatility_scalar=0.0,
        )

        self.assertEqual(len(price_levels(engine, engine.manager.books.bid_maker)), 1)
        self.assertEqual(engine._phase_best_mid, 99.0)

    def test_phase_change_trade_mode_uses_own_fill_price_for_add(self):
        engine = SimpleMakerStrategy(
            make_config(phase_change_position=50.0, phase_mode="trade")
        )
        set_position(engine, qty=1.0, cost=100.0)
        engine._phase_side = 1
        engine._phase_best_mid = 101.0
        engine.manager.books.bid_maker.merge([(1000, 200)])

        engine._on_trade_event(
            trade_time=1,
            is_buyer_maker=True,
            trade_price=99.9,
            trade_qty=0.2,
        )

        self.assertAlmostEqual(engine.manager.position.qty, 1.2)
        self.assertEqual(engine._phase_best_mid, 100.0)

    def test_phase_change_trade_mode_restarts_at_own_fill_price_after_partial_close(self):
        engine = SimpleMakerStrategy(
            make_config(phase_change_position=50.0, phase_mode="trade")
        )
        set_position(engine, qty=1.0, cost=100.0)
        engine._phase_side = 1
        engine._phase_best_mid = 95.0
        engine.manager.books.ask_maker.merge([(1000, 200)])

        engine._on_trade_event(
            trade_time=1,
            is_buyer_maker=False,
            trade_price=100.1,
            trade_qty=0.2,
        )

        self.assertAlmostEqual(engine.manager.position.qty, 0.8)
        self.assertEqual(engine._phase_side, 1)
        self.assertEqual(engine._phase_best_mid, 100.0)

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

    def test_active_profit_grid_is_not_reclipped_when_bbo_moves(self):
        engine = SimpleMakerStrategy(make_config())
        set_position(engine, qty=1.0, cost=100.0)

        engine._on_ticker_event(
            timestamp=1,
            best_bid=100.4,
            best_ask=100.6,
            intensity_value=0.0,
            volatility_scalar=0.0,
        )
        before = price_levels(engine, engine.manager.books.ask_maker)

        engine._on_ticker_event(
            timestamp=2,
            best_bid=101.4,
            best_ask=101.5,
            intensity_value=0.0,
            volatility_scalar=0.0,
        )

        self.assertEqual(price_levels(engine, engine.manager.books.ask_maker), before)

    def test_active_profit_grid_survives_underwater_until_add_fill(self):
        engine = SimpleMakerStrategy(make_config())
        set_position(engine, qty=1.0, cost=100.0)

        engine._on_ticker_event(
            timestamp=1,
            best_bid=100.4,
            best_ask=100.6,
            intensity_value=0.0,
            volatility_scalar=0.0,
        )
        close_grid = price_levels(engine, engine.manager.books.ask_maker)

        engine._on_ticker_event(
            timestamp=2,
            best_bid=98.9,
            best_ask=99.1,
            intensity_value=0.0,
            volatility_scalar=0.0,
        )

        self.assertEqual(price_levels(engine, engine.manager.books.ask_maker), close_grid)
        self.assertEqual(len(price_levels(engine, engine.manager.books.bid_maker)), 1)

        engine._on_trade_event(
            trade_time=3,
            is_buyer_maker=True,
            trade_price=98.8,
            trade_qty=0.1,
        )

        self.assertEqual(price_levels(engine, engine.manager.books.ask_maker), [])

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

    def test_min_quote_distance_bps_clips_short_profit_grid_close_levels(self):
        engine = SimpleMakerStrategy(make_config(min_quote_distance_bps=51.0))
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
            [(98.0, 0.667), (97.0, 0.333)],
        )


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
