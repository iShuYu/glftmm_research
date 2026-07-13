import unittest

from core.position import Position
from sim.report import _limit_plot_frame, _normalize_report_columns, load_report_frame
from sim.strategy import (
    get_minimum_size,
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
        "max_position_usdt": 100000.0,
        "phase_change_position": 0.0,
        "boost_phase_change": 1.0,
        "phase_mode": "market",
        "max_holding_time": -1,
        "adj_spread_intensity": (1.0, 1.0),
        "adj_spread_instructor": 0.0,
        "passive_only": False,
        "adj_spread_volatility": (0.0, 0.0),
        "min_quote_distance_bps": 0.0,
        "inventory_skew": (0.0, 1.0),
        "min_order_qty": 0.0,
        "min_order_notional": 0.0,
        "stoploss": 0.0,
        "takeprofit": 0.0,
        "hold_since": 0.0,
        "open_curve": (1000.0, 1000.0, 1.0),
        "close_curve": (1000.0, 1000.0, 1.0),
        "boost_underwater": (1.0, 1.0),
        "boost_profitzone": (1.0, 1.0),
        "toxic_lock": (0, 0),
        "strict_mode": True,
        "simple_mode": True,
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
        "max_position_usdt": 100000.0,
        "phase_change_position": 0.0,
        "boost_phase_change": 1.0,
        "phase_mode": "market",
        "max_holding_time": -1,
        "adj_spread_intensity": [1.0, 1.0],
        "adj_spread_instructor": 0.0,
        "passive_only": False,
        "adj_spread_volatility": [0.0, 0.0],
        "min_quote_distance_bps": 0.0,
        "min_order_qty": 0.0,
        "min_order_notional": 0.0,
        "stoploss": 0.0,
        "takeprofit": 0.0,
        "hold_since": 0.0,
        "boost_underwater": [1.0, 1.0],
        "boost_profitzone": [1.0, 1.0],
        "toxic_lock": [[0, 0]],
        "strict_mode": True,
        "simple_mode": True,
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
        self.calls = []

    def iter_merged_alpha_trade_tuples(self, **kwargs):
        self.calls.append(kwargs)
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
    def test_kls_pair_lookback_reaches_strategy_loader_spec(self):
        simulation = _build_simulation_config(
            raw_config(name_intensity="kls", lookback_intensity=[120, 12])["simulation"]
        )
        engine = SimpleMakerStrategy(simulation)

        self.assertEqual(simulation.lookback_intensity, "12_120")
        self.assertEqual(
            engine._selected_intensity_spec(),
            {"name": "kls", "lookback": "12_120"},
        )

        sim_map = _normalize_sim_param_map(
            raw_config(name_intensity="kls", lookback_intensity=[120, 12])
        )
        self.assertEqual(sim_map["lookback_intensity"], ["12_120"])

    def test_build_config_accepts_single_max_position_stoploss_takeprofit_and_hold_since(self):
        cfg = _build_simulation_config(
            raw_config(
                max_position_usdt=250.0,
                stoploss=25.0,
                takeprofit=40.0,
                hold_since=15.0,
            )["simulation"]
        )

        self.assertEqual(cfg.max_position_usdt, 250.0)
        self.assertEqual(cfg.total_max_position_usdt, 250.0)
        self.assertEqual(cfg.stoploss, 25.0)
        self.assertEqual(cfg.takeprofit, 40.0)
        self.assertEqual(cfg.hold_since, 15.0)

    def test_rejects_lot_shaped_max_position_stoploss_takeprofit_and_hold_since(self):
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

        with self.assertRaisesRegex(ValueError, "takeprofit must be a number"):
            _build_simulation_config(
                raw_config(
                    max_position_usdt=250.0,
                    takeprofit=[25.0, 75.0],
                )["simulation"]
            )

        with self.assertRaisesRegex(ValueError, "hold_since must be a number"):
            _build_simulation_config(
                raw_config(
                    max_position_usdt=250.0,
                    hold_since=[25.0, 75.0],
                )["simulation"]
            )

    def test_normalizes_max_position_stoploss_and_takeprofit_as_parameter_grids(self):
        sim_map = _normalize_sim_param_map(
            raw_config(
                max_position_usdt=[250.0, 750.0],
                phase_change_position=[100.0, 200.0],
                boost_phase_change=[1.0, 2.0],
                stoploss=[25.0, 75.0],
                takeprofit=[40.0, 80.0],
                hold_since=[15.0, 30.0],
                toxic_lock=[[20, 15]],
            )
        )

        self.assertEqual(sim_map["max_position_usdt"], [250.0, 750.0])
        self.assertEqual(sim_map["phase_change_position"], [100.0, 200.0])
        self.assertEqual(sim_map["boost_phase_change"], [1.0, 2.0])
        self.assertEqual(sim_map["stoploss"], [25.0, 75.0])
        self.assertEqual(sim_map["takeprofit"], [40.0, 80.0])
        self.assertEqual(sim_map["hold_since"], [15.0, 30.0])
        self.assertEqual(sim_map["toxic_lock"], [[20, 15]])

    def test_rejects_bare_toxic_lock_pair_in_parameter_grid(self):
        with self.assertRaisesRegex(ValueError, "list of"):
            _normalize_sim_param_map(raw_config(toxic_lock=[20, 15]))

    def test_normalizes_toxic_lock_grid_rows(self):
        sim_map = _normalize_sim_param_map(
            raw_config(toxic_lock=[[20, 15], [40, 25]])
        )

        self.assertEqual(sim_map["toxic_lock"], [[20, 15], [40, 25]])

    def test_build_config_parses_toxic_lock(self):
        cfg = _build_simulation_config(raw_config(toxic_lock=[[20, 15]])["simulation"])

        self.assertEqual(cfg.toxic_lock, (20, 15))

    def test_build_config_rejects_legacy_cooldown_time(self):
        raw = raw_config()["simulation"]
        raw.pop("toxic_lock", None)
        raw["cooldown_time"] = 30_000

        with self.assertRaisesRegex(ValueError, "toxic_lock"):
            _build_simulation_config(raw)

        with self.assertRaisesRegex(ValueError, "toxic_lock"):
            _normalize_sim_param_map({"simulation": raw})

    def test_build_config_parses_close_curve(self):
        cfg = _build_simulation_config(
            raw_config(close_curve=[0.0, 2.0, 1.0])["simulation"]
        )

        self.assertEqual(cfg.close_curve, (0.0, 2.0, 1.0))

    def test_build_config_parses_open_close_spread_adjustments(self):
        cfg = _build_simulation_config(
            raw_config(
                adj_spread_intensity=[3.0, 1.0],
                adj_spread_volatility=[0.5, 0.0],
            )["simulation"]
        )

        self.assertEqual(cfg.adj_spread_intensity, (3.0, 1.0))
        self.assertEqual(cfg.adj_spread_volatility, (0.5, 0.0))

    def test_build_config_parses_underwater_and_profitzone_boosts(self):
        cfg = _build_simulation_config(
            raw_config(
                boost_underwater=[2.0, 0.5],
                boost_profitzone=[0.25, 3.0],
            )["simulation"]
        )

        self.assertEqual(cfg.boost_underwater, (2.0, 0.5))
        self.assertEqual(cfg.boost_profitzone, (0.25, 3.0))

    def test_build_config_parses_simple_mode(self):
        cfg = _build_simulation_config(raw_config(simple_mode=False)["simulation"])

        self.assertFalse(cfg.simple_mode)

    def test_build_config_parses_strict_mode(self):
        cfg = _build_simulation_config(raw_config(strict_mode=False)["simulation"])

        self.assertFalse(cfg.strict_mode)

    def test_build_config_parses_min_quote_distance_bps(self):
        cfg = _build_simulation_config(
            raw_config(min_quote_distance_bps=20.5)["simulation"]
        )

        self.assertEqual(cfg.min_quote_distance_bps, 20.5)

    def test_build_config_parses_phase_change_controls(self):
        cfg = _build_simulation_config(
            raw_config(
                phase_change_position=250.0,
                boost_phase_change=2.5,
                phase_mode="trade",
            )["simulation"]
        )

        self.assertEqual(cfg.phase_change_position, 250.0)
        self.assertEqual(cfg.boost_phase_change, 2.5)
        self.assertEqual(cfg.phase_mode, "trade")

    def test_build_config_rejects_removed_orderbook_optimizer(self):
        with self.assertRaisesRegex(ValueError, "has been removed"):
            _build_simulation_config(
                raw_config(optimize_by_orderbook=1000)["simulation"]
            )

    def test_build_config_rejects_negative_min_quote_distance_bps(self):
        with self.assertRaisesRegex(ValueError, "min_quote_distance_bps"):
            _build_simulation_config(
                raw_config(min_quote_distance_bps=-0.1)["simulation"]
            )

    def test_build_config_rejects_invalid_phase_change_controls(self):
        with self.assertRaisesRegex(ValueError, "phase_change_position"):
            _build_simulation_config(
                raw_config(phase_change_position=-1.0)["simulation"]
            )
        with self.assertRaisesRegex(ValueError, "boost_phase_change"):
            _build_simulation_config(raw_config(boost_phase_change=-1.0)["simulation"])
        with self.assertRaisesRegex(ValueError, "phase_mode"):
            _build_simulation_config(raw_config(phase_mode="last_open")["simulation"])

    def test_build_config_rejects_legacy_min_quote_distance_ticks(self):
        with self.assertRaisesRegex(ValueError, "min_quote_distance_bps"):
            _build_simulation_config(
                raw_config(min_quote_distance_ticks=20)["simulation"]
            )

    def test_normalizes_close_curve_rows(self):
        sim_map = _normalize_sim_param_map(
            raw_config(close_curve=[[0.0, 1.0, 0.0], [0.0, 2.0, 1.0]])
        )

        self.assertEqual(
            sim_map["close_curve"],
            [[0.0, 1.0, 0.0], [0.0, 2.0, 1.0]],
        )

    def test_normalizes_open_close_spread_adjustment_rows(self):
        sim_map = _normalize_sim_param_map(
            raw_config(
                adj_spread_intensity=[[3.0, 1.0], [4.0, 1.5]],
                adj_spread_volatility=[[0.5, 0.0], [1.0, 0.25]],
            )
        )

        self.assertEqual(
            sim_map["adj_spread_intensity"],
            [[3.0, 1.0], [4.0, 1.5]],
        )
        self.assertEqual(
            sim_map["adj_spread_volatility"],
            [[0.5, 0.0], [1.0, 0.25]],
        )

    def test_normalizes_underwater_and_profitzone_boost_rows(self):
        sim_map = _normalize_sim_param_map(
            raw_config(
                boost_underwater=[[2.0, 0.5], [3.0, 0.25]],
                boost_profitzone=[[0.5, 2.0], [0.25, 3.0]],
            )
        )

        self.assertEqual(
            sim_map["boost_underwater"],
            [[2.0, 0.5], [3.0, 0.25]],
        )
        self.assertEqual(
            sim_map["boost_profitzone"],
            [[0.5, 2.0], [0.25, 3.0]],
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

    def test_build_tasks_crosses_takeprofit_grid(self):
        tasks = _build_tasks(
            raw_task_config(
                mode=[0, 1],
                takeprofit=[50.0, 100.0],
            )
        )

        self.assertEqual(len(tasks), 4)
        self.assertEqual(
            sorted(task.sim_params["takeprofit"] for task in tasks),
            [50.0, 50.0, 100.0, 100.0],
        )

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
            name_volatility="sigma",
            lookback_volatility=[300, 600],
            adj_spread_intensity=[0.0, 0.0],
            adj_spread_instructor=0.0,
            adj_spread_volatility=[0.0, 0.0],
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

    def test_zero_spread_adjustments_skip_feature_specs(self):
        engine = SimpleMakerStrategy(
            make_config(
                name_instructor="trade_imbalance",
                lookback_instructor=1,
                name_volatility="sigma",
                lookback_volatility=300,
                adj_spread_intensity=[0.0, 0.0],
                adj_spread_instructor=0.0,
                adj_spread_volatility=[0.0, 0.0],
            )
        )

        self.assertIsNone(engine._selected_intensity_spec())
        self.assertIsNone(engine._selected_instructor_spec())
        self.assertEqual(engine._selected_volatility_specs(), [])

    def test_zero_instructor_adjustment_does_not_request_instructor_file(self):
        loader = StaticEventLoader([])
        engine = SimpleMakerStrategy(
            make_config(
                name_instructor="bbo_imbalance",
                lookback_instructor=0,
                name_volatility="sigma",
                lookback_volatility=300,
                adj_spread_intensity=[2.0, 1.0],
                adj_spread_instructor=0.0,
                adj_spread_volatility=[0.0, 0.0],
            ),
            loader=loader,
        )

        engine.run_day(symbol="BTCUSDT", date="2025-01-02")

        self.assertEqual(len(loader.calls), 1)
        self.assertIsNotNone(loader.calls[0]["trade_intensity_spec"])
        self.assertIsNone(loader.calls[0]["instructor_spec"])
        self.assertEqual(loader.calls[0]["volatility_specs"], [])


class StrategyQuoteTest(unittest.TestCase):
    def test_get_minimum_size_rounds_up_to_qty_precision(self):
        self.assertEqual(
            get_minimum_size(mid=30_000.0, qty_precision=3, min_order_notional=100.0),
            0.004,
        )
        self.assertEqual(
            get_minimum_size(mid=100.0, qty_precision=3, min_order_notional=0.0),
            0.001,
        )
        self.assertEqual(
            get_minimum_size(
                mid=100.0,
                qty_precision=3,
                min_order_notional=0.0,
                min_order_qty=0.01,
            ),
            0.01,
        )

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

        self.assertGreater(len(price_levels(engine, engine.manager.books.bid_maker)), 0)
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

        self.assertGreater(len(price_levels(engine, engine.manager.books.bid_maker)), 0)
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

        self.assertGreater(len(price_levels(engine, engine.manager.books.ask_maker)), 0)
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

    def test_open_curve_units_use_gross_cost_notional_and_floor(self):
        engine = SimpleMakerStrategy(
            make_config(
                max_position_usdt=100.0,
                open_curve=(0.0, 2.0, 1.0),
            )
        )
        set_position(engine, qty=1.0, cost=25.0)

        self.assertEqual(engine._inventory_open_curve_units(mid=100.0), 1)

    def test_open_curve_units_can_floor_to_zero(self):
        engine = SimpleMakerStrategy(
            make_config(
                max_position_usdt=100.0,
                open_curve=(0.25, 2.0, 1.0),
            )
        )
        set_position(engine, qty=1.0, cost=100.0)

        self.assertEqual(engine._inventory_open_curve_units(mid=100.0), 0)

    def test_open_curve_qty_is_integer_min_bet_multiple(self):
        engine = SimpleMakerStrategy(
            make_config(
                min_order_notional=100.0,
                open_curve=(1.0, 3.0, 1.0),
            )
        )

        engine._on_ticker_event(
            timestamp=1,
            best_bid=99.9,
            best_ask=100.1,
            intensity_value=0.0,
            volatility_scalar=0.0,
        )

        self.assertEqual(
            price_levels(engine, engine.manager.books.ask_maker),
            [(100.1, 3.0)],
        )
        self.assertEqual(
            price_levels(engine, engine.manager.books.bid_maker),
            [(99.9, 3.006)],
        )

    def test_simple_mode_can_be_disabled_for_legacy_level_path(self):
        engine = SimpleMakerStrategy(
            make_config(
                simple_mode=False,
                min_order_notional=100.0,
                open_curve=(1.0, 3.0, 1.0),
            )
        )

        engine._on_ticker_event(
            timestamp=1,
            best_bid=99.9,
            best_ask=100.1,
            intensity_value=0.0,
            volatility_scalar=0.0,
        )

        self.assertEqual(
            price_levels(engine, engine.manager.books.ask_maker),
            [(100.1, 3.0)],
        )
        self.assertEqual(
            price_levels(engine, engine.manager.books.bid_maker),
            [(99.9, 3.0)],
        )

    def test_close_curve_units_use_gross_cost_notional_and_floor(self):
        engine = SimpleMakerStrategy(
            make_config(
                max_position_usdt=100.0,
                min_order_notional=100.0,
                close_curve=(0.0, 2.0, 1.0),
            )
        )
        set_position(engine, qty=1.0, cost=25.0)

        self.assertEqual(engine._inventory_close_curve_units(mid=100.0), 0)

    def test_close_curve_units_can_floor_to_zero(self):
        engine = SimpleMakerStrategy(
            make_config(
                max_position_usdt=100.0,
                close_curve=(0.25, 2.0, 1.0),
            )
        )

        self.assertEqual(engine._inventory_close_curve_units(mid=100.0), 0)

    def test_close_curve_qty_does_not_cap_to_position_qty(self):
        engine = SimpleMakerStrategy(
            make_config(
                max_position_usdt=100.0,
                min_order_notional=100.0,
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

    def test_takeprofit_is_absolute_usdt_for_long_and_short(self):
        engine = SimpleMakerStrategy(make_config(max_position_usdt=100.0, takeprofit=25.0))
        set_position(engine, qty=1.0, cost=100.0)

        self.assertFalse(engine._should_activate_takeprofit(mid=124.9))
        self.assertTrue(engine._should_activate_takeprofit(mid=125.0))

        set_position(engine, qty=-1.0, cost=100.0)

        self.assertFalse(engine._should_activate_takeprofit(mid=75.1))
        self.assertTrue(engine._should_activate_takeprofit(mid=75.0))

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

        self.assertGreater(len(price_levels(engine, engine.manager.books.ask_maker)), 0)
        self.assertEqual(len(price_levels(engine, engine.manager.books.bid_maker)), 1)

    def test_long_underwater_uses_underwater_open_boost(self):
        engine = SimpleMakerStrategy(
            make_config(
                min_order_notional=100.0,
                open_curve=(1.0, 1.0, 1.0),
                boost_underwater=(2.0, 1.0),
                boost_profitzone=(1.0, 1.0),
            )
        )
        set_position(engine, qty=1.0, cost=100.0)

        engine._on_ticker_event(
            timestamp=1,
            best_bid=98.9,
            best_ask=99.1,
            intensity_value=0.0,
            volatility_scalar=0.0,
        )

        self.assertGreater(len(price_levels(engine, engine.manager.books.ask_maker)), 0)
        self.assertEqual(price_levels(engine, engine.manager.books.bid_maker), [(98.9, 2.024)])

    def test_phase_change_gate_opens_simple_long_add(self):
        engine = SimpleMakerStrategy(
            make_config(
                phase_change_position=50.0,
                boost_phase_change=2.0,
                open_curve=(1.0, 1.0, 1.0),
                min_order_notional=100.0,
            )
        )
        set_position(engine, qty=1.0, cost=100.0)
        engine._phase_side = 1
        engine._phase_best_mid = 99.0

        engine._on_ticker_event(
            timestamp=1,
            best_bid=98.9,
            best_ask=99.1,
            intensity_value=0.0,
            volatility_scalar=0.0,
        )

        self.assertGreater(len(price_levels(engine, engine.manager.books.ask_maker)), 0)
        self.assertEqual(price_levels(engine, engine.manager.books.bid_maker), [(98.9, 2.024)])

    def test_phase_change_gate_closed_simple_long_add(self):
        engine = SimpleMakerStrategy(
            make_config(
                phase_change_position=50.0,
                boost_phase_change=2.0,
                open_curve=(1.0, 1.0, 1.0),
                min_order_notional=100.0,
            )
        )
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

        self.assertGreater(len(price_levels(engine, engine.manager.books.ask_maker)), 0)
        self.assertEqual(price_levels(engine, engine.manager.books.bid_maker), [])

    def test_phase_change_gate_opens_level_long_add(self):
        engine = SimpleMakerStrategy(
            make_config(
                phase_change_position=50.0,
                boost_phase_change=2.0,
                simple_mode=False,
                open_curve=(1.0, 1.0, 1.0),
                min_order_notional=100.0,
            )
        )
        set_position(engine, qty=1.0, cost=100.0)
        engine._phase_side = 1
        engine._phase_best_mid = 99.0

        engine._on_ticker_event(
            timestamp=1,
            best_bid=98.9,
            best_ask=99.1,
            intensity_value=0.0,
            volatility_scalar=0.0,
        )

        self.assertGreater(len(price_levels(engine, engine.manager.books.ask_maker)), 0)
        self.assertEqual(price_levels(engine, engine.manager.books.bid_maker), [(98.9, 2.022)])

    def test_phase_change_gate_opens_simple_short_add(self):
        engine = SimpleMakerStrategy(
            make_config(
                phase_change_position=50.0,
                boost_phase_change=2.0,
                open_curve=(1.0, 1.0, 1.0),
                min_order_notional=100.0,
            )
        )
        set_position(engine, qty=-1.0, cost=100.0)
        engine._phase_side = -1
        engine._phase_best_mid = 101.0

        engine._on_ticker_event(
            timestamp=1,
            best_bid=100.9,
            best_ask=101.1,
            intensity_value=0.0,
            volatility_scalar=0.0,
        )

        self.assertEqual(price_levels(engine, engine.manager.books.ask_maker), [(101.1, 1.98)])
        self.assertGreater(len(price_levels(engine, engine.manager.books.bid_maker)), 0)

    def test_phase_change_trade_mode_uses_own_fill_price_for_add(self):
        engine = SimpleMakerStrategy(
            make_config(
                phase_change_position=50.0,
                phase_mode="trade",
            )
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
        self.assertEqual(engine._phase_side, 1)
        self.assertEqual(engine._phase_best_mid, 100.0)

    def test_phase_change_trade_mode_ignores_market_best_without_fill(self):
        engine = SimpleMakerStrategy(
            make_config(
                phase_change_position=50.0,
                phase_mode="trade",
            )
        )
        set_position(engine, qty=1.0, cost=100.0)
        engine._phase_side = 1
        engine._phase_best_mid = 99.0

        engine._on_ticker_event(
            timestamp=1,
            best_bid=98.9,
            best_ask=99.1,
            intensity_value=0.0,
            volatility_scalar=0.0,
        )

        self.assertEqual(price_levels(engine, engine.manager.books.bid_maker), [(98.9, 1.0)])
        self.assertEqual(engine._phase_best_mid, 99.0)

    def test_toxic_lock_mutes_long_open_bid_until_slip_count_recovers(self):
        engine = SimpleMakerStrategy(
            make_config(
                toxic_lock=(3, 2),
                min_order_notional=100.0,
                open_curve=(1.0, 1.0, 1.0),
                close_curve=(1.0, 1.0, 1.0),
                strict_mode=False,
            )
        )
        set_position(engine, qty=1.0, cost=100.0)

        for idx, mid in enumerate([99.0, 98.8, 98.6], start=1):
            engine._on_ticker_event(
                timestamp=idx * 1000,
                best_bid=mid - 0.1,
                best_ask=mid + 0.1,
                intensity_value=0.0,
                volatility_scalar=0.0,
            )
        self.assertGreater(len(price_levels(engine, engine.manager.books.ask_maker)), 0)
        self.assertGreater(len(price_levels(engine, engine.manager.books.bid_maker)), 0)

        engine._on_ticker_event(
            timestamp=4000,
            best_bid=98.3,
            best_ask=98.5,
            intensity_value=0.0,
            volatility_scalar=0.0,
        )
        self.assertTrue(engine.snapshot_state()["toxic_lock_active"])
        self.assertGreater(len(price_levels(engine, engine.manager.books.ask_maker)), 0)
        self.assertEqual(price_levels(engine, engine.manager.books.bid_maker), [])

        engine._on_ticker_event(
            timestamp=5000,
            best_bid=98.4,
            best_ask=98.6,
            intensity_value=0.0,
            volatility_scalar=0.0,
        )
        self.assertTrue(engine.snapshot_state()["toxic_lock_active"])
        self.assertEqual(price_levels(engine, engine.manager.books.bid_maker), [])

        engine._on_ticker_event(
            timestamp=6000,
            best_bid=98.5,
            best_ask=98.7,
            intensity_value=0.0,
            volatility_scalar=0.0,
        )
        self.assertFalse(engine.snapshot_state()["toxic_lock_active"])
        self.assertGreater(len(price_levels(engine, engine.manager.books.bid_maker)), 0)

    def test_toxic_lock_mutes_short_open_ask_until_slip_count_recovers(self):
        engine = SimpleMakerStrategy(
            make_config(
                toxic_lock=(3, 2),
                min_order_notional=100.0,
                open_curve=(1.0, 1.0, 1.0),
                close_curve=(1.0, 1.0, 1.0),
                strict_mode=False,
            )
        )
        set_position(engine, qty=-1.0, cost=100.0)

        for idx, mid in enumerate([101.0, 101.2, 101.4], start=1):
            engine._on_ticker_event(
                timestamp=idx * 1000,
                best_bid=mid - 0.1,
                best_ask=mid + 0.1,
                intensity_value=0.0,
                volatility_scalar=0.0,
            )
        self.assertGreater(len(price_levels(engine, engine.manager.books.ask_maker)), 0)
        self.assertGreater(len(price_levels(engine, engine.manager.books.bid_maker)), 0)

        engine._on_ticker_event(
            timestamp=4000,
            best_bid=101.5,
            best_ask=101.7,
            intensity_value=0.0,
            volatility_scalar=0.0,
        )
        self.assertTrue(engine.snapshot_state()["toxic_lock_active"])
        self.assertEqual(price_levels(engine, engine.manager.books.ask_maker), [])
        self.assertGreater(len(price_levels(engine, engine.manager.books.bid_maker)), 0)

        engine._on_ticker_event(
            timestamp=5000,
            best_bid=101.4,
            best_ask=101.6,
            intensity_value=0.0,
            volatility_scalar=0.0,
        )
        self.assertTrue(engine.snapshot_state()["toxic_lock_active"])
        self.assertEqual(price_levels(engine, engine.manager.books.ask_maker), [])

        engine._on_ticker_event(
            timestamp=6000,
            best_bid=101.3,
            best_ask=101.5,
            intensity_value=0.0,
            volatility_scalar=0.0,
        )
        self.assertFalse(engine.snapshot_state()["toxic_lock_active"])
        self.assertGreater(len(price_levels(engine, engine.manager.books.ask_maker)), 0)

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

        self.assertGreater(len(price_levels(engine, engine.manager.books.bid_maker)), 0)
        self.assertEqual(price_levels(engine, engine.manager.books.ask_maker), [(100.6, 1.0)])

    def test_long_profitzone_uses_profitzone_close_boost(self):
        engine = SimpleMakerStrategy(
            make_config(
                min_order_notional=100.0,
                close_curve=(1.0, 1.0, 1.0),
                boost_underwater=(1.0, 1.0),
                boost_profitzone=(1.0, 2.0),
            )
        )
        set_position(engine, qty=5.0, cost=100.0)

        engine._on_ticker_event(
            timestamp=1,
            best_bid=100.4,
            best_ask=100.6,
            intensity_value=0.0,
            volatility_scalar=0.0,
        )

        self.assertGreater(len(price_levels(engine, engine.manager.books.bid_maker)), 0)
        self.assertEqual(price_levels(engine, engine.manager.books.ask_maker), [(100.6, 1.99)])

    def test_long_above_cost_close_curve_floors_fractional_units(self):
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

        self.assertGreater(len(price_levels(engine, engine.manager.books.bid_maker)), 0)
        self.assertEqual(price_levels(engine, engine.manager.books.ask_maker), [])

    def test_close_curve_qty_is_integer_min_bet_multiple(self):
        engine = SimpleMakerStrategy(
            make_config(
                max_position_usdt=100.0,
                min_order_notional=100.0,
                close_curve=(0.0, 3.0, 1.0),
            )
        )
        set_position(engine, qty=1.0, cost=100.0)

        self.assertEqual(engine._curve_close_qty(mid=100.0, pos_qty=1.0), 3.0)

    def test_close_curve_places_position_cap_after_integer_min_bet_sizing(self):
        engine = SimpleMakerStrategy(
            make_config(
                max_position_usdt=100.0,
                min_order_notional=100.0,
                close_curve=(0.0, 3.0, 1.0),
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
        self.assertEqual(price_levels(engine, engine.manager.books.ask_maker), [(100.6, 1.0)])

    def test_close_curve_allows_full_close_when_position_below_min_bet(self):
        engine = SimpleMakerStrategy(
            make_config(
                min_order_notional=100.0,
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

        self.assertGreater(len(price_levels(engine, engine.manager.books.bid_maker)), 0)
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

        self.assertGreater(len(price_levels(engine, engine.manager.books.bid_maker)), 0)
        self.assertEqual(price_levels(engine, engine.manager.books.ask_maker), [(101.0, 1.0)])

    def test_non_strict_long_underwater_places_close_ask_and_open_bid(self):
        engine = SimpleMakerStrategy(
            make_config(
                strict_mode=False,
                min_order_notional=100.0,
                open_curve=(1.0, 1.0, 1.0),
                close_curve=(1.0, 20.0, 1.0),
            )
        )
        set_position(engine, qty=1.0, cost=100.0)

        engine._on_ticker_event(
            timestamp=1,
            best_bid=98.9,
            best_ask=99.1,
            intensity_value=0.0,
            volatility_scalar=0.0,
        )

        self.assertEqual(price_levels(engine, engine.manager.books.ask_maker), [(99.1, 1.0)])
        self.assertEqual(price_levels(engine, engine.manager.books.bid_maker), [(98.9, 1.012)])

    def test_non_strict_short_above_cost_places_open_ask_and_close_bid(self):
        engine = SimpleMakerStrategy(
            make_config(
                strict_mode=False,
                min_order_notional=100.0,
                open_curve=(1.0, 1.0, 1.0),
                close_curve=(1.0, 20.0, 1.0),
            )
        )
        set_position(engine, qty=-1.0, cost=100.0)

        engine._on_ticker_event(
            timestamp=1,
            best_bid=100.4,
            best_ask=100.6,
            intensity_value=0.0,
            volatility_scalar=0.0,
        )

        self.assertEqual(price_levels(engine, engine.manager.books.ask_maker), [(100.6, 0.995)])
        self.assertEqual(price_levels(engine, engine.manager.books.bid_maker), [(100.4, 0.997)])

    def test_hold_since_blocks_underwater_close_at_position_threshold_but_keeps_open_side(self):
        for simple_mode in (True, False):
            with self.subTest(simple_mode=simple_mode, position="long"):
                engine = SimpleMakerStrategy(
                    make_config(
                        simple_mode=simple_mode,
                        strict_mode=False,
                        hold_since=10.0,
                        open_curve=(1.0, 1.0, 1.0),
                        close_curve=(1.0, 20.0, 1.0),
                    )
                )
                set_position(engine, qty=0.2, cost=100.0)

                engine._on_ticker_event(
                    timestamp=1,
                    best_bid=89.7,
                    best_ask=89.9,
                    intensity_value=0.0,
                    volatility_scalar=0.0,
                )

                self.assertEqual(price_levels(engine, engine.manager.books.ask_maker), [])
                self.assertGreater(len(price_levels(engine, engine.manager.books.bid_maker)), 0)

            with self.subTest(simple_mode=simple_mode, position="short"):
                engine = SimpleMakerStrategy(
                    make_config(
                        simple_mode=simple_mode,
                        strict_mode=False,
                        hold_since=10.0,
                        open_curve=(1.0, 1.0, 1.0),
                        close_curve=(1.0, 20.0, 1.0),
                    )
                )
                set_position(engine, qty=-0.2, cost=100.0)

                engine._on_ticker_event(
                    timestamp=1,
                    best_bid=110.1,
                    best_ask=110.3,
                    intensity_value=0.0,
                    volatility_scalar=0.0,
                )

                self.assertGreater(len(price_levels(engine, engine.manager.books.ask_maker)), 0)
                self.assertEqual(price_levels(engine, engine.manager.books.bid_maker), [])

    def test_hold_since_blocks_close_at_exact_position_threshold(self):
        engine = SimpleMakerStrategy(
            make_config(
                strict_mode=False,
                hold_since=100.0,
                open_curve=(1.0, 1.0, 1.0),
                close_curve=(1.0, 20.0, 1.0),
            )
        )
        set_position(engine, qty=1.0, cost=100.0)

        engine._on_ticker_event(
            timestamp=1,
            best_bid=89.9,
            best_ask=90.1,
            intensity_value=0.0,
            volatility_scalar=0.0,
        )

        self.assertEqual(price_levels(engine, engine.manager.books.ask_maker), [])
        self.assertGreater(len(price_levels(engine, engine.manager.books.bid_maker)), 0)

    def test_hold_since_does_not_block_underwater_close_below_position_threshold(self):
        engine = SimpleMakerStrategy(
            make_config(
                strict_mode=False,
                hold_since=101.0,
                open_curve=(1.0, 1.0, 1.0),
                close_curve=(1.0, 20.0, 1.0),
            )
        )
        set_position(engine, qty=1.0, cost=100.0)

        engine._on_ticker_event(
            timestamp=1,
            best_bid=89.9,
            best_ask=90.1,
            intensity_value=0.0,
            volatility_scalar=0.0,
        )

        self.assertGreater(len(price_levels(engine, engine.manager.books.ask_maker)), 0)
        self.assertGreater(len(price_levels(engine, engine.manager.books.bid_maker)), 0)

    def test_hold_since_keeps_profitzone_close_active(self):
        engine = SimpleMakerStrategy(
            make_config(
                strict_mode=False,
                hold_since=10.0,
                open_curve=(1.0, 1.0, 1.0),
                close_curve=(1.0, 20.0, 1.0),
            )
        )
        set_position(engine, qty=1.0, cost=100.0)

        engine._on_ticker_event(
            timestamp=1,
            best_bid=109.9,
            best_ask=110.1,
            intensity_value=0.0,
            volatility_scalar=0.0,
        )

        self.assertGreater(len(price_levels(engine, engine.manager.books.ask_maker)), 0)

    def test_flat_open_quote_uses_open_spread_adjustments(self):
        engine = SimpleMakerStrategy(
            make_config(
                adj_spread_intensity=(3.0, 1.0),
                adj_spread_volatility=(2.0, 0.0),
            )
        )

        engine._on_ticker_event(
            timestamp=1,
            best_bid=100.4,
            best_ask=100.6,
            intensity_value=0.2,
            volatility_scalar=0.1,
        )

        self.assertEqual(price_levels(engine, engine.manager.books.ask_maker), [(101.4, 1.0)])
        self.assertEqual(price_levels(engine, engine.manager.books.bid_maker), [(99.6, 1.0)])

    def test_flat_open_quote_uses_directional_intensity_by_quote_side(self):
        engine = SimpleMakerStrategy(
            make_config(
                adj_spread_intensity=(3.0, 1.0),
                adj_spread_volatility=(0.0, 0.0),
            )
        )

        engine._on_ticker_event(
            timestamp=1,
            best_bid=100.4,
            best_ask=100.6,
            intensity_positive=0.2,
            intensity_negative=0.5,
            volatility_scalar=0.0,
        )

        self.assertEqual(price_levels(engine, engine.manager.books.ask_maker), [(101.2, 1.0)])
        self.assertEqual(price_levels(engine, engine.manager.books.bid_maker), [(98.9, 1.0)])

    def test_min_quote_distance_bps_floors_simple_open_quotes_to_ticks(self):
        engine = SimpleMakerStrategy(make_config(min_quote_distance_bps=59.0))

        engine._on_ticker_event(
            timestamp=1,
            best_bid=99.8,
            best_ask=100.2,
            intensity_value=0.0,
            volatility_scalar=0.0,
        )

        self.assertEqual(price_levels(engine, engine.manager.books.ask_maker), [(100.7, 1.0)])
        self.assertEqual(price_levels(engine, engine.manager.books.bid_maker), [(99.3, 1.0)])

    def test_min_quote_distance_bps_floors_level_open_quotes_to_ticks(self):
        engine = SimpleMakerStrategy(
            make_config(
                simple_mode=False,
                min_quote_distance_bps=59.0,
            )
        )

        engine._on_ticker_event(
            timestamp=1,
            best_bid=99.8,
            best_ask=100.2,
            intensity_value=0.0,
            volatility_scalar=0.0,
        )

        self.assertEqual(price_levels(engine, engine.manager.books.ask_maker), [(100.7, 1.0)])
        self.assertEqual(price_levels(engine, engine.manager.books.bid_maker), [(99.3, 1.0)])

    def test_long_close_quote_uses_close_spread_adjustments(self):
        engine = SimpleMakerStrategy(
            make_config(
                adj_spread_intensity=(3.0, 1.0),
                adj_spread_volatility=(2.0, 0.0),
            )
        )
        set_position(engine, qty=1.0, cost=100.0)

        engine._on_ticker_event(
            timestamp=1,
            best_bid=100.4,
            best_ask=100.6,
            intensity_value=0.2,
            volatility_scalar=0.1,
        )

        self.assertGreater(len(price_levels(engine, engine.manager.books.bid_maker)), 0)
        self.assertEqual(price_levels(engine, engine.manager.books.ask_maker), [(100.8, 1.0)])

    def test_min_quote_distance_bps_clips_simple_close_quote(self):
        engine = SimpleMakerStrategy(make_config(min_quote_distance_bps=200.0))
        set_position(engine, qty=1.0, cost=100.0)

        engine._on_ticker_event(
            timestamp=1,
            best_bid=100.4,
            best_ask=100.6,
            intensity_value=0.0,
            volatility_scalar=0.0,
        )

        self.assertGreater(len(price_levels(engine, engine.manager.books.bid_maker)), 0)
        self.assertEqual(price_levels(engine, engine.manager.books.ask_maker), [(102.6, 1.0)])

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

        self.assertGreater(len(price_levels(engine, engine.manager.books.bid_maker)), 0)
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

        self.assertGreater(len(price_levels(engine, engine.manager.books.ask_maker)), 0)
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

        self.assertGreater(len(price_levels(engine, engine.manager.books.ask_maker)), 0)
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

        self.assertGreater(len(price_levels(engine, engine.manager.books.ask_maker)), 0)
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
        self.assertEqual(state["takeprofit_usdt"], 0.0)
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
    def test_report_normalization_adds_datetime_for_current_columns(self):
        import pandas as pd

        df = pd.DataFrame(
            {
                "timestamp": [1, 2],
                "total_pnl": [20.0, 20.0],
                "realized_pnl": [0.0, 0.0],
                "unrealized_pnl": [20.0, 20.0],
                "traded_volume": [100.0, 200.0],
                "price": [110.0, 90.0],
                "position": [2.0, -2.0],
                "mark_notional_usdt": [220.0, -180.0],
                "cost_notional_usdt": [200.0, -200.0],
                "gross_cost_notional_usdt": [200.0, 200.0],
            }
        )

        out = _normalize_report_columns(df)

        self.assertIn("datetime", out.columns)
        self.assertEqual(list(out["mark_notional_usdt"]), [220.0, -180.0])
        self.assertEqual(list(out["cost_notional_usdt"]), [200.0, -200.0])
        self.assertEqual(list(out["gross_cost_notional_usdt"]), [200.0, 200.0])

    def test_load_report_frame_samples_before_plotting_across_files(self):
        import tempfile
        from pathlib import Path

        import pandas as pd

        def make_frame(start: int, stop: int) -> pd.DataFrame:
            timestamps = list(range(start, stop))
            return pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "total_pnl": [float(value) for value in timestamps],
                    "realized_pnl": [0.0 for _ in timestamps],
                    "unrealized_pnl": [float(value) for value in timestamps],
                    "traded_volume": [float(value * 10) for value in timestamps],
                    "price": [100.0 for _ in timestamps],
                    "mark_notional_usdt": [0.0 for _ in timestamps],
                    "cost_notional_usdt": [0.0 for _ in timestamps],
                    "gross_cost_notional_usdt": [0.0 for _ in timestamps],
                    "position": [0.0 for _ in timestamps],
                }
            )

        with tempfile.TemporaryDirectory() as tmp_dir:
            folder = Path(tmp_dir)
            make_frame(0, 5).to_parquet(folder / "2025-01-01.parquet", index=False)
            make_frame(5, 10).to_parquet(folder / "2025-01-02.parquet", index=False)

            out = load_report_frame(folder, every=3)

        self.assertEqual(list(out["timestamp"]), [0, 3, 4, 5, 6, 9])
        self.assertIn("datetime", out.columns)

    def test_limit_plot_frame_keeps_boundaries_within_point_cap(self):
        import pandas as pd

        df = pd.DataFrame({"timestamp": list(range(11)), "total_pnl": list(range(11))})

        out = _limit_plot_frame(df, max_points=4)

        self.assertLessEqual(len(out), 4)
        self.assertEqual(int(out["timestamp"].iloc[0]), 0)
        self.assertEqual(int(out["timestamp"].iloc[-1]), 10)

    def test_state_cost_notional_requires_current_state_field(self):
        self.assertEqual(
            _state_cost_notional_usdt(
                {"position": {"qty": 2.0, "cost": 100.0, "cost_notional_usdt": 201.0}}
            ),
            201.0,
        )
        self.assertIsNone(_state_cost_notional_usdt({"position": {"qty": -2.0, "cost": 100.0}}))


if __name__ == "__main__":
    unittest.main()
