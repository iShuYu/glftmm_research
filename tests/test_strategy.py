import unittest

import pandas as pd

from core.position import Position
from run_strategy import (
    _build_simulation_config,
    _build_tasks,
    _normalize_sim_param_map,
    build_param_path_parts,
)
from sim.loader import BinanceEventLoader, build_confirmed_aggtrade_events
from sim.report import (
    _compute_daily_pnl_from_total,
    _parse_daily_parallel_from_folder,
    _stitch_daily_parallel_frame,
    _stitch_daily_parallel_series,
)
from sim.strategy import SimpleMakerStrategy, SimulationConfig, get_minimum_size


def make_config(**overrides):
    values = {
        "latency": 0,
        "price_precision": 1,
        "qty_precision": 3,
        "mode": 1,
        "taker_fee": 0.00025,
        "maker_fee": -0.00003,
        "max_position_usdt": 100000.0,
        "max_open_inventory_utilization": 1.0,
        "max_holding_time": -1,
        "adj_spread_intensity": (1.0, 1.0),
        "ewma_intensity": 1.0,
        "consequtive_sameside": 0,
        "min_quote_distance_bps": 0.0,
        "inventory_skew": (0.0, 1.0),
        "min_order_qty": 0.0,
        "min_order_notional": 0.0,
        "stoploss": 0.0,
        "open_curve": None,
        "close_curve": None,
        "boost_underwater": (1.0, 1.0),
        "boost_profitzone": (1.0, 1.0),
        "cooldown_time": 0,
        "strict_mode": True,
        "simple_mode": True,
        "daily_parallel": False,
    }
    values.update(overrides)
    return SimulationConfig(**values)


def raw_config(**overrides):
    values = {
        "latency": 0,
        "price_precision": 1,
        "qty_precision": 3,
        "mode": 1,
        "taker_fee": 0.00025,
        "maker_fee": -0.00003,
        "max_position_usdt": 100000.0,
        "max_open_inventory_utilization": 1.0,
        "max_holding_time": -1,
        "adj_spread_intensity": [1.0, 1.0],
        "ewma_intensity": 1.0,
        "consequtive_sameside": 0,
        "min_quote_distance_bps": 0.0,
        "min_order_qty": 0.0,
        "min_order_notional": 0.0,
        "stoploss": 0.0,
        "boost_underwater": [1.0, 1.0],
        "boost_profitzone": [1.0, 1.0],
        "cooldown_time": 0,
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


class StaticEventLoader:
    def __init__(self, events):
        self.events = list(events)
        self.calls = []

    def iter_merged_trade_intensity_tuples(self, **kwargs):
        self.calls.append(kwargs)
        yield from self.events


class AggTradeIntensityLoaderTest(unittest.TestCase):
    def test_confirmed_aggtrade_uses_next_opposite_side_trade_as_timestamp(self):
        trades = pd.DataFrame(
            {
                "timestamp": [1, 2, 3, 4],
                "is_buyer_maker": [False, False, True, True],
                "price": [100.0, 102.0, 101.0, 99.0],
                "volume": [1.0, 2.0, 3.0, 4.0],
            }
        )

        events = build_confirmed_aggtrade_events(trades, force_close_timestamp=9)

        self.assertEqual(events["timestamp"].tolist(), [3, 9])
        self.assertEqual(events["is_buyer_maker"].tolist(), [False, True])
        self.assertEqual(events["impact"].tolist(), [2.0, 2.0])
        self.assertEqual(events["volume"].tolist(), [3.0, 7.0])
        self.assertEqual(events["forced_close"].tolist(), [False, True])

    def test_merge_events_emits_trade_before_bbo_before_aggtrade(self):
        trades = pd.DataFrame(
            {
                "timestamp": [1000],
                "is_buyer_maker": [False],
                "price": [100.0],
                "volume": [1.0],
            }
        )
        bookticker = pd.DataFrame(
            {
                "timestamp": [1000],
                "best_bid_price": [99.0],
                "best_ask_price": [101.0],
                "best_bid_qty": [2.0],
                "best_ask_qty": [3.0],
            }
        )
        aggtrades = pd.DataFrame(
            {
                "timestamp": [1000],
                "is_buyer_maker": [False],
                "impact": [1.5],
                "intensity": [1.5],
                "volume": [1.0],
                "first_price": [100.0],
                "last_price": [101.5],
                "forced_close": [False],
            }
        )

        events = list(
            BinanceEventLoader._merge_events(
                trades=trades,
                bookticker=bookticker,
                aggtrades=aggtrades,
            )
        )

        self.assertEqual([event[0] for event in events], ["trade", "bookticker", "aggtrade"])


class PositionNotionalTest(unittest.TestCase):
    def test_position_tracks_mark_and_cost_notional_separately(self):
        pos = Position()
        pos.execute(qty=2.0, price=100.0)
        pos.mark(110.0)

        self.assertEqual(pos.mark_notional_usdt, 220.0)
        self.assertEqual(pos.cost_notional_usdt, 200.0)
        self.assertEqual(pos.gross_cost_notional_usdt, 200.0)
        self.assertEqual(pos.unrealized_pnl, 20.0)

    def test_short_notional_preserves_sign(self):
        pos = Position()
        pos.execute(qty=-2.0, price=100.0)
        pos.mark(90.0)

        self.assertEqual(pos.mark_notional_usdt, -180.0)
        self.assertEqual(pos.cost_notional_usdt, -200.0)
        self.assertEqual(pos.gross_cost_notional_usdt, 200.0)
        self.assertEqual(pos.unrealized_pnl, 20.0)


class StrategyConfigTest(unittest.TestCase):
    def test_build_config_has_no_removed_axes(self):
        cfg = _build_simulation_config(raw_config(adj_spread_intensity=[2.0, 0.5])["simulation"])

        self.assertFalse(hasattr(cfg, "fr" + "eq"))
        self.assertFalse(hasattr(cfg, "name_" + "intensity"))
        self.assertFalse(hasattr(cfg, "lookback_" + "intensity"))
        self.assertFalse(hasattr(cfg, "name_" + "instr" + "uctor"))
        self.assertFalse(hasattr(cfg, "name_" + "vola" + "tility"))
        self.assertFalse(hasattr(cfg, "passive_" + "only"))
        self.assertEqual(cfg.adj_spread_intensity, (2.0, 0.5))
        self.assertEqual(cfg.ewma_intensity, 1.0)
        self.assertEqual(cfg.consequtive_sameside, 0)

    def test_build_config_accepts_ewma_intensity(self):
        sim_map = _normalize_sim_param_map(raw_config(ewma_intensity=[0.2, 1.0]))
        cfg = _build_simulation_config(raw_config(ewma_intensity=0.2)["simulation"])

        self.assertEqual(sim_map["ewma_intensity"], [0.2, 1.0])
        self.assertEqual(cfg.ewma_intensity, 0.2)

    def test_ewma_intensity_validation(self):
        for value in (0.0, -0.1, 1.1, float("inf")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "ewma_intensity"):
                    _normalize_sim_param_map(raw_config(ewma_intensity=value))

    def test_build_config_accepts_consequtive_sameside(self):
        sim_map = _normalize_sim_param_map(raw_config(consequtive_sameside=[0, 3]))
        cfg = _build_simulation_config(
            raw_config(consequtive_sameside=3)["simulation"]
        )

        self.assertEqual(sim_map["consequtive_sameside"], [0, 3])
        self.assertEqual(cfg.consequtive_sameside, 3)

    def test_consequtive_sameside_validation(self):
        for value in (-1, 1.5, True, float("inf")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "consequtive_sameside"):
                    _normalize_sim_param_map(raw_config(consequtive_sameside=value))

    def test_normalize_rejects_removed_simulation_keys(self):
        removed = {
            "fr" + "eq": 1000,
            "name_" + "intensity": "k",
            "lookback_" + "intensity": 300,
            "name_" + "instr" + "uctor": "bbo_imbalance",
            "lookback_" + "instr" + "uctor": 0,
            "adj_spread_" + "instr" + "uctor": 0.0,
            "name_" + "vola" + "tility": "sigma",
            "lookback_" + "vola" + "tility": 300,
            "adj_spread_" + "vola" + "tility": [0.0, 0.0],
            "optimize_by_" + "orderbook": -1,
            "event_driven_" + "quotes": True,
            "passive_" + "only": True,
        }
        for key, value in removed.items():
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, "unknown simulation keys"):
                    _normalize_sim_param_map(raw_config(**{key: value}))

    def test_normalizes_open_close_spread_and_boost_rows(self):
        sim_map = _normalize_sim_param_map(
            raw_config(
                adj_spread_intensity=[[3.0, 1.0], [4.0, 1.5]],
                boost_underwater=[[2.0, 0.5], [3.0, 0.25]],
                boost_profitzone=[[0.5, 2.0], [0.25, 3.0]],
            )
        )

        self.assertEqual(sim_map["adj_spread_intensity"], [[3.0, 1.0], [4.0, 1.5]])
        self.assertEqual(sim_map["boost_underwater"], [[2.0, 0.5], [3.0, 0.25]])
        self.assertEqual(sim_map["boost_profitzone"], [[0.5, 2.0], [0.25, 3.0]])

    def test_build_tasks_crosses_remaining_parameter_grids(self):
        tasks = _build_tasks(
            raw_task_config(
                mode=[0, 1],
                max_position_usdt=[250.0, 500.0],
                stoploss=[50.0, 100.0],
            )
        )

        self.assertEqual(len(tasks), 8)
        self.assertEqual([task.sim_params["mode"] for task in tasks].count(0), 4)
        self.assertEqual([task.sim_params["mode"] for task in tasks].count(1), 4)

    def test_daily_parallel_fans_out_tasks_by_day(self):
        cfg = raw_task_config(mode=[0, 1])
        cfg["date_end"] = "2025-01-03"
        cfg["daily_parallel"] = True

        tasks = _build_tasks(cfg)

        self.assertEqual(len(tasks), 6)
        self.assertTrue(all(len(task.dates) == 1 for task in tasks))
        self.assertEqual(
            sorted((task.sim_params["mode"], task.dates[0]) for task in tasks),
            [
                (0, "2025-01-01"),
                (0, "2025-01-02"),
                (0, "2025-01-03"),
                (1, "2025-01-01"),
                (1, "2025-01-02"),
                (1, "2025-01-03"),
            ],
        )

    def test_non_daily_parallel_keeps_dates_in_one_task(self):
        cfg = raw_task_config(mode=[0, 1])
        cfg["date_end"] = "2025-01-03"

        tasks = _build_tasks(cfg)

        self.assertEqual(len(tasks), 2)
        self.assertTrue(
            all(
                task.dates == ["2025-01-01", "2025-01-02", "2025-01-03"]
                for task in tasks
            )
        )

    def test_daily_parallel_path_uses_parseable_token(self):
        sim_dir, _strat_dir = build_param_path_parts(
            {"mode": 0, "daily_parallel": True}
        )

        self.assertIn("dp1", sim_dir)
        self.assertTrue(
            _parse_daily_parallel_from_folder(f"/tmp/BTCUSDT/{sim_dir}/strat__all")
        )
        self.assertTrue(
            _parse_daily_parallel_from_folder(
                "/tmp/BTCUSDT/sim__daily_paralleltrue__md0/strat__all"
            )
        )
        self.assertFalse(
            _parse_daily_parallel_from_folder("/tmp/BTCUSDT/sim__md0/strat__all")
        )


class ReportDailyParallelTest(unittest.TestCase):
    def test_stitches_independent_daily_pnl_frames(self):
        df = pd.DataFrame(
            {
                "datetime": pd.to_datetime(
                    [
                        "2025-01-01 00:00:00",
                        "2025-01-01 23:59:00",
                        "2025-01-02 00:00:00",
                        "2025-01-02 23:59:00",
                    ]
                ),
                "total_pnl": [0.0, 10.0, 0.0, 5.0],
                "realized_pnl": [0.0, 8.0, 0.0, 4.0],
                "unrealized_pnl": [0.0, 2.0, 0.0, 1.0],
                "traded_volume": [0.0, 100.0, 0.0, 20.0],
            }
        )

        stitched = _stitch_daily_parallel_frame(df)

        self.assertEqual(stitched["total_pnl"].tolist(), [0.0, 10.0, 10.0, 15.0])
        self.assertEqual(stitched["realized_pnl"].tolist(), [0.0, 8.0, 8.0, 12.0])
        self.assertEqual(stitched["traded_volume"].tolist(), [0.0, 100.0, 100.0, 120.0])
        self.assertEqual(stitched["unrealized_pnl"].tolist(), [0.0, 2.0, 0.0, 1.0])

    def test_stitches_daily_state_series_and_keeps_first_daily_pnl(self):
        totals = pd.Series([10.0, 5.0], index=["20250101", "20250102"])
        volumes = pd.Series([100.0, 20.0], index=["20250101", "20250102"])

        stitched_total, stitched_volume = _stitch_daily_parallel_series(totals, volumes)

        self.assertEqual(stitched_total.tolist(), [10.0, 15.0])
        self.assertEqual(stitched_volume.tolist(), [100.0, 120.0])
        self.assertEqual(_compute_daily_pnl_from_total(stitched_total).tolist(), [10.0, 5.0])


class StrategyQuoteTest(unittest.TestCase):
    def test_get_minimum_size_rounds_up_to_qty_precision(self):
        self.assertEqual(
            get_minimum_size(mid=30000.0, qty_precision=3, min_order_notional=100.0),
            0.004,
        )
        self.assertEqual(
            get_minimum_size(mid=100.0, qty_precision=3, min_order_notional=0.0),
            0.001,
        )

    def test_quote_distance_is_pure_intensity(self):
        engine = SimpleMakerStrategy(make_config(adj_spread_intensity=(2.0, 0.5)))

        self.assertEqual(engine._quote_distance(intensity_base=3.0, close=False), 6.0)
        self.assertEqual(engine._quote_distance(intensity_base=3.0, close=True), 1.5)

    def test_run_day_places_only_aggtrade_side_quote(self):
        loader = StaticEventLoader(
            [
                ("bookticker", 1000, 100.0, 101.0, 1.0, 1.0),
                ("aggtrade", 1000, False, 2.0, 2.0, 1.0, 101.0, 103.0),
            ]
        )
        engine = SimpleMakerStrategy(
            make_config(
                adj_spread_intensity=(1.0, 1.0),
                min_order_notional=10.0,
                inventory_skew=None,
            ),
            loader=loader,
        )

        day = engine.run_day(symbol="BTCUSDT", date="2025-01-01")
        books = engine.manager.snapshot_ticks()

        self.assertEqual(loader.calls, [{"symbol": "BTCUSDT", "date": "2025-01-01"}])
        self.assertEqual(len(day), 1)
        self.assertEqual(engine.snapshot_state()["latest_buy_intensity"], 2.0)
        self.assertEqual(engine.snapshot_state()["latest_sell_intensity"], 0.0)
        self.assertEqual(books["ask_maker"][0][0], 1030)
        self.assertGreater(books["ask_maker"][0][1], 0)
        self.assertEqual(books["bid_maker"], [])

    def test_zero_intensity_aggtrade_does_not_place_quotes(self):
        loader = StaticEventLoader(
            [
                ("bookticker", 1000, 100.0, 101.0, 1.0, 1.0),
                ("aggtrade", 1000, False, 0.0, 0.0, 1.0, 101.0, 101.0),
            ]
        )
        engine = SimpleMakerStrategy(
            make_config(
                adj_spread_intensity=(1.0, 1.0),
                min_order_notional=10.0,
                inventory_skew=None,
            ),
            loader=loader,
        )

        day = engine.run_day(symbol="BTCUSDT", date="2025-01-01")
        books = engine.manager.snapshot_ticks()

        self.assertEqual(len(day), 0)
        self.assertEqual(engine.snapshot_state()["latest_buy_intensity"], 0.0)
        self.assertEqual(engine.snapshot_state()["latest_sell_intensity"], 0.0)
        self.assertEqual(books["ask_maker"], [])
        self.assertEqual(books["bid_maker"], [])

    def test_zero_intensity_aggtrade_keeps_existing_quotes_unchanged(self):
        loader = StaticEventLoader(
            [
                ("bookticker", 1000, 100.0, 101.0, 1.0, 1.0),
                ("aggtrade", 1000, False, 2.0, 2.0, 1.0, 101.0, 103.0),
                ("bookticker", 1001, 100.5, 101.5, 1.0, 1.0),
                ("aggtrade", 1001, False, 0.0, 0.0, 1.0, 101.5, 101.5),
            ]
        )
        engine = SimpleMakerStrategy(
            make_config(
                adj_spread_intensity=(1.0, 1.0),
                min_order_notional=10.0,
                inventory_skew=None,
            ),
            loader=loader,
        )

        day = engine.run_day(symbol="BTCUSDT", date="2025-01-01")
        books = engine.manager.snapshot_ticks()

        self.assertEqual(len(day), 1)
        self.assertEqual(engine.snapshot_state()["latest_buy_intensity"], 2.0)
        self.assertEqual(books["ask_maker"][0][0], 1030)
        self.assertEqual(books["bid_maker"], [])

    def test_sell_aggtrade_affects_bid_not_ask(self):
        loader = StaticEventLoader(
            [
                ("bookticker", 1000, 100.0, 101.0, 1.0, 1.0),
                ("aggtrade", 1000, True, 2.0, 2.0, 1.0, 100.0, 98.0),
            ]
        )
        engine = SimpleMakerStrategy(
            make_config(
                adj_spread_intensity=(1.0, 1.0),
                min_order_notional=10.0,
                inventory_skew=None,
            ),
            loader=loader,
        )

        day = engine.run_day(symbol="BTCUSDT", date="2025-01-01")
        books = engine.manager.snapshot_ticks()

        self.assertEqual(len(day), 1)
        self.assertEqual(engine.snapshot_state()["latest_sell_intensity"], 2.0)
        self.assertEqual(engine.snapshot_state()["latest_buy_intensity"], 0.0)
        self.assertEqual(books["ask_maker"], [])
        self.assertEqual(books["bid_maker"][0][0], 980)

    def test_buy_and_sell_intensity_update_independently(self):
        loader = StaticEventLoader(
            [
                ("bookticker", 1000, 100.0, 101.0, 1.0, 1.0),
                ("aggtrade", 1000, False, 2.0, 2.0, 1.0, 101.0, 103.0),
                ("bookticker", 1001, 100.0, 102.0, 1.0, 1.0),
                ("aggtrade", 1001, True, 3.0, 3.0, 1.0, 100.0, 97.0),
            ]
        )
        engine = SimpleMakerStrategy(
            make_config(
                adj_spread_intensity=(1.0, 1.0),
                min_order_notional=10.0,
                inventory_skew=None,
            ),
            loader=loader,
        )

        day = engine.run_day(symbol="BTCUSDT", date="2025-01-01")
        books = engine.manager.snapshot_ticks()

        self.assertEqual(len(day), 2)
        self.assertEqual(engine.snapshot_state()["latest_buy_intensity"], 2.0)
        self.assertEqual(engine.snapshot_state()["latest_sell_intensity"], 3.0)
        self.assertEqual(books["ask_maker"][0][0], 1030)
        self.assertEqual(books["bid_maker"][0][0], 970)

    def test_ewma_intensity_smooths_same_side_and_keeps_sides_separate(self):
        loader = StaticEventLoader(
            [
                ("bookticker", 1000, 100.0, 101.0, 1.0, 1.0),
                ("aggtrade", 1000, False, 2.0, 2.0, 1.0, 101.0, 103.0),
                ("aggtrade", 1001, True, 6.0, 6.0, 1.0, 100.0, 94.0),
                ("aggtrade", 1002, False, 6.0, 6.0, 1.0, 101.0, 107.0),
            ]
        )
        engine = SimpleMakerStrategy(
            make_config(
                ewma_intensity=0.2,
                adj_spread_intensity=(1.0, 1.0),
                min_order_notional=10.0,
                inventory_skew=None,
            ),
            loader=loader,
        )

        engine.run_day(symbol="BTCUSDT", date="2025-01-01")
        state = engine.snapshot_state()
        books = engine.manager.snapshot_ticks()

        self.assertAlmostEqual(state["latest_buy_intensity"], 2.8)
        self.assertAlmostEqual(state["latest_sell_intensity"], 6.0)
        self.assertEqual(books["ask_maker"][0][0], 1038)
        self.assertEqual(books["bid_maker"][0][0], 940)

    def test_consequtive_sameside_blocks_quote_side_after_threshold(self):
        loader = StaticEventLoader(
            [
                ("bookticker", 1000, 100.0, 101.0, 1.0, 1.0),
                ("aggtrade", 1000, False, 2.0, 2.0, 1.0, 101.0, 103.0),
                ("aggtrade", 1001, False, 2.0, 2.0, 1.0, 101.0, 103.0),
            ]
        )
        engine = SimpleMakerStrategy(
            make_config(
                consequtive_sameside=2,
                adj_spread_intensity=(1.0, 1.0),
                min_order_notional=10.0,
                inventory_skew=None,
            ),
            loader=loader,
        )

        engine.run_day(symbol="BTCUSDT", date="2025-01-01")
        state = engine.snapshot_state()
        books = engine.manager.snapshot_ticks()

        self.assertEqual(state["last_nonzero_agg_side"], "ask")
        self.assertEqual(state["same_side_nonzero_count"], 2)
        self.assertTrue(state["blocked_quote_sides"]["ask"])
        self.assertEqual(books["ask_maker"], [])
        self.assertEqual(books["bid_maker"], [])

    def test_reverse_nonzero_unblocks_previous_quote_side(self):
        loader = StaticEventLoader(
            [
                ("bookticker", 1000, 100.0, 101.0, 1.0, 1.0),
                ("aggtrade", 1000, False, 2.0, 2.0, 1.0, 101.0, 103.0),
                ("aggtrade", 1001, False, 2.0, 2.0, 1.0, 101.0, 103.0),
                ("aggtrade", 1002, True, 2.0, 2.0, 1.0, 100.0, 98.0),
                ("aggtrade", 1003, False, 2.0, 2.0, 1.0, 101.0, 103.0),
            ]
        )
        engine = SimpleMakerStrategy(
            make_config(
                consequtive_sameside=2,
                adj_spread_intensity=(1.0, 1.0),
                min_order_notional=10.0,
                inventory_skew=None,
            ),
            loader=loader,
        )

        engine.run_day(symbol="BTCUSDT", date="2025-01-01")
        state = engine.snapshot_state()
        books = engine.manager.snapshot_ticks()

        self.assertEqual(state["last_nonzero_agg_side"], "ask")
        self.assertEqual(state["same_side_nonzero_count"], 1)
        self.assertFalse(state["blocked_quote_sides"]["ask"])
        self.assertFalse(state["blocked_quote_sides"]["bid"])
        self.assertEqual(books["ask_maker"][0][0], 1030)

    def test_consequtive_sameside_blocks_close_side(self):
        position = Position()
        position.execute(1.0, 100.0)
        loader = StaticEventLoader(
            [
                ("bookticker", 1000, 100.0, 101.0, 1.0, 1.0),
                ("aggtrade", 1000, False, 2.0, 2.0, 1.0, 101.0, 103.0),
                ("aggtrade", 1001, False, 2.0, 2.0, 1.0, 101.0, 103.0),
            ]
        )
        engine = SimpleMakerStrategy(
            make_config(
                consequtive_sameside=2,
                adj_spread_intensity=(1.0, 1.0),
                min_order_notional=10.0,
                inventory_skew=None,
            ),
            position=position,
            loader=loader,
        )

        engine.run_day(symbol="BTCUSDT", date="2025-01-01")
        state = engine.snapshot_state()
        books = engine.manager.snapshot_ticks()

        self.assertTrue(state["blocked_quote_sides"]["ask"])
        self.assertEqual(books["ask_maker"], [])
        self.assertEqual(books["bid_maker"], [])

    def test_latency_activates_due_quotes_on_both_sides(self):
        loader = StaticEventLoader(
            [
                ("bookticker", 1000, 100.0, 101.0, 1.0, 1.0),
                ("aggtrade", 1000, False, 2.0, 2.0, 1.0, 101.0, 103.0),
                ("aggtrade", 1000, True, 3.0, 3.0, 1.0, 100.0, 97.0),
                ("bookticker", 1010, 100.0, 101.0, 1.0, 1.0),
            ]
        )
        engine = SimpleMakerStrategy(
            make_config(
                latency=10,
                adj_spread_intensity=(1.0, 1.0),
                min_order_notional=10.0,
                inventory_skew=None,
            ),
            loader=loader,
        )

        engine.run_day(symbol="BTCUSDT", date="2025-01-01")
        books = engine.manager.snapshot_ticks()

        self.assertEqual(books["ask_maker"][0][0], 1030)
        self.assertEqual(books["bid_maker"][0][0], 970)

    def test_side_quote_close_side_is_capped_at_placement(self):
        position = Position()
        position.execute(1.0, 100.0)
        engine = SimpleMakerStrategy(make_config(), position=position)

        engine._activate_simple_maker_quote(
            (
                0,
                "ask",
                1200,
                2000,
                None,
                0,
                1100,
                900,
                False,
            )
        )
        books = engine.manager.snapshot_ticks()

        self.assertEqual(books["ask_maker"], [(1200, 1000)])

    def test_full_quote_caps_only_close_side_at_placement(self):
        position = Position()
        position.execute(1.0, 100.0)
        engine = SimpleMakerStrategy(make_config(), position=position)

        engine._activate_simple_maker_quote(
            (
                0,
                None,
                1200,
                2000,
                800,
                2000,
                1100,
                900,
                False,
            )
        )
        books = engine.manager.snapshot_ticks()

        self.assertEqual(books["ask_maker"], [(1200, 1000)])
        self.assertEqual(books["bid_maker"], [(800, 2000)])

    def test_daily_parallel_stoploss_flattens_then_stops_trading_until_day_end(self):
        loader = StaticEventLoader(
            [
                ("bookticker", 1000, 98.0, 99.0, 1.0, 1.0),
                ("aggtrade", 1000, False, 1.0, 1.0, 1.0, 99.0, 98.0),
                ("trade", 1001, True, 98.0, 1.0),
                ("bookticker", 1002, 100.0, 101.0, 1.0, 1.0),
                ("aggtrade", 1002, False, 2.0, 2.0, 1.0, 101.0, 103.0),
            ]
        )
        position = Position()
        position.execute(1.0, 100.0)
        engine = SimpleMakerStrategy(
            make_config(
                mode=0,
                stoploss=1.0,
                daily_parallel=True,
                min_order_notional=10.0,
                inventory_skew=None,
            ),
            loader=loader,
            position=position,
        )

        day = engine.run_day(symbol="BTCUSDT", date="2025-01-01")
        state = engine.snapshot_state()
        books = engine.manager.snapshot_ticks()

        self.assertEqual(day["timestamp"].tolist(), [1000, 1002])
        self.assertTrue(state["daily_stop_trading"])
        self.assertEqual(state["daily_stop_date"], "2025-01-01")
        self.assertAlmostEqual(state["position"]["qty"], 0.0)
        self.assertEqual(books["ask_maker"], [])
        self.assertEqual(books["bid_maker"], [])
        self.assertEqual(books["ask_taker"], [])
        self.assertEqual(books["bid_taker"], [])

    def test_non_daily_parallel_stoploss_can_quote_again_after_flattening(self):
        loader = StaticEventLoader(
            [
                ("bookticker", 1000, 98.0, 99.0, 1.0, 1.0),
                ("aggtrade", 1000, False, 1.0, 1.0, 1.0, 99.0, 98.0),
                ("trade", 1001, True, 98.0, 1.0),
                ("bookticker", 1002, 100.0, 101.0, 1.0, 1.0),
                ("aggtrade", 1002, False, 2.0, 2.0, 1.0, 101.0, 103.0),
            ]
        )
        position = Position()
        position.execute(1.0, 100.0)
        engine = SimpleMakerStrategy(
            make_config(
                mode=0,
                stoploss=1.0,
                daily_parallel=False,
                min_order_notional=10.0,
                inventory_skew=None,
            ),
            loader=loader,
            position=position,
        )

        engine.run_day(symbol="BTCUSDT", date="2025-01-01")
        state = engine.snapshot_state()
        books = engine.manager.snapshot_ticks()

        self.assertFalse(state["daily_stop_trading"])
        self.assertAlmostEqual(state["position"]["qty"], 0.0)
        self.assertGreater(len(books["ask_maker"]), 0)
        self.assertEqual(books["bid_maker"], [])


if __name__ == "__main__":
    unittest.main()
