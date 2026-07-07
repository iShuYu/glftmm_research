import tempfile
import unittest
from pathlib import Path

import pandas as pd

from sampler.intensity import build_tasks as build_intensity_tasks
from sampler.intensity import intensity_output_path
from sampler.instructor import (
    build_trade_instructor_frame,
    instructor_output_path,
    validate_instructor_frame,
)
from sampler.resample import (
    TickerResampler,
    build_tasks as build_resample_tasks,
    day_timestamp_grid,
    output_path as sampled_ticker_path,
)
from sampler.volatility import volatility_output_path
from sampler.run import build_stage_configs
from sim.loader import BinanceEventLoader


class SamplerInstructorTest(unittest.TestCase):
    def test_resample_scheme_shift_moves_grid_and_path(self):
        date = "2025-01-01"
        base_grid = day_timestamp_grid(date, 1000)
        shifted_grid = day_timestamp_grid(date, 1000, scheme_shift_ms=250)

        self.assertEqual(int(shifted_grid[0]), int(base_grid[0]) + 250)
        self.assertEqual(len(shifted_grid), len(base_grid))
        self.assertEqual(
            sampled_ticker_path(Path("/tmp/cache"), "BTCUSDT", 1000, date, 250),
            Path("/tmp/cache/BTCUSDT/resample/freq_1000ms/scheme_shift_250ms/2025-01-01.parquet"),
        )

        raw = pd.DataFrame(
            {
                "timestamp": [int(base_grid[0]) + 100, int(base_grid[0]) + 1500],
                "best_bid_price": [100.0, 101.0],
                "best_ask_price": [100.5, 101.5],
                "best_bid_qty": [1.0, 2.0],
                "best_ask_qty": [1.0, 2.0],
            }
        )
        out = TickerResampler(freq_ms=1000, scheme_shift_ms=250).resample(raw, date)

        self.assertEqual(int(out.loc[0, "timestamp"]), int(base_grid[0]) + 250)
        self.assertEqual(float(out.loc[0, "best_bid_price"]), 100.0)
        self.assertEqual(float(out.loc[1, "best_bid_price"]), 100.0)
        self.assertEqual(float(out.loc[2, "best_bid_price"]), 101.0)

    def test_trade_imbalance_uses_prior_buckets_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "TRADE"
            symbol = "BTCUSDT"
            date = "2025-01-01"
            day_start = int(day_timestamp_grid(date, 1000)[0])
            trade_dir = root / symbol
            trade_dir.mkdir(parents=True)
            pd.DataFrame(
                {
                    "exchange_timestamp": [
                        day_start,
                        day_start + 1000,
                        day_start + 1000,
                        day_start + 2000,
                    ],
                    "price": [100.0, 99.9, 100.1, 99.8],
                    "volume": [3.0, 1.0, 1.0, 2.0],
                    "is_buyer_maker": [False, True, False, True],
                }
            ).to_parquet(trade_dir / f"{symbol}--TRADE--{date}.parquet")

            out = build_trade_instructor_frame(
                symbol=symbol,
                date=date,
                freq_ms=1000,
                lookback=2,
                indicator="trade_imbalance",
                trade_roots=(root,),
            )

            self.assertEqual(len(out), 86400)
            self.assertEqual(out.columns.tolist(), ["timestamp", "instructor"])
            self.assertEqual(out.loc[0, "instructor"], 0.0)
            self.assertAlmostEqual(out.loc[1, "instructor"], -1.0)
            self.assertAlmostEqual(out.loc[2, "instructor"], -0.6)
            self.assertAlmostEqual(out.loc[3, "instructor"], 0.5)
            validate_instructor_frame(out, date_str=date, freq_ms=1000)

    def test_bbo_imbalance_uses_sampled_ticker_quantities(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "cache"
            symbol = "BTCUSDT"
            date = "2025-01-01"
            freq_ms = 60000
            timestamps = day_timestamp_grid(date, freq_ms).tolist()
            bid_qty = [1.0] * len(timestamps)
            ask_qty = [1.0] * len(timestamps)
            bid_qty[:3] = [3.0, 0.0, 2.0]
            ask_qty[:3] = [1.0, 5.0, 0.0]

            ticker_path = sampled_ticker_path(root, symbol, freq_ms, date)
            ticker_path.parent.mkdir(parents=True)
            pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "best_bid_price": [100.0] * len(timestamps),
                    "best_ask_price": [100.5] * len(timestamps),
                    "best_bid_qty": bid_qty,
                    "best_ask_qty": ask_qty,
                }
            ).to_parquet(ticker_path, index=False)

            out = build_trade_instructor_frame(
                symbol=symbol,
                date=date,
                freq_ms=freq_ms,
                lookback=0,
                indicator="bbo_imbalance",
                trade_roots=(),
                ticker_cache_root=root,
            )

            self.assertEqual(len(out), 1440)
            self.assertEqual(out.columns.tolist(), ["timestamp", "instructor"])
            self.assertAlmostEqual(out.loc[0, "instructor"], 0.5)
            self.assertAlmostEqual(out.loc[1, "instructor"], -1.0)
            self.assertAlmostEqual(out.loc[2, "instructor"], 1.0)
            self.assertEqual(out.loc[3, "instructor"], 0.0)
            validate_instructor_frame(out, date_str=date, freq_ms=freq_ms)

            with self.assertRaisesRegex(ValueError, "only supports lookback 0"):
                build_trade_instructor_frame(
                    symbol=symbol,
                    date=date,
                    freq_ms=freq_ms,
                    lookback=1,
                    indicator="bbo_imbalance",
                    trade_roots=(),
                    ticker_cache_root=root,
                )

    def test_loader_merges_instructor_into_alpha_frame(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "cache"
            symbol = "BTCUSDT"
            date = "2025-01-01"
            freq_ms = 1000
            ts0 = int(day_timestamp_grid(date, freq_ms)[0])
            timestamps = [ts0, ts0 + freq_ms]

            ticker_path = sampled_ticker_path(root, symbol, freq_ms, date)
            ticker_path.parent.mkdir(parents=True)
            pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "best_bid_price": [100.0, 101.0],
                    "best_ask_price": [100.5, 101.5],
                    "best_bid_qty": [1.0, 1.0],
                    "best_ask_qty": [1.0, 1.0],
                }
            ).to_parquet(ticker_path, index=False)

            instructor_path = instructor_output_path(
                root, symbol, "trade_imbalance", freq_ms, 5, date
            )
            instructor_path.parent.mkdir(parents=True)
            pd.DataFrame(
                {"timestamp": timestamps, "instructor": [0.25, -0.5]}
            ).to_parquet(instructor_path, index=False)

            intensity_path = intensity_output_path(root, symbol, "k", freq_ms, 300, date)
            intensity_path.parent.mkdir(parents=True)
            pd.DataFrame(
                {"timestamp": timestamps, "intensity": [1.5, 2.5]}
            ).to_parquet(intensity_path, index=False)

            volatility_path = volatility_output_path(root, symbol, "sigma", freq_ms, 300, date)
            volatility_path.parent.mkdir(parents=True)
            pd.DataFrame(
                {"timestamp": timestamps, "volatility": [0.1, 0.2]}
            ).to_parquet(volatility_path, index=False)

            loader = BinanceEventLoader(
                ticker_cache_root=root,
                instructor_cache_root=root,
                trade_intensity_cache_root=root,
                volatility_cache_root=root,
                scheme_shift=0,
            )
            alpha = loader._read_alpha_frame(
                symbol=symbol,
                date=date,
                freq_ms=freq_ms,
                trade_intensity_spec={"name": "k", "lookback": 300},
                volatility_specs=[{"name": "sigma", "lookback": 300}],
                instructor_spec={"name": "trade_imbalance", "lookback": 5},
            )

            self.assertEqual(
                alpha.columns.tolist(),
                [
                    "timestamp",
                    "best_bid_price",
                    "best_ask_price",
                    "instructor",
                    "intensity",
                    "volatility_scalar",
                ],
            )
            self.assertEqual(alpha["instructor"].tolist(), [0.25, -0.5])
            self.assertEqual(alpha["intensity"].tolist(), [1.5, 2.5])
            self.assertEqual(alpha["volatility_scalar"].tolist(), [0.1, 0.2])

    def test_loader_broadcasts_slower_alpha_features_to_fastest_grid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "cache"
            symbol = "BTCUSDT"
            date = "2025-01-01"
            fast_freq_ms = 1000
            instructor_freq_ms = 2000
            volatility_freq_ms = 3000
            ts0 = int(day_timestamp_grid(date, fast_freq_ms)[0])
            fast_timestamps = [ts0 + i * fast_freq_ms for i in range(4)]

            ticker_path = sampled_ticker_path(root, symbol, fast_freq_ms, date)
            ticker_path.parent.mkdir(parents=True)
            pd.DataFrame(
                {
                    "timestamp": fast_timestamps,
                    "best_bid_price": [100.0, 101.0, 102.0, 103.0],
                    "best_ask_price": [100.5, 101.5, 102.5, 103.5],
                    "best_bid_qty": [1.0, 1.0, 1.0, 1.0],
                    "best_ask_qty": [1.0, 1.0, 1.0, 1.0],
                }
            ).to_parquet(ticker_path, index=False)

            intensity_path = intensity_output_path(root, symbol, "k", fast_freq_ms, 300, date)
            intensity_path.parent.mkdir(parents=True)
            pd.DataFrame(
                {"timestamp": fast_timestamps, "intensity": [1.0, 2.0, 3.0, 4.0]}
            ).to_parquet(intensity_path, index=False)

            instructor_path = instructor_output_path(
                root,
                symbol,
                "trade_imbalance",
                instructor_freq_ms,
                5,
                date,
            )
            instructor_path.parent.mkdir(parents=True)
            pd.DataFrame(
                {
                    "timestamp": [ts0, ts0 + instructor_freq_ms],
                    "instructor": [0.25, -0.5],
                }
            ).to_parquet(instructor_path, index=False)

            volatility_path = volatility_output_path(
                root,
                symbol,
                "sigma",
                volatility_freq_ms,
                300,
                date,
            )
            volatility_path.parent.mkdir(parents=True)
            pd.DataFrame(
                {
                    "timestamp": [ts0, ts0 + volatility_freq_ms],
                    "volatility": [0.1, 0.2],
                }
            ).to_parquet(volatility_path, index=False)

            loader = BinanceEventLoader(cache_root=root, scheme_shift=0)
            alpha = loader._read_alpha_frame(
                symbol=symbol,
                date=date,
                freq_ms=60000,
                trade_intensity_spec={
                    "name": "k",
                    "lookback": 300,
                    "freq_ms": fast_freq_ms,
                },
                volatility_specs=[
                    {
                        "name": "sigma",
                        "lookback": 300,
                        "freq_ms": volatility_freq_ms,
                    }
                ],
                instructor_spec={
                    "name": "trade_imbalance",
                    "lookback": 5,
                    "freq_ms": instructor_freq_ms,
                },
            )

            self.assertEqual(alpha["timestamp"].tolist(), fast_timestamps)
            self.assertEqual(alpha["intensity"].tolist(), [1.0, 2.0, 3.0, 4.0])
            self.assertEqual(alpha["instructor"].tolist(), [0.25, 0.25, -0.5, -0.5])
            self.assertEqual(alpha["volatility_scalar"].tolist(), [0.1, 0.1, 0.1, 0.2])

    def test_loader_uses_shifted_cache_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "cache"
            symbol = "BTCUSDT"
            date = "2025-01-01"
            freq_ms = 1000
            scheme_shift = 250
            ts0 = int(day_timestamp_grid(date, freq_ms, scheme_shift)[0])
            timestamps = [ts0, ts0 + freq_ms]

            ticker_path = sampled_ticker_path(root, symbol, freq_ms, date, scheme_shift)
            ticker_path.parent.mkdir(parents=True)
            pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "best_bid_price": [100.0, 101.0],
                    "best_ask_price": [100.5, 101.5],
                    "best_bid_qty": [1.0, 1.0],
                    "best_ask_qty": [1.0, 1.0],
                }
            ).to_parquet(ticker_path, index=False)

            intensity_path = intensity_output_path(
                root,
                symbol,
                "k",
                freq_ms,
                300,
                date,
                scheme_shift,
            )
            intensity_path.parent.mkdir(parents=True)
            pd.DataFrame(
                {"timestamp": timestamps, "intensity": [1.5, 2.5]}
            ).to_parquet(intensity_path, index=False)

            loader = BinanceEventLoader(cache_root=root, scheme_shift=scheme_shift)
            alpha = loader._read_alpha_frame(
                symbol=symbol,
                date=date,
                freq_ms=freq_ms,
                trade_intensity_spec={"name": "k", "lookback": 300},
                volatility_specs=[],
                instructor_spec=None,
            )

            self.assertEqual(alpha["timestamp"].tolist(), timestamps)
            self.assertEqual(alpha["intensity"].tolist(), [1.5, 2.5])

    def test_loader_allows_bbo_imbalance_zero_lookback(self):
        self.assertEqual(
            BinanceEventLoader._parse_spec(
                {"name": "bbo_imbalance", "lookback": 0},
                default_name="trade_imbalance",
            ),
            ("bbo_imbalance", 0),
        )
        with self.assertRaisesRegex(ValueError, "bbo_imbalance lookback must be 0"):
            BinanceEventLoader._parse_spec(
                {"name": "bbo_imbalance", "lookback": 1},
                default_name="trade_imbalance",
            )

    def test_build_stage_configs_accepts_bbo_zero_lookback(self):
        cfg = {
            "symbols": ["BTCUSDT"],
            "date_start": "2025-01-01",
            "input_path": "/tmp/input",
            "output_path": "/tmp/output",
            "freq_ms_instructor": [1000],
            "freq_ms_intensity": [500],
            "freq_ms_volatility": [2000],
            "scheme_shift": [0, 250],
            "name_instructor": ["bbo_imbalance"],
            "lookback_instructor": [0],
            "name_intensity": ["k"],
            "lookback_intensity": [300],
            "name_volatility": ["sigma"],
            "lookback_volatility": [600],
        }

        resample_cfg, instructor_cfg, intensity_cfg, volatility_cfg = build_stage_configs(cfg)

        self.assertEqual(
            instructor_cfg["instructor"]["indicator"],
            ["bbo_imbalance"],
        )
        self.assertEqual(resample_cfg["sampler"]["freq"], [500, 1000, 2000])
        self.assertEqual(instructor_cfg["instructor"]["freq"], [1000])
        self.assertEqual(intensity_cfg["intensity"]["freq"], [500])
        self.assertEqual(volatility_cfg["volatility"]["freq"], [2000])
        self.assertEqual(instructor_cfg["instructor"]["lookback"], [0])
        self.assertEqual(instructor_cfg["instructor"]["scheme_shift"], [0, 250])
        self.assertEqual(instructor_cfg["paths"]["ticker_cache_root"], "/tmp/output")
        self.assertEqual(intensity_cfg["intensity"]["lookback"], [300])
        self.assertEqual(intensity_cfg["intensity"]["scheme_shift"], [0, 250])
        self.assertEqual(volatility_cfg["volatility"]["lookback"], [600])
        self.assertEqual(volatility_cfg["volatility"]["scheme_shift"], [0, 250])
        self.assertEqual(
            [task.scheme_shift_ms for task in build_resample_tasks(resample_cfg)],
            [0, 250, 0, 250, 0, 250],
        )
        self.assertEqual(
            [task.scheme_shift_ms for task in build_intensity_tasks(intensity_cfg)],
            [0, 250],
        )
        self.assertEqual(
            instructor_output_path(
                Path("/tmp/output"),
                "BTCUSDT",
                "bbo_imbalance",
                1000,
                0,
                "2025-01-01",
                250,
            ),
            Path("/tmp/output/BTCUSDT/instructor/freq_1000ms/scheme_shift_250ms/bbo_imbalance/lookback_0/2025-01-01.parquet"),
        )

        bad_cfg = dict(cfg)
        bad_cfg["scheme_shift"] = [0, 1000]
        with self.assertRaisesRegex(ValueError, "scheme_shift"):
            build_stage_configs(bad_cfg)

    def test_build_stage_configs_only_requires_active_stage_frequencies(self):
        cfg = {
            "symbols": ["BTCUSDT"],
            "date_start": "2025-01-01",
            "input_path": "/tmp/input",
            "output_path": "/tmp/output",
            "freq_ms_intensity": [1000],
            "name_intensity": ["k"],
            "lookback_intensity": [300],
        }

        resample_cfg, instructor_cfg, intensity_cfg, volatility_cfg = build_stage_configs(cfg)

        self.assertEqual(resample_cfg["sampler"]["freq"], [1000])
        self.assertEqual(instructor_cfg["instructor"]["freq"], [])
        self.assertEqual(intensity_cfg["intensity"]["freq"], [1000])
        self.assertEqual(volatility_cfg["volatility"]["freq"], [])


if __name__ == "__main__":
    unittest.main()
