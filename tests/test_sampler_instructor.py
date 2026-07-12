import tempfile
import unittest
from pathlib import Path

import pandas as pd

from sampler.intensity import build_tasks as build_intensity_tasks
from sampler.intensity import build_trade_intensity_frame
from sampler.intensity import compute_k_decay, compute_kls
from sampler.intensity import intensity_output_path
from sampler.instructor import (
    build_tasks as build_instructor_tasks,
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
from sampler.volatility import VolatilityCalculator, VolatilityConfig, volatility_output_path
from sampler.run import build_stage_configs
from sim.loader import BinanceEventLoader


class SamplerInstructorTest(unittest.TestCase):
    def test_kls_uses_larger_short_and_long_decay(self):
        break_dist = pd.Series([0.0, 4.0, 0.0, 0.0, 1.0])

        actual = compute_kls(break_dist, [3, 1])
        expected = pd.concat(
            [
                compute_k_decay(break_dist, 1),
                compute_k_decay(break_dist, 3),
            ],
            axis=1,
        ).max(axis=1)

        self.assertEqual(actual.tolist(), expected.tolist())

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
                    "trade_type": 0,
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

    def test_volume_zscore_compares_previous_bar_to_older_history(self):
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
                        day_start + 2000,
                        day_start + 3000,
                    ],
                    "price": [100.0, 100.0, 100.0, 100.0],
                    "volume": [1.0, 2.0, 3.0, 7.0],
                    "is_buyer_maker": [True, True, True, True],
                    "trade_type": 0,
                }
            ).to_parquet(trade_dir / f"{symbol}--TRADE--{date}.parquet")

            out = build_trade_instructor_frame(
                symbol=symbol,
                date=date,
                freq_ms=1000,
                lookback=4,
                indicator="volume_zscore",
                trade_roots=(root,),
            )

            expected = (7.0 - 2.0) / 1.0
            self.assertEqual(len(out), 86400)
            self.assertEqual(out.columns.tolist(), ["timestamp", "instructor"])
            self.assertEqual(out.loc[0, "instructor"], 0.0)
            self.assertAlmostEqual(out.loc[4, "instructor"], expected)
            self.assertGreater(abs(out.loc[4, "instructor"]), 1.0)
            validate_instructor_frame(
                out,
                date_str=date,
                freq_ms=1000,
                indicator="volume_zscore",
            )
            with self.assertRaisesRegex(ValueError, "inside \\[-1, 1\\]"):
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
        out = TickerResampler(freq_ms=1000, scheme_shift_ms=250).resample(
            raw,
            date,
        )

        self.assertEqual(int(out.loc[0, "timestamp"]), int(base_grid[0]) + 250)
        self.assertEqual(float(out.loc[0, "best_bid_price"]), 100.0)
        self.assertEqual(float(out.loc[1, "best_bid_price"]), 100.0)
        self.assertEqual(float(out.loc[2, "best_bid_price"]), 101.0)

    def test_volatility_sigma_outputs_price_units(self):
        base = pd.DataFrame(
            {
                "timestamp": [0, 1000],
                "mid": [100.0, 110.0],
                "high": [100.0, 110.0],
                "low": [100.0, 110.0],
                "close": [100.0, 110.0],
                "open": [100.0, 110.0],
            }
        )
        frame = VolatilityCalculator(base=base, freq_ms=1000).compute(
            "sigma",
            VolatilityConfig(lookback=2, freq_ms=1000, min_periods=1),
        )

        self.assertEqual(float(frame.loc[0, "volatility"]), 0.0)
        self.assertAlmostEqual(
            float(frame.loc[1, "volatility"]),
            110.0 * pd.Series([0.0, 0.1]).std(),
        )

    def test_intensity_outputs_price_break_distance(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_root = Path(tmpdir) / "cache"
            trade_root = Path(tmpdir) / "TRADE"
            symbol = "BTCUSDT"
            date = "2025-01-01"
            freq_ms = 1000
            timestamps = day_timestamp_grid(date, freq_ms)[:2].tolist()

            ticker_path = sampled_ticker_path(cache_root, symbol, freq_ms, date)
            ticker_path.parent.mkdir(parents=True)
            pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "best_bid_price": [99.0, 99.0],
                    "best_ask_price": [101.0, 101.0],
                    "best_bid_qty": [1.0, 1.0],
                    "best_ask_qty": [1.0, 1.0],
                }
            ).to_parquet(ticker_path, index=False)

            trade_dir = trade_root / symbol
            trade_dir.mkdir(parents=True)
            pd.DataFrame(
                {
                    "exchange_timestamp": [timestamps[0]],
                    "price": [103.0],
                    "volume": [1.0],
                    "trade_type": 0,
                }
            ).to_parquet(trade_dir / f"{symbol}--TRADE--{date}.parquet", index=False)

            out = build_trade_intensity_frame(
                symbol=symbol,
                date=date,
                freq_ms=freq_ms,
                lookback=1,
                indicator="k",
                ticker_cache_root=cache_root,
                trade_roots=(trade_root,),
            )

            self.assertEqual(float(out.loc[0, "intensity"]), 0.0)
            self.assertEqual(float(out.loc[1, "intensity"]), 2.0)

            kls = build_trade_intensity_frame(
                symbol=symbol,
                date=date,
                freq_ms=freq_ms,
                lookback=[3, 1],
                indicator="kls",
                ticker_cache_root=cache_root,
                trade_roots=(trade_root,),
            )

            self.assertEqual(kls.columns.tolist(), ["timestamp", "intensity"])
            self.assertEqual(float(kls.loc[0, "intensity"]), 0.0)
            self.assertEqual(float(kls.loc[1, "intensity"]), 1.0)

    def test_kw_vol_weights_nonzero_break_bins_by_bucket_volume(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_root = Path(tmpdir) / "cache"
            trade_root = Path(tmpdir) / "TRADE"
            symbol = "BTCUSDT"
            date = "2025-01-01"
            freq_ms = 1000
            timestamps = day_timestamp_grid(date, freq_ms)[:4].tolist()

            ticker_path = sampled_ticker_path(cache_root, symbol, freq_ms, date)
            ticker_path.parent.mkdir(parents=True)
            pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "best_bid_price": [99.0, 99.0, 99.0, 99.0],
                    "best_ask_price": [101.0, 101.0, 101.0, 101.0],
                    "best_bid_qty": [1.0, 1.0, 1.0, 1.0],
                    "best_ask_qty": [1.0, 1.0, 1.0, 1.0],
                }
            ).to_parquet(ticker_path, index=False)

            trade_dir = trade_root / symbol
            trade_dir.mkdir(parents=True)
            pd.DataFrame(
                {
                    "exchange_timestamp": [timestamps[0], timestamps[1], timestamps[2]],
                    "price": [103.0, 95.0, 104.0],
                    "volume": [1.0, 3.0, 2.0],
                    "trade_type": 0,
                }
            ).to_parquet(trade_dir / f"{symbol}--TRADE--{date}.parquet", index=False)

            out = build_trade_intensity_frame(
                symbol=symbol,
                date=date,
                freq_ms=freq_ms,
                lookback=2,
                indicator="kw_vol",
                ticker_cache_root=cache_root,
                trade_roots=(trade_root,),
            )

            self.assertEqual(out.columns.tolist(), [
                "timestamp",
                "intensity_positive",
                "intensity_negative",
            ])
            self.assertEqual(float(out.loc[0, "intensity_positive"]), 0.0)
            self.assertEqual(float(out.loc[0, "intensity_negative"]), 0.0)
            self.assertEqual(float(out.loc[1, "intensity_positive"]), 2.0)
            self.assertEqual(float(out.loc[1, "intensity_negative"]), 0.0)
            self.assertEqual(float(out.loc[2, "intensity_positive"]), 2.0)
            self.assertEqual(float(out.loc[2, "intensity_negative"]), 4.0)
            self.assertEqual(float(out.loc[3, "intensity_positive"]), 3.0)
            self.assertEqual(float(out.loc[3, "intensity_negative"]), 4.0)

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
                cache_root=root,
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
                    "intensity_positive",
                    "intensity_negative",
                    "volatility_scalar",
                ],
            )
            self.assertEqual(alpha["instructor"].tolist(), [0.25, -0.5])
            self.assertEqual(alpha["intensity_positive"].tolist(), [1.5, 2.5])
            self.assertEqual(alpha["intensity_negative"].tolist(), [1.5, 2.5])
            self.assertEqual(alpha["volatility_scalar"].tolist(), [0.1, 0.2])

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

            loader = BinanceEventLoader(
                cache_root=root,
                scheme_shift=scheme_shift,
            )
            alpha = loader._read_alpha_frame(
                symbol=symbol,
                date=date,
                freq_ms=freq_ms,
                trade_intensity_spec={"name": "k", "lookback": 300},
                volatility_specs=[],
                instructor_spec=None,
            )

            self.assertEqual(alpha["timestamp"].tolist(), timestamps)
            self.assertEqual(alpha["intensity_positive"].tolist(), [1.5, 2.5])
            self.assertEqual(alpha["intensity_negative"].tolist(), [1.5, 2.5])

    def test_loader_reads_kls_pair_lookback_path(self):
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

            intensity_path = intensity_output_path(
                root,
                symbol,
                "kls",
                freq_ms,
                "12_120",
                date,
            )
            intensity_path.parent.mkdir(parents=True)
            pd.DataFrame(
                {"timestamp": timestamps, "intensity": [1.5, 2.5]}
            ).to_parquet(intensity_path, index=False)

            loader = BinanceEventLoader(cache_root=root, scheme_shift=0)
            alpha = loader._read_alpha_frame(
                symbol=symbol,
                date=date,
                freq_ms=freq_ms,
                trade_intensity_spec={"name": "kls", "lookback": [120, 12]},
                volatility_specs=[],
                instructor_spec=None,
            )

            self.assertEqual(alpha["intensity_positive"].tolist(), [1.5, 2.5])
            self.assertEqual(alpha["intensity_negative"].tolist(), [1.5, 2.5])

    def test_loader_maps_split_kw_vol_columns_into_directional_intensity(self):
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

            intensity_path = intensity_output_path(root, symbol, "kw_vol", freq_ms, 300, date)
            intensity_path.parent.mkdir(parents=True)
            pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "kw_vol_positive": [1.5, 2.5],
                    "kw_vol_negative": [3.5, 4.5],
                }
            ).to_parquet(intensity_path, index=False)

            loader = BinanceEventLoader(cache_root=root, scheme_shift=0)
            alpha = loader._read_alpha_frame(
                symbol=symbol,
                date=date,
                freq_ms=freq_ms,
                trade_intensity_spec={"name": "kw_vol", "lookback": 300},
                volatility_specs=[],
                instructor_spec=None,
            )

            self.assertEqual(alpha["intensity_positive"].tolist(), [1.5, 2.5])
            self.assertEqual(alpha["intensity_negative"].tolist(), [3.5, 4.5])

    def test_loader_pairs_split_kw_vol_files_into_directional_intensity(self):
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

            for name, values in (
                ("kw_vol_positive", [1.5, 2.5]),
                ("kw_vol_negative", [3.5, 4.5]),
            ):
                intensity_path = intensity_output_path(root, symbol, name, freq_ms, 300, date)
                intensity_path.parent.mkdir(parents=True)
                pd.DataFrame(
                    {"timestamp": timestamps, "intensity": values}
                ).to_parquet(intensity_path, index=False)

            loader = BinanceEventLoader(cache_root=root, scheme_shift=0)
            alpha = loader._read_alpha_frame(
                symbol=symbol,
                date=date,
                freq_ms=freq_ms,
                trade_intensity_spec={"name": "kw_vol", "lookback": 300},
                volatility_specs=[],
                instructor_spec=None,
            )

            self.assertEqual(alpha["intensity_positive"].tolist(), [1.5, 2.5])
            self.assertEqual(alpha["intensity_negative"].tolist(), [3.5, 4.5])

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
        self.assertEqual(
            BinanceEventLoader._parse_spec(
                {"name": "volume_zscore", "lookback": 4},
                default_name="trade_imbalance",
            ),
            ("volume_zscore", 4),
        )
        with self.assertRaisesRegex(ValueError, "volume_zscore lookback must be >= 2"):
            BinanceEventLoader._parse_spec(
                {"name": "volume_zscore", "lookback": 1},
                default_name="trade_imbalance",
            )

    def test_build_stage_configs_accepts_volume_zscore_lookback(self):
        cfg = {
            "symbols": ["BTCUSDT"],
            "date_start": "2025-01-01",
            "input_path": "/tmp/input",
            "output_path": "/tmp/output",
            "freq_ms": [1000],
            "name_instructor": ["volume_zscore"],
            "lookback_instructor": [4],
        }

        _, instructor_cfg, _, _ = build_stage_configs(cfg)
        tasks = build_instructor_tasks(instructor_cfg)

        self.assertEqual(instructor_cfg["instructor"]["indicator"], ["volume_zscore"])
        self.assertEqual(instructor_cfg["instructor"]["lookback"], [4])
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].indicator, "volume_zscore")
        self.assertEqual(tasks[0].lookback, 4)

    def test_build_stage_configs_accepts_bbo_zero_lookback(self):
        cfg = {
            "symbols": ["BTCUSDT"],
            "date_start": "2025-01-01",
            "input_path": "/tmp/input",
            "output_path": "/tmp/output",
            "freq_ms": [1000],
            "scheme_shift": [0, 250],
            "name_instructor": ["bbo_imbalance"],
            "lookback_instructor": [0],
            "name_intensity": ["kw_vol"],
            "lookback_intensity": [300],
            "name_volatility": ["sigma"],
            "lookback_volatility": [600],
        }

        resample_cfg, instructor_cfg, intensity_cfg, volatility_cfg = build_stage_configs(cfg)

        self.assertEqual(
            instructor_cfg["instructor"]["indicator"],
            ["bbo_imbalance"],
        )
        self.assertEqual(instructor_cfg["instructor"]["lookback"], [0])
        self.assertEqual(instructor_cfg["instructor"]["scheme_shift"], [0, 250])
        self.assertEqual(instructor_cfg["paths"]["ticker_cache_root"], "/tmp/output")
        self.assertEqual(intensity_cfg["intensity"]["indicator"], ["kw_vol"])
        self.assertEqual(intensity_cfg["intensity"]["lookback"], [300])
        self.assertEqual(intensity_cfg["intensity"]["scheme_shift"], [0, 250])
        self.assertEqual(volatility_cfg["volatility"]["lookback"], [600])
        self.assertEqual(volatility_cfg["volatility"]["scheme_shift"], [0, 250])
        self.assertEqual(
            [task.scheme_shift_ms for task in build_resample_tasks(resample_cfg)],
            [0, 250],
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

    def test_build_stage_configs_accepts_kls_pair_lookback(self):
        cfg = {
            "symbols": ["BTCUSDT"],
            "date_start": "2025-01-01",
            "input_path": "/tmp/input",
            "output_path": "/tmp/output",
            "freq_ms": [1000],
            "scheme_shift": [0],
            "name_intensity": ["kls"],
            "lookback_intensity": [[120, 12]],
        }

        _, _, intensity_cfg, _ = build_stage_configs(cfg)
        tasks = build_intensity_tasks(intensity_cfg)

        self.assertEqual(intensity_cfg["intensity"]["lookback"], ["12_120"])
        self.assertEqual([(task.indicator, task.lookback) for task in tasks], [("kls", "12_120")])


if __name__ == "__main__":
    unittest.main()
