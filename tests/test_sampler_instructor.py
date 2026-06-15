import tempfile
import unittest
from pathlib import Path

import pandas as pd

from sampler.intensity import intensity_output_path
from sampler.instructor import (
    build_trade_instructor_frame,
    instructor_output_path,
    validate_instructor_frame,
)
from sampler.resample import day_timestamp_grid, output_path as sampled_ticker_path
from sampler.volatility import volatility_output_path
from sampler.run import build_stage_configs
from sim.loader import BinanceEventLoader


class SamplerInstructorTest(unittest.TestCase):
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

    def test_build_stage_configs_splits_lookbacks(self):
        cfg = {
            "symbols": ["BTCUSDT"],
            "date_start": "2025-01-01",
            "input_path": "/tmp/input",
            "output_path": "/tmp/output",
            "freq_ms": [1000],
            "name_instructor": ["trade_imbalance"],
            "lookback_instructor": [5, 10],
            "name_intensity": ["k"],
            "lookback_intensity": [300],
            "name_volatility": ["sigma"],
            "lookback_volatility": [600],
        }

        _, instructor_cfg, intensity_cfg, volatility_cfg = build_stage_configs(cfg)

        self.assertEqual(instructor_cfg["instructor"]["lookback"], [5, 10])
        self.assertEqual(intensity_cfg["intensity"]["lookback"], [300])
        self.assertEqual(volatility_cfg["volatility"]["lookback"], [600])
        self.assertEqual(
            instructor_output_path(
                Path("/tmp/output"),
                "BTCUSDT",
                "trade_imbalance",
                1000,
                5,
                "2025-01-01",
            ),
            Path("/tmp/output/BTCUSDT/instructor/freq_1000ms/trade_imbalance/lookback_5/2025-01-01.parquet"),
        )


if __name__ == "__main__":
    unittest.main()
