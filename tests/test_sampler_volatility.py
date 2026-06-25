import argparse
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from sampler.volatility import (
    VolatilityCalculator,
    VolatilityConfig,
    build_config_from_args,
    build_tasks,
)


class SamplerVolatilityTest(unittest.TestCase):
    def test_sigma_is_scaled_to_price_units(self):
        base = pd.DataFrame(
            {
                "timestamp": [1000, 2000, 3000, 4000, 5000],
                "mid": [100.0, 101.0, 99.0, 102.0, 98.0],
                "high": [100.0, 101.0, 99.0, 102.0, 98.0],
                "low": [100.0, 101.0, 99.0, 102.0, 98.0],
                "close": [100.0, 101.0, 99.0, 102.0, 98.0],
                "open": [100.0, 101.0, 99.0, 102.0, 98.0],
            }
        )

        config = VolatilityConfig(
            lookback=3,
            freq_ms=1000,
            min_periods=2,
            annualize=False,
        )
        out = VolatilityCalculator(base=base, freq_ms=1000).compute_sigma(config)

        mid = base["mid"].to_numpy(dtype="float64")
        returns = np.zeros(len(mid), dtype="float64")
        returns[1:] = np.diff(mid) / mid[:-1]
        up_returns = np.where(returns > 0.0, returns, 0.0)
        down_returns = np.where(returns < 0.0, -returns, 0.0)
        nonzero_returns = np.where(returns != 0.0, returns, 0.0)

        def expected(values: np.ndarray) -> np.ndarray:
            return (
                pd.Series(values)
                .rolling(window=3, min_periods=2)
                .std()
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
                .clip(lower=0.0)
                .to_numpy(dtype="float64")
                * mid
            )

        np.testing.assert_allclose(out["volatility"], expected(nonzero_returns))
        np.testing.assert_allclose(out["volatility_up"], expected(up_returns))
        np.testing.assert_allclose(out["volatility_down"], expected(down_returns))

    def test_direct_config_accepts_pipeline_top_level_keys(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            path.write_text(
                json.dumps(
                    {
                        "symbols": ["BTCUSDT"],
                        "date_start": "2025-01-01",
                        "date_end": "2025-01-01",
                        "input_path": "/tmp/raw",
                        "input_backup_path": "/tmp/raw_backup",
                        "output_path": "/tmp/cache",
                        "freq_ms": [1000],
                        "name_volatility": ["sigma"],
                        "lookback_volatility": [300, 600],
                        "ticker_category": "BOOKTICKER",
                        "trade_category": "TRADE",
                        "num_worker": 10,
                        "overwrite": False,
                    }
                ),
                encoding="utf-8",
            )

            cfg = build_config_from_args(
                argparse.Namespace(
                    config=str(path),
                    ticker_cache_root=None,
                    bookticker_root=None,
                    bookticker_backup_root=None,
                    ticker_category=None,
                    trade_root=None,
                    trade_backup_root=None,
                    trade_category=None,
                    output_root=None,
                    overwrite=True,
                    auto_resample=False,
                    annualize=False,
                )
            )
            tasks = build_tasks(cfg)

        self.assertEqual(len(tasks), 2)
        self.assertEqual({task.lookback for task in tasks}, {300, 600})
        self.assertTrue(all(task.overwrite for task in tasks))
        self.assertEqual(cfg["parallel"]["num_workers"], 10)
        self.assertEqual(tasks[0].ticker_cache_root, Path("/tmp/cache"))
        self.assertEqual(tasks[0].output_root, Path("/tmp/cache"))
        self.assertEqual(
            tasks[0].bookticker_roots,
            (
                Path("/tmp/raw/BOOKTICKER"),
                Path("/tmp/raw_backup/BOOKTICKER"),
            ),
        )
        self.assertEqual(
            tasks[0].trade_roots,
            (
                Path("/tmp/raw/TRADE"),
                Path("/tmp/raw_backup/TRADE"),
            ),
        )


if __name__ == "__main__":
    unittest.main()
