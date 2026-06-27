from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal, TypeAlias

import numpy as np
import pandas as pd

from sampler.instructor import instructor_output_path
from sampler.intensity import intensity_output_path
from sampler.resample import normalize_scheme_shift
from sampler.resample import output_path as sampled_ticker_path
from sampler.resample import resolve_existing_input_path
from sampler.volatility import volatility_output_path


DateLike: TypeAlias = str | dt.date | dt.datetime | pd.Timestamp
TradeTuple: TypeAlias = tuple[Literal["trade"], int, bool, float, float]
AlphaTuple: TypeAlias = tuple[
    Literal["ticker"],
    int,
    float,
    float,
    float | None,
    float | None,
    float,
    Any,
    Any,
    Any,
    Any,
]
MergedEventTuple: TypeAlias = TradeTuple | AlphaTuple

RAW_TRADE_TIME_COLUMN = "exchange_timestamp"
RAW_TRADE_COLUMNS = (
    RAW_TRADE_TIME_COLUMN,
    "price",
    "volume",
    "is_buyer_maker",
)
TICKER_COLUMNS = (
    "timestamp",
    "best_bid_price",
    "best_ask_price",
)
DEFAULT_CACHE_ROOT = Path("/data/users/kang/backtest/glftmm/cached")
DEFAULT_DATA_ROOT = Path("/data/users/data-helper/PROCESSED/TARDIS/BINANCE/UFUTURES")
DEFAULT_BACKUP_DATA_ROOT = Path("/home/kang/data_helper/PROCESSED/DATA_RECORDER/BINANCE/UFUTURES")
DEFAULT_ORDERBOOK_REPLAY_ROOT = Path("/home/kang/data/wallmaker/cached")


@dataclass(frozen=True)
class OrderBookReplayConfig:
    root: Path
    replay_levels: int = 1000
    sample_interval_ms: int = 1000
    tick_size: float = 0.1
    depth_price_min: float = 20_000.0
    depth_price_max: float = 200_000.0
    raw_quantity_max: float = 10_000.0


@dataclass(frozen=True)
class OrderBookReplayFrame:
    timestamps: np.ndarray
    bid_ticks: np.ndarray
    ask_ticks: np.ndarray
    bid_notional: np.ndarray
    ask_notional: np.ndarray


_ORDERBOOK_REPLAY_CACHE: dict[tuple[object, ...], OrderBookReplayFrame] = {}
_ORDERBOOK_REPLAY_CACHE_ORDER: list[tuple[object, ...]] = []
_ORDERBOOK_REPLAY_CACHE_MAX_DAYS = 1


def normalize_date(value: DateLike) -> str:
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if isinstance(value, dt.datetime):
        return value.date().isoformat()
    if isinstance(value, dt.date):
        return value.isoformat()

    text = str(value).strip()
    if not text:
        raise ValueError("date must not be empty")
    if len(text) == 8 and text.isdigit():
        return dt.datetime.strptime(text, "%Y%m%d").date().isoformat()
    return dt.date.fromisoformat(text).isoformat()


def _path_safe_value(value: object) -> str:
    if isinstance(value, bool):
        text = "true" if value else "false"
    elif isinstance(value, float):
        text = f"{value:.12g}"
    else:
        text = str(value)
    return text.replace("-", "m").replace("+", "").replace(".", "p")


def parse_orderbook_replay_config(raw: dict[str, Any] | None) -> OrderBookReplayConfig:
    if raw is None:
        raise ValueError("orderbook_path config is required when optimize_by_orderbook is enabled")
    if not isinstance(raw, dict):
        raise ValueError("orderbook config must be an object")

    replay_raw = raw.get("orderbook_replay", {})
    if replay_raw is None:
        replay_raw = {}
    if not isinstance(replay_raw, dict):
        raise ValueError("orderbook_replay config must be an object")

    root_raw = raw.get("orderbook_path")
    if root_raw in (None, "", []):
        root_raw = replay_raw.get("root") or replay_raw.get("output_path") or replay_raw.get("cache_root")
    if root_raw in (None, "", []):
        raise ValueError("orderbook_path config is required when optimize_by_orderbook is enabled")

    cfg = OrderBookReplayConfig(
        root=Path(root_raw),
        replay_levels=int(replay_raw.get("replay_levels", 1000)),
        sample_interval_ms=int(replay_raw.get("sample_interval_ms", 1000)),
        tick_size=float(replay_raw.get("tick_size", 0.1)),
        depth_price_min=float(replay_raw.get("depth_price_min", 20_000.0)),
        depth_price_max=float(replay_raw.get("depth_price_max", 200_000.0)),
        raw_quantity_max=float(replay_raw.get("raw_quantity_max", 10_000.0)),
    )
    validate_orderbook_replay_config(cfg)
    return cfg


def validate_orderbook_replay_config(cfg: OrderBookReplayConfig) -> None:
    if cfg.replay_levels < 1:
        raise ValueError("orderbook_replay.replay_levels must be >= 1")
    if cfg.sample_interval_ms < 1:
        raise ValueError("orderbook_replay.sample_interval_ms must be >= 1")
    if cfg.tick_size <= 0.0:
        raise ValueError("orderbook_replay.tick_size must be > 0")
    if cfg.depth_price_max <= cfg.depth_price_min:
        raise ValueError("orderbook_replay.depth_price_max must be > depth_price_min")
    if cfg.raw_quantity_max <= 0.0:
        raise ValueError("orderbook_replay.raw_quantity_max must be > 0")


def orderbook_replay_path(
    root: Path,
    symbol: str,
    date_str: str,
    cfg: OrderBookReplayConfig,
) -> Path:
    path = root / "ORDERBOOK_REPLAY" / "depth_update" / symbol
    for name in (
        "replay_levels",
        "sample_interval_ms",
        "tick_size",
        "depth_price_min",
        "depth_price_max",
        "raw_quantity_max",
    ):
        path = path / f"{name}-{_path_safe_value(getattr(cfg, name))}"
    return path / f"{date_str}.parquet"


class BinanceEventLoader:
    """
    Load the cached sampler outputs and raw trades as a single timestamp stream.

    Trade events are yielded before ticker/alpha events when timestamps are
    identical. That avoids letting a zero-latency quote react to a sampled book
    and fill against a trade at the same millisecond.
    """

    def __init__(
        self,
        cache_root: str | Path | None = None,
        input_path: str | Path | None = None,
        input_backup_path: str | Path | None = None,
        trade_roots: Iterable[str | Path] | None = None,
        trade_category: str | None = None,
        scheme_shift: int = 0,
        orderbook_replay_config: OrderBookReplayConfig | dict[str, Any] | None = None,
    ) -> None:
        category = str(trade_category or "TRADE").strip().upper()
        if not category:
            raise ValueError("trade_category must not be empty")

        default_cache_root = Path(cache_root) if cache_root is not None else DEFAULT_CACHE_ROOT
        self.cache_root = default_cache_root
        self.scheme_shift = int(scheme_shift)
        self.trade_category = category
        if trade_roots is not None:
            roots = tuple(Path(path) for path in trade_roots)
        elif input_path is not None:
            root_list = [Path(input_path) / category]
            if input_backup_path not in (None, "", []):
                root_list.append(Path(input_backup_path) / category)
            roots = tuple(root_list)
        else:
            roots = (DEFAULT_DATA_ROOT / category, DEFAULT_BACKUP_DATA_ROOT / category)
        self.trade_roots = roots
        if isinstance(orderbook_replay_config, OrderBookReplayConfig):
            self.orderbook_replay_config = orderbook_replay_config
        elif orderbook_replay_config is None:
            self.orderbook_replay_config = None
        else:
            self.orderbook_replay_config = parse_orderbook_replay_config(orderbook_replay_config)

    def iter_merged_alpha_trade_tuples(
        self,
        symbol: str,
        date: DateLike,
        freq: int,
        trade_intensity_spec: dict[str, int | str] | None = None,
        volatility_specs: Iterable[dict[str, int | str]] | None = None,
        instructor_spec: dict[str, int | str] | None = None,
    ) -> Iterator[MergedEventTuple]:
        symbol = str(symbol).upper()
        date_str = normalize_date(date)
        freq_ms = int(freq)
        if freq_ms <= 0:
            raise ValueError("freq must be > 0")

        alpha = self._read_alpha_frame(
            symbol=symbol,
            date=date_str,
            freq_ms=freq_ms,
            trade_intensity_spec=trade_intensity_spec,
            volatility_specs=volatility_specs,
            instructor_spec=instructor_spec,
        )
        replay = self._read_orderbook_replay_frame(symbol=symbol, date=date_str)
        if replay is not None:
            alpha_ts = alpha["timestamp"].to_numpy(dtype="int64", copy=False)
            replay = self._align_orderbook_replay_frame(
                symbol=symbol,
                date=date_str,
                alpha_timestamps=alpha_ts,
                replay=replay,
            )
        trades = self._read_trade_frame(symbol=symbol, date=date_str)
        yield from self._merge_sorted(alpha=alpha, trades=trades, replay=replay)

    def _read_alpha_frame(
        self,
        symbol: str,
        date: str,
        freq_ms: int,
        trade_intensity_spec: dict[str, int | str] | None,
        volatility_specs: Iterable[dict[str, int | str]] | None,
        instructor_spec: dict[str, int | str] | None,
    ) -> pd.DataFrame:
        ticker = self._read_sampled_ticker(symbol=symbol, date=date, freq_ms=freq_ms)
        alpha = ticker.loc[:, TICKER_COLUMNS].copy()
        alpha["instructor"] = 0.0
        alpha["intensity"] = 0.0
        alpha["volatility_scalar"] = 0.0

        if instructor_spec is not None:
            instructor = self._read_instructor_frame(
                symbol=symbol,
                date=date,
                freq_ms=freq_ms,
                spec=instructor_spec,
            )
            alpha = alpha.merge(instructor, on="timestamp", how="left", suffixes=("", "_new"))
            alpha["instructor"] = alpha["instructor_new"].fillna(alpha["instructor"])
            alpha = alpha.drop(columns=["instructor_new"])

        if trade_intensity_spec is not None:
            intensity = self._read_intensity_frame(
                symbol=symbol,
                date=date,
                freq_ms=freq_ms,
                spec=trade_intensity_spec,
            )
            alpha = alpha.merge(intensity, on="timestamp", how="left", suffixes=("", "_new"))
            alpha["intensity"] = alpha["intensity_new"].fillna(alpha["intensity"])
            alpha = alpha.drop(columns=["intensity_new"])

        for idx, spec in enumerate(volatility_specs or ()):
            vol = self._read_volatility_frame(
                symbol=symbol,
                date=date,
                freq_ms=freq_ms,
                spec=spec,
            )
            column = f"volatility_scalar_{idx}"
            vol = vol.rename(columns={"volatility": column})
            alpha = alpha.merge(vol, on="timestamp", how="left")
            alpha["volatility_scalar"] = (
                alpha["volatility_scalar"] + alpha[column].fillna(0.0)
            )
            alpha = alpha.drop(columns=[column])

        alpha["timestamp"] = alpha["timestamp"].astype("int64")
        for column in (
            "best_bid_price",
            "best_ask_price",
            "instructor",
            "intensity",
            "volatility_scalar",
        ):
            alpha[column] = alpha[column].astype("float64")
        return alpha.sort_values("timestamp", kind="mergesort", ignore_index=True)

    def _read_sampled_ticker(self, symbol: str, date: str, freq_ms: int) -> pd.DataFrame:
        path = sampled_ticker_path(
            root=self.cache_root,
            symbol=symbol,
            freq_ms=freq_ms,
            date_str=date,
            scheme_shift_ms=normalize_scheme_shift(self.scheme_shift, freq_ms),
        )
        if not path.exists():
            raise FileNotFoundError(f"missing sampled ticker: {path}")

        frame = pd.read_parquet(path, columns=list(TICKER_COLUMNS))
        missing = set(TICKER_COLUMNS) - set(frame.columns)
        if missing:
            raise ValueError(f"sampled ticker missing columns: {sorted(missing)}")
        return frame

    def _read_instructor_frame(
        self,
        symbol: str,
        date: str,
        freq_ms: int,
        spec: dict[str, int | str],
    ) -> pd.DataFrame:
        name, lookback = self._parse_spec(spec=spec, default_name="trade_imbalance")
        path = instructor_output_path(
            root=self.cache_root,
            symbol=symbol,
            indicator=name,
            freq_ms=freq_ms,
            lookback=lookback,
            date_str=date,
            scheme_shift_ms=normalize_scheme_shift(self.scheme_shift, freq_ms),
        )
        if not path.exists():
            raise FileNotFoundError(f"missing instructor: {path}")

        frame = pd.read_parquet(path, columns=["timestamp", "instructor"])
        frame["timestamp"] = frame["timestamp"].astype("int64")
        frame["instructor"] = frame["instructor"].astype("float64")
        return frame.loc[:, ["timestamp", "instructor"]]

    def _read_intensity_frame(
        self,
        symbol: str,
        date: str,
        freq_ms: int,
        spec: dict[str, int | str],
    ) -> pd.DataFrame:
        name, lookback = self._parse_spec(spec=spec, default_name="k")
        path = intensity_output_path(
            root=self.cache_root,
            symbol=symbol,
            indicator=name,
            freq_ms=freq_ms,
            lookback=lookback,
            date_str=date,
            scheme_shift_ms=normalize_scheme_shift(self.scheme_shift, freq_ms),
        )
        if not path.exists():
            raise FileNotFoundError(f"missing intensity: {path}")

        frame = pd.read_parquet(path, columns=["timestamp", "intensity"])
        frame["timestamp"] = frame["timestamp"].astype("int64")
        frame["intensity"] = frame["intensity"].astype("float64")
        return frame.loc[:, ["timestamp", "intensity"]]

    def _read_volatility_frame(
        self,
        symbol: str,
        date: str,
        freq_ms: int,
        spec: dict[str, int | str],
    ) -> pd.DataFrame:
        name, lookback = self._parse_spec(spec=spec, default_name="sigma")
        path = volatility_output_path(
            root=self.cache_root,
            symbol=symbol,
            indicator=name,
            freq_ms=freq_ms,
            lookback=lookback,
            date_str=date,
            scheme_shift_ms=normalize_scheme_shift(self.scheme_shift, freq_ms),
        )
        if not path.exists():
            raise FileNotFoundError(f"missing volatility: {path}")

        frame = pd.read_parquet(path, columns=["timestamp", "volatility"])
        frame["timestamp"] = frame["timestamp"].astype("int64")
        frame["volatility"] = frame["volatility"].astype("float64")
        return frame.loc[:, ["timestamp", "volatility"]]

    def _read_trade_frame(self, symbol: str, date: str) -> pd.DataFrame:
        path = resolve_existing_input_path(
            roots=self.trade_roots,
            symbol=symbol,
            date_str=date,
            category=self.trade_category,
        )
        raw = pd.read_parquet(path, columns=list(RAW_TRADE_COLUMNS))
        missing = set(RAW_TRADE_COLUMNS) - set(raw.columns)
        if missing:
            raise ValueError(f"trade frame missing columns: {sorted(missing)}")

        frame = raw.rename(columns={RAW_TRADE_TIME_COLUMN: "timestamp"})
        frame = frame.dropna(subset=["timestamp", "price", "volume", "is_buyer_maker"])
        frame = frame[(frame["price"] > 0.0) & (frame["volume"] > 0.0)]
        if frame.empty:
            return pd.DataFrame(
                {
                    "timestamp": pd.Series(dtype="int64"),
                    "is_buyer_maker": pd.Series(dtype="bool"),
                    "price": pd.Series(dtype="float64"),
                    "volume": pd.Series(dtype="float64"),
                }
            )

        frame["timestamp"] = frame["timestamp"].astype("int64")
        frame["is_buyer_maker"] = frame["is_buyer_maker"].astype("bool")
        frame["price"] = frame["price"].astype("float64")
        frame["volume"] = frame["volume"].astype("float64")
        return frame.loc[:, ["timestamp", "is_buyer_maker", "price", "volume"]].sort_values(
            "timestamp",
            kind="mergesort",
            ignore_index=True,
        )

    def _read_orderbook_replay_frame(
        self,
        symbol: str,
        date: str,
    ) -> OrderBookReplayFrame | None:
        cfg = self.orderbook_replay_config
        if cfg is None:
            return None

        key = (
            str(cfg.root),
            symbol,
            date,
            cfg.replay_levels,
            cfg.sample_interval_ms,
            cfg.tick_size,
            cfg.depth_price_min,
            cfg.depth_price_max,
            cfg.raw_quantity_max,
        )
        cached = _ORDERBOOK_REPLAY_CACHE.get(key)
        if cached is not None:
            return cached

        path = orderbook_replay_path(root=cfg.root, symbol=symbol, date_str=date, cfg=cfg)
        if not path.exists():
            raise FileNotFoundError(f"missing orderbook replay snapshot: {path}")

        columns = ["timestamp"]
        for side in ("bid", "ask"):
            for idx in range(cfg.replay_levels):
                columns.append(f"{side}_price_{idx}")
                columns.append(f"{side}_qty_{idx}")
        frame = pd.read_parquet(path, columns=columns)
        missing = set(columns) - set(frame.columns)
        if missing:
            raise ValueError(f"orderbook replay frame missing columns: {sorted(missing)}")

        timestamps = frame["timestamp"].astype("int64").to_numpy(copy=True)

        def read_side(side: str) -> tuple[np.ndarray, np.ndarray]:
            price_cols = [f"{side}_price_{idx}" for idx in range(cfg.replay_levels)]
            qty_cols = [f"{side}_qty_{idx}" for idx in range(cfg.replay_levels)]
            prices = frame.loc[:, price_cols].to_numpy(dtype="float64", copy=False)
            qtys = frame.loc[:, qty_cols].to_numpy(dtype="float64", copy=False)
            valid = np.isfinite(prices) & np.isfinite(qtys) & (prices > 0.0) & (qtys > 0.0)
            raw_ticks = np.rint(prices / cfg.tick_size)
            ticks = np.where(valid, raw_ticks, -1).astype(np.int32, copy=False)
            notional = np.where(valid, prices * qtys, 0.0).astype(np.float32, copy=False)
            return np.ascontiguousarray(ticks), np.ascontiguousarray(notional)

        bid_ticks, bid_notional = read_side("bid")
        ask_ticks, ask_notional = read_side("ask")

        replay = OrderBookReplayFrame(
            timestamps=timestamps,
            bid_ticks=bid_ticks,
            ask_ticks=ask_ticks,
            bid_notional=bid_notional,
            ask_notional=ask_notional,
        )
        _ORDERBOOK_REPLAY_CACHE[key] = replay
        _ORDERBOOK_REPLAY_CACHE_ORDER.append(key)
        while len(_ORDERBOOK_REPLAY_CACHE_ORDER) > _ORDERBOOK_REPLAY_CACHE_MAX_DAYS:
            old_key = _ORDERBOOK_REPLAY_CACHE_ORDER.pop(0)
            if old_key != key:
                _ORDERBOOK_REPLAY_CACHE.pop(old_key, None)
        return replay

    @staticmethod
    def _align_orderbook_replay_frame(
        *,
        symbol: str,
        date: str,
        alpha_timestamps: np.ndarray,
        replay: OrderBookReplayFrame,
    ) -> OrderBookReplayFrame:
        if len(alpha_timestamps) == len(replay.timestamps) and np.array_equal(
            alpha_timestamps,
            replay.timestamps,
        ):
            return replay

        indices = np.searchsorted(replay.timestamps, alpha_timestamps)
        if (
            indices.shape[0] != alpha_timestamps.shape[0]
            or np.any(indices >= len(replay.timestamps))
            or not np.array_equal(replay.timestamps[indices], alpha_timestamps)
        ):
            raise ValueError(
                "orderbook replay timestamp grid does not match sampled ticker "
                f"for {symbol} {date}"
            )
        return OrderBookReplayFrame(
            timestamps=np.array(alpha_timestamps, dtype=np.int64, copy=True),
            bid_ticks=np.ascontiguousarray(replay.bid_ticks[indices]),
            ask_ticks=np.ascontiguousarray(replay.ask_ticks[indices]),
            bid_notional=np.ascontiguousarray(replay.bid_notional[indices]),
            ask_notional=np.ascontiguousarray(replay.ask_notional[indices]),
        )

    @staticmethod
    def _parse_spec(spec: dict[str, int | str], default_name: str) -> tuple[str, int]:
        name = str(spec.get("name", default_name)).strip().lower()
        if not name:
            raise ValueError("indicator name must not be empty")
        lookback = int(spec.get("lookback", 0))
        if name == "bbo_imbalance":
            if lookback != 0:
                raise ValueError("bbo_imbalance lookback must be 0")
        elif lookback <= 0:
            raise ValueError(f"{name} lookback must be > 0")
        return name, lookback

    @staticmethod
    def _merge_sorted(
        alpha: pd.DataFrame,
        trades: pd.DataFrame,
        replay: OrderBookReplayFrame | None = None,
    ) -> Iterator[MergedEventTuple]:
        alpha_iter = iter(
            alpha.loc[
                :,
                [
                    "timestamp",
                    "best_bid_price",
                    "best_ask_price",
                    "instructor",
                    "intensity",
                    "volatility_scalar",
                ],
            ].itertuples(index=False, name=None)
        )
        trade_iter = iter(
            trades.loc[:, ["timestamp", "is_buyer_maker", "price", "volume"]].itertuples(
                index=False,
                name=None,
            )
        )

        alpha_row = next(alpha_iter, None)
        alpha_idx = 0
        trade_row = next(trade_iter, None)

        while alpha_row is not None or trade_row is not None:
            if trade_row is not None and (
                alpha_row is None or int(trade_row[0]) <= int(alpha_row[0])
            ):
                yield (
                    "trade",
                    int(trade_row[0]),
                    bool(trade_row[1]),
                    float(trade_row[2]),
                    float(trade_row[3]),
                )
                trade_row = next(trade_iter, None)
                continue

            assert alpha_row is not None
            bid = float(alpha_row[1])
            ask = float(alpha_row[2])
            instructor = float(alpha_row[3]) if math.isfinite(float(alpha_row[3])) else 0.0
            intensity = float(alpha_row[4]) if math.isfinite(float(alpha_row[4])) else 0.0
            volatility = float(alpha_row[5]) if math.isfinite(float(alpha_row[5])) else 0.0
            bid_replay_ticks = None if replay is None else replay.bid_ticks[alpha_idx]
            ask_replay_ticks = None if replay is None else replay.ask_ticks[alpha_idx]
            bid_replay_notional = None if replay is None else replay.bid_notional[alpha_idx]
            ask_replay_notional = None if replay is None else replay.ask_notional[alpha_idx]
            yield (
                "ticker",
                int(alpha_row[0]),
                bid,
                ask,
                instructor,
                intensity,
                volatility,
                bid_replay_ticks,
                ask_replay_ticks,
                bid_replay_notional,
                ask_replay_notional,
            )
            alpha_row = next(alpha_iter, None)
            alpha_idx += 1
