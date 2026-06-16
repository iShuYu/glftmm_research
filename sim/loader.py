from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal, TypeAlias

import pandas as pd

from sampler.instructor import instructor_output_path
from sampler.intensity import intensity_output_path
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
PROJECT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "sampler" / "config.json"
DEFAULT_CACHE_ROOT = Path("/data/users/kang/backtest/glftmm_lot/cached")
DEFAULT_DATA_ROOT = Path("/data/users/data-helper/PROCESSED/TARDIS/BINANCE/UFUTURES")
DEFAULT_BACKUP_DATA_ROOT = Path("/home/kang/data_helper/PROCESSED/DATA_RECORDER/BINANCE/UFUTURES")


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


def _load_project_config() -> dict[str, Any]:
    try:
        with open(PROJECT_CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except FileNotFoundError:
        return {}
    if not isinstance(cfg, dict):
        return {}
    return cfg


def _configured_cache_root() -> Path:
    cfg = _load_project_config()
    value = cfg.get("output_path")
    if value not in (None, "", []):
        return Path(value)
    return DEFAULT_CACHE_ROOT


def _configured_trade_roots(trade_category: str) -> tuple[Path, ...]:
    cfg = _load_project_config()
    roots: list[Path] = []

    explicit_trade_path = cfg.get("trade_path")
    if explicit_trade_path not in (None, "", []):
        roots.append(Path(explicit_trade_path))
    else:
        input_path = cfg.get("input_path")
        if input_path not in (None, "", []):
            roots.append(Path(input_path) / trade_category)

    explicit_backup_path = cfg.get("trade_backup_path")
    if explicit_backup_path not in (None, "", []):
        roots.append(Path(explicit_backup_path))
    else:
        input_backup_path = cfg.get("input_backup_path")
        if input_backup_path not in (None, "", []):
            roots.append(Path(input_backup_path) / trade_category)

    if not roots:
        roots = [DEFAULT_DATA_ROOT / trade_category, DEFAULT_BACKUP_DATA_ROOT / trade_category]

    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root)
        if key in seen:
            continue
        seen.add(key)
        unique.append(root)
    return tuple(unique)


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
        trades_root: str | Path | None = None,
        bookticker_root: str | Path | None = None,
        ticker_cache_root: str | Path | None = None,
        instructor_cache_root: str | Path | None = None,
        trade_intensity_cache_root: str | Path | None = None,
        intensity_cache_root: str | Path | None = None,
        volatility_cache_root: str | Path | None = None,
        trade_roots: Iterable[str | Path] | None = None,
        trade_category: str | None = None,
        ticker_category: str | None = None,
    ) -> None:
        project_cfg = _load_project_config()
        category = str(
            trade_category
            or project_cfg.get("trade_category")
            or "TRADE"
        ).strip().upper()
        if not category:
            raise ValueError("trade_category must not be empty")
        ticker_category_norm = str(
            ticker_category
            or project_cfg.get("ticker_category")
            or "BOOKTICKER"
        ).strip().upper()
        if not ticker_category_norm:
            raise ValueError("ticker_category must not be empty")

        default_cache_root = Path(cache_root) if cache_root is not None else _configured_cache_root()
        self.cache_root = default_cache_root
        if bookticker_root is not None:
            self.bookticker_root = Path(bookticker_root)
        elif input_path is not None:
            self.bookticker_root = Path(input_path) / ticker_category_norm
        else:
            self.bookticker_root = None
        self.ticker_cache_root = (
            Path(ticker_cache_root) if ticker_cache_root is not None else default_cache_root
        )
        self.instructor_cache_root = (
            Path(instructor_cache_root)
            if instructor_cache_root is not None
            else default_cache_root
        )
        resolved_intensity_root = (
            trade_intensity_cache_root
            if trade_intensity_cache_root is not None
            else intensity_cache_root
        )
        self.trade_intensity_cache_root = (
            Path(resolved_intensity_root)
            if resolved_intensity_root is not None
            else default_cache_root
        )
        self.volatility_cache_root = (
            Path(volatility_cache_root)
            if volatility_cache_root is not None
            else default_cache_root
        )
        self.trade_category = category
        if trade_roots is not None:
            roots = tuple(Path(path) for path in trade_roots)
        elif trades_root is not None:
            roots = (Path(trades_root),)
        elif input_path is not None:
            root_list = [Path(input_path) / category]
            if input_backup_path not in (None, "", []):
                root_list.append(Path(input_backup_path) / category)
            roots = tuple(root_list)
        else:
            roots = _configured_trade_roots(category)
        self.trade_roots = roots

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
        trades = self._read_trade_frame(symbol=symbol, date=date_str)
        yield from self._merge_sorted(alpha=alpha, trades=trades)

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
            root=self.ticker_cache_root,
            symbol=symbol,
            freq_ms=freq_ms,
            date_str=date,
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
            root=self.instructor_cache_root,
            symbol=symbol,
            indicator=name,
            freq_ms=freq_ms,
            lookback=lookback,
            date_str=date,
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
            root=self.trade_intensity_cache_root,
            symbol=symbol,
            indicator=name,
            freq_ms=freq_ms,
            lookback=lookback,
            date_str=date,
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
            root=self.volatility_cache_root,
            symbol=symbol,
            indicator=name,
            freq_ms=freq_ms,
            lookback=lookback,
            date_str=date,
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
    def _merge_sorted(alpha: pd.DataFrame, trades: pd.DataFrame) -> Iterator[MergedEventTuple]:
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
            yield (
                "ticker",
                int(alpha_row[0]),
                bid,
                ask,
                instructor,
                intensity,
                volatility,
            )
            alpha_row = next(alpha_iter, None)


MarketDataLoader = BinanceEventLoader
