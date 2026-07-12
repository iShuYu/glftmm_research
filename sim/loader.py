from __future__ import annotations

import datetime as dt
import math
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal, NamedTuple, TypeAlias

import numpy as np
import pandas as pd

from sampler.instructor import instructor_output_path
from sampler.intensity import intensity_output_path, normalize_kls_lookback
from sampler.resample import normalize_scheme_shift
from sampler.resample import output_path as sampled_ticker_path
from sampler.resample import resolve_existing_input_path
from sampler.volatility import volatility_output_path


DateLike: TypeAlias = str | dt.date | dt.datetime | pd.Timestamp


class TradeEvent(NamedTuple):
    kind: Literal["trade"]
    timestamp: int
    is_buyer_maker: bool
    price: float
    volume: float


class AlphaEvent(NamedTuple):
    kind: Literal["ticker"]
    timestamp: int
    best_bid_price: float
    best_ask_price: float
    instructor: float | None
    intensity_positive: float | None
    intensity_negative: float | None
    volatility_scalar: float


MergedEventTuple: TypeAlias = TradeEvent | AlphaEvent

RAW_TRADE_TIME_COLUMN = "exchange_timestamp"
TRADE_TYPE_COLUMN = "trade_type"
RAW_TRADE_COLUMNS = (
    RAW_TRADE_TIME_COLUMN,
    "price",
    "volume",
    "is_buyer_maker",
    TRADE_TYPE_COLUMN,
)
TICKER_COLUMNS = (
    "timestamp",
    "best_bid_price",
    "best_ask_price",
)
INTENSITY_COLUMNS = ("intensity_positive", "intensity_negative")
INTENSITY_COLUMN_ALIASES = {
    "intensity_positive": ("intensity_positive", "kw_vol_positive"),
    "intensity_negative": ("intensity_negative", "kw_vol_negative"),
}
DEFAULT_CACHE_ROOT = Path("/data/users/kang/backtest/glftmm/cached")
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
        alpha["intensity_positive"] = 0.0
        alpha["intensity_negative"] = 0.0
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
            for column in INTENSITY_COLUMNS:
                new_column = f"{column}_new"
                alpha[column] = alpha[new_column].fillna(alpha[column])
                alpha = alpha.drop(columns=[new_column])

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
            "intensity_positive",
            "intensity_negative",
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
        spec: dict[str, Any],
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
            frame = self._read_directional_intensity_pair(
                symbol=symbol,
                date=date,
                freq_ms=freq_ms,
                name=name,
                lookback=lookback,
            )
            if frame is not None:
                return frame
            raise FileNotFoundError(f"missing intensity: {path}")

        frame = pd.read_parquet(path)
        frame = self._normalize_intensity_columns(frame, source=str(path))
        return frame.loc[:, ["timestamp", *INTENSITY_COLUMNS]]

    def _read_directional_intensity_pair(
        self,
        symbol: str,
        date: str,
        freq_ms: int,
        name: str,
        lookback: int | str,
    ) -> pd.DataFrame | None:
        if name not in ("kw_vol", "kw_vol_positive", "kw_vol_negative"):
            return None

        positive = self._read_single_intensity_file(
            symbol=symbol,
            date=date,
            freq_ms=freq_ms,
            name="kw_vol_positive",
            lookback=lookback,
        )
        negative = self._read_single_intensity_file(
            symbol=symbol,
            date=date,
            freq_ms=freq_ms,
            name="kw_vol_negative",
            lookback=lookback,
        )
        if positive is None or negative is None:
            return None

        frame = positive.merge(negative, on="timestamp", how="outer")
        frame["intensity_positive"] = frame["intensity_positive"].fillna(0.0)
        frame["intensity_negative"] = frame["intensity_negative"].fillna(0.0)
        frame["timestamp"] = frame["timestamp"].astype("int64")
        frame["intensity_positive"] = frame["intensity_positive"].astype("float64")
        frame["intensity_negative"] = frame["intensity_negative"].astype("float64")
        return frame.loc[:, ["timestamp", *INTENSITY_COLUMNS]].sort_values(
            "timestamp",
            kind="mergesort",
            ignore_index=True,
        )

    def _read_single_intensity_file(
        self,
        symbol: str,
        date: str,
        freq_ms: int,
        name: str,
        lookback: int | str,
    ) -> pd.DataFrame | None:
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
            return None

        side = name.rsplit("_", 1)[-1]
        column = f"intensity_{side}"
        frame = pd.read_parquet(path)
        if "timestamp" not in frame.columns or "intensity" not in frame.columns:
            raise ValueError(f"single-side intensity frame has invalid schema: {path}")
        frame["timestamp"] = frame["timestamp"].astype("int64")
        frame[column] = frame["intensity"].astype("float64")
        return frame.loc[:, ["timestamp", column]]

    @staticmethod
    def _normalize_intensity_columns(frame: pd.DataFrame, source: str) -> pd.DataFrame:
        if "timestamp" not in frame.columns:
            raise ValueError(f"intensity frame missing timestamp column: {source}")

        out = frame.loc[:, ["timestamp"]].copy()
        if "intensity" in frame.columns:
            out["intensity_positive"] = frame["intensity"]
            out["intensity_negative"] = frame["intensity"]
        else:
            for output_column, aliases in INTENSITY_COLUMN_ALIASES.items():
                for alias in aliases:
                    if alias in frame.columns:
                        out[output_column] = frame[alias]
                        break
                else:
                    raise ValueError(
                        f"intensity frame missing {output_column} column: {source}"
                    )

        out["timestamp"] = out["timestamp"].astype("int64")
        out["intensity_positive"] = out["intensity_positive"].astype("float64")
        out["intensity_negative"] = out["intensity_negative"].astype("float64")
        return out

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
        raw = raw.loc[raw[TRADE_TYPE_COLUMN] == 0]

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
    def _parse_spec(spec: dict[str, Any], default_name: str) -> tuple[str, int | str]:
        name = str(spec.get("name", default_name)).strip().lower()
        if not name:
            raise ValueError("indicator name must not be empty")
        raw_lookback = spec.get("lookback", 0)
        if name == "bbo_imbalance":
            lookback = int(raw_lookback)
            if lookback != 0:
                raise ValueError("bbo_imbalance lookback must be 0")
        elif name == "volume_zscore":
            lookback = int(raw_lookback)
            if lookback < 2:
                raise ValueError("volume_zscore lookback must be >= 2")
        elif name == "kls":
            lookback = normalize_kls_lookback(raw_lookback)
        else:
            lookback = int(raw_lookback)
            if lookback <= 0:
                raise ValueError(f"{name} lookback must be > 0")
        return name, lookback

    @staticmethod
    def _merge_sorted(
        alpha: pd.DataFrame,
        trades: pd.DataFrame,
    ) -> Iterator[MergedEventTuple]:
        alpha_iter = iter(
            alpha.loc[
                :,
                [
                    "timestamp",
                    "best_bid_price",
                    "best_ask_price",
                    "instructor",
                    "intensity_positive",
                    "intensity_negative",
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
                yield TradeEvent(
                    kind="trade",
                    timestamp=int(trade_row[0]),
                    is_buyer_maker=bool(trade_row[1]),
                    price=float(trade_row[2]),
                    volume=float(trade_row[3]),
                )
                trade_row = next(trade_iter, None)
                continue

            assert alpha_row is not None
            bid = float(alpha_row[1])
            ask = float(alpha_row[2])
            instructor = float(alpha_row[3]) if math.isfinite(float(alpha_row[3])) else 0.0
            intensity_positive = (
                float(alpha_row[4]) if math.isfinite(float(alpha_row[4])) else 0.0
            )
            intensity_negative = (
                float(alpha_row[5]) if math.isfinite(float(alpha_row[5])) else 0.0
            )
            volatility = float(alpha_row[6]) if math.isfinite(float(alpha_row[6])) else 0.0
            yield AlphaEvent(
                kind="ticker",
                timestamp=int(alpha_row[0]),
                best_bid_price=bid,
                best_ask_price=ask,
                instructor=instructor,
                intensity_positive=intensity_positive,
                intensity_negative=intensity_negative,
                volatility_scalar=volatility,
            )
            alpha_row = next(alpha_iter, None)
