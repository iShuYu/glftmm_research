from __future__ import annotations

import datetime as dt
import math
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal, TypeAlias

import numpy as np
import pandas as pd


DateLike: TypeAlias = str | dt.date | dt.datetime | pd.Timestamp
TradeTuple: TypeAlias = tuple[Literal["trade"], int, bool, float, float]
BookTickerTuple: TypeAlias = tuple[
    Literal["bookticker"],
    int,
    float,
    float,
    float,
    float,
]
AggTradeTuple: TypeAlias = tuple[
    Literal["aggtrade"],
    int,
    bool,
    float,
    float,
    float,
    float,
    float,
]
MergedEventTuple: TypeAlias = TradeTuple | BookTickerTuple | AggTradeTuple

MS_IN_SECOND = 1000
MS_IN_DAY = 24 * 60 * 60 * MS_IN_SECOND

DEFAULT_DATA_ROOT = Path("/data/users/data-helper/PROCESSED/TARDIS/BINANCE/UFUTURES")
DEFAULT_BACKUP_DATA_ROOT = Path(
    "/home/kang/data_helper/PROCESSED/DATA_RECORDER/BINANCE/UFUTURES"
)

RAW_TRADE_TIME_COLUMN = "exchange_timestamp"
RAW_TRADE_COLUMNS = (
    RAW_TRADE_TIME_COLUMN,
    "price",
    "volume",
    "is_buyer_maker",
)
RAW_TICKER_COLUMNS = (
    "exchange_timestamp",
    "price_bid",
    "price_ask",
    "volume_bid",
    "volume_ask",
)
BBO_COLUMNS = (
    "timestamp",
    "best_bid_price",
    "best_ask_price",
    "best_bid_qty",
    "best_ask_qty",
)
AGGTRADE_COLUMNS = (
    "timestamp",
    "is_buyer_maker",
    "impact",
    "intensity",
    "volume",
    "first_price",
    "last_price",
    "forced_close",
)


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


def day_start_timestamp_ms(date_str: str) -> int:
    date_obj = dt.date.fromisoformat(date_str)
    day_start = dt.datetime.combine(date_obj, dt.time.min)
    epoch = dt.datetime(1970, 1, 1)
    return int((day_start - epoch).total_seconds() * MS_IN_SECOND)


def day_end_timestamp_ms(date_str: str) -> int:
    return day_start_timestamp_ms(date_str) + MS_IN_DAY - 1


def _ensure_paths(value: Any) -> tuple[Path, ...]:
    if value is None:
        return ()
    if isinstance(value, Path):
        items = [value]
    elif isinstance(value, str):
        items = [value]
    elif isinstance(value, Iterable):
        items = list(value)
    else:
        items = [value]

    paths: list[Path] = []
    seen: set[str] = set()
    for item in items:
        if item is None:
            continue
        text = str(item).strip()
        if not text:
            continue
        if text in seen:
            continue
        seen.add(text)
        paths.append(Path(text))
    return tuple(paths)


def input_path(root: Path, symbol: str, date_str: str, category: str) -> Path:
    return root / symbol / f"{symbol}--{category}--{date_str}.parquet"


def resolve_existing_input_path(
    roots: Any,
    symbol: str,
    date_str: str,
    category: str,
) -> Path:
    candidates = [
        input_path(root=root, symbol=symbol, date_str=date_str, category=category)
        for root in _ensure_paths(roots)
    ]
    for path in candidates:
        if path.exists():
            return path
    if not candidates:
        raise FileNotFoundError("no input roots configured")
    raise FileNotFoundError(
        "missing input file; tried: " + ", ".join(str(path) for path in candidates)
    )


def empty_aggtrade_events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.Series(dtype="int64"),
            "is_buyer_maker": pd.Series(dtype="bool"),
            "impact": pd.Series(dtype="float64"),
            "intensity": pd.Series(dtype="float64"),
            "volume": pd.Series(dtype="float64"),
            "first_price": pd.Series(dtype="float64"),
            "last_price": pd.Series(dtype="float64"),
            "forced_close": pd.Series(dtype="bool"),
        }
    )


def build_confirmed_aggtrade_events(
    trades: pd.DataFrame,
    force_close_timestamp: int | None = None,
) -> pd.DataFrame:
    if trades.empty:
        return empty_aggtrade_events()

    side = trades["is_buyer_maker"].to_numpy(dtype="bool", copy=False)
    ts = trades["timestamp"].to_numpy(dtype="int64", copy=False)
    price = trades["price"].to_numpy(dtype="float64", copy=False)
    volume = trades["volume"].to_numpy(dtype="float64", copy=False)

    starts = np.r_[0, np.flatnonzero(side[1:] != side[:-1]) + 1]
    if len(starts) < 2 and force_close_timestamp is None:
        return empty_aggtrade_events()
    ends = np.r_[starts[1:], len(ts)]

    group_starts = starts[:-1].astype("int64", copy=False)
    group_ends = ends[:-1].astype("int64", copy=False)
    confirm_ts = ts[starts[1:]].astype("int64", copy=False)
    forced_close = np.zeros(len(confirm_ts), dtype="bool")

    if force_close_timestamp is not None:
        group_starts = np.r_[group_starts, int(starts[-1])]
        group_ends = np.r_[group_ends, len(ts)]
        confirm_ts = np.r_[confirm_ts, int(force_close_timestamp)].astype(
            "int64",
            copy=False,
        )
        forced_close = np.r_[forced_close, True].astype("bool", copy=False)

    group_side = side[group_starts]
    first_price = price[group_starts]
    last_price = price[group_ends - 1]
    volume_cumsum = np.r_[0.0, np.cumsum(volume, dtype="float64")]
    group_volume = volume_cumsum[group_ends] - volume_cumsum[group_starts]

    buy_impact = last_price - first_price
    sell_impact = first_price - last_price
    impact = np.where(group_side, sell_impact, buy_impact)
    impact = np.maximum(impact, 0.0).astype("float64", copy=False)

    return pd.DataFrame(
        {
            "timestamp": confirm_ts.astype("int64", copy=False),
            "is_buyer_maker": group_side.astype("bool", copy=False),
            "impact": impact,
            "volume": group_volume.astype("float64", copy=False),
            "first_price": first_price.astype("float64", copy=False),
            "last_price": last_price.astype("float64", copy=False),
            "forced_close": forced_close,
        }
    )


class BinanceEventLoader:
    """
    Stream raw trades, raw bookticker, and confirmed aggtrade impact events.

    Same-timestamp public trades are emitted before book state and aggtrade
    updates so resting maker orders cannot react before they are eligible to fill.
    """

    def __init__(
        self,
        input_path: str | Path | None = None,
        input_backup_path: str | Path | None = None,
        bookticker_roots: Iterable[str | Path] | None = None,
        ticker_category: str | None = None,
        trade_roots: Iterable[str | Path] | None = None,
        trade_category: str | None = None,
    ) -> None:
        trade_cat = str(trade_category or "TRADE").strip().upper()
        ticker_cat = str(ticker_category or "BOOKTICKER").strip().upper()
        if not trade_cat:
            raise ValueError("trade_category must not be empty")
        if not ticker_cat:
            raise ValueError("ticker_category must not be empty")

        self.trade_category = trade_cat
        self.ticker_category = ticker_cat

        if trade_roots is not None:
            self.trade_roots = tuple(Path(path) for path in trade_roots)
        elif input_path is not None:
            roots = [Path(input_path) / trade_cat]
            if input_backup_path not in (None, "", []):
                roots.append(Path(input_backup_path) / trade_cat)
            self.trade_roots = tuple(roots)
        else:
            self.trade_roots = (
                DEFAULT_DATA_ROOT / trade_cat,
                DEFAULT_BACKUP_DATA_ROOT / trade_cat,
            )

        if bookticker_roots is not None:
            self.bookticker_roots = tuple(Path(path) for path in bookticker_roots)
        elif input_path is not None:
            roots = [Path(input_path) / ticker_cat]
            if input_backup_path not in (None, "", []):
                roots.append(Path(input_backup_path) / ticker_cat)
            self.bookticker_roots = tuple(roots)
        else:
            self.bookticker_roots = (
                DEFAULT_DATA_ROOT / ticker_cat,
                DEFAULT_BACKUP_DATA_ROOT / ticker_cat,
            )

    def iter_merged_trade_intensity_tuples(
        self,
        symbol: str,
        date: DateLike,
    ) -> Iterator[MergedEventTuple]:
        symbol = str(symbol).upper()
        date_str = normalize_date(date)
        trades = self._read_trade_frame(symbol=symbol, date=date_str)
        bookticker = self._read_bookticker_frame(symbol=symbol, date=date_str)
        aggtrades = self._build_aggtrade_intensity_frame(
            trades=trades,
            force_close_timestamp=day_end_timestamp_ms(date_str),
        )
        yield from self._merge_events(
            trades=trades,
            bookticker=bookticker,
            aggtrades=aggtrades,
        )

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

    def _read_bookticker_frame(self, symbol: str, date: str) -> pd.DataFrame:
        path = resolve_existing_input_path(
            roots=self.bookticker_roots,
            symbol=symbol,
            date_str=date,
            category=self.ticker_category,
        )
        raw = pd.read_parquet(path, columns=list(RAW_TICKER_COLUMNS))
        missing = set(RAW_TICKER_COLUMNS) - set(raw.columns)
        if missing:
            raise ValueError(f"bookticker frame missing columns: {sorted(missing)}")

        frame = raw.rename(
            columns={
                "exchange_timestamp": "timestamp",
                "price_bid": "best_bid_price",
                "price_ask": "best_ask_price",
                "volume_bid": "best_bid_qty",
                "volume_ask": "best_ask_qty",
            }
        )
        frame = frame.dropna(subset=list(BBO_COLUMNS))
        frame = frame[
            (frame["best_bid_price"] > 0.0)
            & (frame["best_ask_price"] > 0.0)
            & (frame["best_ask_price"] >= frame["best_bid_price"])
            & (frame["best_bid_qty"] >= 0.0)
            & (frame["best_ask_qty"] >= 0.0)
        ]
        if frame.empty:
            return pd.DataFrame({column: pd.Series(dtype="float64") for column in BBO_COLUMNS})

        frame["timestamp"] = frame["timestamp"].astype("int64")
        for column in BBO_COLUMNS[1:]:
            frame[column] = frame[column].astype("float64")
        return frame.loc[:, BBO_COLUMNS].sort_values(
            "timestamp",
            kind="mergesort",
            ignore_index=True,
        )

    @staticmethod
    def _build_aggtrade_intensity_frame(
        *,
        trades: pd.DataFrame,
        force_close_timestamp: int | None = None,
    ) -> pd.DataFrame:
        columns = {
            "timestamp": pd.Series(dtype="int64"),
            "is_buyer_maker": pd.Series(dtype="bool"),
            "impact": pd.Series(dtype="float64"),
            "intensity": pd.Series(dtype="float64"),
            "volume": pd.Series(dtype="float64"),
            "first_price": pd.Series(dtype="float64"),
            "last_price": pd.Series(dtype="float64"),
            "forced_close": pd.Series(dtype="bool"),
        }
        if trades.empty:
            return pd.DataFrame(columns)

        events = build_confirmed_aggtrade_events(
            trades=trades,
            force_close_timestamp=force_close_timestamp,
        )
        if events.empty:
            return pd.DataFrame(columns)
        events = events.sort_values("timestamp", kind="mergesort", ignore_index=True)
        events["intensity"] = events["impact"].astype("float64")
        return events.loc[:, AGGTRADE_COLUMNS]

    @staticmethod
    def _merge_events(
        trades: pd.DataFrame,
        bookticker: pd.DataFrame,
        aggtrades: pd.DataFrame,
    ) -> Iterator[MergedEventTuple]:
        trade_iter = iter(
            trades.loc[:, ["timestamp", "is_buyer_maker", "price", "volume"]].itertuples(
                index=False,
                name=None,
            )
        )
        bbo_iter = iter(bookticker.loc[:, BBO_COLUMNS].itertuples(index=False, name=None))
        agg_iter = iter(
            aggtrades.loc[
                :,
                [
                    "timestamp",
                    "is_buyer_maker",
                    "impact",
                    "intensity",
                    "volume",
                    "first_price",
                    "last_price",
                ],
            ].itertuples(index=False, name=None)
        )

        trade_row = next(trade_iter, None)
        bbo_row = next(bbo_iter, None)
        agg_row = next(agg_iter, None)
        while trade_row is not None or bbo_row is not None or agg_row is not None:
            candidates: list[int] = []
            if trade_row is not None:
                candidates.append(int(trade_row[0]))
            if bbo_row is not None:
                candidates.append(int(bbo_row[0]))
            if agg_row is not None:
                candidates.append(int(agg_row[0]))
            ts = min(candidates)

            while trade_row is not None and int(trade_row[0]) == ts:
                row = trade_row
                yield (
                    "trade",
                    int(row[0]),
                    bool(row[1]),
                    float(row[2]),
                    float(row[3]),
                )
                trade_row = next(trade_iter, None)

            while bbo_row is not None and int(bbo_row[0]) == ts:
                row = bbo_row
                bid = float(row[1])
                ask = float(row[2])
                bid_qty = float(row[3]) if math.isfinite(float(row[3])) else 0.0
                ask_qty = float(row[4]) if math.isfinite(float(row[4])) else 0.0
                yield (
                    "bookticker",
                    int(row[0]),
                    bid,
                    ask,
                    bid_qty,
                    ask_qty,
                )
                bbo_row = next(bbo_iter, None)

            while agg_row is not None and int(agg_row[0]) == ts:
                row = agg_row
                yield (
                    "aggtrade",
                    int(row[0]),
                    bool(row[1]),
                    float(row[2]),
                    float(row[3]),
                    float(row[4]),
                    float(row[5]),
                    float(row[6]),
                )
                agg_row = next(agg_iter, None)
