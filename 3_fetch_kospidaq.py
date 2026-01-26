#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd
from pykrx import stock
from tqdm import tqdm

DEFAULT_START = "20140102"
DEFAULT_END = "20191230"
DEFAULT_OUTPUT_DB = "data/kospidaq.sqlite"
DEFAULT_SLEEP_SEC = 0.2
DEFAULT_SAVE_EVERY = 5000

KOR_OPEN = "\uc2dc\uac00"
KOR_HIGH = "\uace0\uac00"
KOR_LOW = "\uc800\uac00"
KOR_CLOSE = "\uc885\uac00"
KOR_VOLUME = "\uac70\ub798\ub7c9"
KOR_TURNOVER = "\uac70\ub798\ub300\uae08"
KOR_MCAP = "\uc2dc\uac00\ucd1d\uc561"


def _ensure_tables(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS market_data (
            market TEXT NOT NULL,
            date TEXT NOT NULL,
            ticker TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            turnover REAL,
            market_cap REAL,
            per REAL,
            pbr REAL,
            PRIMARY KEY (market, date, ticker)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ticker_info (
            ticker TEXT PRIMARY KEY,
            market TEXT NOT NULL,
            name TEXT,
            sector TEXT,
            industry TEXT,
            list_date TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT
        )
        """
    )
    conn.commit()


def _to_date_str(index: pd.Index) -> pd.Series:
    return pd.to_datetime(index).strftime("%Y-%m-%d")


def _fetch_ticker_df(ticker: str, start: str, end: str) -> pd.DataFrame:
    try:
        ohlcv = stock.get_market_ohlcv_by_date(start, end, ticker)
        cap = stock.get_market_cap_by_date(start, end, ticker)
        fundamental = stock.get_market_fundamental_by_date(start, end, ticker)
    except Exception:
        return pd.DataFrame()

    if ohlcv is None or ohlcv.empty:
        return pd.DataFrame()

    df = ohlcv.join(cap, how="left").join(fundamental, how="left")
    df.index = _to_date_str(df.index)

    rename_map = {
        KOR_OPEN: "open",
        KOR_HIGH: "high",
        KOR_LOW: "low",
        KOR_CLOSE: "close",
        KOR_VOLUME: "volume",
        KOR_TURNOVER: "turnover",
        KOR_MCAP: "market_cap",
        "PER": "per",
        "PBR": "pbr",
    }
    df = df.rename(columns=rename_map)

    keep_cols = ["open", "high", "low", "close", "volume", "turnover", "market_cap", "per", "pbr"]
    for col in keep_cols:
        if col not in df.columns:
            df[col] = None
    return df[keep_cols]


def _iter_rows(
    market: str, ticker: str, df: pd.DataFrame
) -> Iterable[Tuple[str, str, str, float, float, float, float, float, float, float, float, float]]:
    for date_str, row in df.iterrows():
        yield (
            market,
            date_str,
            ticker,
            row.get("open"),
            row.get("high"),
            row.get("low"),
            row.get("close"),
            row.get("volume"),
            row.get("turnover"),
            row.get("market_cap"),
            row.get("per"),
            row.get("pbr"),
        )


def _insert_market_rows(conn: sqlite3.Connection, rows: List[Tuple]) -> None:
    conn.executemany(
        """
        INSERT OR REPLACE INTO market_data (
            market, date, ticker, open, high, low, close,
            volume, turnover, market_cap, per, pbr
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


def _load_tickers(conn: sqlite3.Connection) -> List[Tuple[str, str]]:
    cur = conn.cursor()
    cur.execute("SELECT ticker, market FROM ticker_info ORDER BY ticker")
    return [(row[0], row[1]) for row in cur.fetchall()]


def main() -> None:
    output_path = Path(DEFAULT_OUTPUT_DB).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(output_path)
    _ensure_tables(conn)

    tickers = _load_tickers(conn)
    if not tickers:
        raise RuntimeError(
            "ticker_info is empty. Populate ticker_info first; pykrx ticker list endpoints are failing."
        )

    total_inserted = 0
    empty_tickers = 0
    batch: List[Tuple] = []

    for ticker, market in tqdm(tickers, desc="Tickers", unit="ticker"):
        df = _fetch_ticker_df(ticker, DEFAULT_START, DEFAULT_END)
        if df.empty:
            empty_tickers += 1
            continue
        for row in _iter_rows(market, ticker, df):
            batch.append(row)
            if DEFAULT_SAVE_EVERY and len(batch) >= DEFAULT_SAVE_EVERY:
                _insert_market_rows(conn, batch)
                conn.commit()
                total_inserted += len(batch)
                batch = []
        time.sleep(DEFAULT_SLEEP_SEC)

    if batch:
        _insert_market_rows(conn, batch)
        conn.commit()
        total_inserted += len(batch)

    conn.execute(
        "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
        ("last_run", time.strftime("%Y-%m-%d %H:%M:%S")),
    )
    conn.commit()
    conn.close()
    print(
        f"Done. Inserted/updated {total_inserted} rows into {output_path}. "
        f"Empty tickers: {empty_tickers}/{len(tickers)}"
    )


if __name__ == "__main__":
    main()
