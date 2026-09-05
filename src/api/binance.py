"""
Fetch BTC hourly OHLCV data from Binance and cache it in SQLite.

Usage from the command line::

    python -m src.api.binance            # ensure last 10 years of BTCUSDT 1h bars
    python -m src.api.binance --years 3  # shorter window

Usage from Python::

    from src.api.binance import load_prices
    df = load_prices()                   # pandas DataFrame indexed by UTC timestamp

Every call goes through `ensure_prices`, which checks what is already in the
``prices`` table of ``data/technical_analysis.sqlite`` and only calls the Binance API
for the parts that are missing (an empty database, a gap at the front, or new bars at
the back). Binance spot BTCUSDT history starts 2017-08-17, so a 10-year request will
in practice begin there; the earliest available timestamp is recorded in a ``meta``
table so we do not re-request pre-listing history on every run.
"""

from __future__ import annotations
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import argparse
import logging
import sqlite3
import time
import pandas as pd
import requests

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

DEFAULT_DB_PATH = DATA_DIR / "technical_analysis.sqlite"
DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_INTERVAL = "1h"
DEFAULT_YEARS = 10
TABLE = "prices"

# Binance market-data hosts, tried in order. ``data-api.binance.vision`` serves public
# market data only and is usually reachable from regions where the main host is not.
BINANCE_HOSTS = (
    "https://api.binance.com",
    "https://data-api.binance.vision",
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com",
    "https://api4.binance.com",
)
KLINES_PATH = "/api/v3/klines"
KLINES_LIMIT = 1000  # max rows per request
REQUEST_TIMEOUT = 30  # seconds
MAX_RETRIES = 5
POLITE_SLEEP = 0.1  # seconds between paginated requests

INTERVAL_MS = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "8h": 28_800_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
}

COLUMNS = [
    "symbol",
    "interval",
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades",
    "taker_buy_base",
    "taker_buy_quote",
]

# --------------------------------------------------------------------------------------
# SQLite
# --------------------------------------------------------------------------------------

def get_connection(db_path: Path | str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Open (creating if needed) the SQLite database and make sure the schema exists."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    init_db(conn)
    return conn

def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        f"""
        CREATE TABLE IF NOT EXISTS {TABLE} (
            symbol          TEXT    NOT NULL,
            interval        TEXT    NOT NULL,
            open_time       INTEGER NOT NULL,   -- ms since epoch, UTC
            open            REAL    NOT NULL,
            high            REAL    NOT NULL,
            low             REAL    NOT NULL,
            close           REAL    NOT NULL,
            volume          REAL    NOT NULL,   -- base asset volume
            close_time      INTEGER NOT NULL,
            quote_volume    REAL,
            trades          INTEGER,
            taker_buy_base  REAL,
            taker_buy_quote REAL,
            PRIMARY KEY (symbol, interval, open_time)
        );
        CREATE INDEX IF NOT EXISTS idx_{TABLE}_open_time ON {TABLE} (open_time);

        -- Per (symbol, interval) bookkeeping, e.g. the earliest timestamp the exchange has.
        CREATE TABLE IF NOT EXISTS meta (
            symbol   TEXT NOT NULL,
            interval TEXT NOT NULL,
            key      TEXT NOT NULL,
            value    TEXT,
            PRIMARY KEY (symbol, interval, key)
        );
        """
    )
    conn.commit()

def _get_meta(conn: sqlite3.Connection, symbol: str, interval: str, key: str) -> str | None:
    row = conn.execute(
        "SELECT value FROM meta WHERE symbol=? AND interval=? AND key=?",
        (symbol, interval, key),
    ).fetchone()
    return row[0] if row else None

def _set_meta(conn: sqlite3.Connection, symbol: str, interval: str, key: str, value) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO meta (symbol, interval, key, value) VALUES (?, ?, ?, ?)",
        (symbol, interval, key, str(value)),
    )
    conn.commit()

def db_range(conn: sqlite3.Connection, symbol: str, interval: str) -> tuple[int | None, int | None, int]:
    """Return (min_open_time, max_open_time, row_count) currently stored."""
    row = conn.execute(
        f"SELECT MIN(open_time), MAX(open_time), COUNT(*) FROM {TABLE} WHERE symbol=? AND interval=?",
        (symbol, interval),
    ).fetchone()
    return row[0], row[1], row[2]

def insert_rows(conn: sqlite3.Connection, rows: Iterable[tuple]) -> int:
    rows = list(rows)
    if not rows:
        return 0
    placeholders = ",".join("?" * len(COLUMNS))
    conn.executemany(
        f"INSERT OR REPLACE INTO {TABLE} ({','.join(COLUMNS)}) VALUES ({placeholders})",
        rows,
    )
    conn.commit()
    return len(rows)

# --------------------------------------------------------------------------------------
# Binance API
# --------------------------------------------------------------------------------------

class BinanceError(RuntimeError):
    pass

def _request_klines(session: requests.Session, params: dict) -> list[list]:
    """Single klines request with host fallback, retries and rate-limit handling."""
    last_err: Exception | None = None
    for attempt in range(MAX_RETRIES):
        for host in BINANCE_HOSTS:
            try:
                resp = session.get(host + KLINES_PATH, params=params, timeout=REQUEST_TIMEOUT)
            except requests.RequestException as e:
                last_err = e
                log.debug("request to %s failed: %s", host, e)
                continue

            if resp.status_code == 200:
                return resp.json()

            if resp.status_code in (418, 429):
                wait = int(resp.headers.get("Retry-After", 2 ** attempt))
                log.warning("rate limited by %s (%s); sleeping %ss", host, resp.status_code, wait)
                time.sleep(wait)
                last_err = BinanceError(f"{resp.status_code} from {host}")
                break  # retry same host list after sleeping

            if resp.status_code == 451:
                # Geo-restricted host; try the next one.
                last_err = BinanceError(f"451 (region blocked) from {host}")
                log.debug("%s", last_err)
                continue

            last_err = BinanceError(f"{resp.status_code} from {host}: {resp.text[:200]}")
            log.debug("%s", last_err)
        else:
            # Exhausted every host without a rate-limit break: back off before retrying.
            time.sleep(2 ** attempt)
    raise BinanceError(f"klines request failed after {MAX_RETRIES} attempts: {last_err}")

def _kline_to_row(k: list, symbol: str, interval: str) -> tuple:
    # Binance kline layout:
    # [open_time, open, high, low, close, volume, close_time, quote_volume,
    #  trades, taker_buy_base, taker_buy_quote, ignore]
    return (
        symbol,
        interval,
        int(k[0]),
        float(k[1]),
        float(k[2]),
        float(k[3]),
        float(k[4]),
        float(k[5]),
        int(k[6]),
        float(k[7]),
        int(k[8]),
        float(k[9]),
        float(k[10]),
    )

def fetch_klines(
    start_ms: int,
    end_ms: int,
    symbol: str = DEFAULT_SYMBOL,
    interval: str = DEFAULT_INTERVAL,
    session: requests.Session | None = None,
) -> list[tuple]:
    """Fetch all klines with ``start_ms <= open_time < end_ms``, paginating as needed."""
    if interval not in INTERVAL_MS:
        raise ValueError(f"unsupported interval {interval!r}")
    session = session or requests.Session()
    step = INTERVAL_MS[interval]
    rows: list[tuple] = []
    cursor = start_ms
    while cursor < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": cursor,
            "endTime": end_ms - 1,
            "limit": KLINES_LIMIT,
        }
        batch = _request_klines(session, params)
        if not batch:
            break
        rows.extend(_kline_to_row(k, symbol, interval) for k in batch)
        last_open = int(batch[-1][0])
        log.info(
            "fetched %5d bars  %s .. %s",
            len(batch),
            _fmt(int(batch[0][0])),
            _fmt(last_open),
        )
        cursor = last_open + step
        if len(batch) < KLINES_LIMIT:
            break
        time.sleep(POLITE_SLEEP)
    return rows

# --------------------------------------------------------------------------------------
# Ensure / load
# --------------------------------------------------------------------------------------

def _fmt(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")

def _now_ms() -> int:
    return int(time.time() * 1000)

def _floor_to_interval(ms: int, interval: str) -> int:
    step = INTERVAL_MS[interval]
    return ms - (ms % step)

def ensure_prices(
    years: float = DEFAULT_YEARS,
    symbol: str = DEFAULT_SYMBOL,
    interval: str = DEFAULT_INTERVAL,
    db_path: Path | str = DEFAULT_DB_PATH,
    conn: sqlite3.Connection | None = None,
) -> tuple[int, int, int]:
    """Make sure the DB holds bars from ``now - years`` up to the last closed bar.

    Only missing ranges are fetched from Binance. Returns ``(min_open_time,
    max_open_time, row_count)`` for the stored series after any fetching.
    """
    own_conn = conn is None
    conn = conn or get_connection(db_path)
    try:
        step = INTERVAL_MS[interval]
        now = _now_ms()
        # Last fully closed bar: exclude the bar currently forming.
        target_end = _floor_to_interval(now, interval)
        target_start = _floor_to_interval(
            int((datetime.now(timezone.utc) - timedelta(days=365.25 * years)).timestamp() * 1000),
            interval,
        )

        earliest_known = _get_meta(conn, symbol, interval, "earliest_available")
        if earliest_known is not None:
            target_start = max(target_start, int(earliest_known))

        db_min, db_max, n = db_range(conn, symbol, interval)
        session = requests.Session()

        if db_min is None:
            log.info("no %s %s data in %s; fetching %s .. %s",
                     symbol, interval, db_path, _fmt(target_start), _fmt(target_end))
            rows = fetch_klines(target_start, target_end, symbol, interval, session)
            insert_rows(conn, rows)
            if rows and rows[0][2] > target_start:
                # Exchange has no history before this point; remember it.
                _set_meta(conn, symbol, interval, "earliest_available", rows[0][2])
        else:
            # Backfill at the front.
            if db_min > target_start:
                log.info("backfilling %s .. %s", _fmt(target_start), _fmt(db_min))
                rows = fetch_klines(target_start, db_min, symbol, interval, session)
                inserted = insert_rows(conn, rows)
                first = rows[0][2] if rows else db_min
                if first > target_start or inserted == 0:
                    _set_meta(conn, symbol, interval, "earliest_available", first)
            # Extend at the back.
            if db_max + step < target_end:
                log.info("extending %s .. %s", _fmt(db_max + step), _fmt(target_end))
                rows = fetch_klines(db_max + step, target_end, symbol, interval, session)
                insert_rows(conn, rows)

        db_min, db_max, n = db_range(conn, symbol, interval)
        log.info("%s %s: %d bars, %s .. %s", symbol, interval, n,
                 _fmt(db_min) if db_min else "-", _fmt(db_max) if db_max else "-")
        return db_min, db_max, n
    finally:
        if own_conn:
            conn.close()

def load_prices(
    years: float = DEFAULT_YEARS,
    symbol: str = DEFAULT_SYMBOL,
    interval: str = DEFAULT_INTERVAL,
    db_path: Path | str = DEFAULT_DB_PATH,
    refresh: bool = True,
) -> pd.DataFrame:
    """Return OHLCV bars as a DataFrame indexed by UTC ``timestamp``.

    With ``refresh=True`` (default) the database is first brought up to date via
    `ensure_prices`; set it to ``False`` for a read-only load.
    """
    conn = get_connection(db_path)
    try:
        if refresh:
            ensure_prices(years, symbol, interval, db_path, conn=conn)
        start_ms = int((datetime.now(timezone.utc) - timedelta(days=365.25 * years)).timestamp() * 1000)
        df = pd.read_sql_query(
            f"""
            SELECT open_time, open, high, low, close, volume,
                   quote_volume, trades, taker_buy_base, taker_buy_quote
            FROM {TABLE}
            WHERE symbol=? AND interval=? AND open_time >= ?
            ORDER BY open_time
            """,
            conn,
            params=(symbol, interval, start_ms),
        )
    finally:
        conn.close()
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return df.set_index("timestamp").drop(columns="open_time")

# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--years", type=float, default=DEFAULT_YEARS, help="lookback window in years")
    p.add_argument("--symbol", default=DEFAULT_SYMBOL)
    p.add_argument("--interval", default=DEFAULT_INTERVAL, choices=sorted(INTERVAL_MS))
    p.add_argument("--db", default=str(DEFAULT_DB_PATH), help="path to SQLite database")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    ensure_prices(args.years, args.symbol, args.interval, args.db)

if __name__ == "__main__":
    main()
