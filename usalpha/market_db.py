from __future__ import annotations

from pathlib import Path
import sqlite3
from typing import Iterable

import pandas as pd


CN_BAR_COLUMNS = ["open", "high", "low", "close", "volume"]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def get_market_db_path() -> Path:
    data_dir = _project_root() / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / "cn_market.sqlite3"


def duckdb_available() -> bool:
    return True


def _normalize_symbol(symbol: str) -> str:
    raw = str(symbol or "").strip().upper()
    if raw.startswith(("SH", "SZ", "BJ")):
        raw = raw[2:]
    digits = "".join(ch for ch in raw if ch.isdigit())
    return digits.zfill(6)


def _board_from_symbol(symbol: str) -> tuple[str, str]:
    s = _normalize_symbol(symbol)
    if s.startswith("6"):
        return "SH", "main"
    if s.startswith(("0", "3")):
        return "SZ", "main" if s.startswith("0") else "chinext"
    if s.startswith(("4", "8", "9")):
        return "BJ", "bse"
    return "", ""


def connect_market_db(read_only: bool = False) -> sqlite3.Connection:
    path = get_market_db_path()
    if read_only:
        uri = f"file:{path}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, detect_types=sqlite3.PARSE_DECLTYPES)
    else:
        conn = sqlite3.connect(str(path), detect_types=sqlite3.PARSE_DECLTYPES)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA mmap_size=268435456")
    return conn


def init_market_db() -> None:
    with connect_market_db(read_only=False) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cn_symbol_master (
                symbol TEXT PRIMARY KEY,
                name TEXT,
                exchange TEXT,
                board TEXT,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cn_daily_bars (
                symbol TEXT NOT NULL,
                trade_date TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                PRIMARY KEY(symbol, trade_date)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cn_daily_valuation (
                symbol TEXT NOT NULL,
                trade_date TEXT NOT NULL,
                total_value_yi REAL,
                PRIMARY KEY(symbol, trade_date)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cn_sync_meta (
                meta_key TEXT PRIMARY KEY,
                meta_value TEXT,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_cn_daily_bars_date ON cn_daily_bars(trade_date)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_cn_daily_valuation_date ON cn_daily_valuation(trade_date)")
        conn.commit()


def upsert_symbol_master(frame: pd.DataFrame) -> int:
    if frame is None or frame.empty:
        return 0
    init_market_db()
    out = frame.copy()
    out["symbol"] = out["symbol"].map(_normalize_symbol)
    if "name" in out.columns:
        out["name"] = out["name"].astype(str)
    else:
        out["name"] = ""
    out = out[out["symbol"].str.match(r"^\d{6}$", na=False)].drop_duplicates("symbol")
    if out.empty:
        return 0
    rows: list[tuple[str, str, str, str]] = []
    for _, row in out.iterrows():
        symbol = str(row["symbol"])
        exchange, board = _board_from_symbol(symbol)
        rows.append((symbol, str(row.get("name", "")), exchange, board))
    with connect_market_db(read_only=False) as conn:
        conn.executemany(
            """
            INSERT INTO cn_symbol_master(symbol, name, exchange, board)
            VALUES(?, ?, ?, ?)
            ON CONFLICT(symbol) DO UPDATE SET
                name = excluded.name,
                exchange = excluded.exchange,
                board = excluded.board,
                updated_at = CURRENT_TIMESTAMP
            """,
            rows,
        )
        conn.commit()
    return len(rows)


def upsert_daily_bars(symbol: str, frame: pd.DataFrame) -> int:
    if frame is None or frame.empty:
        return 0
    normalized = _normalize_symbol(symbol)
    out = frame.copy()
    out.index = pd.to_datetime(out.index).tz_localize(None)
    missing = [col for col in CN_BAR_COLUMNS if col not in out.columns]
    if missing:
        raise ValueError(f"daily bars missing columns for {normalized}: {missing}")
    out = out[CN_BAR_COLUMNS].copy().sort_index()
    out = out[~out.index.duplicated(keep="last")]
    out = out.dropna(subset=["open", "high", "low", "close"])
    for col in CN_BAR_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float32")
    out["trade_date"] = pd.to_datetime(out.index).normalize().strftime("%Y-%m-%d")
    rows = [
        (
            normalized,
            str(trade_date),
            None if pd.isna(open_) else float(open_),
            None if pd.isna(high) else float(high),
            None if pd.isna(low) else float(low),
            None if pd.isna(close) else float(close),
            None if pd.isna(volume) else float(volume),
        )
        for trade_date, open_, high, low, close, volume in zip(
            out["trade_date"], out["open"], out["high"], out["low"], out["close"], out["volume"]
        )
    ]
    if not rows:
        return 0
    init_market_db()
    with connect_market_db(read_only=False) as conn:
        conn.executemany(
            """
            INSERT INTO cn_daily_bars(symbol, trade_date, open, high, low, close, volume)
            VALUES(?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol, trade_date) DO UPDATE SET
                open = excluded.open,
                high = excluded.high,
                low = excluded.low,
                close = excluded.close,
                volume = excluded.volume
            """,
            rows,
        )
        conn.commit()
    return len(rows)


def upsert_daily_valuation(symbol: str, series: pd.Series) -> int:
    if series is None or len(series) == 0:
        return 0
    normalized = _normalize_symbol(symbol)
    out = pd.DataFrame({"total_value_yi": pd.to_numeric(series, errors="coerce")})
    out.index = pd.to_datetime(out.index).tz_localize(None)
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    out = out.dropna(subset=["total_value_yi"])
    rows = [
        (
            normalized,
            pd.Timestamp(idx).normalize().strftime("%Y-%m-%d"),
            float(value),
        )
        for idx, value in out["total_value_yi"].items()
    ]
    if not rows:
        return 0
    init_market_db()
    with connect_market_db(read_only=False) as conn:
        conn.executemany(
            """
            INSERT INTO cn_daily_valuation(symbol, trade_date, total_value_yi)
            VALUES(?, ?, ?)
            ON CONFLICT(symbol, trade_date) DO UPDATE SET
                total_value_yi = excluded.total_value_yi
            """,
            rows,
        )
        conn.commit()
    return len(rows)


def load_daily_bars(symbol: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    path = get_market_db_path()
    if not path.exists():
        return pd.DataFrame(columns=CN_BAR_COLUMNS)
    normalized = _normalize_symbol(symbol)
    clauses = ["symbol = ?"]
    params: list[object] = [normalized]
    if start:
        clauses.append("trade_date >= ?")
        params.append(pd.Timestamp(start).normalize().strftime("%Y-%m-%d"))
    if end:
        clauses.append("trade_date <= ?")
        params.append(pd.Timestamp(end).normalize().strftime("%Y-%m-%d"))
    sql = (
        "SELECT trade_date, open, high, low, close, volume "
        "FROM cn_daily_bars WHERE " + " AND ".join(clauses) + " ORDER BY trade_date"
    )
    with connect_market_db(read_only=True) as conn:
        out = pd.read_sql_query(sql, conn, params=params)
    if out.empty:
        return pd.DataFrame(columns=CN_BAR_COLUMNS)
    out["trade_date"] = pd.to_datetime(out["trade_date"]).dt.tz_localize(None)
    out = out.set_index("trade_date").sort_index()
    out.index.name = None
    for col in CN_BAR_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float32")
    return out[CN_BAR_COLUMNS]


def load_daily_valuation(symbol: str) -> pd.Series:
    path = get_market_db_path()
    if not path.exists():
        return pd.Series(dtype=float)
    normalized = _normalize_symbol(symbol)
    with connect_market_db(read_only=True) as conn:
        out = pd.read_sql_query(
            """
            SELECT trade_date, total_value_yi
            FROM cn_daily_valuation
            WHERE symbol = ?
            ORDER BY trade_date
            """,
            conn,
            params=[normalized],
        )
    if out.empty:
        return pd.Series(dtype=float)
    out["trade_date"] = pd.to_datetime(out["trade_date"]).dt.tz_localize(None)
    series = pd.Series(pd.to_numeric(out["total_value_yi"], errors="coerce").to_numpy(), index=out["trade_date"])
    series = series.dropna().sort_index()
    series.index.name = None
    return series


def load_symbol_master() -> pd.DataFrame:
    path = get_market_db_path()
    if not path.exists():
        return pd.DataFrame(columns=["symbol", "name", "exchange", "board"])
    with connect_market_db(read_only=True) as conn:
        out = pd.read_sql_query(
            "SELECT symbol, name, exchange, board FROM cn_symbol_master ORDER BY symbol",
            conn,
        )
    if out.empty:
        return pd.DataFrame(columns=["symbol", "name", "exchange", "board"])
    out["symbol"] = out["symbol"].astype(str).map(_normalize_symbol)
    out["name"] = out["name"].fillna("").astype(str)
    return out


def market_db_status() -> dict[str, object]:
    path = get_market_db_path()
    base = {
        "db_path": str(path),
        "exists": path.exists(),
        "size_bytes": int(path.stat().st_size) if path.exists() else 0,
        "symbols": 0,
        "bars": 0,
        "valuation_rows": 0,
        "valuation_symbols": 0,
    }
    if not path.exists():
        return base
    with connect_market_db(read_only=True) as conn:
        symbol_count = int(conn.execute("SELECT COUNT(*) FROM cn_symbol_master").fetchone()[0])
        bar_count = int(conn.execute("SELECT COUNT(*) FROM cn_daily_bars").fetchone()[0])
        valuation_count = int(conn.execute("SELECT COUNT(*) FROM cn_daily_valuation").fetchone()[0])
        valuation_symbol_count = int(conn.execute("SELECT COUNT(DISTINCT symbol) FROM cn_daily_valuation").fetchone()[0])
    base["symbols"] = symbol_count
    base["bars"] = bar_count
    base["valuation_rows"] = valuation_count
    base["valuation_symbols"] = valuation_symbol_count
    return base


def load_symbols_missing_bars(symbols: Iterable[str], *, start: str, end: str) -> list[str]:
    normalized = [_normalize_symbol(symbol) for symbol in symbols if str(symbol).strip()]
    normalized = list(dict.fromkeys([symbol for symbol in normalized if len(symbol) == 6]))
    path = get_market_db_path()
    if not normalized or not path.exists():
        return normalized
    placeholders = ", ".join(["?"] * len(normalized))
    sql = (
        "SELECT symbol, MIN(trade_date) AS min_date, MAX(trade_date) AS max_date, COUNT(*) AS cnt "
        f"FROM cn_daily_bars WHERE symbol IN ({placeholders}) GROUP BY symbol"
    )
    with connect_market_db(read_only=True) as conn:
        rows = pd.read_sql_query(sql, conn, params=normalized)
    if rows.empty:
        return normalized
    end_date = pd.Timestamp(end).normalize().strftime("%Y-%m-%d")
    good: set[str] = set()
    for _, row in rows.iterrows():
        symbol = str(row["symbol"])
        max_date = str(row["max_date"]) if pd.notna(row["max_date"]) else ""
        count = int(row["cnt"]) if pd.notna(row["cnt"]) else 0
        # For incremental maintenance and backtests, a symbol is usable as long as:
        # 1) it has some bars in the DB, and
        # 2) its latest bar reaches the requested end date.
        #
        # Do not require min_date <= start_date here. Later-listed stocks naturally
        # start after the backtest start, and treating them as "missing" forces
        # unnecessary local rebuild / network refresh for most of the universe.
        if count > 0 and max_date >= end_date:
            good.add(symbol)
    return [symbol for symbol in normalized if symbol not in good]


def load_symbol_latest_bar_dates(symbols: Iterable[str]) -> dict[str, str]:
    normalized = [_normalize_symbol(symbol) for symbol in symbols if str(symbol).strip()]
    normalized = list(dict.fromkeys([symbol for symbol in normalized if len(symbol) == 6]))
    path = get_market_db_path()
    if not normalized or not path.exists():
        return {}
    placeholders = ", ".join(["?"] * len(normalized))
    sql = (
        "SELECT symbol, MAX(trade_date) AS max_date "
        f"FROM cn_daily_bars WHERE symbol IN ({placeholders}) GROUP BY symbol"
    )
    with connect_market_db(read_only=True) as conn:
        rows = pd.read_sql_query(sql, conn, params=normalized)
    if rows.empty:
        return {}
    return {
        str(row["symbol"]): (str(row["max_date"]) if pd.notna(row["max_date"]) else "")
        for _, row in rows.iterrows()
    }


def load_symbol_latest_valuation_dates(symbols: Iterable[str]) -> dict[str, str]:
    normalized = [_normalize_symbol(symbol) for symbol in symbols if str(symbol).strip()]
    normalized = list(dict.fromkeys([symbol for symbol in normalized if len(symbol) == 6]))
    path = get_market_db_path()
    if not normalized or not path.exists():
        return {}
    placeholders = ", ".join(["?"] * len(normalized))
    sql = (
        "SELECT symbol, MAX(trade_date) AS max_date "
        f"FROM cn_daily_valuation WHERE symbol IN ({placeholders}) GROUP BY symbol"
    )
    with connect_market_db(read_only=True) as conn:
        rows = pd.read_sql_query(sql, conn, params=normalized)
    if rows.empty:
        return {}
    return {
        str(row["symbol"]): (str(row["max_date"]) if pd.notna(row["max_date"]) else "")
        for _, row in rows.iterrows()
    }
