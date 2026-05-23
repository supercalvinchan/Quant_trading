#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from usalpha.market_db import (
    duckdb_available,
    get_market_db_path,
    init_market_db,
    market_db_status,
    upsert_daily_bars,
    upsert_daily_valuation,
    upsert_symbol_master,
)

CACHE_ROOT = PROJECT_ROOT / ".cache"
CN_AKSHARE_DIR = CACHE_ROOT / "cn_akshare"
CN_VALUATION_DIR = CACHE_ROOT / "cn_valuation_total_value"
CN_SYMBOL_NAME_PATH = CACHE_ROOT / "cn_symbol_names.parquet"


def _normalize_symbol(symbol: str) -> str:
    raw = str(symbol or "").strip().upper()
    if raw.startswith(("SH", "SZ", "BJ")):
        raw = raw[2:]
    digits = "".join(ch for ch in raw if ch.isdigit())
    return digits.zfill(6)


def _load_symbol_master_frame() -> pd.DataFrame:
    if not CN_SYMBOL_NAME_PATH.exists():
        return pd.DataFrame(columns=["symbol", "name"])
    frame = pd.read_parquet(CN_SYMBOL_NAME_PATH)
    if frame.empty or "symbol" not in frame.columns:
        return pd.DataFrame(columns=["symbol", "name"])
    out = pd.DataFrame()
    out["symbol"] = frame["symbol"].astype(str).map(_normalize_symbol)
    out["name"] = frame["name"].astype(str) if "name" in frame.columns else ""
    out = out[out["symbol"].str.match(r"^\d{6}$", na=False)].drop_duplicates("symbol")
    return out.sort_values("symbol").reset_index(drop=True)


def _load_daily_bars_from_cache(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if frame.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    frame.index = pd.to_datetime(frame.index).tz_localize(None)
    keep = [col for col in ["open", "high", "low", "close", "volume"] if col in frame.columns]
    if len(keep) < 5:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    out = frame[["open", "high", "low", "close", "volume"]].copy()
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    out = out.dropna(subset=["open", "high", "low", "close"])
    return out


def _load_daily_valuation_from_cache(path: Path) -> pd.Series:
    frame = pd.read_parquet(path)
    if frame.empty or "total_value_yi" not in frame.columns:
        return pd.Series(dtype=float)
    frame.index = pd.to_datetime(frame.index).tz_localize(None)
    series = pd.to_numeric(frame["total_value_yi"], errors="coerce").dropna()
    series.index = pd.to_datetime(series.index).tz_localize(None)
    return series.sort_index()


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild compact CN market main DB from existing cache files")
    parser.add_argument("--limit-bars", type=int, default=0, help="仅导入前 N 只股票日线；0 表示全部")
    parser.add_argument("--limit-valuations", type=int, default=0, help="仅导入前 N 只估值；0 表示全部")
    args = parser.parse_args()

    if not duckdb_available():
        print("主库后端不可用")
        return 2

    init_market_db()
    master = _load_symbol_master_frame()
    master_count = upsert_symbol_master(master)

    bar_files = [
        path for path in sorted(CN_AKSHARE_DIR.glob("*.parquet"))
        if path.is_file() and not path.name.startswith("index_") and ".bad." not in path.name
    ]
    val_files = [path for path in sorted(CN_VALUATION_DIR.glob("*.parquet")) if path.is_file()]
    if args.limit_bars > 0:
        bar_files = bar_files[: args.limit_bars]
    if args.limit_valuations > 0:
        val_files = val_files[: args.limit_valuations]

    bar_symbols = 0
    bar_rows = 0
    bad_bar_files: list[str] = []
    for i, path in enumerate(bar_files, start=1):
        symbol = _normalize_symbol(path.stem)
        try:
            frame = _load_daily_bars_from_cache(path)
            rows = upsert_daily_bars(symbol, frame)
            if rows > 0:
                bar_symbols += 1
                bar_rows += rows
        except Exception as exc:  # pylint: disable=broad-except
            bad_bar_files.append(f"{path.name}: {exc}")
        if i % 200 == 0 or i == len(bar_files):
            print(f"[bars] {i}/{len(bar_files)} imported symbols={bar_symbols} rows={bar_rows}")

    valuation_symbols = 0
    valuation_rows = 0
    bad_val_files: list[str] = []
    for i, path in enumerate(val_files, start=1):
        symbol = _normalize_symbol(path.stem)
        try:
            series = _load_daily_valuation_from_cache(path)
            rows = upsert_daily_valuation(symbol, series)
            if rows > 0:
                valuation_symbols += 1
                valuation_rows += rows
        except Exception as exc:  # pylint: disable=broad-except
            bad_val_files.append(f"{path.name}: {exc}")
        if i % 200 == 0 or i == len(val_files):
            print(f"[valuation] {i}/{len(val_files)} imported symbols={valuation_symbols} rows={valuation_rows}")

    status = market_db_status()
    print("db_path:", get_market_db_path())
    print("symbol_master_upserted:", master_count)
    print("bar_symbols_imported:", bar_symbols)
    print("bar_rows_imported:", bar_rows)
    print("valuation_symbols_imported:", valuation_symbols)
    print("valuation_rows_imported:", valuation_rows)
    print("db_status:", status)
    if bad_bar_files:
        print("bad_bar_files_sample:", bad_bar_files[:20])
    if bad_val_files:
        print("bad_valuation_files_sample:", bad_val_files[:20])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
