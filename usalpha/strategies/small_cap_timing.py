from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import akshare as ak
except Exception:  # pylint: disable=broad-except
    ak = None

from ..market_db import load_daily_valuation, load_symbol_master, upsert_daily_valuation, upsert_symbol_master
from .base import BaseStrategy


_CACHE_TTL_HOURS = 24 * 7


def _normalize_cn_symbol(symbol: str) -> str:
    raw = str(symbol or "").strip().upper()
    if raw.startswith(("SH", "SZ", "BJ")):
        raw = raw[2:]
    return "".join(ch for ch in raw if ch.isdigit()).zfill(6)


def _cache_dir() -> Path:
    root = Path(__file__).resolve().parents[2]
    path = root / ".cache" / "cn_valuation_total_value"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _cache_path(symbol: str) -> Path:
    return _cache_dir() / f"{symbol}.parquet"


def _name_cache_path() -> Path:
    return _cache_dir().parent / "cn_symbol_names.parquet"


def _is_cache_fresh(path: Path, ttl_hours: int) -> bool:
    if not path.exists():
        return False
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return datetime.now(timezone.utc) - mtime <= timedelta(hours=ttl_hours)


def _read_cached_total_value(symbol: str) -> pd.Series:
    path = _cache_path(symbol)
    if not path.exists():
        return pd.Series(dtype=float)
    frame = pd.read_parquet(path)
    if frame.empty or "total_value_yi" not in frame.columns:
        return pd.Series(dtype=float)
    frame.index = pd.to_datetime(frame.index).tz_localize(None)
    frame = frame.sort_index()
    return pd.to_numeric(frame["total_value_yi"], errors="coerce").dropna()


def _write_cached_total_value(symbol: str, series: pd.Series) -> None:
    path = _cache_path(symbol)
    frame = pd.DataFrame({"total_value_yi": pd.to_numeric(series, errors="coerce")})
    frame = frame.dropna().sort_index()
    frame.to_parquet(path, compression="zstd", index=True)


def _fetch_total_value_history(symbol: str) -> pd.Series:
    if ak is None:
        raise RuntimeError("small_cap_timing requires akshare for CN total-value history")
    errors: list[str] = []
    for period in ["全部", "近十年", "近五年", "近三年", "近一年"]:
        try:
            raw = ak.stock_zh_valuation_baidu(symbol=symbol, indicator="总市值", period=period)
            if raw is None or raw.empty:
                continue
            frame = raw.copy()
            frame.columns = [str(col).strip().lower() for col in frame.columns]
            if "date" not in frame.columns or "value" not in frame.columns:
                continue
            frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None)
            frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
            frame = frame.dropna(subset=["date", "value"]).drop_duplicates(subset=["date"], keep="last")
            if frame.empty:
                continue
            series = frame.set_index("date")["value"].sort_index()
            if len(series) > 0:
                return series
        except Exception as exc:  # pylint: disable=broad-except
            errors.append(f"{period}: {exc}")
    raise RuntimeError(f"failed to fetch total_value history for {symbol}: {'; '.join(errors)}")


def _read_cached_symbol_names() -> pd.DataFrame:
    path = _name_cache_path()
    if not path.exists():
        return pd.DataFrame(columns=["symbol", "name"])
    frame = pd.read_parquet(path)
    if frame.empty:
        return pd.DataFrame(columns=["symbol", "name"])
    frame["symbol"] = frame["symbol"].astype(str).map(_normalize_cn_symbol)
    frame["name"] = frame["name"].astype(str)
    return frame.drop_duplicates("symbol")


def _write_cached_symbol_names(frame: pd.DataFrame) -> None:
    path = _name_cache_path()
    out = frame.copy()
    out["symbol"] = out["symbol"].astype(str).map(_normalize_cn_symbol)
    out["name"] = out["name"].astype(str)
    out = out[["symbol", "name"]].drop_duplicates("symbol").sort_values("symbol")
    out.to_parquet(path, compression="zstd", index=False)


def get_cn_symbol_name_map_cached() -> dict[str, str]:
    db_master = load_symbol_master()
    if not db_master.empty:
        return dict(zip(db_master["symbol"], db_master["name"]))
    path = _name_cache_path()
    if _is_cache_fresh(path, _CACHE_TTL_HOURS):
        cached = _read_cached_symbol_names()
        if not cached.empty:
            return dict(zip(cached["symbol"], cached["name"]))
    cached = _read_cached_symbol_names()
    if ak is None:
        return dict(zip(cached["symbol"], cached["name"])) if not cached.empty else {}
    try:
        raw = ak.stock_info_a_code_name()
        if raw is None or raw.empty:
            raise ValueError("empty stock_info_a_code_name")
        code_col = "code" if "code" in raw.columns else ("代码" if "代码" in raw.columns else None)
        name_col = "name" if "name" in raw.columns else ("名称" if "名称" in raw.columns else None)
        if code_col is None or name_col is None:
            raise ValueError("stock_info_a_code_name missing code/name columns")
        frame = pd.DataFrame(
            {
                "symbol": raw[code_col].astype(str).map(_normalize_cn_symbol),
                "name": raw[name_col].astype(str),
            }
        )
        frame = frame[frame["symbol"].str.match(r"^\d{6}$", na=False)]
        _write_cached_symbol_names(frame)
        upsert_symbol_master(frame[["symbol", "name"]])
        return dict(zip(frame["symbol"], frame["name"]))
    except Exception:
        return dict(zip(cached["symbol"], cached["name"])) if not cached.empty else {}


def get_total_value_history_cached(symbol: str) -> pd.Series:
    normalized = _normalize_cn_symbol(symbol)
    db_series = load_daily_valuation(normalized)
    if len(db_series) > 0:
        return db_series
    path = _cache_path(normalized)
    if _is_cache_fresh(path, _CACHE_TTL_HOURS):
        cached = _read_cached_total_value(normalized)
        if len(cached) > 0:
            return cached
    cached = _read_cached_total_value(normalized)
    try:
        series = _fetch_total_value_history(normalized)
        _write_cached_total_value(normalized, series)
        upsert_daily_valuation(normalized, series)
        return series
    except Exception:
        if len(cached) > 0:
            return cached
        raise


def refresh_total_value_history(symbol: str) -> pd.Series:
    normalized = _normalize_cn_symbol(symbol)
    series = _fetch_total_value_history(normalized)
    _write_cached_total_value(normalized, series)
    upsert_daily_valuation(normalized, series)
    return series


def align_daily_series_to_frame_index(series: pd.Series, frame_index: pd.Index) -> pd.Series:
    normalized_index = pd.DatetimeIndex(pd.to_datetime(frame_index)).normalize()
    aligned = series.reindex(normalized_index).ffill()
    aligned.index = pd.DatetimeIndex(pd.to_datetime(frame_index))
    return aligned


class SmallCapTimingStrategy(BaseStrategy):
    type_name = "small_cap_timing"

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "min_total_value_yi": {"type": "number", "minimum": 0.0, "maximum": 1000000.0},
                "max_total_value_yi": {"type": "number", "minimum": 0.0, "maximum": 1000000.0},
                "min_avg_amount": {"type": "number", "minimum": 0.0},
                "amount_window": {"type": "integer", "minimum": 1, "maximum": 250},
                "exclude_st": {"type": "boolean"},
                "exclude_delisting": {"type": "boolean"},
            },
            "required": [],
            "additionalProperties": False,
            "x_constraints": [
                {
                    "type": "compare",
                    "left": "min_total_value_yi",
                    "op": "<=",
                    "right": "max_total_value_yi",
                    "message": "strategy.parameters.min_total_value_yi must be <= strategy.parameters.max_total_value_yi",
                }
            ],
        }

    def generate_score_matrix(
        self,
        data_by_symbol: dict[str, pd.DataFrame],
        parameters: dict[str, Any],
    ) -> pd.DataFrame:
        resolved = self.resolve_params(parameters)
        min_total_value_yi = float(resolved.get("min_total_value_yi", 3.0))
        max_total_value_yi = float(resolved.get("max_total_value_yi", 1000.0))
        min_avg_amount = float(resolved.get("min_avg_amount", 0.0))
        amount_window = int(resolved.get("amount_window", 20))
        exclude_st = bool(resolved.get("exclude_st", True))
        exclude_delisting = bool(resolved.get("exclude_delisting", True))
        symbol_name_map = get_cn_symbol_name_map_cached() if (exclude_st or exclude_delisting) else {}

        per_symbol: dict[str, pd.Series] = {}
        failures: list[str] = []
        for symbol, frame in data_by_symbol.items():
            normalized = _normalize_cn_symbol(symbol)
            if len(normalized) != 6:
                failures.append(str(symbol))
                continue
            name = str(symbol_name_map.get(normalized, "")).upper()
            if exclude_st and "ST" in name:
                continue
            if exclude_delisting and "退" in name:
                continue
            total_value = get_total_value_history_cached(normalized)
            aligned = align_daily_series_to_frame_index(total_value, frame.index)
            if len(aligned) == 0:
                continue
            score = -pd.to_numeric(aligned, errors="coerce")
            cap_mask = (aligned >= min_total_value_yi) & (aligned <= max_total_value_yi)
            score = score.where(cap_mask)
            if min_avg_amount > 0 and "amount" in frame.columns:
                avg_amount = pd.to_numeric(frame["amount"], errors="coerce").rolling(amount_window, min_periods=1).mean()
                score = score.where(avg_amount >= min_avg_amount)
            per_symbol[str(symbol)] = score.astype(float)

        if not per_symbol:
            if failures and len(failures) == len(data_by_symbol):
                raise ValueError("small_cap_timing currently supports CN A-share symbols only")
            return pd.DataFrame()
        return pd.DataFrame(per_symbol).sort_index()
