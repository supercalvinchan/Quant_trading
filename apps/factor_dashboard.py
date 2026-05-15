from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import socket
from urllib.parse import parse_qs, urlparse
import signal
import threading
import time
from typing import Any
import uuid

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from usalpha.config import USAlphaConfig
from usalpha.dashboard_service import EvolutionUIConfig, run_dashboard_workflow, run_dashboard_workflow_with_bundle
from usalpha.data import MarketDataBundle

from pandas.tseries.offsets import BDay


def _today_date():
    """Today's date (date object), for default values."""
    return pd.Timestamp.now().normalize().date()


def _next_bday_date():
    """Next business day (date object), for default predict date."""
    return (pd.Timestamp.now() + BDay(1)).date()


def _effective_cn_daily_end(end: str | pd.Timestamp) -> pd.Timestamp:
    """Use the latest completed CN trading day for daily bars.

    If the requested end date is today or in the future, daily bar data is usually
    only complete through the previous business day, so avoid forcing a full-market
    refresh for a date that does not exist yet.
    """
    requested = pd.Timestamp(end).normalize()
    today = pd.Timestamp.now().normalize()
    if requested >= today:
        return pd.Timestamp(today - BDay(1)).normalize()
    return requested

try:
    import plotly.express as px
except Exception:  # pylint: disable=broad-except
    px = None

try:
    import plotly.graph_objects as go
except Exception:  # pylint: disable=broad-except
    go = None

try:
    from plotly.subplots import make_subplots
except Exception:  # pylint: disable=broad-except
    make_subplots = None

try:
    import yfinance as yf
except Exception:  # pylint: disable=broad-except
    yf = None

try:
    import akshare as ak
except Exception:  # pylint: disable=broad-except
    ak = None

try:
    from mootdx.quotes import Quotes
except Exception:  # pylint: disable=broad-except
    Quotes = None

try:
    import requests
except Exception:  # pylint: disable=broad-except
    requests = None

try:
    from py_mini_racer import py_mini_racer as _pmr
except Exception:  # pylint: disable=broad-except
    _pmr = None


def _patch_py_mini_racer_destructor() -> None:
    if _pmr is None or getattr(_pmr.MiniRacer, "_usalpha_safe_del", False):
        return

    def _safe_del(self) -> None:
        ext = getattr(self, "ext", None)
        ctx = getattr(self, "ctx", None)
        free_fn = getattr(ext, "mr_free_context", None) if ext is not None else None
        if free_fn is None:
            return
        try:
            free_fn(ctx)
        except Exception:  # pylint: disable=broad-except
            return

    _pmr.MiniRacer.__del__ = _safe_del
    _pmr.MiniRacer._usalpha_safe_del = True


_patch_py_mini_racer_destructor()


st.set_page_config(page_title="USalpha 因子训练可视化", layout="wide")


CN_CACHE_COLUMNS = ["open", "high", "low", "close", "volume"]
CN_DERIVED_COLUMNS = ["amount", "vwap", "ret"]
CN_CACHE_ALL_COLUMNS = CN_CACHE_COLUMNS + CN_DERIVED_COLUMNS
CN_AKSHARE_MAX_RETRIES = 3
CN_AKSHARE_RETRY_SLEEP_SECONDS = 0.8
CN_AKSHARE_NETWORK_TIMEOUT_SECONDS = 20
CN_INCREMENTAL_BACKFILL_DAYS = 30
_REQUESTS_TIMEOUT_PATCHED = False
_MOOTDX_CLIENT = None


def _browser_auto_shutdown_enabled() -> bool:
    value = str(os.getenv("USALPHA_ENABLE_BROWSER_AUTO_SHUTDOWN", "")).strip().lower()
    return value in {"1", "true", "yes", "on"}


def _browser_idle_timeout_sec() -> float:
    raw = str(os.getenv("USALPHA_BROWSER_IDLE_TIMEOUT_SEC", "1800")).strip()
    try:
        return max(float(raw), 10.0)
    except Exception:  # pylint: disable=broad-except
        return 1800.0


class _BrowserShutdownMonitor:
    """Stop the Streamlit process after all browser tabs disappear."""

    def __init__(self, *, preferred_port: int = 8765, idle_timeout_sec: float = 1800.0):
        self.idle_timeout_sec = float(idle_timeout_sec)
        self.active_tabs: dict[str, float] = {}
        self.ever_connected = False
        self.last_seen = time.time()
        self.shutdown_requested = False
        self.lock = threading.Lock()
        self.port = self._start_server(preferred_port)
        self._start_idle_watcher()

    def _start_server(self, preferred_port: int) -> int:
        monitor = self

        class Handler(BaseHTTPRequestHandler):
            def _send_ok(self) -> None:
                self.send_response(204)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                self.end_headers()

            def do_OPTIONS(self) -> None:  # noqa: N802
                self._send_ok()

            def do_GET(self) -> None:  # noqa: N802
                monitor.handle_request(self.path)
                self._send_ok()

            def do_POST(self) -> None:  # noqa: N802
                monitor.handle_request(self.path)
                self._send_ok()

            def log_message(self, *_args: Any) -> None:
                return

        last_exc: Exception | None = None
        for port in range(int(preferred_port), int(preferred_port) + 50):
            try:
                server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
                server.daemon_threads = True
                thread = threading.Thread(target=server.serve_forever, name="usalpha-browser-shutdown", daemon=True)
                thread.start()
                return port
            except OSError as exc:
                last_exc = exc
                continue
        raise RuntimeError(f"无法启动浏览器关闭监听端口: {last_exc}")

    def handle_request(self, path: str) -> None:
        parsed = urlparse(path)
        sid = parse_qs(parsed.query).get("sid", [""])[0] or str(uuid.uuid4())
        now = time.time()
        with self.lock:
            self.ever_connected = True
            self.last_seen = now
            if parsed.path.endswith("/close"):
                self.active_tabs.pop(sid, None)
            else:
                self.active_tabs[sid] = now

    def _start_idle_watcher(self) -> None:
        def watch() -> None:
            while True:
                time.sleep(1.0)
                now = time.time()
                should_stop = False
                with self.lock:
                    stale = [
                        sid
                        for sid, seen_at in self.active_tabs.items()
                        if now - seen_at > self.idle_timeout_sec
                    ]
                    for sid in stale:
                        self.active_tabs.pop(sid, None)
                    should_stop = (
                        self.ever_connected
                        and not self.active_tabs
                        and now - self.last_seen > self.idle_timeout_sec
                        and not self.shutdown_requested
                    )
                    if should_stop:
                        self.shutdown_requested = True
                if should_stop:
                    os.kill(os.getpid(), signal.SIGTERM)

        threading.Thread(target=watch, name="usalpha-idle-port-release", daemon=True).start()


@st.cache_resource
def _get_browser_shutdown_monitor() -> _BrowserShutdownMonitor:
    return _BrowserShutdownMonitor(idle_timeout_sec=_browser_idle_timeout_sec())


def _install_browser_shutdown_hook() -> None:
    if not _browser_auto_shutdown_enabled():
        return
    monitor = _get_browser_shutdown_monitor()
    components.html(
        f"""
        <script>
        (function() {{
          const endpoint = "http://127.0.0.1:{monitor.port}";
          const sid = (window.crypto && crypto.randomUUID)
            ? crypto.randomUUID()
            : String(Date.now()) + "_" + String(Math.random());

          function ping() {{
            fetch(endpoint + "/ping?sid=" + encodeURIComponent(sid), {{
              method: "GET",
              mode: "cors",
              keepalive: true
            }}).catch(function() {{}});
          }}

          function closeTab() {{
            const url = endpoint + "/close?sid=" + encodeURIComponent(sid);
            if (navigator.sendBeacon) {{
              navigator.sendBeacon(url, "");
            }} else {{
              fetch(url, {{method: "POST", mode: "cors", keepalive: true}}).catch(function() {{}});
            }}
          }}

          ping();
          window.setInterval(ping, 3000);
          window.addEventListener("pagehide", closeTab);
          window.addEventListener("beforeunload", closeTab);
        }})();
        </script>
        """,
        height=0,
        width=0,
    )


def _metric(v: Any, digits: int = 4) -> str:
    try:
        f = float(v)
    except Exception:  # pylint: disable=broad-except
        return str(v)
    if pd.isna(f) or np.isinf(f):
        return "NaN"
    return f"{f:.{digits}f}"


def _normalize_stock_history(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    out = raw.copy()
    fields = {"open", "high", "low", "close", "volume", "adj close"}
    if isinstance(out.columns, pd.MultiIndex):
        field_level = None
        for level in range(out.columns.nlevels):
            vals = {str(x).strip().lower() for x in out.columns.get_level_values(level)}
            if len(vals & fields) >= 4:
                field_level = level
                break
        if field_level is not None:
            out.columns = [str(x).strip().lower() for x in out.columns.get_level_values(field_level)]
        else:
            ticker_upper = ticker.upper()
            for level in range(out.columns.nlevels):
                vals = {str(x).strip().upper() for x in out.columns.get_level_values(level)}
                if ticker_upper in vals:
                    out = out.xs(ticker_upper, axis=1, level=level)
                    break
            out.columns = [str(c).strip().lower() for c in out.columns]
    else:
        out.columns = [str(c).strip().lower() for c in out.columns]

    rename = {
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
    }
    out = out.rename(columns=rename)
    required = ["open", "high", "low", "close", "volume"]
    missing = [col for col in required if col not in out.columns]
    if missing:
        raise ValueError(f"缺少行情字段: {missing}")

    out = out[required].copy()
    out.index = pd.to_datetime(out.index).tz_localize(None)
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    out = out.dropna(subset=["open", "high", "low", "close"])
    out["volume"] = out["volume"].fillna(0.0)
    return out


def _normalize_akshare_history(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    out = raw.copy()
    rename_map = {
        "日期": "date",
        "开盘": "open",
        "最高": "high",
        "最低": "low",
        "收盘": "close",
        "成交量": "volume",
        "成交额": "amount",
        "date": "date",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
    }
    out = out.rename(columns={col: rename_map.get(str(col).strip(), str(col).strip().lower()) for col in out.columns})
    required = ["date", "open", "high", "low", "close", "volume"]
    missing = [col for col in required if col not in out.columns]
    if missing:
        raise ValueError(f"AKShare 返回数据缺少字段: {missing}")

    out = out[required].copy()
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
    for col in ["open", "high", "low", "close", "volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["date", "open", "high", "low", "close"])
    out["volume"] = out["volume"].fillna(0.0)
    out = out.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    out = out.set_index("date")
    out.index.name = None
    return out[["open", "high", "low", "close", "volume"]]


def _with_usalpha_fields(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in CN_CACHE_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("float32")
    out["amount"] = out["close"].astype(float) * out["volume"].astype(float)
    out["vwap"] = (out["high"] + out["low"] + out["close"]) / 3.0
    out["ret"] = out["close"].pct_change()
    for col in CN_DERIVED_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float32")
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_stock_history_cached(ticker: str, start: str, end: str) -> pd.DataFrame:
    if yf is None:
        raise RuntimeError("未安装 yfinance，无法加载个股K线")

    # yfinance 的 end 是开区间，这里加一天确保用户选择的结束日能被覆盖。
    end_exclusive = (pd.Timestamp(end) + pd.Timedelta(days=1)).date().isoformat()
    raw = yf.download(
        tickers=ticker,
        start=start,
        end=end_exclusive,
        interval="1d",
        auto_adjust=False,
        actions=False,
        progress=False,
        threads=False,
    )
    return _normalize_stock_history(raw, ticker)


def _normalize_cn_symbol(symbol: str) -> str:
    s = str(symbol).strip().lower()
    for prefix in ("sh", "sz", "bj"):
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    return s


def _cn_cache_dir() -> Path:
    path = Path(__file__).resolve().parents[1] / ".cache" / "cn_akshare"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _cn_cache_path(symbol: str) -> Path:
    return _cn_cache_dir() / f"{_normalize_cn_symbol(symbol).zfill(6)}.parquet"


def _cn_index_cache_path(symbol: str) -> Path:
    return _cn_cache_dir() / f"index_{_normalize_cn_symbol(symbol).zfill(6)}.parquet"


def _empty_cn_history() -> pd.DataFrame:
    return pd.DataFrame(columns=CN_CACHE_ALL_COLUMNS)


def _install_akshare_timeouts() -> None:
    global _REQUESTS_TIMEOUT_PATCHED
    socket.setdefaulttimeout(CN_AKSHARE_NETWORK_TIMEOUT_SECONDS)
    if requests is None or _REQUESTS_TIMEOUT_PATCHED:
        return
    original_request = requests.sessions.Session.request

    def _request_with_timeout(self, method, url, **kwargs):
        if kwargs.get("timeout") is None:
            kwargs["timeout"] = CN_AKSHARE_NETWORK_TIMEOUT_SECONDS
        return original_request(self, method, url, **kwargs)

    requests.sessions.Session.request = _request_with_timeout
    _REQUESTS_TIMEOUT_PATCHED = True


def _quarantine_bad_cache(path: Path, reason: str) -> None:
    try:
        bad_path = path.with_suffix(path.suffix + f".bad.{int(time.time())}")
        path.rename(bad_path)
        print(f"[USalpha-CN] moved bad cache {path.name} -> {bad_path.name}: {reason}")
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[USalpha-CN] failed to quarantine bad cache {path}: {exc}; original error: {reason}")


def _compact_cn_history_for_cache(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    keep = [col for col in CN_CACHE_COLUMNS if col in out.columns]
    out = out[keep]
    for col in CN_CACHE_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    out = out[CN_CACHE_COLUMNS]
    for col in CN_CACHE_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float32")
    out.index = pd.to_datetime(out.index).tz_localize(None)
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out.dropna(subset=["open", "high", "low", "close"])


def _read_cn_cached_history(symbol: str) -> pd.DataFrame:
    path = _cn_cache_path(symbol)
    if not path.exists():
        return _empty_cn_history()
    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        _quarantine_bad_cache(path, str(exc))
        return _empty_cn_history()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    compact = _compact_cn_history_for_cache(df)
    return _with_usalpha_fields(compact.sort_index())


def _write_cn_cached_history(symbol: str, frame: pd.DataFrame) -> None:
    compact = _compact_cn_history_for_cache(frame)
    compact.to_parquet(_cn_cache_path(symbol), compression="zstd", index=True)


def _prefix_cn_symbol(symbol: str) -> str:
    s = _normalize_cn_symbol(symbol)
    if not s.isdigit():
        return str(symbol).strip().lower()
    if s.startswith("6"):
        return f"sh{s}"
    if s.startswith(("0", "3")):
        return f"sz{s}"
    if s.startswith(("4", "8")):
        return f"bj{s}"
    return s


def _get_mootdx_quotes_client():
    global _MOOTDX_CLIENT
    if Quotes is None:
        raise RuntimeError("未安装 mootdx，无法使用通达信行情源；请先执行 pip install mootdx")
    if _MOOTDX_CLIENT is None:
        _MOOTDX_CLIENT = Quotes.factory(market="std")
    return _MOOTDX_CLIENT


def _normalize_mootdx_history(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty:
        return _empty_cn_history()
    out = raw.copy()
    if "datetime" not in out.columns:
        raise ValueError("mootdx bars missing datetime column")
    out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce").dt.tz_localize(None)
    out = out.dropna(subset=["datetime"]).set_index("datetime").sort_index()
    if "volume" not in out.columns and "vol" in out.columns:
        out["volume"] = out["vol"]
    required = ["open", "high", "low", "close", "volume"]
    missing = [col for col in required if col not in out.columns]
    if missing:
        raise ValueError(f"mootdx bars missing columns: {missing}")
    out = out[required].copy()
    return _with_usalpha_fields(out)


def _fetch_cn_stock_history_mootdx(symbol: str, start: str, end: str) -> pd.DataFrame:
    client = _get_mootdx_quotes_client()
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()

    frames: list[pd.DataFrame] = []
    seen_ranges: set[tuple[pd.Timestamp, pd.Timestamp]] = set()
    for start_offset in range(0, 6400, 800):
        raw = client.bars(symbol=_normalize_cn_symbol(symbol).zfill(6), frequency=9, start=start_offset, offset=800)
        frame = _normalize_mootdx_history(raw)
        if frame.empty:
            break
        frame = frame.loc[(frame.index >= start_ts) & (frame.index <= end_ts)].copy()
        if not frame.empty:
            key = (pd.Timestamp(frame.index.min()), pd.Timestamp(frame.index.max()))
            if key in seen_ranges:
                break
            seen_ranges.add(key)
            frames.append(frame)

        raw_index = pd.to_datetime(raw["datetime"], errors="coerce").dropna()
        if raw_index.empty or raw_index.min().normalize() <= start_ts:
            break

    if not frames:
        return _empty_cn_history()

    merged = pd.concat(frames).sort_index()
    merged = merged[~merged.index.duplicated(keep="last")]
    return merged.loc[(merged.index >= start_ts) & (merged.index <= end_ts)].copy()


def _df_item_value_to_dict(df: pd.DataFrame) -> dict[str, Any]:
    if df is None or df.empty:
        return {}
    if "item" in df.columns and "value" in df.columns:
        return {
            str(row["item"]).strip(): row["value"]
            for _, row in df.iterrows()
            if str(row.get("item", "")).strip()
        }
    if "item" not in df.columns and "value" not in df.columns and len(df) == 1:
        return df.iloc[0].to_dict()
    if df.shape[1] >= 2:
        key_col, value_col = df.columns[:2]
        return {
            str(row[key_col]).strip(): row[value_col]
            for _, row in df.iterrows()
            if str(row.get(key_col, "")).strip()
        }
    return {
        str(col).strip(): df.iloc[0][col]
        for col in df.columns
        if str(col).strip()
    }


def _safe_call_akshare(fn_name: str, **kwargs: Any) -> tuple[Any | None, str | None]:
    if ak is None:
        return None, "akshare 未安装"
    _install_akshare_timeouts()
    fn = getattr(ak, fn_name, None)
    if fn is None:
        return None, f"当前 akshare 版本没有 {fn_name}"
    try:
        return fn(**kwargs), None
    except Exception as exc:  # pylint: disable=broad-except
        return None, str(exc)


def _is_nonempty_df(value: Any) -> bool:
    return isinstance(value, pd.DataFrame) and not value.empty


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def _fetch_cn_all_a_symbols_cached() -> pd.DataFrame:
    cols = ["symbol", "name"]
    if ak is None:
        return pd.DataFrame(columns=cols)

    raw, err = _safe_call_akshare("stock_info_a_code_name")
    if err or not isinstance(raw, pd.DataFrame) or raw.empty:
        return pd.DataFrame(columns=cols)

    code_col = "code" if "code" in raw.columns else ("代码" if "代码" in raw.columns else None)
    name_col = "name" if "name" in raw.columns else ("名称" if "名称" in raw.columns else None)
    if code_col is None:
        return pd.DataFrame(columns=cols)

    out = pd.DataFrame()
    out["symbol"] = raw[code_col].astype(str).map(lambda x: _normalize_cn_symbol(x).zfill(6))
    out["name"] = raw[name_col].astype(str) if name_col is not None else ""
    out = out[out["symbol"].str.match(r"^\d{6}$", na=False)].drop_duplicates("symbol")
    return out.sort_values("symbol").reset_index(drop=True)


def _format_bytes(num_bytes: float) -> str:
    value = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(value) < 1024.0:
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} PB"


def _cn_cache_status() -> dict[str, Any]:
    cache_dir = _cn_cache_dir()
    files = [p for p in cache_dir.glob("*.parquet") if p.is_file() and not p.name.startswith("index_")]
    sizes = [p.stat().st_size for p in files]
    total_size = int(sum(sizes))
    avg_size = float(np.mean(sizes)) if sizes else 0.0
    return {
        "file_count": len(files),
        "total_size": total_size,
        "avg_size": avg_size,
        "cache_dir": str(cache_dir),
    }


def _pick_latest_record(df: pd.DataFrame) -> dict[str, Any]:
    if df is None or df.empty:
        return {}

    out = df.copy()
    if isinstance(out.index, pd.DatetimeIndex):
        out = out.sort_index()
        record = out.iloc[-1].to_dict()
        record.setdefault("trade_date", out.index[-1])
        return record

    date_cols = [
        col
        for col in out.columns
        if str(col).lower() in {"trade_date", "date", "日期", "报告期", "公告日期"}
    ]
    if date_cols:
        col = date_cols[0]
        out[col] = pd.to_datetime(out[col], errors="coerce")
        out = out.sort_values(col)
        return out.iloc[-1].to_dict()

    return out.iloc[0].to_dict()


def _format_cn_market_value(value: Any) -> str:
    if value is None:
        return "-"
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return "-"
    if "亿" in text:
        return text
    try:
        number = float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return text
    if not np.isfinite(number):
        return "-"
    return f"{number / 100_000_000:.2f} 亿"


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_cn_stock_info_cached(ticker: str) -> dict[str, Any]:
    if ak is None:
        raise RuntimeError("未安装 akshare，无法加载中国股市个股信息；请先执行 pip install akshare")

    symbol = _normalize_cn_symbol(ticker)
    prefixed = _prefix_cn_symbol(symbol)
    prefixed_upper = prefixed.upper()
    errors: dict[str, str] = {}

    individual_raw, err = _safe_call_akshare("stock_individual_info_em", symbol=symbol)
    if err:
        errors["stock_individual_info_em"] = err
    individual = _df_item_value_to_dict(individual_raw) if isinstance(individual_raw, pd.DataFrame) else {}

    stock_info_raw, err = _safe_call_akshare("stock_info_a_code_name")
    if err:
        errors["stock_info_a_code_name"] = err
    if isinstance(stock_info_raw, pd.DataFrame) and not stock_info_raw.empty and "code" in stock_info_raw.columns:
        hit = stock_info_raw[stock_info_raw["code"].astype(str).str.zfill(6) == symbol.zfill(6)]
        if len(hit) > 0:
            for key, value in hit.iloc[0].to_dict().items():
                individual.setdefault(str(key), value)

    basic_raw, err = _safe_call_akshare("stock_individual_basic_info_xq", symbol=prefixed_upper)
    if err:
        errors["stock_individual_basic_info_xq"] = err
    basic = _df_item_value_to_dict(basic_raw) if isinstance(basic_raw, pd.DataFrame) else {}

    if not basic:
        alt_basic_raw, err = _safe_call_akshare("stock_individual_info_xq", symbol=prefixed)
        if err:
            errors["stock_individual_info_xq"] = err
        basic = _df_item_value_to_dict(alt_basic_raw) if isinstance(alt_basic_raw, pd.DataFrame) else {}

    profile_raw, err = _safe_call_akshare("stock_profile_cninfo", symbol=symbol)
    if err:
        errors["stock_profile_cninfo"] = err
    profile = _df_item_value_to_dict(profile_raw) if isinstance(profile_raw, pd.DataFrame) else {}
    for key, value in profile.items():
        basic.setdefault(str(key), value)

    business_raw, err = _safe_call_akshare("stock_zyjs_ths", symbol=symbol)
    if err:
        errors["stock_zyjs_ths"] = err
    business: dict[str, Any] = {}
    if isinstance(business_raw, pd.DataFrame) and not business_raw.empty:
        business = business_raw.iloc[0].to_dict()

    spot_latest: dict[str, Any] = {}
    spot_errors: list[str] = []
    spot_candidates = [
        ("stock_zh_a_spot_em", {}),
        ("stock_sh_a_spot_em", {}),
        ("stock_sz_a_spot_em", {}),
        ("stock_bj_a_spot_em", {}),
        ("stock_kc_a_spot_em", {}),
    ]
    for fn_name, kwargs in spot_candidates:
        spot_raw, err = _safe_call_akshare(fn_name, **kwargs)
        if err:
            spot_errors.append(f"{fn_name}: {err}")
            continue
        if isinstance(spot_raw, pd.DataFrame) and not spot_raw.empty and "代码" in spot_raw.columns:
            hit = spot_raw[spot_raw["代码"].astype(str).str.zfill(6) == symbol.zfill(6)]
            if len(hit) > 0:
                spot_latest = hit.iloc[0].to_dict()
                break
    if spot_errors and not spot_latest:
        errors["spot_em"] = "；".join(spot_errors)

    individual_spot: dict[str, Any] = {}
    xq_symbol_candidates = [prefixed_upper, prefixed]
    if prefixed_upper.startswith("SH") and symbol.startswith(("0", "3")):
        xq_symbol_candidates.insert(0, f"SZ{symbol}")
    if prefixed_upper.startswith("SZ") and symbol.startswith("6"):
        xq_symbol_candidates.insert(0, f"SH{symbol}")
    for xq_symbol in dict.fromkeys(xq_symbol_candidates):
        individual_spot_raw, err = _safe_call_akshare("stock_individual_spot_xq", symbol=xq_symbol)
        if err:
            errors[f"stock_individual_spot_xq:{xq_symbol}"] = err
            continue
        individual_spot = _df_item_value_to_dict(individual_spot_raw) if isinstance(individual_spot_raw, pd.DataFrame) else {}
        if individual_spot:
            break

    for key, value in individual_spot.items():
        spot_latest.setdefault(str(key), value)

    value_em_raw, err = _safe_call_akshare("stock_value_em", symbol=symbol)
    if err:
        errors["stock_value_em"] = err
    value_em_latest = _pick_latest_record(value_em_raw) if _is_nonempty_df(value_em_raw) else {}

    valuation_raw = None
    valuation_errs: list[str] = []
    valuation_attempts = [
        ("stock_a_lg_indicator", {"stock": symbol}),
        ("stock_a_lg_indicator", {"symbol": symbol}),
        ("stock_a_indicator_lg", {"symbol": symbol}),
        ("stock_a_indicator_lg", {"stock": symbol}),
    ]
    for fn_name, kwargs in valuation_attempts:
        valuation_raw, err = _safe_call_akshare(fn_name, **kwargs)
        if err:
            valuation_errs.append(f"{fn_name}({kwargs}): {err}")
            continue
        if _is_nonempty_df(valuation_raw):
            break
    else:
        valuation_raw = pd.DataFrame()
    if valuation_errs and not _is_nonempty_df(valuation_raw):
        errors["valuation_indicator"] = "；".join(valuation_errs)

    valuation_latest = _pick_latest_record(valuation_raw) if _is_nonempty_df(valuation_raw) else {}
    for key, value in value_em_latest.items():
        valuation_latest.setdefault(str(key), value)
    # Real-time Eastmoney fields are a useful fallback and often more stable
    # than the historical valuation endpoints.
    for src_key, dst_key in {
        "市盈率-动态": "市盈率",
        "市净率": "市净率",
        "总市值": "总市值",
        "流通市值": "流通市值",
        "市盈率(TTM)": "市盈率TTM",
        "市盈率(动)": "市盈率",
        "市盈率(静)": "市盈率静态",
        "股息率(TTM)": "股息率TTM",
        "资产净值/总市值": "市净率",
        "现价": "最新价",
    }.items():
        if src_key in spot_latest and dst_key not in valuation_latest:
            valuation_latest[dst_key] = spot_latest[src_key]

    return {
        "symbol": symbol,
        "prefixed_symbol": prefixed,
        "individual": individual,
        "basic": basic,
        "business": business,
        "valuation_latest": valuation_latest,
        "spot_latest": spot_latest,
        "errors": errors,
    }


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_cn_stock_history_cached(ticker: str, start: str, end: str) -> pd.DataFrame:
    symbol = _normalize_cn_symbol(ticker)
    start_ts = pd.Timestamp(start).normalize()
    end_ts = _effective_cn_daily_end(end)
    cached = _read_cn_cached_history(symbol)
    if not cached.empty:
        cached_start = pd.Timestamp(cached.index.min()).normalize()
        cached_end = pd.Timestamp(cached.index.max()).normalize()
        if cached_start <= start_ts and cached_end >= end_ts:
            return cached.loc[(cached.index >= start_ts) & (cached.index <= end_ts)].copy()

    download_start_ts = start_ts
    if not cached.empty:
        cached_end = pd.Timestamp(cached.index.max()).normalize()
        if cached_end >= start_ts:
            # Match alpha_mining's incremental semantics: keep overlap and
            # backfill a short window so adjusted prices can be refreshed.
            download_start_ts = max(start_ts, cached_end - pd.Timedelta(days=CN_INCREMENTAL_BACKFILL_DAYS))

    errors: list[str] = []
    live = pd.DataFrame()
    try:
        live = _fetch_cn_stock_history_mootdx(symbol, download_start_ts.date().isoformat(), end_ts.date().isoformat())
    except Exception as exc:  # pylint: disable=broad-except
        errors.append(f"mootdx: {exc}")

    if not live.empty:
        merged = live if cached.empty else pd.concat([cached, live]).sort_index()
        merged = merged[~merged.index.duplicated(keep="last")]
        _write_cn_cached_history(symbol, merged)
        return merged.loc[(merged.index >= start_ts) & (merged.index <= end_ts)].copy()

    if ak is None:
        if not cached.empty:
            sliced = cached.loc[(cached.index >= start_ts) & (cached.index <= end_ts)].copy()
            if not sliced.empty:
                return sliced
        raise RuntimeError(
            "未安装 akshare 且 mootdx 取数失败，无法加载中国股市K线；"
            "请先执行 pip install akshare mootdx"
        )

    _install_akshare_timeouts()
    start_yyyymmdd = pd.Timestamp(start).strftime("%Y%m%d")
    end_yyyymmdd = pd.Timestamp(end).strftime("%Y%m%d")
    download_start_yyyymmdd = download_start_ts.strftime("%Y%m%d")
    raw = pd.DataFrame()

    # 参照 alpha_mining：先 stock_zh_a_daily，再 stock_zh_a_hist，最后 stock_zh_a_hist_tx；每个源做短重试。
    for attempt in range(1, CN_AKSHARE_MAX_RETRIES + 1):
        try:
            daily_raw = ak.stock_zh_a_daily(symbol=_prefix_cn_symbol(symbol), adjust="qfq")
            daily_raw = daily_raw.copy()
            daily_raw["date"] = pd.to_datetime(daily_raw["date"], errors="coerce")
            raw = daily_raw[
                (daily_raw["date"] >= download_start_ts)
                & (daily_raw["date"] <= end_ts)
            ]
            if not raw.empty:
                break
        except Exception as exc:
            errors.append(f"stock_zh_a_daily attempt={attempt}: {exc}")
            if attempt < CN_AKSHARE_MAX_RETRIES:
                time.sleep(CN_AKSHARE_RETRY_SLEEP_SECONDS * attempt)

    if raw.empty:
        for attempt in range(1, CN_AKSHARE_MAX_RETRIES + 1):
            try:
                raw = ak.stock_zh_a_hist(
                    symbol=symbol,
                    period="daily",
                    start_date=download_start_yyyymmdd,
                    end_date=end_yyyymmdd,
                    adjust="qfq",
                )
                if raw is not None and not raw.empty:
                    break
            except Exception as exc:
                errors.append(f"stock_zh_a_hist attempt={attempt}: {exc}")
                if attempt < CN_AKSHARE_MAX_RETRIES:
                    time.sleep(CN_AKSHARE_RETRY_SLEEP_SECONDS * attempt)

    if raw.empty:
        for attempt in range(1, CN_AKSHARE_MAX_RETRIES + 1):
            try:
                raw = ak.stock_zh_a_hist_tx(
                    symbol=_prefix_cn_symbol(symbol),
                    start_date=download_start_yyyymmdd,
                    end_date=end_yyyymmdd,
                    adjust="qfq",
                )
                if raw is not None and not raw.empty:
                    break
            except Exception as exc:
                errors.append(f"stock_zh_a_hist_tx attempt={attempt}: {exc}")
                if attempt < CN_AKSHARE_MAX_RETRIES:
                    time.sleep(CN_AKSHARE_RETRY_SLEEP_SECONDS * attempt)

    if raw.empty:
        if not cached.empty:
            sliced = cached.loc[(cached.index >= start_ts) & (cached.index <= end_ts)].copy()
            if not sliced.empty:
                return sliced
        raise ValueError(f"未获取到 {ticker} 在 {start_yyyymmdd}~{end_yyyymmdd} 的A股行情: {'; '.join(errors)}")

    new_data = _with_usalpha_fields(_normalize_akshare_history(raw))
    merged = new_data if cached.empty else pd.concat([cached, new_data]).sort_index()
    merged = merged[~merged.index.duplicated(keep="last")]
    _write_cn_cached_history(symbol, merged)
    return merged.loc[(merged.index >= start_ts) & (merged.index <= end_ts)].copy()


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_cn_index_history_cached(index_code: str, start: str, end: str) -> pd.DataFrame:
    if ak is None:
        raise RuntimeError("未安装 akshare，无法加载中国股市指数行情；请先执行 pip install akshare")

    _install_akshare_timeouts()
    symbol = _normalize_cn_symbol(index_code).zfill(6)
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    path = _cn_index_cache_path(symbol)
    cached = pd.DataFrame(columns=["open", "high", "low", "close", "volume", "amount", "vwap", "ret"])
    if path.exists():
        try:
            cached = pd.read_parquet(path)
            cached.index = pd.to_datetime(cached.index).tz_localize(None)
            cached = cached.sort_index()
        except Exception as exc:
            _quarantine_bad_cache(path, str(exc))
            cached = pd.DataFrame(columns=["open", "high", "low", "close", "volume", "amount", "vwap", "ret"])

    if not cached.empty and cached.index.min() <= start_ts and cached.index.max() >= end_ts:
        return cached.loc[(cached.index >= start_ts) & (cached.index <= end_ts)].copy()

    download_start_ts = start_ts
    if not cached.empty and cached.index.max() >= start_ts:
        download_start_ts = pd.Timestamp(cached.index.max()).normalize()

    start_yyyymmdd = download_start_ts.strftime("%Y%m%d")
    end_yyyymmdd = end_ts.strftime("%Y%m%d")
    errors: list[str] = []
    raw = pd.DataFrame()

    for fn_name, kwargs in [
        ("stock_zh_index_daily", {"symbol": f"sh{symbol}" if symbol == "000001" else symbol}),
        ("index_zh_a_hist", {"symbol": symbol, "period": "daily", "start_date": start_yyyymmdd, "end_date": end_yyyymmdd}),
    ]:
        try:
            fn = getattr(ak, fn_name)
            raw = fn(**kwargs)
            if raw is not None and not raw.empty:
                break
        except Exception as exc:  # pylint: disable=broad-except
            errors.append(f"{fn_name}: {exc}")
            raw = pd.DataFrame()

    if raw.empty:
        if not cached.empty:
            sliced = cached.loc[(cached.index >= start_ts) & (cached.index <= end_ts)].copy()
            if not sliced.empty:
                return sliced
        raise ValueError(f"未获取到指数 {symbol} 在 {start_yyyymmdd}~{end_yyyymmdd} 的行情: {'; '.join(errors)}")

    new_data = _with_usalpha_fields(_normalize_akshare_history(raw))
    merged = new_data if cached.empty else pd.concat([cached, new_data]).sort_index()
    merged = merged[~merged.index.duplicated(keep="last")]
    _compact_cn_history_for_cache(merged).to_parquet(path, compression="zstd", index=True)
    merged = _with_usalpha_fields(_compact_cn_history_for_cache(merged))
    return merged.loc[(merged.index >= start_ts) & (merged.index <= end_ts)].copy()


def _build_cn_market_bundle(tickers: list[str], *, benchmark: str, start: str, end: str) -> MarketDataBundle:
    per_ticker: dict[str, pd.DataFrame] = {}
    failures: dict[str, str] = {}
    for ticker in tickers:
        symbol = _normalize_cn_symbol(ticker).zfill(6)
        try:
            per_ticker[symbol] = _fetch_cn_stock_history_cached(symbol, start, end)
        except Exception as exc:  # pylint: disable=broad-except
            failures[symbol] = str(exc)

    if not per_ticker:
        raise RuntimeError(f"A股行情下载失败: {failures}")

    panel = pd.concat(per_ticker, axis=1)
    panel.columns.names = ["instrument", "field"]
    panel = panel.sort_index()

    bench_symbol = _normalize_cn_symbol(benchmark or tickers[0]).zfill(6)
    try:
        benchmark_df = _fetch_cn_stock_history_cached(bench_symbol, start, end)
    except Exception:
        benchmark_df = next(iter(per_ticker.values())).copy()

    if failures:
        print(f"[USalpha-CN] warning: failed tickers excluded: {failures}")
    return MarketDataBundle(panel=panel, benchmark=benchmark_df)


def _resample_ohlcv(daily: pd.DataFrame, freq: str) -> pd.DataFrame:
    if daily.empty:
        return daily
    # pandas 2.2+ recommends "ME" for month-end, while older pandas versions
    # only accept "M". Keep the app compatible with both.
    effective_freq = "M" if freq == "ME" else freq
    out = daily.resample(effective_freq).agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    return out.dropna(subset=["open", "high", "low", "close"])


def _add_macd(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    close = out["close"].astype(float)
    ema12 = close.ewm(span=12, adjust=False, min_periods=1).mean()
    ema26 = close.ewm(span=26, adjust=False, min_periods=1).mean()
    out["macd"] = ema12 - ema26
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False, min_periods=1).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]
    return out


def _add_bollinger(frame: pd.DataFrame, window: int = 20, width: float = 2.0) -> pd.DataFrame:
    out = frame.copy()
    mid = out["close"].rolling(window, min_periods=1).mean()
    std = out["close"].rolling(window, min_periods=1).std(ddof=0)
    out["boll_mid"] = mid
    out["boll_upper"] = mid + width * std
    out["boll_lower"] = mid - width * std
    return out


def _add_kdj(frame: pd.DataFrame, n: int = 9) -> pd.DataFrame:
    out = frame.copy()
    low_n = out["low"].rolling(n, min_periods=1).min()
    high_n = out["high"].rolling(n, min_periods=1).max()
    rsv = (out["close"] - low_n) / (high_n - low_n).replace(0, np.nan) * 100.0
    out["kdj_k"] = rsv.ewm(alpha=1 / 3, adjust=False, min_periods=1).mean().fillna(50.0)
    out["kdj_d"] = out["kdj_k"].ewm(alpha=1 / 3, adjust=False, min_periods=1).mean().fillna(50.0)
    out["kdj_j"] = 3 * out["kdj_k"] - 2 * out["kdj_d"]
    return out


def _add_rsi(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    delta = out["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    for window in (6, 12, 24):
        avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=1).mean()
        avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=1).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        out[f"rsi_{window}"] = (100 - 100 / (1 + rs)).fillna(100.0)
    return out


def _add_dmi(frame: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    out = frame.copy()
    high = out["high"]
    low = out["low"]
    close = out["close"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=out.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=out.index)
    tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / n, adjust=False, min_periods=1).mean().replace(0, np.nan)
    out["dmi_pdi"] = 100 * plus_dm.ewm(alpha=1 / n, adjust=False, min_periods=1).mean() / atr
    out["dmi_mdi"] = 100 * minus_dm.ewm(alpha=1 / n, adjust=False, min_periods=1).mean() / atr
    dx = ((out["dmi_pdi"] - out["dmi_mdi"]).abs() / (out["dmi_pdi"] + out["dmi_mdi"]).replace(0, np.nan)) * 100
    out["dmi_adx"] = dx.ewm(alpha=1 / n, adjust=False, min_periods=1).mean()
    out["dmi_adxr"] = (out["dmi_adx"] + out["dmi_adx"].shift(n)) / 2
    return out


def _add_wr(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for window in (10, 20):
        high_n = out["high"].rolling(window, min_periods=1).max()
        low_n = out["low"].rolling(window, min_periods=1).min()
        out[f"wr_{window}"] = (high_n - out["close"]) / (high_n - low_n).replace(0, np.nan) * 100.0
    return out


def _add_indicator(frame: pd.DataFrame, indicator: str) -> pd.DataFrame:
    out = _add_macd(frame)
    if indicator == "KDJ":
        out = _add_kdj(out)
    elif indicator == "RSI":
        out = _add_rsi(out)
    elif indicator == "DMI":
        out = _add_dmi(out)
    elif indicator == "威廉指标":
        out = _add_wr(out)
    return out


def _score_to_action(score: float) -> str:
    if score >= 25:
        return "买入"
    if score <= -25:
        return "卖出"
    return "中性"


def _score_technical_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    cols = ["指标", "信号", "分数", "说明"]
    if frame.empty or len(frame) < 2:
        return pd.DataFrame(columns=cols)

    data = _add_wr(_add_dmi(_add_rsi(_add_kdj(_add_macd(frame)))))
    close = data["close"]
    latest = data.iloc[-1]
    prev = data.iloc[-2]
    rows: list[dict[str, Any]] = []

    def add_row(name: str, score: float, detail: str) -> None:
        score = float(np.clip(score, -100, 100))
        rows.append({"指标": name, "信号": _score_to_action(score), "分数": round(score, 1), "说明": detail})

    macd_score = 0.0
    if latest["macd"] > latest["macd_signal"]:
        macd_score += 35
    else:
        macd_score -= 35
    if prev["macd"] <= prev["macd_signal"] and latest["macd"] > latest["macd_signal"]:
        macd_score += 45
    elif prev["macd"] >= prev["macd_signal"] and latest["macd"] < latest["macd_signal"]:
        macd_score -= 45
    macd_score += 20 if latest["macd_hist"] > prev["macd_hist"] else -20
    add_row("MACD", macd_score, f"DIF={latest['macd']:.3f}, DEA={latest['macd_signal']:.3f}, 柱={latest['macd_hist']:.3f}")

    kdj_score = (50 - abs(float(latest["kdj_k"]) - 50)) * 0.4
    if latest["kdj_k"] > latest["kdj_d"]:
        kdj_score += 25
    else:
        kdj_score -= 25
    if latest["kdj_j"] < 20:
        kdj_score += 35
    elif latest["kdj_j"] > 80:
        kdj_score -= 35
    add_row("KDJ", kdj_score, f"K={latest['kdj_k']:.1f}, D={latest['kdj_d']:.1f}, J={latest['kdj_j']:.1f}")

    rsi6 = float(latest["rsi_6"])
    if rsi6 < 30:
        rsi_score = 70
    elif rsi6 > 70:
        rsi_score = -70
    else:
        rsi_score = (50 - rsi6) * 1.2
    if latest["rsi_6"] > prev["rsi_6"]:
        rsi_score += 15
    else:
        rsi_score -= 15
    add_row("RSI", rsi_score, f"RSI6={latest['rsi_6']:.1f}, RSI12={latest['rsi_12']:.1f}, RSI24={latest['rsi_24']:.1f}")

    dmi_score = 30 if latest["dmi_pdi"] > latest["dmi_mdi"] else -30
    dmi_score += min(float(latest["dmi_adx"]), 40) if latest["dmi_pdi"] > latest["dmi_mdi"] else -min(float(latest["dmi_adx"]), 40)
    add_row("DMI", dmi_score, f"PDI={latest['dmi_pdi']:.1f}, MDI={latest['dmi_mdi']:.1f}, ADX={latest['dmi_adx']:.1f}")

    wr = float(latest["wr_10"])
    if wr > 80:
        wr_score = 70
    elif wr < 20:
        wr_score = -70
    else:
        wr_score = (50 - wr) * -1.2
    add_row("威廉指标", wr_score, f"WR10={latest['wr_10']:.1f}, WR20={latest['wr_20']:.1f}")

    ma_windows = [5, 10, 20, 120]
    ma_values = {w: close.rolling(w, min_periods=1).mean().iloc[-1] for w in ma_windows}
    ma_score = 0.0
    ma_score += 25 if close.iloc[-1] > ma_values[5] else -25
    ma_score += 25 if ma_values[5] > ma_values[10] else -25
    ma_score += 25 if ma_values[10] > ma_values[20] else -25
    ma_score += 25 if close.iloc[-1] > ma_values[120] else -25
    add_row("均线", ma_score, f"收盘={close.iloc[-1]:.2f}, MA5={ma_values[5]:.2f}, MA10={ma_values[10]:.2f}, MA20={ma_values[20]:.2f}, MA120={ma_values[120]:.2f}")

    scored = pd.DataFrame(rows, columns=cols)
    total_score = float(scored["分数"].mean()) if len(scored) else 0.0
    summary = pd.DataFrame([{"指标": "综合", "信号": _score_to_action(total_score), "分数": round(total_score, 1), "说明": "各技术指标分数简单平均"}])
    return pd.concat([summary, scored], ignore_index=True)


def _technical_strategy_one_score_series(frame: pd.DataFrame) -> pd.Series:
    if frame.empty or len(frame) < 2:
        return pd.Series(dtype=float)

    data = _add_wr(_add_dmi(_add_rsi(_add_kdj(_add_macd(frame[["open", "high", "low", "close", "volume"]])))))
    close = data["close"].astype(float)
    prev = data.shift(1)

    macd_score = pd.Series(np.where(data["macd"] > data["macd_signal"], 35.0, -35.0), index=data.index)
    macd_score += np.where(
        (prev["macd"] <= prev["macd_signal"]) & (data["macd"] > data["macd_signal"]),
        45.0,
        np.where((prev["macd"] >= prev["macd_signal"]) & (data["macd"] < data["macd_signal"]), -45.0, 0.0),
    )
    macd_score += np.where(data["macd_hist"] > prev["macd_hist"], 20.0, -20.0)

    kdj_score = (50.0 - (data["kdj_k"].astype(float) - 50.0).abs()) * 0.4
    kdj_score += np.where(data["kdj_k"] > data["kdj_d"], 25.0, -25.0)
    kdj_score += np.where(data["kdj_j"] < 20, 35.0, np.where(data["kdj_j"] > 80, -35.0, 0.0))

    rsi6 = data["rsi_6"].astype(float)
    rsi_score = pd.Series(np.where(rsi6 < 30, 70.0, np.where(rsi6 > 70, -70.0, (50.0 - rsi6) * 1.2)), index=data.index)
    rsi_score += np.where(data["rsi_6"] > prev["rsi_6"], 15.0, -15.0)

    dmi_score = pd.Series(np.where(data["dmi_pdi"] > data["dmi_mdi"], 30.0, -30.0), index=data.index)
    adx = data["dmi_adx"].astype(float).clip(upper=40.0).fillna(0.0)
    dmi_score += np.where(data["dmi_pdi"] > data["dmi_mdi"], adx, -adx)

    wr = data["wr_10"].astype(float)
    wr_score = pd.Series(np.where(wr > 80, 70.0, np.where(wr < 20, -70.0, (50.0 - wr) * -1.2)), index=data.index)

    ma5 = close.rolling(5, min_periods=1).mean()
    ma10 = close.rolling(10, min_periods=1).mean()
    ma20 = close.rolling(20, min_periods=1).mean()
    ma120 = close.rolling(120, min_periods=1).mean()
    ma_score = pd.Series(0.0, index=data.index)
    ma_score += np.where(close > ma5, 25.0, -25.0)
    ma_score += np.where(ma5 > ma10, 25.0, -25.0)
    ma_score += np.where(ma10 > ma20, 25.0, -25.0)
    ma_score += np.where(close > ma120, 25.0, -25.0)

    score = pd.concat(
        [
            macd_score.clip(-100, 100),
            kdj_score.clip(-100, 100),
            rsi_score.clip(-100, 100),
            dmi_score.clip(-100, 100),
            wr_score.clip(-100, 100),
            ma_score.clip(-100, 100),
        ],
        axis=1,
    ).mean(axis=1)
    score.iloc[:1] = np.nan
    return score.replace([np.inf, -np.inf], np.nan)


def _predict_factor_trade_points(
    *,
    ticker: str,
    ohlcv: pd.DataFrame,
    timeframe: str,
    factor_context: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Reserved hook for factor-driven buy/sell predictions on the stock page.

    Expected output schema for future implementations:
    - datetime: signal date, aligned to ``ohlcv.index``
    - signal: "buy" or "sell"
    - price: marker price; if missing, chart will use close
    - confidence: optional 0~1 score
    - factor_name: optional source factor/model name
    - reason: optional explanation

    Current implementation intentionally returns an empty DataFrame so the
    stock viewer can be used now while leaving a stable integration point for
    factor/model-generated trade points later.
    """
    _ = (ticker, ohlcv, timeframe, factor_context)
    return pd.DataFrame(columns=["datetime", "signal", "price", "confidence", "factor_name", "reason"])


def _prepare_trade_signal_markers(signals: pd.DataFrame | None, frame: pd.DataFrame) -> pd.DataFrame:
    cols = ["datetime", "signal", "price", "confidence", "factor_name", "reason"]
    if signals is None or len(signals) == 0:
        return pd.DataFrame(columns=cols)

    out = signals.copy()
    if "datetime" not in out.columns or "signal" not in out.columns:
        return pd.DataFrame(columns=cols)

    out["datetime"] = pd.to_datetime(out["datetime"]).dt.tz_localize(None)
    out["signal"] = out["signal"].astype(str).str.lower()
    out = out[out["signal"].isin(["buy", "sell"])].copy()
    if out.empty:
        return pd.DataFrame(columns=cols)

    close = frame["close"].copy()
    out = out.sort_values("datetime")
    valid_dates = pd.DatetimeIndex(frame.index).unique().sort_values()
    out["datetime"] = out["datetime"].map(lambda dt: valid_dates[valid_dates <= dt].max() if (valid_dates <= dt).any() else pd.NaT)
    out = out.dropna(subset=["datetime"])
    if "price" not in out.columns:
        out["price"] = np.nan
    out["price"] = out.apply(
        lambda row: close.loc[row["datetime"]] if pd.isna(row["price"]) and row["datetime"] in close.index else row["price"],
        axis=1,
    )
    for col in cols:
        if col not in out.columns:
            out[col] = np.nan if col in {"price", "confidence"} else ""
    return out[cols]


def _render_trade_signal_table(signals: pd.DataFrame, timeframe: str) -> None:
    st.markdown(f"#### 因子买卖点预测（{timeframe}）")
    if signals.empty:
        st.info("已预留因子买点/卖点预测接口；当前尚未接入具体因子信号，因此暂无预测点。")
        return
    st.dataframe(signals.sort_values("datetime", ascending=False), use_container_width=True, hide_index=True)


def _render_technical_score_table(frame: pd.DataFrame, timeframe: str) -> None:
    st.markdown(f"#### 技术指标买卖评分（{timeframe}）")
    scores = _score_technical_indicators(frame)
    if scores.empty:
        st.info("数据不足，暂无法计算技术指标评分。")
        return
    st.dataframe(scores, use_container_width=True, hide_index=True)


def _render_ohlcv_macd_chart(
    frame: pd.DataFrame,
    *,
    ticker: str,
    title: str,
    trade_signals: pd.DataFrame | None = None,
    ma_windows: list[int] | None = None,
    indicator: str = "MACD",
    show_bollinger: bool = False,
) -> None:
    if frame.empty:
        st.warning("没有可展示的行情数据")
        return
    score_source = frame.copy()
    frame = _add_indicator(frame, indicator)
    if show_bollinger:
        frame = _add_bollinger(frame)
    signals = _prepare_trade_signal_markers(trade_signals, frame)

    if go is None or make_subplots is None:
        st.warning("Plotly 不可用，暂以收盘价折线替代K线图。")
        st.line_chart(frame[["close"]])
        st.bar_chart(frame[["volume"]])
        st.line_chart(frame[["macd", "macd_signal", "macd_hist"]])
        _render_technical_score_table(score_source, title)
        _render_trade_signal_table(signals, title)
        return

    up_color = "#2ca02c"
    down_color = "#d62728"
    candle_colors = np.where(frame["close"] >= frame["open"], up_color, down_color)
    hist_colors = np.where(frame["macd_hist"] >= 0, up_color, down_color)
    ma_windows_clean = sorted({int(x) for x in (ma_windows or []) if int(x) > 0})
    # Use a categorical x-axis so weekends and US market holidays are not shown
    # as empty gaps between trading bars.
    x_values = pd.DatetimeIndex(frame.index).strftime("%Y-%m-%d")

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.58, 0.18, 0.24],
        subplot_titles=(f"{ticker.upper()} {title} 蜡烛线", "成交量", indicator),
    )

    ma_colors = ["#f39c12", "#3498db", "#9b59b6", "#34495e", "#00a087", "#e377c2", "#7f7f7f"]
    for i, window in enumerate(ma_windows_clean):
        ma = frame["close"].rolling(window, min_periods=1).mean()
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=ma,
                mode="lines",
                name=f"MA{window}",
                line={"width": 1.4, "color": ma_colors[i % len(ma_colors)]},
            ),
            row=1,
            col=1,
        )
    if show_bollinger:
        for col, name, color in [
            ("boll_upper", "BOLL上轨", "#6c757d"),
            ("boll_mid", "BOLL中轨", "#495057"),
            ("boll_lower", "BOLL下轨", "#6c757d"),
        ]:
            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=frame[col],
                    mode="lines",
                    name=name,
                    line={"width": 1.1, "color": color, "dash": "dot" if col != "boll_mid" else "solid"},
                ),
                row=1,
                col=1,
            )
    fig.add_trace(
        go.Candlestick(
            x=x_values,
            open=frame["open"],
            high=frame["high"],
            low=frame["low"],
            close=frame["close"],
            name="K线",
            increasing_line_color=up_color,
            decreasing_line_color=down_color,
        ),
        row=1,
        col=1,
    )
    if not signals.empty:
        signal_x = pd.to_datetime(signals["datetime"]).dt.strftime("%Y-%m-%d")
        buy = signals[signals["signal"] == "buy"]
        sell = signals[signals["signal"] == "sell"]
        if len(buy) > 0:
            fig.add_trace(
                go.Scatter(
                    x=signal_x.loc[buy.index],
                    y=buy["price"],
                    mode="markers",
                    name="预测买点",
                    marker={"symbol": "triangle-up", "size": 13, "color": "#00cc66", "line": {"width": 1, "color": "white"}},
                    customdata=buy[["confidence", "factor_name", "reason"]],
                    hovertemplate="买点 %{x}<br>价格 %{y:.2f}<br>置信度 %{customdata[0]}<br>来源 %{customdata[1]}<br>%{customdata[2]}<extra></extra>",
                ),
                row=1,
                col=1,
            )
        if len(sell) > 0:
            fig.add_trace(
                go.Scatter(
                    x=signal_x.loc[sell.index],
                    y=sell["price"],
                    mode="markers",
                    name="预测卖点",
                    marker={"symbol": "triangle-down", "size": 13, "color": "#ff3333", "line": {"width": 1, "color": "white"}},
                    customdata=sell[["confidence", "factor_name", "reason"]],
                    hovertemplate="卖点 %{x}<br>价格 %{y:.2f}<br>置信度 %{customdata[0]}<br>来源 %{customdata[1]}<br>%{customdata[2]}<extra></extra>",
                ),
                row=1,
                col=1,
            )
    fig.add_trace(
        go.Bar(x=x_values, y=frame["volume"], name="成交量", marker_color=candle_colors),
        row=2,
        col=1,
    )
    if indicator == "MACD":
        fig.add_trace(go.Bar(x=x_values, y=frame["macd_hist"], name="MACD柱", marker_color=hist_colors), row=3, col=1)
        fig.add_trace(go.Scatter(x=x_values, y=frame["macd"], name="DIF", line={"color": "#1f77b4"}), row=3, col=1)
        fig.add_trace(go.Scatter(x=x_values, y=frame["macd_signal"], name="DEA", line={"color": "#ff7f0e"}), row=3, col=1)
    elif indicator == "KDJ":
        fig.add_trace(go.Scatter(x=x_values, y=frame["kdj_k"], name="K", line={"color": "#1f77b4"}), row=3, col=1)
        fig.add_trace(go.Scatter(x=x_values, y=frame["kdj_d"], name="D", line={"color": "#ff7f0e"}), row=3, col=1)
        fig.add_trace(go.Scatter(x=x_values, y=frame["kdj_j"], name="J", line={"color": "#9467bd"}), row=3, col=1)
    elif indicator == "RSI":
        fig.add_trace(go.Scatter(x=x_values, y=frame["rsi_6"], name="RSI6", line={"color": "#1f77b4"}), row=3, col=1)
        fig.add_trace(go.Scatter(x=x_values, y=frame["rsi_12"], name="RSI12", line={"color": "#ff7f0e"}), row=3, col=1)
        fig.add_trace(go.Scatter(x=x_values, y=frame["rsi_24"], name="RSI24", line={"color": "#9467bd"}), row=3, col=1)
    elif indicator == "DMI":
        fig.add_trace(go.Scatter(x=x_values, y=frame["dmi_pdi"], name="PDI", line={"color": "#2ca02c"}), row=3, col=1)
        fig.add_trace(go.Scatter(x=x_values, y=frame["dmi_mdi"], name="MDI", line={"color": "#d62728"}), row=3, col=1)
        fig.add_trace(go.Scatter(x=x_values, y=frame["dmi_adx"], name="ADX", line={"color": "#1f77b4"}), row=3, col=1)
        fig.add_trace(go.Scatter(x=x_values, y=frame["dmi_adxr"], name="ADXR", line={"color": "#ff7f0e"}), row=3, col=1)
    elif indicator == "威廉指标":
        fig.add_trace(go.Scatter(x=x_values, y=frame["wr_10"], name="WR10", line={"color": "#1f77b4"}), row=3, col=1)
        fig.add_trace(go.Scatter(x=x_values, y=frame["wr_20"], name="WR20", line={"color": "#ff7f0e"}), row=3, col=1)

    fig.update_layout(
        height=780,
        margin={"l": 20, "r": 20, "t": 70, "b": 20},
        hovermode="x unified",
        showlegend=True,
    )
    fig.update_xaxes(rangeslider_visible=False, type="category")
    fig.update_yaxes(title_text="价格", row=1, col=1)
    fig.update_yaxes(title_text="成交量", row=2, col=1)
    fig.update_yaxes(title_text=indicator, row=3, col=1)
    st.plotly_chart(fig, use_container_width=True)
    _render_technical_score_table(score_source, title)
    _render_trade_signal_table(signals, title)


def _render_stock_technical_panel(
    *,
    header: str = "个股K线 / 成交量 / MACD",
    caption: str = "输入任意 Yahoo Finance 可识别的美股代码，展示日线、周线、月线；图表已预留因子预测买点/卖点叠加接口。",
    default_ticker: str = "AAPL",
    key_prefix: str = "technical",
    fetch_history_fn: Any = _fetch_stock_history_cached,
    show_ticker_input: bool = True,
    selected_ticker: str | None = None,
) -> None:
    st.header(header)
    st.caption(caption)
    widget_default_ticker = st.session_state.get(f"{key_prefix}_ticker", default_ticker)

    with st.container(border=True):
        today = pd.Timestamp(_today_date())
        default_day_start = (today - pd.Timedelta(days=30)).date()
        default_week_start = (today - pd.DateOffset(years=1)).date()
        default_month_start = (today - pd.DateOffset(years=5)).date()

        if show_ticker_input:
            c1, c2 = st.columns([1.2, 1])
            with c1:
                ticker = st.text_input("股票代码", value=widget_default_ticker, key=f"{key_prefix}_ticker").strip().upper()
            with c2:
                end = st.date_input("K线结束日期", value=_today_date(), key=f"{key_prefix}_end")
        else:
            ticker = str(selected_ticker or widget_default_ticker).strip().upper()
            c1, c2 = st.columns([1.2, 1])
            with c1:
                st.text_input("股票代码", value=ticker, disabled=True, key=f"{key_prefix}_ticker_display")
            with c2:
                end = st.date_input("K线结束日期", value=_today_date(), key=f"{key_prefix}_end")

        d1, d2, d3 = st.columns(3)
        with d1:
            day_start = st.date_input("日线开始日期", value=default_day_start, key=f"{key_prefix}_day_start")
        with d2:
            week_start = st.date_input("周线开始日期", value=default_week_start, key=f"{key_prefix}_week_start")
        with d3:
            month_start = st.date_input("月线开始日期", value=default_month_start, key=f"{key_prefix}_month_start")

        st.markdown("#### 均线设置")
        ma_col1, ma_col2 = st.columns([1, 2])
        with ma_col1:
            enable_custom_ma = st.checkbox("显示自定义N日均线", value=False, key=f"{key_prefix}_enable_custom_ma")
        with ma_col2:
            custom_ma_raw = st.text_input(
                "自定义N日均线（可填一个或多个，用逗号分隔）",
                value="60",
                key=f"{key_prefix}_custom_ma",
                disabled=not enable_custom_ma,
            )

        ma_windows = [5, 10, 20, 120]
        if enable_custom_ma:
            try:
                custom_ma = [
                    int(x.strip())
                    for x in custom_ma_raw.split(",")
                    if x.strip()
                ]
                ma_windows.extend([x for x in custom_ma if x > 0])
            except ValueError:
                st.warning("自定义均线请输入正整数，例如：60 或 30,60,250。")
                return

        st.markdown("#### 指标设置")
        ind_col1, ind_col2 = st.columns([1, 1])
        with ind_col1:
            indicator = st.selectbox(
                "下方技术指标",
                options=["MACD", "KDJ", "RSI", "DMI", "威廉指标"],
                index=0,
                key=f"{key_prefix}_indicator",
            )
        with ind_col2:
            show_bollinger = st.checkbox("在蜡烛图显示布林带", value=False, key=f"{key_prefix}_show_bollinger")

        if not ticker:
            st.info("请输入股票代码。")
            return
        starts = {
            "日线": pd.Timestamp(day_start),
            "周线": pd.Timestamp(week_start),
            "月线": pd.Timestamp(month_start),
        }
        invalid_starts = [name for name, value in starts.items() if value > pd.Timestamp(end)]
        if invalid_starts:
            st.warning(f"{'、'.join(invalid_starts)}开始日期不能晚于结束日期。")
            return
        fetch_start = min(starts.values()).date().isoformat()

        try:
            with st.spinner(f"加载 {ticker} 行情中..."):
                full_daily = fetch_history_fn(ticker, fetch_start, str(end))
        except Exception as exc:  # pylint: disable=broad-except
            st.error(f"加载 {ticker} 行情失败：{exc}")
            return

        if full_daily.empty:
            st.warning(f"{ticker} 在所选日期区间没有行情数据。")
            return

        daily = full_daily.loc[full_daily.index >= starts["日线"]]
        weekly_source = full_daily.loc[full_daily.index >= starts["周线"]]
        monthly_source = full_daily.loc[full_daily.index >= starts["月线"]]

        latest = full_daily.iloc[-1]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("最新日期", pd.Timestamp(full_daily.index[-1]).date().isoformat())
        m2.metric("收盘价", _metric(latest["close"], 2))
        m3.metric("日线区间涨跌幅", _metric(daily["close"].iloc[-1] / daily["close"].iloc[0] - 1.0, 4) if len(daily) else "NaN")
        m4.metric("最新成交量", f"{float(latest['volume']):,.0f}")

        with st.expander("因子买卖点预测接口说明", expanded=False):
            st.write(
                "后续把因子/模型信号接入 `_predict_factor_trade_points(...)` 即可在K线上叠加预测买点和卖点。"
            )
            st.code(
                "columns = ['datetime', 'signal', 'price', 'confidence', 'factor_name', 'reason']\n"
                "signal 取值: 'buy' 或 'sell'",
                language="python",
            )

        weekly = _resample_ohlcv(weekly_source, "W-FRI")
        monthly = _resample_ohlcv(monthly_source, "ME")
        day_signals = _predict_factor_trade_points(ticker=ticker, ohlcv=daily, timeframe="1d")
        week_signals = _predict_factor_trade_points(ticker=ticker, ohlcv=weekly, timeframe="1w")
        month_signals = _predict_factor_trade_points(ticker=ticker, ohlcv=monthly, timeframe="1mo")

        tab_day, tab_week, tab_month = st.tabs(["日线", "周线", "月线"])
        with tab_day:
            _render_ohlcv_macd_chart(
                daily,
                ticker=ticker,
                title="日线",
                trade_signals=day_signals,
                ma_windows=ma_windows,
                indicator=indicator,
                show_bollinger=show_bollinger,
            )
        with tab_week:
            _render_ohlcv_macd_chart(
                weekly,
                ticker=ticker,
                title="周线",
                trade_signals=week_signals,
                ma_windows=ma_windows,
                indicator=indicator,
                show_bollinger=show_bollinger,
            )
        with tab_month:
            _render_ohlcv_macd_chart(
                monthly,
                ticker=ticker,
                title="月线",
                trade_signals=month_signals,
                ma_windows=ma_windows,
                indicator=indicator,
                show_bollinger=show_bollinger,
            )


def _first_present(mapping: dict[str, Any], names: list[str]) -> Any:
    for name in names:
        value = mapping.get(name)
        if value is not None and str(value).strip() not in {"", "nan", "None"}:
            return value
    return None


def _compact_value(value: Any, *, digits: int = 2) -> str:
    if value is None:
        return "-"
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return "-"
    try:
        return _metric(value, digits)
    except Exception:  # pylint: disable=broad-except
        return text


def _small_info_card(label: str, value: Any) -> None:
    st.markdown(
        f"""
        <div style="padding:6px 0;">
          <div style="font-size:12px;color:#6b7280;line-height:1.1;">{label}</div>
          <div style="font-size:15px;font-weight:600;line-height:1.25;word-break:break-word;">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_cn_stock_info_panel(ticker: str) -> None:
    st.header("个股信息")
    try:
        with st.spinner(f"加载 {ticker} 个股信息中..."):
            info = _fetch_cn_stock_info_cached(ticker)
    except Exception as exc:  # pylint: disable=broad-except
        st.warning(f"个股信息加载失败：{exc}")
        return

    individual = info.get("individual", {})
    basic = info.get("basic", {})
    business_info = info.get("business", {})
    valuation = info.get("valuation_latest", {})
    spot = info.get("spot_latest", {})

    display_name = (
        _first_present(individual, ["股票简称", "股票名称", "简称", "名称", "security_name", "name"])
        or spot.get("名称")
        or _first_present(business_info, ["股票简称", "股票名称", "名称"])
        or ticker
    )
    industry = (
        _first_present(individual, ["行业", "所处行业", "板块", "industry", "所属行业"])
        or _first_present(basic, ["所属行业", "行业"])
        or spot.get("所处行业")
    )
    listing_date = _first_present(individual, ["上市时间", "上市日期", "上市日期 ", "上市日"]) or _first_present(basic, ["上市日期"])
    total_mv = _first_present(individual, ["总市值"]) or valuation.get("总市值") or spot.get("总市值") or spot.get("资产净值/总市值")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        _small_info_card("股票名称", str(display_name))
    with c2:
        _small_info_card("股票代码", info.get("symbol", ticker))
    with c3:
        _small_info_card("行业/板块", str(industry or "-"))
    with c4:
        _small_info_card("上市日期", str(listing_date or "-"))

    v1, v2, v3, v4 = st.columns(4)
    pe = _first_present(
        valuation,
        ["PE(TTM)", "pe", "pe_ttm", "peTTM", "pe_ttm_lyr", "市盈率", "市盈率TTM", "市盈率-动态", "市盈率(TTM)", "市盈率(动)"],
    ) or _first_present(spot, ["市盈率(TTM)", "市盈率(动)", "市盈率-动态"])
    pb = _first_present(valuation, ["pb", "pb_mrq", "市净率", "市净率(MRQ)"]) or _first_present(spot, ["市净率", "资产净值/总市值"])
    ps = _first_present(valuation, ["ps", "ps_ttm", "psTTM", "市销率", "市销率TTM"])
    dividend = _first_present(valuation, ["dv_ratio", "dv_ttm", "股息率", "股息率TTM", "dividend_yield"]) or spot.get("股息率(TTM)")
    with v1:
        _small_info_card("市盈率", _compact_value(pe))
    with v2:
        _small_info_card("市净率", _compact_value(pb))
    with v3:
        _small_info_card("市销率", _compact_value(ps))
    with v4:
        _small_info_card("股息率", _compact_value(dividend))

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        _small_info_card("总市值", _format_cn_market_value(total_mv or spot.get("总市值")))
    with m2:
        _small_info_card("流通市值", _format_cn_market_value(_first_present(individual, ["流通市值"]) or valuation.get("流通市值") or spot.get("流通市值") or spot.get("流通值")))
    with m3:
        _small_info_card("最新价", str(spot.get("最新价", spot.get("现价", "-"))))
    with m4:
        _small_info_card("涨跌幅", str(spot.get("涨跌幅", spot.get("涨幅", "-"))))

    left, right = st.columns(2)
    with left:
        st.markdown("#### 公司概况")
        overview_rows = []
        for label, keys in {
            "公司名称": ["公司名称", "公司全称", "org_name_cn"],
            "英文名称": ["英文名称", "org_name_en"],
            "注册地址": ["注册地址", "reg_address_cn"],
            "办公地址": ["办公地址", "office_address_cn"],
            "法人代表": ["法人代表", "法定代表人", "legal_representative"],
            "董事长": ["董事长", "chairman"],
            "总经理": ["总经理", "manager"],
            "所属地域": ["地域", "所在区域", "省份", "province"],
        }.items():
            value = _first_present(basic, keys) or _first_present(individual, keys)
            if value is not None:
                overview_rows.append({"项目": label, "内容": value})
        if overview_rows:
            st.dataframe(pd.DataFrame(overview_rows), use_container_width=True, hide_index=True)
        else:
            st.info("当前 AKShare 接口未返回公司概况。")

    with right:
        st.markdown("#### 业务与板块")
        business = (
            _first_present(basic, ["主营业务", "经营范围", "公司简介", "main_operation_business", "operating_scope", "org_cn_introduction"])
            or _first_present(individual, ["主营业务", "经营范围"])
            or _first_present(business_info, ["主营业务", "经营范围", "业务范围", "产品类型", "产品名称", "主营构成"])
        )
        if business is not None:
            rows = []
            for label in ["主营业务", "产品类型", "产品名称", "经营范围"]:
                if business_info.get(label) is not None:
                    rows.append({"项目": label, "内容": business_info[label]})
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            else:
                st.write(str(business))
        else:
            st.info("当前 AKShare 接口未返回主营业务/经营范围。")

    with st.expander("AKShare 原始字段", expanded=False):
        tabs = st.tabs(["个股基础", "公司概况", "估值指标", "实时行情", "接口提示"])
        with tabs[0]:
            st.json(individual)
        with tabs[1]:
            st.json(basic)
        with tabs[2]:
            st.json({str(k): str(v) for k, v in valuation.items()})
        with tabs[3]:
            st.json({str(k): str(v) for k, v in spot.items()})
        with tabs[4]:
            st.json({"business": {str(k): str(v) for k, v in business_info.items()}, "errors": info.get("errors", {})})


def _build_config_from_ui(*, market: str = "US") -> tuple[USAlphaConfig, EvolutionUIConfig | None, int, dict[str, str]]:
    cfg = USAlphaConfig()

    with st.sidebar:
        st.header("运行参数")
        default_tickers = "600000,000001,300750,600519,000858" if market == "CN" else ",".join(cfg.data.tickers)
        tickers_raw = st.text_area(
            "股票池（逗号分隔）",
            value=default_tickers,
            height=80,
        )
        if market == "CN":
            st.caption("A股代码示例：600000,000001,300750。为控制运行时间，建议先使用 5-20 只股票。")
            cfg.data.tickers = [_normalize_cn_symbol(x).zfill(6) for x in tickers_raw.split(",") if x.strip()]
            cfg.data.max_tickers = int(
                st.number_input(
                    "最大股票数（防止OOM/断连）",
                    min_value=3,
                    max_value=200,
                    value=min(max(len(cfg.data.tickers), 5), 40),
                    step=5,
                )
            )
        else:
            st.caption("默认 `NASDAQ_ALL` 表示自动拉取纳斯达克全部股票代码（带本地缓存）。")
            cfg.data.tickers = [x.strip().upper() for x in tickers_raw.split(",") if x.strip()]
            cfg.data.max_tickers = int(
                st.number_input(
                    "最大股票数（防止OOM/断连）",
                    min_value=10,
                    max_value=500,
                    value=int(cfg.data.max_tickers),
                    step=10,
                )
            )

        c1, c2 = st.columns(2)
        with c1:
            cfg.data.start = str(st.date_input("开始日期", value=pd.to_datetime(cfg.data.start).date()))
        with c2:
            cfg.data.end = str(st.date_input("结束日期", value=_today_date()))

        st.markdown("### 日期控制")
        tc1, tc2 = st.columns(2)
        with tc1:
            cfg.train.train_start = str(
                st.date_input("训练开始日期", value=pd.to_datetime(cfg.train.train_start).date())
            )
        with tc2:
            cfg.train.train_end = str(
                st.date_input("训练结束日期", value=pd.to_datetime(cfg.train.train_end).date())
            )

        bc1, bc2 = st.columns(2)
        with bc1:
            backtest_start = str(
                st.date_input("回测开始日期", value=pd.to_datetime(cfg.train.train_end).date())
            )
        with bc2:
            backtest_end = str(
                st.date_input("回测结束日期", value=_today_date())
            )

        predict_asof = str(
            st.date_input("预测日期(按该日收盘后给次日建议)", value=_next_bday_date())
        )

        if market == "CN":
            cfg.data.benchmark = _normalize_cn_symbol(
                st.text_input("基准/对照代码", value="000001", help="当前先使用A股代码作为基准行情来源。")
            ).zfill(6)
        else:
            benchmark_options = {
                "^IXIC": "纳斯达克综合指数 (^IXIC)",
                "^GSPC": "标普500 (^GSPC)",
                "^DJI": "道琼斯 (^DJI)",
                "SPY": "SPDR S&P500 ETF (SPY)",
                "QQQ": "纳斯达克100 ETF (QQQ)",
            }
            default_bench_key = cfg.data.benchmark if cfg.data.benchmark in benchmark_options else "^IXIC"
            selected_bench = st.selectbox(
                "基准",
                options=list(benchmark_options.keys()),
                format_func=lambda k: benchmark_options[k],
                index=list(benchmark_options.keys()).index(default_bench_key),
            )
            cfg.data.benchmark = selected_bench
        cfg.train.label_horizon = int(st.number_input("预测步长(天)", min_value=1, max_value=10, value=cfg.train.label_horizon))
        cfg.train.top_quantile = float(st.slider("回测多空分位", min_value=0.05, max_value=0.45, value=cfg.train.top_quantile, step=0.05))

        st.divider()
        st.subheader("新因子演进")
        enable_evolution = st.checkbox("启用LLM新因子挖掘", value=True)

        evo_cfg = None
        if enable_evolution:
            api_key = st.text_input("GLM API Key", type="password")
            model = st.text_input("模型", value="glm-5")
            temperature = float(st.slider("温度", min_value=0.0, max_value=1.5, value=0.8, step=0.1))
            num_candidates = int(st.number_input("候选因子数", min_value=10, max_value=200, value=72, step=2))
            top_k_accept = int(st.number_input("入库数量", min_value=1, max_value=50, value=12))
            require_glm = st.checkbox("必须真实GLM输出（失败则重试）", value=False)
            max_retries = int(st.number_input("最大重试轮次(0=无限)", min_value=0, max_value=100, value=6))
            retry_wait_sec = int(st.number_input("重试间隔(秒)", min_value=1, max_value=300, value=20))
            feedback = st.text_area("历史反馈（可选）", value="")

            evo_cfg = EvolutionUIConfig(
                api_key=api_key,
                model=model,
                temperature=temperature,
                num_candidates=num_candidates,
                top_k_accept=top_k_accept,
                require_glm_api=require_glm,
                max_retries=max_retries,
                retry_wait_sec=retry_wait_sec,
                history_feedback=feedback or None,
            )

        top_stock_k = int(st.number_input("明日候选股票数量", min_value=3, max_value=30, value=10))

    date_controls = {
        "backtest_start": backtest_start,
        "backtest_end": backtest_end,
        "predict_asof_date": predict_asof,
    }
    return cfg, evo_cfg, top_stock_k, date_controls


def _render_backtest(daily: pd.DataFrame, benchmark_daily: pd.DataFrame | None = None) -> None:
    st.subheader("回测曲线")
    if daily is None or len(daily) == 0:
        st.warning("无回测数据")
        return

    view = daily.reset_index()

    # --- cumulative NAV dual-line chart ---
    has_benchmark = "cum_benchmark" in daily.columns and daily["cum_benchmark"].notna().any()
    if go is not None:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=view["datetime"], y=view["cum_ls"],
            mode="lines", name="策略 Long-Short",
        ))
        if has_benchmark:
            fig.add_trace(go.Scatter(
                x=view["datetime"], y=view["cum_benchmark"],
                mode="lines", name="基准 (^IXIC)",
            ))
        fig.update_layout(title="累计净值（策略 vs 基准）", xaxis_title="日期", yaxis_title="累计净值")
        st.plotly_chart(fig, use_container_width=True)
    elif px is not None:
        fig = px.line(view, x="datetime", y="cum_ls", title="Long-Short 累计净值")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.line_chart(view.set_index("datetime")["cum_ls"])

    if px is not None:
        fig2 = px.bar(view, x="datetime", y="ls_ret", title="日度多空收益")
        st.plotly_chart(fig2, use_container_width=True)

        fig3 = px.line(view, x="datetime", y="ic", title="日度IC")
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.bar_chart(view.set_index("datetime")["ls_ret"])
        st.line_chart(view.set_index("datetime")["ic"])

    # --- holdings panel ---
    _render_holdings_panel(daily)


def _render_holdings_panel(daily: pd.DataFrame) -> None:
    st.subheader("持仓明细（点击查看某日持仓）")
    if "long_stocks" not in daily.columns and "short_stocks" not in daily.columns:
        st.info("持仓明细数据不可用")
        return

    dates = [str(d.date()) for d in daily.index]
    if not dates:
        st.info("无可用日期")
        return

    selected = st.selectbox("选择日期", dates, key="holdings_date_select")
    if not selected:
        return

    row = daily.loc[pd.Timestamp(selected)]

    col_l, col_s = st.columns(2)
    with col_l:
        st.markdown("#### 多头持仓")
        longs = row.get("long_stocks")
        if longs and isinstance(longs, list) and len(longs) > 0:
            df_long = pd.DataFrame(longs)
            df_long = df_long.sort_values("pred", ascending=False).reset_index(drop=True)
            st.dataframe(df_long, use_container_width=True, hide_index=True)
        else:
            st.info("该日无多头持仓数据")

    with col_s:
        st.markdown("#### 空头持仓")
        shorts = row.get("short_stocks")
        if shorts and isinstance(shorts, list) and len(shorts) > 0:
            df_short = pd.DataFrame(shorts)
            df_short = df_short.sort_values("pred", ascending=True).reset_index(drop=True)
            st.dataframe(df_short, use_container_width=True, hide_index=True)
        else:
            st.info("该日无空头持仓数据")


def _technical_strategy_one_scores(
    data_by_symbol: dict[str, pd.DataFrame],
    asof_date: pd.Timestamp,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    asof_ts = pd.Timestamp(asof_date).normalize()
    for symbol, frame in data_by_symbol.items():
        history = frame.loc[frame.index <= asof_ts].copy()
        if len(history) < 2:
            continue
        scores = _score_technical_indicators(history[["open", "high", "low", "close", "volume"]])
        if scores.empty:
            continue
        hit = scores[scores["指标"] == "综合"]
        if hit.empty:
            continue
        score = pd.to_numeric(hit.iloc[0]["分数"], errors="coerce")
        if pd.isna(score):
            continue
        rows.append(
            {
                "symbol": symbol,
                "score": float(score),
                "signal": str(hit.iloc[0]["信号"]),
                "asof_date": asof_ts,
            }
        )
    if not rows:
        return pd.DataFrame(columns=["symbol", "score", "signal", "asof_date"])
    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)


def _strategy_one_selector(
    data_by_symbol: dict[str, pd.DataFrame],
    asof_date: pd.Timestamp,
    *,
    top_n: int = 10,
) -> pd.DataFrame:
    scores = _technical_strategy_one_scores(data_by_symbol, asof_date)
    if scores.empty:
        return scores
    return scores.head(int(top_n)).copy()


STRATEGY_REGISTRY = {
    "策略一：技术指标平均分": _strategy_one_selector,
}


def _build_backtest_matrices(data_by_symbol: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    open_prices: dict[str, pd.Series] = {}
    close_prices: dict[str, pd.Series] = {}
    scores: dict[str, pd.Series] = {}
    returns: dict[str, pd.Series] = {}

    for symbol, frame in data_by_symbol.items():
        if frame.empty:
            continue
        clean = frame.sort_index()
        open_series = pd.to_numeric(clean["open"], errors="coerce").astype("float32")
        close_series = pd.to_numeric(clean["close"], errors="coerce").astype("float32")
        score_series = _technical_strategy_one_score_series(clean).astype("float32")
        ret_series = (close_series / open_series - 1.0).astype("float32")
        open_prices[symbol] = open_series
        close_prices[symbol] = close_series
        scores[symbol] = score_series
        returns[symbol] = ret_series

    return {
        "open": pd.DataFrame(open_prices, dtype="float32").sort_index(),
        "close": pd.DataFrame(close_prices, dtype="float32").sort_index(),
        "score": pd.DataFrame(scores, dtype="float32").sort_index(),
        "intraday_return": pd.DataFrame(returns, dtype="float32").sort_index(),
    }


def _select_top_from_score_matrix(score_matrix: pd.DataFrame, asof_date: pd.Timestamp, top_n: int) -> pd.DataFrame:
    ts = pd.Timestamp(asof_date).normalize()
    if score_matrix.empty:
        return pd.DataFrame(columns=["symbol", "score", "signal", "asof_date"])
    available = score_matrix.loc[score_matrix.index <= ts]
    if available.empty:
        return pd.DataFrame(columns=["symbol", "score", "signal", "asof_date"])
    row = available.iloc[-1].dropna().nlargest(int(top_n))
    return pd.DataFrame(
        {
            "symbol": row.index.astype(str),
            "score": row.to_numpy(dtype=float),
            "signal": [_score_to_action(float(x)) for x in row.to_numpy(dtype=float)],
            "asof_date": available.index[-1],
        }
    )


def _price_on_date(frame: pd.DataFrame, date: pd.Timestamp, field: str) -> float | None:
    ts = pd.Timestamp(date).normalize()
    if ts not in frame.index or field not in frame.columns:
        return None
    value = pd.to_numeric(frame.loc[ts, field], errors="coerce")
    if pd.isna(value) or float(value) <= 0:
        return None
    return float(value)


def _matrix_price(prices: pd.DataFrame, symbol: str, date: pd.Timestamp) -> float | None:
    ts = pd.Timestamp(date).normalize()
    if prices.empty or ts not in prices.index or symbol not in prices.columns:
        return None
    value = pd.to_numeric(prices.at[ts, symbol], errors="coerce")
    if pd.isna(value) or float(value) <= 0:
        return None
    return float(value)


def _rebalance_equal_weight_lot100_matrix(
    *,
    prices: pd.DataFrame,
    positions: dict[str, int],
    cash: float,
    trade_date: pd.Timestamp,
    selected: list[str],
) -> tuple[dict[str, int], float, list[dict[str, Any]]]:
    trades: list[dict[str, Any]] = []
    new_cash = float(cash)

    for symbol, shares in list(positions.items()):
        price = _matrix_price(prices, symbol, trade_date)
        if price is None or shares <= 0:
            continue
        new_cash += shares * price
        trades.append(
            {
                "date": pd.Timestamp(trade_date),
                "symbol": symbol,
                "side": "sell",
                "shares": int(shares),
                "price": price,
                "turnover": shares * price,
            }
        )

    buy_candidates = [(symbol, _matrix_price(prices, symbol, trade_date)) for symbol in selected]
    buy_candidates = [(symbol, price) for symbol, price in buy_candidates if price is not None]
    next_positions: dict[str, int] = {}
    if buy_candidates:
        budget = new_cash / len(buy_candidates)
        for symbol, price in buy_candidates:
            shares = int((budget // (price * 100)) * 100)
            if shares <= 0:
                continue
            cost = shares * price
            if cost > new_cash:
                shares = int((new_cash // (price * 100)) * 100)
                cost = shares * price
            if shares <= 0:
                continue
            new_cash -= cost
            next_positions[symbol] = next_positions.get(symbol, 0) + shares
            trades.append(
                {
                    "date": pd.Timestamp(trade_date),
                    "symbol": symbol,
                    "side": "buy",
                    "shares": int(shares),
                    "price": price,
                    "turnover": cost,
                }
            )

    return next_positions, new_cash, trades


def _value_portfolio_matrix(
    prices: pd.DataFrame,
    positions: dict[str, int],
    cash: float,
    date: pd.Timestamp,
) -> float:
    value = float(cash)
    for symbol, shares in positions.items():
        price = _matrix_price(prices, symbol, date)
        if price is not None:
            value += shares * price
    return value


def _rebalance_equal_weight_lot100(
    *,
    data_by_symbol: dict[str, pd.DataFrame],
    positions: dict[str, int],
    cash: float,
    trade_date: pd.Timestamp,
    selected: list[str],
    price_field: str,
) -> tuple[dict[str, int], float, list[dict[str, Any]]]:
    trades: list[dict[str, Any]] = []
    new_cash = float(cash)

    for symbol, shares in list(positions.items()):
        price = _price_on_date(data_by_symbol.get(symbol, pd.DataFrame()), trade_date, price_field)
        if price is None or shares <= 0:
            continue
        new_cash += shares * price
        trades.append(
            {
                "date": pd.Timestamp(trade_date),
                "symbol": symbol,
                "side": "sell",
                "shares": int(shares),
                "price": price,
                "turnover": shares * price,
                "session": "open" if price_field == "open" else "close",
            }
        )

    buy_candidates: list[tuple[str, float]] = []
    for symbol in selected:
        price = _price_on_date(data_by_symbol.get(symbol, pd.DataFrame()), trade_date, price_field)
        if price is not None:
            buy_candidates.append((symbol, price))

    next_positions: dict[str, int] = {}
    if buy_candidates:
        budget = new_cash / len(buy_candidates)
        for symbol, price in buy_candidates:
            shares = int((budget // (price * 100)) * 100)
            if shares <= 0:
                continue
            cost = shares * price
            if cost > new_cash:
                shares = int((new_cash // (price * 100)) * 100)
                cost = shares * price
            if shares <= 0:
                continue
            new_cash -= cost
            next_positions[symbol] = next_positions.get(symbol, 0) + shares
            trades.append(
                {
                    "date": pd.Timestamp(trade_date),
                    "symbol": symbol,
                    "side": "buy",
                    "shares": int(shares),
                    "price": price,
                    "turnover": cost,
                    "session": "open" if price_field == "open" else "close",
                }
            )

    return next_positions, new_cash, trades


def _value_portfolio(
    data_by_symbol: dict[str, pd.DataFrame],
    positions: dict[str, int],
    cash: float,
    date: pd.Timestamp,
    price_field: str,
) -> float:
    value = float(cash)
    for symbol, shares in positions.items():
        price = _price_on_date(data_by_symbol.get(symbol, pd.DataFrame()), date, price_field)
        if price is not None:
            value += shares * price
    return value


def _spearman_corr(left: pd.Series, right: pd.Series) -> float:
    pair = pd.concat([left.rename("left"), right.rename("right")], axis=1).dropna()
    if len(pair) < 2 or pair["left"].nunique() < 2 or pair["right"].nunique() < 2:
        return float("nan")
    ranked = pair.rank(method="average")
    return float(ranked["left"].corr(ranked["right"]))


def _mean_spearman_ic(rows: list[dict[str, Any]]) -> tuple[float, pd.DataFrame]:
    ic_rows: list[dict[str, Any]] = []
    for row in rows:
        scores = row.get("scores")
        returns = row.get("returns")
        if not isinstance(scores, pd.Series) or not isinstance(returns, pd.Series):
            continue
        pair = pd.concat([scores.rename("score"), returns.rename("return")], axis=1).dropna()
        if len(pair) < 3 or pair["score"].nunique() < 2 or pair["return"].nunique() < 2:
            continue
        ic_rows.append({"date": row["date"], "ic": _spearman_corr(pair["score"], pair["return"])})
    ic_daily = pd.DataFrame(ic_rows)
    if ic_daily.empty:
        return float("nan"), ic_daily
    return float(ic_daily["ic"].mean()), ic_daily


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return float("nan")
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    return float(drawdown.min())


def _run_cn_backtest_strategy_one_mode_one(
    *,
    data_by_symbol: dict[str, pd.DataFrame],
    benchmark: pd.DataFrame,
    initial_cash: float,
    start: str,
    end: str,
    strategy_fn: Any = _strategy_one_selector,
    top_n: int = 10,
) -> dict[str, Any]:
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    matrices = _build_backtest_matrices(data_by_symbol)
    open_matrix = matrices["open"]
    close_matrix = matrices["close"]
    score_matrix = matrices["score"]
    intraday_return_matrix = matrices["intraday_return"]
    dates = list(close_matrix.loc[(close_matrix.index >= start_ts) & (close_matrix.index <= end_ts)].index)
    if not dates:
        raise ValueError("回测区间内没有可交易行情。")

    positions: dict[str, int] = {}
    cash = float(initial_cash)
    daily_rows: list[dict[str, Any]] = []
    all_trades: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    _ = strategy_fn

    for idx, trade_date in enumerate(dates):
        open_asof = dates[idx - 1] if idx > 0 else trade_date
        open_scores = _select_top_from_score_matrix(score_matrix, open_asof, top_n)
        open_selected = open_scores["symbol"].tolist()

        positions, cash, open_trades = _rebalance_equal_weight_lot100_matrix(
            prices=open_matrix,
            positions=positions,
            cash=cash,
            trade_date=trade_date,
            selected=open_selected,
        )
        for trade in open_trades:
            trade["session"] = "open"
        all_trades.extend(open_trades)

        close_equity_before_rebalance = _value_portfolio_matrix(close_matrix, positions, cash, trade_date)
        close_scores = _select_top_from_score_matrix(score_matrix, trade_date, top_n)
        close_selected = close_scores["symbol"].tolist()
        positions, cash, close_trades = _rebalance_equal_weight_lot100_matrix(
            prices=close_matrix,
            positions=positions,
            cash=cash,
            trade_date=trade_date,
            selected=close_selected,
        )
        for trade in close_trades:
            trade["session"] = "close"
        all_trades.extend(close_trades)
        close_equity = _value_portfolio_matrix(close_matrix, positions, cash, trade_date)

        holdings = [
            {
                "symbol": symbol,
                "shares": int(shares),
                "close": _matrix_price(close_matrix, symbol, trade_date),
                "market_value": int(shares) * (_matrix_price(close_matrix, symbol, trade_date) or 0.0),
            }
            for symbol, shares in positions.items()
            if shares > 0
        ]
        daily_rows.append(
            {
                "datetime": trade_date,
                "equity": close_equity,
                "cash": cash,
                "open_selected": open_selected,
                "close_selected": close_selected,
                "holdings": holdings,
                "close_equity_before_close_rebalance": close_equity_before_rebalance,
            }
        )

        for rank, item in close_scores.head(top_n).reset_index(drop=True).iterrows():
            selections.append(
                {
                    "date": trade_date,
                    "rank": int(rank) + 1,
                    "symbol": item["symbol"],
                    "score": item["score"],
                    "signal": item["signal"],
                }
            )

    ic_rows: list[dict[str, Any]] = []
    common_dates = score_matrix.index.intersection(intraday_return_matrix.index)
    for date in common_dates[(common_dates >= start_ts) & (common_dates <= end_ts)]:
        pair = pd.concat(
            [score_matrix.loc[date].rename("score"), intraday_return_matrix.loc[date].rename("return")],
            axis=1,
        ).dropna()
        if len(pair) >= 3 and pair["score"].nunique() >= 2 and pair["return"].nunique() >= 2:
            ic_rows.append({"date": date, "ic": _spearman_corr(pair["score"], pair["return"])})
    ic_daily = pd.DataFrame(ic_rows)
    ic_mean = float(ic_daily["ic"].mean()) if not ic_daily.empty else float("nan")

    daily = pd.DataFrame(daily_rows).set_index("datetime").sort_index()
    daily["daily_return"] = daily["equity"].pct_change().fillna(daily["equity"] / float(initial_cash) - 1.0)
    daily["cum_strategy"] = daily["equity"] / float(initial_cash)

    benchmark_curve = benchmark.loc[(benchmark.index >= start_ts) & (benchmark.index <= end_ts)].copy()
    if not benchmark_curve.empty:
        benchmark_curve = benchmark_curve.reindex(daily.index).ffill().dropna(subset=["close"])
        if not benchmark_curve.empty:
            daily["benchmark_equity"] = benchmark_curve["close"] / float(benchmark_curve["close"].iloc[0]) * float(initial_cash)
            daily["cum_benchmark"] = daily["benchmark_equity"] / float(initial_cash)
            daily["benchmark_return"] = daily["benchmark_equity"].pct_change().fillna(0.0)

    ret = daily["daily_return"].dropna()
    sharpe = float(ret.mean() / ret.std(ddof=0) * np.sqrt(252)) if len(ret) > 1 and ret.std(ddof=0) > 0 else float("nan")
    metrics = {
        "final_equity": float(daily["equity"].iloc[-1]),
        "total_return": float(daily["cum_strategy"].iloc[-1] - 1.0),
        "benchmark_return": float(daily["cum_benchmark"].iloc[-1] - 1.0) if "cum_benchmark" in daily.columns else float("nan"),
        "sharpe": sharpe,
        "ic": ic_mean,
        "max_drawdown": _max_drawdown(daily["equity"]),
    }
    return {
        "daily": daily,
        "metrics": metrics,
        "trades": pd.DataFrame(all_trades),
        "selections": pd.DataFrame(selections),
        "ic_daily": ic_daily,
    }


TRADE_MODE_REGISTRY = {
    "交易模式一：全仓等权买入前十": _run_cn_backtest_strategy_one_mode_one,
}


def _render_standalone_backtest(result: dict[str, Any]) -> None:
    daily = result["daily"]
    metrics = result["metrics"]
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("最终资产", f"{metrics['final_equity']:,.2f}")
    c2.metric("策略收益", f"{metrics['total_return']:.2%}")
    c3.metric("沪指收益", "-" if pd.isna(metrics["benchmark_return"]) else f"{metrics['benchmark_return']:.2%}")
    c4.metric("Sharpe", _metric(metrics["sharpe"], 3))
    c5.metric("IC", _metric(metrics["ic"], 4))
    c6.metric("最大回撤", "-" if pd.isna(metrics["max_drawdown"]) else f"{metrics['max_drawdown']:.2%}")

    view = daily.reset_index()
    if go is not None:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=view["datetime"], y=view["cum_strategy"], mode="lines", name="策略一 / 交易模式一"))
        if "cum_benchmark" in view.columns:
            fig.add_trace(go.Scatter(x=view["datetime"], y=view["cum_benchmark"], mode="lines", name="沪指基线"))
        fig.update_layout(title="收益曲线", xaxis_title="日期", yaxis_title="累计净值")
        st.plotly_chart(fig, use_container_width=True)
    else:
        cols = ["cum_strategy"] + (["cum_benchmark"] if "cum_benchmark" in daily.columns else [])
        st.line_chart(daily[cols])

    tabs = st.tabs(["每日选股", "持仓", "交易流水", "日度IC"])
    with tabs[0]:
        selections = result.get("selections", pd.DataFrame())
        if selections is None or selections.empty:
            st.info("暂无每日选股记录。")
        else:
            st.dataframe(selections.sort_values(["date", "rank"], ascending=[False, True]), use_container_width=True, hide_index=True)
    with tabs[1]:
        selected_date = st.selectbox(
            "选择日期",
            [str(pd.Timestamp(x).date()) for x in daily.index],
            index=len(daily.index) - 1,
            key="standalone_bt_holdings_date",
        )
        holdings = daily.loc[pd.Timestamp(selected_date)].get("holdings", [])
        if holdings:
            st.dataframe(pd.DataFrame(holdings), use_container_width=True, hide_index=True)
        else:
            st.info("该日无持仓。")
    with tabs[2]:
        trades = result.get("trades", pd.DataFrame())
        if trades is None or trades.empty:
            st.info("暂无交易流水。")
        else:
            st.dataframe(trades.sort_values("date", ascending=False), use_container_width=True, hide_index=True)
    with tabs[3]:
        ic_daily = result.get("ic_daily", pd.DataFrame())
        if ic_daily is None or ic_daily.empty:
            st.info("暂无可计算的日度IC。")
        elif px is not None:
            st.plotly_chart(px.line(ic_daily, x="date", y="ic", title="日度IC"), use_container_width=True)
        else:
            st.line_chart(ic_daily.set_index("date")["ic"])


def _render_factor_section(result) -> None:
    st.subheader("新因子挖掘")
    evo = result.evolution_result
    if evo is None:
        st.info("本次未启用LLM新因子挖掘")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("候选数", str(evo.candidate_count))
    c2.metric("有效数", str(evo.valid_count))
    c3.metric("入库数", str(evo.accepted_count))

    st.caption(f"generation_mode={evo.generation_mode}; run_dir={evo.run_dir}")

    if len(result.accepted_factors) == 0:
        st.warning("本轮没有可入库因子")
        return

    rows = []
    for item in result.accepted_factors:
        m = item.get("metrics", {})
        rows.append(
            {
                "name": item.get("name"),
                "expression": item.get("expression"),
                "source": item.get("source"),
                "score": m.get("score"),
                "ic_mean": m.get("ic_mean"),
                "sharpe": m.get("sharpe"),
                "annual_return": m.get("annual_return"),
                "max_drawdown": m.get("max_drawdown"),
            }
        )
    df = pd.DataFrame(rows).sort_values("score", ascending=False)
    st.dataframe(df, use_container_width=True)

    best = result.best_factor
    if best is None:
        return

    st.markdown("### 最优新因子参数")
    st.code(best.get("expression", ""), language="text")

    pcols = st.columns(2)
    with pcols[0]:
        st.write("解析参数")
        st.json(best.get("parsed_params", {}))
    with pcols[1]:
        st.write("评估指标")
        st.json(best.get("metrics", {}))

    sig = result.best_factor_signal
    if len(sig) > 0:
        st.markdown("### 最优因子最新截面分布")
        if px is not None:
            fig = px.histogram(sig, x="value", nbins=30, title="因子值分布")
            st.plotly_chart(fig, use_container_width=True)

            top = sig.sort_values("value", ascending=False).head(10)
            fig2 = px.bar(top, x="instrument", y="value", title="Top10 因子值")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.bar_chart(sig.set_index("instrument")["value"].head(10))


def _render_tomorrow_candidates(result) -> None:
    st.subheader("明日值得关注股票")

    st.caption(f"预测日期: {result.predict_asof_date}，下一个交易日: {result.predicted_trade_date}")

    factor_pick = result.tomorrow_candidates
    if len(factor_pick) > 0:
        direction = factor_pick["direction"].iloc[0]
        st.success(f"因子方向: {direction}。以下是因子视角下明日优先关注股票。")
        st.dataframe(factor_pick, use_container_width=True)
    else:
        st.warning("当前无因子候选股票")

    st.markdown("### 模型视角（对照）")
    st.dataframe(result.model_tomorrow_candidates, use_container_width=True)


def _render_stock_page() -> None:
    st.title("USalpha 美股股票查看")
    _render_stock_technical_panel(
        header="美股K线 / 成交量 / MACD",
        caption="输入任意 Yahoo Finance 可识别的美股代码，展示日线、周线、月线；图表已预留因子预测买点/卖点叠加接口。",
        default_ticker="AAPL",
        key_prefix="us_technical",
        fetch_history_fn=_fetch_stock_history_cached,
    )


def _render_china_stock_page() -> None:
    st.title("USalpha 中国股市股票查看")
    if "cn_selected_ticker" not in st.session_state:
        st.session_state["cn_selected_ticker"] = "600000"
    ticker = st.text_input(
        "A股股票代码",
        key="cn_selected_ticker",
        help="支持 600000、000001、300750，也支持 sh600000/sz000001。",
    ).strip().upper()
    if not ticker:
        st.info("请输入A股股票代码。")
        return

    _render_cn_stock_info_panel(ticker)
    st.divider()
    _render_stock_technical_panel(
        header="A股K线 / 成交量 / MACD",
        caption=(
            "输入A股代码，例如 600000、000001、300750，也支持 sh600000/sz000001；"
            "数据源使用 AKShare 前复权日线，功能与美股股票查看一致。"
        ),
        default_ticker="600000",
        key_prefix="cn_technical",
        fetch_history_fn=_fetch_cn_stock_history_cached,
        show_ticker_input=False,
        selected_ticker=ticker,
    )


def _render_us_factor_page() -> None:
    st.title("美国股市因子训练与挖掘")
    st.write("点击一次即可执行：数据加载 -> 526因子计算 -> 训练回测 -> 新因子挖掘 -> 明日股票建议。")

    cfg, evo_cfg, top_stock_k, date_controls = _build_config_from_ui(market="US")

    run_clicked = st.button("🚀 一键运行训练流程", use_container_width=True)

    if run_clicked:
        with st.spinner("流程运行中，请稍候..."):
            try:
                result = run_dashboard_workflow(
                    cfg,
                    evolution_cfg=evo_cfg,
                    top_stock_k=top_stock_k,
                    backtest_start=date_controls["backtest_start"],
                    backtest_end=date_controls["backtest_end"],
                    predict_asof_date=date_controls["predict_asof_date"],
                )
            except Exception as exc:  # pylint: disable=broad-except
                st.error(f"流程失败：{exc}")
                st.exception(exc)
                return
        st.session_state["dashboard_result"] = result

    result = st.session_state.get("dashboard_result")
    if result is None:
        st.info("请先点击“🚀 一键运行训练流程”。")
        return

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("数据模式", result.data_mode)
    m2.metric("总耗时(秒)", _metric(result.runtime_sec, 2))
    m3.metric("526因子数", str(result.factor_stats.get("factor_count", "-")))
    m4.metric("回测Sharpe", _metric(result.backtest_result.metrics.get("sharpe", float("nan")), 3))

    m5, m6, m7, m8 = st.columns(4)
    m5.metric("Train IC", _metric(result.model_result.metrics.get("train_ic_mean", float("nan")), 4))
    m6.metric("Test IC", _metric(result.model_result.metrics.get("test_ic_mean", float("nan")), 4))
    m7.metric("Train RMSE", _metric(result.model_result.metrics.get("train_rmse", float("nan")), 5))
    m8.metric("Test RMSE", _metric(result.model_result.metrics.get("test_rmse", float("nan")), 5))

    d1, d2, d3 = st.columns(3)
    d1.metric("训练区间", f"{result.train_window['train_start']} ~ {result.train_window['train_end']}")
    d2.metric("回测区间", f"{result.backtest_window['backtest_start']} ~ {result.backtest_window['backtest_end']}")
    d3.metric("预测日期", f"{result.predict_asof_date} -> {result.predicted_trade_date}")

    st.subheader("训练过程日志")
    st.dataframe(pd.DataFrame(result.stage_logs), use_container_width=True)

    _render_backtest(result.backtest_result.daily, benchmark_daily=result.benchmark_daily)
    _render_factor_section(result)
    _render_tomorrow_candidates(result)

    with st.expander("原始指标JSON"):
        st.json(
            {
                "factor_stats": result.factor_stats,
                "model_metrics": result.model_result.metrics,
                "backtest_metrics": result.backtest_result.metrics,
            }
        )


def _render_cn_factor_page() -> None:
    st.title("中国股市因子训练与挖掘")
    st.write("点击一次即可执行：AKShare数据加载 -> 526因子计算 -> 训练回测 -> 新因子挖掘 -> 明日股票建议。")

    cfg, evo_cfg, top_stock_k, date_controls = _build_config_from_ui(market="CN")

    run_clicked = st.button("🚀 一键运行中国股市训练流程", use_container_width=True)

    if run_clicked:
        with st.spinner("流程运行中，请稍候..."):
            try:
                bundle = _build_cn_market_bundle(
                    cfg.data.tickers[: cfg.data.max_tickers],
                    benchmark=cfg.data.benchmark,
                    start=cfg.data.start,
                    end=cfg.data.end,
                )
                result = run_dashboard_workflow_with_bundle(
                    cfg,
                    bundle,
                    data_mode="akshare_cn",
                    evolution_cfg=evo_cfg,
                    top_stock_k=top_stock_k,
                    backtest_start=date_controls["backtest_start"],
                    backtest_end=date_controls["backtest_end"],
                    predict_asof_date=date_controls["predict_asof_date"],
                )
            except Exception as exc:  # pylint: disable=broad-except
                st.error(f"流程失败：{exc}")
                st.exception(exc)
                return
        st.session_state["cn_dashboard_result"] = result

    result = st.session_state.get("cn_dashboard_result")
    if result is None:
        st.info("请先点击“🚀 一键运行中国股市训练流程”。")
        return

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("数据模式", result.data_mode)
    m2.metric("总耗时(秒)", _metric(result.runtime_sec, 2))
    m3.metric("526因子数", str(result.factor_stats.get("factor_count", "-")))
    m4.metric("回测Sharpe", _metric(result.backtest_result.metrics.get("sharpe", float("nan")), 3))

    m5, m6, m7, m8 = st.columns(4)
    m5.metric("Train IC", _metric(result.model_result.metrics.get("train_ic_mean", float("nan")), 4))
    m6.metric("Test IC", _metric(result.model_result.metrics.get("test_ic_mean", float("nan")), 4))
    m7.metric("Train RMSE", _metric(result.model_result.metrics.get("train_rmse", float("nan")), 5))
    m8.metric("Test RMSE", _metric(result.model_result.metrics.get("test_rmse", float("nan")), 5))

    d1, d2, d3 = st.columns(3)
    d1.metric("训练区间", f"{result.train_window['train_start']} ~ {result.train_window['train_end']}")
    d2.metric("回测区间", f"{result.backtest_window['backtest_start']} ~ {result.backtest_window['backtest_end']}")
    d3.metric("预测日期", f"{result.predict_asof_date} -> {result.predicted_trade_date}")

    st.subheader("训练过程日志")
    st.dataframe(pd.DataFrame(result.stage_logs), use_container_width=True)

    _render_backtest(result.backtest_result.daily, benchmark_daily=result.benchmark_daily)
    _render_factor_section(result)
    _render_tomorrow_candidates(result)

    with st.expander("原始指标JSON"):
        st.json(
            {
                "factor_stats": result.factor_stats,
                "model_metrics": result.model_result.metrics,
                "backtest_metrics": result.backtest_result.metrics,
            }
        )


def _parse_symbol_pool(raw: str) -> list[str]:
    tokens = str(raw).replace("\n", ",").replace("，", ",").split(",")
    return [_normalize_cn_symbol(x).zfill(6) for x in tokens if x.strip()]


def _load_cn_histories_for_backtest(
    tickers: list[str],
    *,
    start: str,
    end: str,
    max_workers: int,
    cache_only: bool = False,
    progress_slot: Any | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    data_by_symbol: dict[str, pd.DataFrame] = {}
    failures: dict[str, str] = {}
    symbols = list(dict.fromkeys([_normalize_cn_symbol(x).zfill(6) for x in tickers if x.strip()]))
    if not symbols:
        return data_by_symbol, failures

    total = len(symbols)
    done = 0
    workers = max(1, int(max_workers))

    def fetch_one(symbol: str) -> tuple[str, pd.DataFrame | None, str | None]:
        try:
            if cache_only:
                frame = _read_cn_cached_history(symbol)
                if not frame.empty:
                    start_ts = pd.Timestamp(start).normalize()
                    end_ts = pd.Timestamp(end).normalize()
                    frame = frame.loc[(frame.index >= start_ts) & (frame.index <= end_ts)].copy()
            else:
                frame = _fetch_cn_stock_history_cached(symbol, start, end)
            if frame.empty:
                return symbol, None, "空行情"
            return symbol, frame, None
        except Exception as exc:  # pylint: disable=broad-except
            return symbol, None, str(exc)

    if progress_slot is not None:
        progress_slot.progress(0.0, text=f"加载行情 0/{total}")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(fetch_one, symbol) for symbol in symbols]
        for future in as_completed(futures):
            symbol, frame, err = future.result()
            done += 1
            if frame is not None:
                data_by_symbol[symbol] = frame
            else:
                failures[symbol] = err or "未知错误"
            if progress_slot is not None:
                progress_slot.progress(done / total, text=f"加载行情 {done}/{total}，成功 {len(data_by_symbol)}，失败 {len(failures)}")

    return data_by_symbol, failures


def _render_cn_backtest_page() -> None:
    st.title("中国股市回测")
    st.write("当前框架先接入策略一和交易模式一；后续策略和交易模式可直接挂到对应 registry。")

    today = _today_date()
    default_start = (pd.Timestamp(today) - pd.DateOffset(years=1)).date()
    default_pool = "600000,000001,300750,600519,000858,601318,600036,601398,002594,002415,600276,601899"
    all_a = _fetch_cn_all_a_symbols_cached()
    cache_status = _cn_cache_status()

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("A股代码表", str(len(all_a)) if not all_a.empty else "-")
    s2.metric("已缓存股票", str(cache_status["file_count"]))
    s3.metric("缓存占用", _format_bytes(cache_status["total_size"]))
    estimated_full_size = cache_status["avg_size"] * (len(all_a) if not all_a.empty else 5300)
    s4.metric("全市场估算", _format_bytes(estimated_full_size) if estimated_full_size > 0 else "-")
    st.caption(
        f"缓存目录：{cache_status['cache_dir']}。当前只落盘 open/high/low/close/volume 五个必要字段，"
        "amount/vwap/ret 在读取时计算；目标是把全市场日线缓存控制在 10G 以内。"
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        initial_cash = float(st.number_input("初始资金", min_value=10000.0, value=1_000_000.0, step=10000.0))
    with c2:
        start_date = st.date_input("起始日期", value=default_start, key="cn_bt_start")
    with c3:
        end_date = st.date_input("结束日期", value=today, key="cn_bt_end")

    c4, c5, c6 = st.columns(3)
    with c4:
        strategy_name = st.selectbox("策略", options=list(STRATEGY_REGISTRY.keys()), index=0)
    with c5:
        trade_mode_name = st.selectbox("交易模式", options=list(TRADE_MODE_REGISTRY.keys()), index=0)
    with c6:
        top_n = int(st.number_input("买入股票数量", min_value=1, max_value=50, value=10, step=1))

    d1, d2, d3 = st.columns(3)
    with d1:
        universe_mode = st.selectbox("股票池范围", options=["示例股票池", "全A股"], index=1)
    with d2:
        if universe_mode == "全A股":
            limit_full_universe = st.checkbox(
                "限制加载股票数",
                value=False,
                help="默认关闭时直接使用全A股代码表全部股票；需要试跑时再打开限制。",
            )
            if limit_full_universe:
                max_symbols = int(
                    st.number_input(
                        "加载股票数上限",
                        min_value=10,
                        max_value=6000,
                        value=1000,
                        step=100,
                        help="用于试跑或限流场景；关闭上方勾选则直接使用全A股全部股票。",
                    )
                )
            else:
                max_symbols = int(len(all_a)) if not all_a.empty else 0
        else:
            limit_full_universe = False
            max_symbols = int(
                st.number_input(
                    "最多加载股票数",
                    min_value=10,
                    max_value=6000,
                    value=200,
                    step=100,
                    help="手动股票池模式下，仅取前 N 只股票参与回测。",
                )
            )
    with d3:
        max_workers = int(
            st.number_input(
                "并发下载数",
                min_value=1,
                max_value=12,
                value=4,
                step=1,
                help="AKShare 数据源容易限流，MacBook 上通常 3-6 比较稳。",
            )
        )

    if universe_mode == "全A股":
        cache_only_run = st.checkbox(
            "运行回测时仅读取本地缓存",
            value=True,
            help="默认开启，避免在全A股回测时同步联网更新 5000+ 股票导致超时。先点“仅建立/更新行情缓存”，再运行回测。",
        )
    else:
        cache_only_run = st.checkbox(
            "运行回测时仅读取本地缓存",
            value=False,
            help="开启后不会联网补数据，只使用本地缓存中已有行情。",
        )

    pool_raw = st.text_area(
        "股票池（逗号或换行分隔）",
        value=default_pool,
        height=90,
        disabled=universe_mode == "全A股",
        help="策略一会对股票池内股票逐日计算技术指标综合分，并选取得分最高的股票。",
    )
    if universe_mode == "全A股":
        if all_a.empty:
            st.warning("未能通过 AKShare 获取全A股代码表，当前只能使用手动股票池。")
            tickers = _parse_symbol_pool(pool_raw)
        else:
            if limit_full_universe:
                tickers = all_a["symbol"].head(max_symbols).tolist()
                st.caption(f"当前按试跑上限取前 {len(tickers)} 只；全A股代码表总数 {len(all_a)}。")
            else:
                tickers = all_a["symbol"].tolist()
                st.caption(f"当前默认使用全A股全部股票，共 {len(tickers)} 只。")
    else:
        tickers = _parse_symbol_pool(pool_raw)[:max_symbols]
    if not tickers:
        st.info("请输入至少一只A股股票。")
        return
    if pd.Timestamp(start_date) > pd.Timestamp(end_date):
        st.error("起始日期不能晚于结束日期。")
        return

    st.caption(
        "策略一：股票查看中的 MACD、KDJ、RSI、DMI、威廉指标、均线评分取平均。"
        "交易模式一：每日开盘按上一交易日信号调仓，收盘按当日信号再调仓；全仓等权买入前十，最小成交单位 100 股。"
    )

    warmup_start = (pd.Timestamp(start_date) - pd.DateOffset(years=1)).date().isoformat()
    start_str = pd.Timestamp(start_date).date().isoformat()
    effective_end_ts = _effective_cn_daily_end(end_date)
    end_str = effective_end_ts.date().isoformat()
    if effective_end_ts.date() != pd.Timestamp(end_date).date():
        st.caption(f"A股日线当前按最近已完成交易日加载：{end_str}。")
    b1, b2 = st.columns(2)
    with b1:
        build_cache_clicked = st.button("仅建立/更新行情缓存", use_container_width=True)
    with b2:
        run_clicked = st.button("运行回测", type="primary", use_container_width=True)

    if build_cache_clicked:
        with st.spinner("正在建立/更新A股行情缓存..."):
            progress_slot = st.progress(0.0, text="加载行情 0/0")
            data_by_symbol, failures = _load_cn_histories_for_backtest(
                tickers,
                start=warmup_start,
                end=end_str,
                max_workers=max_workers,
                cache_only=False,
                progress_slot=progress_slot,
            )
            progress_slot.empty()
        status = _cn_cache_status()
        st.success(
            f"缓存更新完成：成功 {len(data_by_symbol)}，失败 {len(failures)}，"
            f"当前缓存 {status['file_count']} 只，占用 {_format_bytes(status['total_size'])}。"
        )
        if failures:
            with st.expander("缓存更新失败股票", expanded=False):
                st.json(failures)

    if run_clicked:

        with st.spinner("正在加载A股行情、计算策略分数并执行回测..."):
            progress_slot = st.progress(0.0, text="加载行情 0/0")
            data_by_symbol, failures = _load_cn_histories_for_backtest(
                tickers,
                start=warmup_start,
                end=end_str,
                max_workers=max_workers,
                cache_only=cache_only_run,
                progress_slot=progress_slot,
            )
            progress_slot.empty()
            if not data_by_symbol:
                st.error(f"股票池行情全部加载失败：{failures}")
                return
            if cache_only_run and len(data_by_symbol) < len(tickers):
                st.warning(
                    f"当前为只读缓存模式：目标股票 {len(tickers)} 只，缓存命中 {len(data_by_symbol)} 只，"
                    f"缺失 {len(failures)} 只。若要补齐，请先点击“仅建立/更新行情缓存”。"
                )
            try:
                benchmark = _fetch_cn_index_history_cached("000001", start_str, end_str)
            except Exception as exc:  # pylint: disable=broad-except
                benchmark = pd.DataFrame()
                failures["沪指000001"] = str(exc)
            try:
                backtest_fn = TRADE_MODE_REGISTRY[trade_mode_name]
                result = backtest_fn(
                    data_by_symbol=data_by_symbol,
                    benchmark=benchmark,
                    initial_cash=initial_cash,
                    start=start_str,
                    end=end_str,
                    strategy_fn=STRATEGY_REGISTRY[strategy_name],
                    top_n=top_n,
                )
            except Exception as exc:  # pylint: disable=broad-except
                st.error(f"回测失败：{exc}")
                st.exception(exc)
                return

        st.session_state["cn_backtest_result"] = result
        st.session_state["cn_backtest_meta"] = {
            "strategy": strategy_name,
            "trade_mode": trade_mode_name,
            "symbols": list(data_by_symbol.keys()),
            "failures": failures,
            "cache_only_run": bool(cache_only_run),
            "start": start_str,
            "end": end_str,
            "cache_status": _cn_cache_status(),
        }

    result = st.session_state.get("cn_backtest_result")
    meta = st.session_state.get("cn_backtest_meta", {})
    if result is None:
        st.info("请先运行回测。")
        return

    st.subheader("回测结果")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("策略", meta.get("strategy", strategy_name))
    m2.metric("交易模式", meta.get("trade_mode", trade_mode_name))
    m3.metric("有效股票数", str(len(meta.get("symbols", []))))
    m4.metric("区间", f"{meta.get('start', '')} ~ {meta.get('end', '')}")
    failures = meta.get("failures", {})
    if failures:
        with st.expander("行情加载失败/基线提示", expanded=False):
            st.json(failures)
    _render_standalone_backtest(result)


def _render_us_backtest_page() -> None:
    st.title("美国股市回测")
    st.info("回测框架已先接入中国股市版本；美股版本会复用同一策略/交易模式接口，后续可直接接入 yfinance 数据源。")


def main() -> None:
    _install_browser_shutdown_hook()

    st.title("USalpha")
    market = st.segmented_control(
        "市场",
        options=["中国股市", "美国股市"],
        default="中国股市",
        key="top_market",
    )
    subpage = st.segmented_control(
        "功能",
        options=["股票查看", "因子训练与挖掘", "回测"],
        default="股票查看",
        key="top_subpage",
    )
    st.divider()

    with st.sidebar:
        if _browser_auto_shutdown_enabled():
            st.caption(f"关闭全部浏览器标签页后，后端会在约 {int(_browser_idle_timeout_sec())} 秒后自动退出并释放 8501 端口。")
        else:
            st.caption("当前已关闭浏览器心跳自动关停，长任务期间后端不会因为标签页心跳中断而自停。")
        st.divider()

    if market == "中国股市" and subpage == "股票查看":
        _render_china_stock_page()
    elif market == "中国股市" and subpage == "因子训练与挖掘":
        _render_cn_factor_page()
    elif market == "中国股市" and subpage == "回测":
        _render_cn_backtest_page()
    elif market == "美国股市" and subpage == "股票查看":
        _render_stock_page()
    elif market == "美国股市" and subpage == "回测":
        _render_us_backtest_page()
    else:
        _render_us_factor_page()


if __name__ == "__main__":
    main()
