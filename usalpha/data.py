from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
import re
from typing import Iterable

import pandas as pd
import yfinance as yf

from .config import NASDAQ_ALL_TOKEN


DEFAULT_FALLBACK_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "AMD", "NFLX",
    "COST", "ADBE", "CRM", "INTC", "QCOM", "AMAT", "TXN", "CSCO", "INTU", "MU",
]

CORE_LIQUID_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "AMD", "NFLX",
    "COST", "ADBE", "CRM", "INTC", "QCOM", "AMAT", "TXN", "CSCO", "INTU", "MU",
    "PEP", "SBUX", "AMGN", "BKNG", "ISRG", "GILD", "ADP", "REGN", "VRTX", "MAR",
]

_MAX_BATCH_SIZE = 100
_UNIVERSE_CACHE_TTL_HOURS = 24
_TICKER_RE = re.compile(r"^[A-Z0-9\.\-]+$")


@dataclass
class MarketDataBundle:
    panel: pd.DataFrame
    benchmark: pd.DataFrame


def _cache_dir() -> Path:
    root = Path(__file__).resolve().parents[1]
    path = root / ".cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _cache_path_nasdaq() -> Path:
    return _cache_dir() / "nasdaq_all_tickers.txt"


def _read_cached_tickers(path: Path) -> list[str]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    return [x.strip().upper() for x in text.splitlines() if x.strip()]


def _is_cache_fresh(path: Path, ttl_hours: int) -> bool:
    if not path.exists():
        return False
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return datetime.now(timezone.utc) - mtime <= timedelta(hours=ttl_hours)


def _fetch_nasdaq_symbols_live() -> list[str]:
    # Official NASDAQ symbol directory (pipe-separated TXT).
    url = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
    table = pd.read_csv(url, sep="|")
    if "Symbol" not in table.columns:
        raise ValueError("missing Symbol column from nasdaq source")

    symbols = table["Symbol"].astype(str).str.strip().str.upper()
    mask = symbols.notna() & (symbols != "") & (symbols != "FILE CREATION TIME")

    # Use official flags when available.
    if "Test Issue" in table.columns:
        test_issue = table["Test Issue"].astype(str).str.strip().str.upper()
        mask &= test_issue == "N"
    if "ETF" in table.columns:
        etf = table["ETF"].astype(str).str.strip().str.upper()
        mask &= etf == "N"
    if "NextShares" in table.columns:
        nxt = table["NextShares"].astype(str).str.strip().str.upper()
        mask &= nxt == "N"

    symbols = symbols[mask]
    # Keep common tradable forms.
    symbols = symbols[symbols.str.match(r"^[A-Z0-9\.\-]+$", na=False)]
    # Drop common unit/right/warrant suffixes.
    symbols = symbols[~symbols.str.match(r"^[A-Z]{1,4}(W|U|R)$", na=False)]
    symbols = symbols[~symbols.str.match(r"^[A-Z]{5}(W|U|R)$", na=False)]

    out = sorted(set(symbols.tolist()))
    if len(out) < 500:
        raise ValueError(f"unexpectedly small nasdaq symbol count: {len(out)}")
    return out


@lru_cache(maxsize=1)
def get_nasdaq_all_tickers() -> list[str]:
    cache_path = _cache_path_nasdaq()
    if _is_cache_fresh(cache_path, _UNIVERSE_CACHE_TTL_HOURS):
        cached = _read_cached_tickers(cache_path)
        if len(cached) > 0:
            return cached

    try:
        symbols = _fetch_nasdaq_symbols_live()
        cache_path.write_text("\n".join(symbols) + "\n", encoding="utf-8")
        return symbols
    except Exception as exc:  # pylint: disable=broad-except
        cached = _read_cached_tickers(cache_path)
        if len(cached) > 0:
            print(f"[USalpha] warning: NASDAQ list refresh failed, using cache: {exc}")
            return cached
        print(f"[USalpha] warning: NASDAQ list unavailable, using fallback pool: {exc}")
        return list(DEFAULT_FALLBACK_TICKERS)


def resolve_tickers(tickers: Iterable[str]) -> list[str]:
    cleaned = [str(t).strip().upper() for t in tickers if str(t).strip()]
    if len(cleaned) == 0:
        raise ValueError("tickers cannot be empty")

    out: list[str] = []
    for t in cleaned:
        if t == NASDAQ_ALL_TOKEN:
            out.extend(get_nasdaq_all_tickers())
        else:
            out.append(t)

    deduped: list[str] = []
    seen: set[str] = set()
    for t in out:
        if t in seen:
            continue
        seen.add(t)
        deduped.append(t)
    cleaned = []
    for t in deduped:
        if not t or not _TICKER_RE.match(t):
            continue
        if len(t) <= 2 and t not in CORE_LIQUID_TICKERS:
            continue
        if len(t) == 5 and t.endswith(("W", "U", "R")) and t not in CORE_LIQUID_TICKERS:
            continue
        cleaned.append(t)
    return cleaned


def _prioritize_tickers(tickers: list[str]) -> list[str]:
    priority_rank = {t: i for i, t in enumerate(CORE_LIQUID_TICKERS)}

    def _sort_key(sym: str) -> tuple[int, int, str]:
        if sym in priority_rank:
            return (0, priority_rank[sym], sym)
        # Symbols ending with W/U/R are often warrants/units/rights; deprioritize.
        suffix_penalty = 1 if (len(sym) == 5 and sym[-1] in {"W", "U", "R"}) else 0
        return (1 + suffix_penalty, len(sym), sym)

    return sorted(tickers, key=_sort_key)


def _apply_ticker_cap(tickers: list[str], max_tickers: int | None) -> list[str]:
    if max_tickers is None:
        return tickers
    cap = int(max_tickers)
    if cap <= 0 or len(tickers) <= cap:
        return tickers
    picked = _prioritize_tickers(tickers)[:cap]
    print(f"[USalpha] warning: ticker universe capped from {len(tickers)} to {len(picked)} for stability.")
    return picked


def resolve_tickers_limited(tickers: Iterable[str], *, max_tickers: int | None) -> list[str]:
    deduped = resolve_tickers(tickers)
    return _apply_ticker_cap(deduped, max_tickers)


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise ValueError("empty dataframe returned by yfinance")

    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [str(c[-1]) for c in out.columns]
    out.columns = [str(c).lower() for c in out.columns]

    required = ["open", "high", "low", "close", "volume"]
    missing = [col for col in required if col not in out.columns]
    if missing:
        raise ValueError(f"missing ohlcv columns: {missing}")

    out = out[required].copy()
    out.index = pd.to_datetime(out.index).tz_localize(None)
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]

    out["volume"] = out["volume"].fillna(0.0).astype(float)
    out["amount"] = out["close"].astype(float) * out["volume"].astype(float)
    out["vwap"] = (out["high"] + out["low"] + out["close"]) / 3.0
    out["ret"] = out["close"].pct_change()
    return out


def fetch_ticker_history(
    ticker: str,
    *,
    start: str,
    end: str,
    interval: str,
    auto_adjust: bool,
) -> pd.DataFrame:
    hist = yf.Ticker(ticker).history(
        start=start,
        end=end,
        interval=interval,
        auto_adjust=auto_adjust,
        actions=False,
    )
    return _normalize_ohlcv(hist)


def _fetch_batch_history(
    tickers: list[str],
    *,
    start: str,
    end: str,
    interval: str,
    auto_adjust: bool,
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    # Use one batched request first to reduce Yahoo rate-limit pressure.
    hist = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=auto_adjust,
        actions=False,
        group_by="ticker",
        threads=False,
        progress=False,
    )

    fetched: dict[str, pd.DataFrame] = {}
    failures: dict[str, str] = {}
    if hist.empty:
        for ticker in tickers:
            failures[ticker] = "empty batch dataframe from yfinance"
        return fetched, failures

    if isinstance(hist.columns, pd.MultiIndex):
        level0 = {str(x).upper() for x in hist.columns.get_level_values(0)}
        if all(t in level0 for t in tickers):
            for ticker in tickers:
                try:
                    fetched[ticker] = _normalize_ohlcv(hist[ticker])
                except Exception as exc:  # pylint: disable=broad-except
                    failures[ticker] = str(exc)
            return fetched, failures

        # Single ticker case in some yfinance responses.
        if len(tickers) == 1:
            try:
                fetched[tickers[0]] = _normalize_ohlcv(hist)
            except Exception as exc:  # pylint: disable=broad-except
                failures[tickers[0]] = str(exc)
            return fetched, failures

    failures = {ticker: "unexpected batch format from yfinance" for ticker in tickers}
    return fetched, failures


def fetch_index_data(
    ticker: str,
    *,
    start: str,
    end: str,
    interval: str = "1d",
    auto_adjust: bool = False,
) -> pd.DataFrame:
    """Download index / ETF data and return a normalised OHLCV DataFrame with a ``ret`` column."""
    return fetch_ticker_history(
        ticker,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=auto_adjust,
    )


def fetch_us_market_data(
    tickers: Iterable[str],
    *,
    benchmark: str,
    start: str,
    end: str,
    interval: str,
    auto_adjust: bool = False,
    max_tickers: int | None = None,
) -> MarketDataBundle:
    cleaned = resolve_tickers_limited(tickers, max_tickers=max_tickers)
    if len(cleaned) == 0:
        raise ValueError("tickers cannot be empty")

    per_ticker: dict[str, pd.DataFrame] = {}
    failures: dict[str, str] = {}
    for i in range(0, len(cleaned), _MAX_BATCH_SIZE):
        chunk = cleaned[i: i + _MAX_BATCH_SIZE]
        fetched_chunk, failures_chunk = _fetch_batch_history(
            chunk,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=auto_adjust,
        )
        per_ticker.update(fetched_chunk)
        failures.update(failures_chunk)

    # Fallback to per-ticker mode for failed symbols.
    for ticker in list(failures.keys()):
        try:
            per_ticker[ticker] = fetch_ticker_history(
                ticker,
                start=start,
                end=end,
                interval=interval,
                auto_adjust=auto_adjust,
            )
            failures.pop(ticker, None)
        except Exception as exc:  # pylint: disable=broad-except
            failures[ticker] = str(exc)

    if len(per_ticker) == 0:
        raise RuntimeError(f"failed to download all tickers: {failures}")

    panel = pd.concat(per_ticker, axis=1)
    panel.columns.names = ["instrument", "field"]
    panel = panel.sort_index()

    benchmark_df = fetch_ticker_history(
        benchmark,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=auto_adjust,
    )

    if failures:
        print(f"[USalpha] warning: failed tickers excluded: {failures}")

    return MarketDataBundle(panel=panel, benchmark=benchmark_df)
