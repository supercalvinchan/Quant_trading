from __future__ import annotations

import numpy as np
import pandas as pd

from .data import MarketDataBundle


def build_synthetic_market_bundle(
    tickers: list[str],
    *,
    benchmark: str,
    start: str,
    end: str,
    seed: int = 20260420,
) -> MarketDataBundle:
    dates = pd.bdate_range(start=start, end=end)
    if len(dates) < 120:
        dates = pd.bdate_range(end=pd.Timestamp.today(), periods=240)

    rng = np.random.default_rng(seed)
    per_ticker: dict[str, pd.DataFrame] = {}

    for ticker in tickers:
        base = 80 + np.cumsum(rng.normal(0, 1.2, len(dates)))
        close = pd.Series(base, index=dates).clip(lower=5)
        open_ = close * (1 + rng.normal(0, 0.004, len(dates)))
        high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.006, len(dates))))
        low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.006, len(dates))))
        volume = rng.lognormal(mean=15.0, sigma=0.35, size=len(dates))

        frame = pd.DataFrame(index=dates)
        frame["open"] = open_.to_numpy()
        frame["high"] = high.to_numpy()
        frame["low"] = low.to_numpy()
        frame["close"] = close.to_numpy()
        frame["volume"] = volume
        frame["amount"] = frame["close"] * frame["volume"]
        frame["vwap"] = (frame["high"] + frame["low"] + frame["close"]) / 3.0
        frame["ret"] = frame["close"].pct_change()
        per_ticker[str(ticker).upper()] = frame

    panel = pd.concat(per_ticker, axis=1)
    panel.columns.names = ["instrument", "field"]

    bclose = 400 + np.cumsum(rng.normal(0, 0.8, len(dates)))
    benchmark_df = pd.DataFrame(index=dates)
    benchmark_df["open"] = bclose * (1 + rng.normal(0, 0.002, len(dates)))
    benchmark_df["high"] = np.maximum(benchmark_df["open"], bclose) * (1 + np.abs(rng.normal(0, 0.004, len(dates))))
    benchmark_df["low"] = np.minimum(benchmark_df["open"], bclose) * (1 - np.abs(rng.normal(0, 0.004, len(dates))))
    benchmark_df["close"] = bclose
    benchmark_df["volume"] = rng.lognormal(mean=16.0, sigma=0.3, size=len(dates))
    benchmark_df["amount"] = benchmark_df["close"] * benchmark_df["volume"]
    benchmark_df["vwap"] = (benchmark_df["high"] + benchmark_df["low"] + benchmark_df["close"]) / 3.0
    benchmark_df["ret"] = benchmark_df["close"].pct_change()

    benchmark_df.attrs["symbol"] = benchmark
    return MarketDataBundle(panel=panel, benchmark=benchmark_df)
