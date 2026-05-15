#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd

from usalpha.config import USAlphaConfig


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run one LLM factor evolution round")
    p.add_argument("--api-key", type=str, default=None, help="GLM API key, default from GLM_API_KEY env")
    p.add_argument("--model", type=str, default="glm-5")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--num-candidates", type=int, default=72)
    p.add_argument("--top-k-accept", type=int, default=12)
    p.add_argument("--tickers", type=str, default=None, help="Comma-separated tickers or NASDAQ_ALL")
    p.add_argument("--max-tickers", type=int, default=None, help="Cap ticker count for stable runtime, e.g. 40")
    p.add_argument("--benchmark", type=str, default=None)
    p.add_argument("--start", type=str, default=None)
    p.add_argument("--end", type=str, default=None)
    p.add_argument("--interval", type=str, default=None)
    p.add_argument("--label-horizon", type=int, default=1)
    p.add_argument("--top-quantile", type=float, default=0.2)
    p.add_argument("--history-feedback", type=str, default=None)
    p.add_argument("--output-dir", type=str, default="./artifacts")
    p.add_argument(
        "--require-glm-api",
        action="store_true",
        help="Retry until generation_mode becomes glm_api",
    )
    p.add_argument(
        "--max-retries",
        type=int,
        default=0,
        help="Max retry rounds when --require-glm-api is set; 0 means unlimited",
    )
    p.add_argument(
        "--retry-wait-sec",
        type=int,
        default=25,
        help="Sleep seconds between retry rounds",
    )
    return p


def _build_synthetic_bundle(
    tickers: list[str],
    *,
    start: str,
    end: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range(start=start, end=end)
    if len(dates) < 120:
        dates = pd.bdate_range(end=pd.Timestamp.today(), periods=240)

    rng = np.random.default_rng(20260420)
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
        per_ticker[ticker] = frame

    panel = pd.concat(per_ticker, axis=1)
    panel.columns.names = ["instrument", "field"]

    bclose = 400 + np.cumsum(rng.normal(0, 0.8, len(dates)))
    benchmark = pd.DataFrame(index=dates)
    benchmark["open"] = bclose * (1 + rng.normal(0, 0.002, len(dates)))
    benchmark["high"] = np.maximum(benchmark["open"], bclose) * (1 + np.abs(rng.normal(0, 0.004, len(dates))))
    benchmark["low"] = np.minimum(benchmark["open"], bclose) * (1 - np.abs(rng.normal(0, 0.004, len(dates))))
    benchmark["close"] = bclose
    benchmark["volume"] = rng.lognormal(mean=16.0, sigma=0.3, size=len(dates))
    benchmark["amount"] = benchmark["close"] * benchmark["volume"]
    benchmark["vwap"] = (benchmark["high"] + benchmark["low"] + benchmark["close"]) / 3.0
    benchmark["ret"] = benchmark["close"].pct_change()

    return panel, benchmark


def main() -> None:
    args = _build_parser().parse_args()
    from usalpha.data import fetch_us_market_data, resolve_tickers_limited
    from usalpha.factor_evolution import run_evolution_round

    api_key = args.api_key or os.getenv("GLM_API_KEY", "")
    if not api_key:
        raise SystemExit("missing api key: set --api-key or GLM_API_KEY")

    cfg = USAlphaConfig()
    if args.tickers:
        cfg.data.tickers = [x.strip().upper() for x in args.tickers.split(",") if x.strip()]
    if args.benchmark:
        cfg.data.benchmark = args.benchmark.strip().upper()
    if args.start:
        cfg.data.start = args.start
    if args.end:
        cfg.data.end = args.end
    if args.interval:
        cfg.data.interval = args.interval
    if args.max_tickers is not None:
        cfg.data.max_tickers = int(args.max_tickers)

    project_root = Path(__file__).resolve().parents[1]
    alpha526_path = cfg.resolve_alpha526_path(project_root)

    resolved_tickers = resolve_tickers_limited(cfg.data.tickers, max_tickers=cfg.data.max_tickers)

    print("[USalpha-LLM] step 1/3 downloading market data...")
    data_mode = "yfinance_live"
    try:
        bundle = fetch_us_market_data(
            resolved_tickers,
            benchmark=cfg.data.benchmark,
            start=cfg.data.start,
            end=cfg.data.end,
            interval=cfg.data.interval,
            auto_adjust=cfg.data.auto_adjust,
            max_tickers=cfg.data.max_tickers,
        )
        panel = bundle.panel
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[USalpha-LLM] warning: yfinance unavailable, fallback to synthetic data: {exc}")
        data_mode = "synthetic_fallback"
        panel, _ = _build_synthetic_bundle(resolved_tickers, start=cfg.data.start, end=cfg.data.end)

    print("[USalpha-LLM] step 2/3 generating candidates via GLM and scoring...")
    attempt = 0
    result = None
    while True:
        attempt += 1
        result = run_evolution_round(
            panel=panel,
            alpha526_path=alpha526_path,
            api_key=api_key,
            num_candidates=args.num_candidates,
            top_k_accept=args.top_k_accept,
            label_horizon=args.label_horizon,
            top_quantile=args.top_quantile,
            model=args.model,
            temperature=args.temperature,
            output_dir=(project_root / args.output_dir).resolve(),
            history_feedback=args.history_feedback,
            persist_library_on_fallback=not args.require_glm_api,
        )

        if not args.require_glm_api or result.generation_mode == "glm_api":
            break

        max_retries = int(args.max_retries)
        if max_retries > 0 and attempt >= max_retries:
            print(
                f"[USalpha-LLM] reached max retries ({max_retries}) without glm_api success, "
                "keeping last fallback result"
            )
            break

        wait_sec = max(int(args.retry_wait_sec), 1)
        print(
            f"[USalpha-LLM] attempt {attempt} got generation_mode={result.generation_mode}, "
            f"retrying in {wait_sec}s..."
        )
        time.sleep(wait_sec)

    assert result is not None

    summary = {
        "run_dir": result.run_dir,
        "data_mode": data_mode,
        "generation_mode": result.generation_mode,
        "attempts": attempt,
        "candidate_count": result.candidate_count,
        "valid_count": result.valid_count,
        "accepted_count": result.accepted_count,
        "accepted_factors": result.accepted_factors,
    }

    print("[USalpha-LLM] step 3/3 done")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
