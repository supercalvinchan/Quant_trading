#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from usalpha.config import USAlphaConfig


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run USalpha end-to-end pipeline")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config file")
    parser.add_argument(
        "--tickers",
        type=str,
        default=None,
        help="Comma-separated tickers, e.g. AAPL,MSFT,NVDA or NASDAQ_ALL",
    )
    parser.add_argument("--benchmark", type=str, default=None, help="Benchmark ticker, default SPY")
    parser.add_argument("--max-tickers", type=int, default=None, help="Cap ticker count for stable runtime, e.g. 40")
    parser.add_argument("--start", type=str, default=None, help="Start date, e.g. 2020-01-01")
    parser.add_argument("--end", type=str, default=None, help="End date, e.g. 2025-12-31")
    parser.add_argument("--interval", type=str, default=None, help="yfinance interval, e.g. 1d or 1h")
    parser.add_argument("--train-end", type=str, default=None, help="Train end date")
    parser.add_argument("--output-dir", type=str, default=None, help="Output root directory")
    return parser


def _load_config(path: str | None) -> USAlphaConfig:
    if not path:
        return USAlphaConfig()
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return USAlphaConfig.from_dict(payload)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    from usalpha.pipeline import run_pipeline

    cfg = _load_config(args.config)

    if args.tickers:
        cfg.data.tickers = [x.strip().upper() for x in args.tickers.split(",") if x.strip()]
    if args.benchmark:
        cfg.data.benchmark = args.benchmark.strip().upper()
    if args.max_tickers is not None:
        cfg.data.max_tickers = int(args.max_tickers)
    if args.start:
        cfg.data.start = args.start
    if args.end:
        cfg.data.end = args.end
    if args.interval:
        cfg.data.interval = args.interval
    if args.train_end:
        cfg.train.train_end = args.train_end
    if args.output_dir:
        cfg.io.output_dir = args.output_dir

    summary = run_pipeline(cfg)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
