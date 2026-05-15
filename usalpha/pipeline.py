from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .backtest import run_long_short_backtest
from .config import USAlphaConfig
from .data import fetch_us_market_data
from .factors import compute_526_factors
from .model import train_factor_model


def _resolve_output_dir(project_root: Path, output_dir: str) -> Path:
    path = Path(output_dir).expanduser()
    if not path.is_absolute():
        path = (project_root / path).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")


def run_pipeline(config: USAlphaConfig | dict[str, Any] | None = None) -> dict[str, Any]:
    if config is None:
        cfg = USAlphaConfig()
    elif isinstance(config, dict):
        cfg = USAlphaConfig.from_dict(config)
    else:
        cfg = config

    project_root = Path(__file__).resolve().parents[1]
    alpha526_path = cfg.resolve_alpha526_path(project_root)

    output_root = _resolve_output_dir(project_root, cfg.io.output_dir)
    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print("[USalpha] step 1/4: downloading market data...")
    bundle = fetch_us_market_data(
        cfg.data.tickers,
        benchmark=cfg.data.benchmark,
        start=cfg.data.start,
        end=cfg.data.end,
        interval=cfg.data.interval,
        auto_adjust=cfg.data.auto_adjust,
        max_tickers=cfg.data.max_tickers,
    )

    print("[USalpha] step 2/4: computing 526 factors...")
    factor_set = compute_526_factors(bundle.panel, bundle.benchmark, alpha526_path=alpha526_path)

    print("[USalpha] step 3/4: training model...")
    model_result = train_factor_model(factor_set.features, bundle.panel, cfg.train)

    print("[USalpha] step 4/4: running long-short backtest...")
    backtest_result = run_long_short_backtest(
        model_result.predictions,
        top_quantile=cfg.train.top_quantile,
        split="test",
    )

    factor_stats_path = run_dir / "factor_stats.json"
    model_metrics_path = run_dir / "model_metrics.json"
    backtest_metrics_path = run_dir / "backtest_metrics.json"
    backtest_daily_path = run_dir / "backtest_daily.csv"
    pred_path = run_dir / "predictions.csv"
    cfg_path = run_dir / "run_config.json"

    _write_json(factor_stats_path, factor_set.stats)
    _write_json(model_metrics_path, model_result.metrics)
    _write_json(backtest_metrics_path, backtest_result.metrics)
    _write_json(cfg_path, asdict(cfg))

    backtest_result.daily.to_csv(backtest_daily_path)

    if cfg.io.save_predictions:
        model_result.predictions.to_csv(pred_path)

    if cfg.io.save_factor_snapshot:
        factor_set.features.to_parquet(run_dir / "factor_snapshot.parquet")

    summary = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "factor_count": factor_set.stats["factor_count"],
        "feature_shape": factor_set.stats["feature_shape"],
        "model_metrics": model_result.metrics,
        "backtest_metrics": backtest_result.metrics,
        "files": {
            "config": str(cfg_path),
            "factor_stats": str(factor_stats_path),
            "model_metrics": str(model_metrics_path),
            "backtest_metrics": str(backtest_metrics_path),
            "backtest_daily": str(backtest_daily_path),
            "predictions": str(pred_path) if cfg.io.save_predictions else None,
        },
    }
    _write_json(run_dir / "summary.json", summary)
    return summary
