from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

from .backtest import BacktestResult, run_long_short_backtest
from .config import USAlphaConfig
from .data import MarketDataBundle, fetch_us_market_data, resolve_tickers_limited
from .factor_evolution import EvolutionResult, run_evolution_round
from .factors import _eval_base_expression, compute_526_factors
from .model import ModelResult, train_factor_model
from .synthetic_data import build_synthetic_market_bundle


@dataclass
class EvolutionUIConfig:
    api_key: str
    model: str = "glm-5"
    temperature: float = 0.8
    num_candidates: int = 72
    top_k_accept: int = 12
    require_glm_api: bool = False
    max_retries: int = 0
    retry_wait_sec: int = 20
    history_feedback: str | None = None


@dataclass
class DashboardResult:
    data_mode: str
    panel: pd.DataFrame
    factor_stats: dict[str, Any]
    model_result: ModelResult
    backtest_result: BacktestResult
    evolution_result: EvolutionResult | None
    accepted_factors: list[dict[str, Any]]
    best_factor: dict[str, Any] | None
    best_factor_signal: pd.DataFrame
    tomorrow_candidates: pd.DataFrame
    model_tomorrow_candidates: pd.DataFrame
    train_window: dict[str, str]
    backtest_window: dict[str, str]
    predict_asof_date: str
    predicted_trade_date: str
    stage_logs: list[dict[str, Any]]
    runtime_sec: float
    benchmark_daily: pd.DataFrame


def _latest_model_candidates(
    predictions: pd.DataFrame,
    *,
    asof_date: pd.Timestamp | None = None,
    top_k: int = 10,
) -> pd.DataFrame:
    if len(predictions) == 0:
        return pd.DataFrame(columns=["instrument", "pred", "label"])

    all_dates = pd.DatetimeIndex(predictions.index.get_level_values("datetime")).unique().sort_values()
    if asof_date is None:
        picked_dt = all_dates.max()
    else:
        valid = all_dates[all_dates <= pd.Timestamp(asof_date)]
        picked_dt = valid.max() if len(valid) > 0 else all_dates.min()

    latest = predictions.loc[predictions.index.get_level_values("datetime") == picked_dt].copy()
    latest = latest.reset_index().sort_values("pred", ascending=False)
    out = latest[["instrument", "pred", "label"]].head(top_k).reset_index(drop=True)
    out.insert(0, "asof_date", picked_dt.date().isoformat())
    return out


def _extract_factor_params(expression: str) -> dict[str, Any]:
    funcs = sorted(set(re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", expression)))
    nums = re.findall(r"-?\d+(?:\.\d+)?", expression)
    fields = sorted(set(re.findall(r"\$[A-Za-z_]+", expression)))
    return {
        "functions": funcs,
        "numbers": nums,
        "fields": fields,
    }


def _factor_latest_signal(
    panel: pd.DataFrame,
    expression: str,
    *,
    asof_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    tickers = list(panel.columns.get_level_values(0).unique())
    rows: list[dict[str, Any]] = []

    for ticker in tickers:
        frame = panel[ticker].copy()
        frame.columns = [str(c).lower() for c in frame.columns]
        s = _eval_base_expression(expression, frame)
        if len(s) == 0:
            continue
        if asof_date is None:
            last = s.dropna()
        else:
            last = s.loc[s.index <= pd.Timestamp(asof_date)].dropna()
        if len(last) == 0:
            continue
        rows.append(
            {
                "instrument": ticker,
                "value": float(last.iloc[-1]),
                "last_date": pd.Timestamp(last.index[-1]),
            }
        )

    if len(rows) == 0:
        return pd.DataFrame(columns=["instrument", "value", "zscore", "rank_pct", "last_date"])

    out = pd.DataFrame(rows)
    mu = out["value"].mean()
    sigma = out["value"].std(ddof=0)
    if sigma <= 1e-12:
        out["zscore"] = 0.0
    else:
        out["zscore"] = (out["value"] - mu) / sigma
    out["rank_pct"] = out["value"].rank(pct=True)
    out = out.sort_values("value", ascending=False).reset_index(drop=True)
    return out


def _tomorrow_candidates_from_factor(best_factor: dict[str, Any] | None, signal: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
    cols = ["instrument", "value", "zscore", "rank_pct", "direction", "reason"]
    if best_factor is None or len(signal) == 0:
        return pd.DataFrame(columns=cols)

    ic_mean = float(best_factor.get("metrics", {}).get("ic_mean", np.nan))
    direction = "long_high" if np.isnan(ic_mean) or ic_mean >= 0 else "long_low"

    ordered = signal.sort_values("value", ascending=(direction == "long_low")).head(top_k).copy()
    ordered["direction"] = direction
    ordered["reason"] = (
        f"best_factor={best_factor.get('name')}"
        + f", ic_mean={ic_mean:.4f}, expression={best_factor.get('expression')}"
    )
    return ordered[cols].reset_index(drop=True)


def _resolve_asof_date(date_index: pd.DatetimeIndex, desired: str | None) -> pd.Timestamp:
    ordered = pd.DatetimeIndex(date_index).unique().sort_values()
    if len(ordered) == 0:
        raise ValueError("empty date index")
    if desired is None:
        return ordered.max()
    target = pd.Timestamp(desired)
    valid = ordered[ordered <= target]
    if len(valid) == 0:
        return ordered.min()
    return valid.max()


def _filter_predictions_by_window(
    predictions: pd.DataFrame,
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
) -> pd.DataFrame:
    idx = predictions.index.get_level_values("datetime")
    mask = pd.Series(True, index=predictions.index)
    if start is not None:
        mask &= idx >= pd.Timestamp(start)
    if end is not None:
        mask &= idx <= pd.Timestamp(end)
    return predictions.loc[mask.to_numpy()]


def _load_market_bundle(cfg: USAlphaConfig) -> tuple[MarketDataBundle, str]:
    resolved_tickers = resolve_tickers_limited(cfg.data.tickers, max_tickers=cfg.data.max_tickers)
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
        return bundle, "yfinance_live"
    except Exception:
        bundle = build_synthetic_market_bundle(
            resolved_tickers,
            benchmark=cfg.data.benchmark,
            start=cfg.data.start,
            end=cfg.data.end,
        )
        return bundle, "synthetic_fallback"


def run_dashboard_workflow(
    cfg: USAlphaConfig,
    *,
    evolution_cfg: EvolutionUIConfig | None = None,
    top_stock_k: int = 10,
    backtest_start: str | None = None,
    backtest_end: str | None = None,
    predict_asof_date: str | None = None,
) -> DashboardResult:
    t0 = datetime.now()
    logs: list[dict[str, Any]] = []

    project_root = Path(__file__).resolve().parents[1]
    alpha526_path = cfg.resolve_alpha526_path(project_root)

    t_stage = datetime.now()
    bundle, data_mode = _load_market_bundle(cfg)
    logs.append(
        {
            "stage": "load_market_data",
            "seconds": (datetime.now() - t_stage).total_seconds(),
            "data_mode": data_mode,
            "rows": int(len(bundle.panel)),
            "instruments": int(len(bundle.panel.columns.get_level_values(0).unique())),
        }
    )

    t_stage = datetime.now()
    factor_set = compute_526_factors(bundle.panel, bundle.benchmark, alpha526_path=alpha526_path)
    logs.append(
        {
            "stage": "compute_526_factors",
            "seconds": (datetime.now() - t_stage).total_seconds(),
            "factor_count": int(factor_set.stats.get("factor_count", 0)),
        }
    )

    t_stage = datetime.now()
    model_result = train_factor_model(factor_set.features, bundle.panel, cfg.train)
    logs.append(
        {
            "stage": "train_model",
            "seconds": (datetime.now() - t_stage).total_seconds(),
            "samples_total": int(model_result.metrics.get("samples_total", 0)),
            "test_ic_mean": float(model_result.metrics.get("test_ic_mean", np.nan)),
        }
    )

    pred_dates = pd.DatetimeIndex(model_result.predictions.index.get_level_values("datetime")).unique().sort_values()
    bt_start_ts = pd.Timestamp(backtest_start) if backtest_start is not None else pred_dates.min()
    bt_end_ts = pd.Timestamp(backtest_end) if backtest_end is not None else pred_dates.max()
    if bt_start_ts > bt_end_ts:
        bt_start_ts, bt_end_ts = bt_end_ts, bt_start_ts

    backtest_pred = _filter_predictions_by_window(model_result.predictions, bt_start_ts, bt_end_ts)

    t_stage = datetime.now()
    backtest_result = run_long_short_backtest(
        backtest_pred,
        top_quantile=cfg.train.top_quantile,
        split="",
        benchmark_daily=bundle.benchmark,
    )
    logs.append(
        {
            "stage": "run_backtest",
            "seconds": (datetime.now() - t_stage).total_seconds(),
            "days": int(backtest_result.metrics.get("days", 0)),
            "sharpe": float(backtest_result.metrics.get("sharpe", np.nan)),
            "backtest_start": bt_start_ts.date().isoformat(),
            "backtest_end": bt_end_ts.date().isoformat(),
        }
    )

    evolution_result: EvolutionResult | None = None
    accepted_factors: list[dict[str, Any]] = []

    if evolution_cfg is not None and evolution_cfg.api_key.strip():
        attempt = 0
        t_stage = datetime.now()
        while True:
            attempt += 1
            evolution_result = run_evolution_round(
                panel=bundle.panel,
                alpha526_path=alpha526_path,
                api_key=evolution_cfg.api_key,
                num_candidates=evolution_cfg.num_candidates,
                top_k_accept=evolution_cfg.top_k_accept,
                label_horizon=cfg.train.label_horizon,
                top_quantile=cfg.train.top_quantile,
                model=evolution_cfg.model,
                temperature=evolution_cfg.temperature,
                output_dir=(project_root / cfg.io.output_dir).resolve(),
                history_feedback=evolution_cfg.history_feedback,
                persist_library_on_fallback=not evolution_cfg.require_glm_api,
            )
            if not evolution_cfg.require_glm_api or evolution_result.generation_mode == "glm_api":
                break

            if evolution_cfg.max_retries > 0 and attempt >= evolution_cfg.max_retries:
                break

            wait_sec = max(1, int(evolution_cfg.retry_wait_sec))
            import time

            time.sleep(wait_sec)

        accepted_factors = evolution_result.accepted_factors
        logs.append(
            {
                "stage": "evolve_factors",
                "seconds": (datetime.now() - t_stage).total_seconds(),
                "generation_mode": evolution_result.generation_mode,
                "accepted_count": int(evolution_result.accepted_count),
            }
        )

    best_factor = accepted_factors[0] if accepted_factors else None
    effective_predict_asof = _resolve_asof_date(pred_dates, predict_asof_date)
    best_signal = pd.DataFrame(columns=["instrument", "value", "zscore", "rank_pct", "last_date"])
    if best_factor is not None:
        best_signal = _factor_latest_signal(bundle.panel, best_factor["expression"], asof_date=effective_predict_asof)
        params = _extract_factor_params(best_factor["expression"])
        best_factor["parsed_params"] = params

    tomorrow_candidates = _tomorrow_candidates_from_factor(best_factor, best_signal, top_stock_k)
    model_candidates = _latest_model_candidates(
        model_result.predictions,
        asof_date=effective_predict_asof,
        top_k=top_stock_k,
    )

    runtime = (datetime.now() - t0).total_seconds()
    predicted_trade_date = (effective_predict_asof + BDay(1)).date().isoformat()
    return DashboardResult(
        data_mode=data_mode,
        panel=bundle.panel,
        factor_stats=factor_set.stats,
        model_result=model_result,
        backtest_result=backtest_result,
        evolution_result=evolution_result,
        accepted_factors=accepted_factors,
        best_factor=best_factor,
        best_factor_signal=best_signal,
        tomorrow_candidates=tomorrow_candidates,
        model_tomorrow_candidates=model_candidates,
        train_window={
            "train_start": str(cfg.train.train_start),
            "train_end": str(cfg.train.train_end),
        },
        backtest_window={
            "backtest_start": bt_start_ts.date().isoformat(),
            "backtest_end": bt_end_ts.date().isoformat(),
        },
        predict_asof_date=effective_predict_asof.date().isoformat(),
        predicted_trade_date=predicted_trade_date,
        stage_logs=logs,
        runtime_sec=runtime,
        benchmark_daily=bundle.benchmark,
    )


def run_dashboard_workflow_with_bundle(
    cfg: USAlphaConfig,
    bundle: MarketDataBundle,
    *,
    data_mode: str,
    evolution_cfg: EvolutionUIConfig | None = None,
    top_stock_k: int = 10,
    backtest_start: str | None = None,
    backtest_end: str | None = None,
    predict_asof_date: str | None = None,
) -> DashboardResult:
    """Run the dashboard workflow with a preloaded market bundle."""
    t0 = datetime.now()
    logs: list[dict[str, Any]] = [
        {
            "stage": "load_market_data",
            "seconds": 0.0,
            "data_mode": data_mode,
            "rows": int(len(bundle.panel)),
            "instruments": int(len(bundle.panel.columns.get_level_values(0).unique())),
        }
    ]

    project_root = Path(__file__).resolve().parents[1]
    alpha526_path = cfg.resolve_alpha526_path(project_root)

    t_stage = datetime.now()
    factor_set = compute_526_factors(bundle.panel, bundle.benchmark, alpha526_path=alpha526_path)
    logs.append(
        {
            "stage": "compute_526_factors",
            "seconds": (datetime.now() - t_stage).total_seconds(),
            "factor_count": int(factor_set.stats.get("factor_count", 0)),
        }
    )

    t_stage = datetime.now()
    model_result = train_factor_model(factor_set.features, bundle.panel, cfg.train)
    logs.append(
        {
            "stage": "train_model",
            "seconds": (datetime.now() - t_stage).total_seconds(),
            "samples_total": int(model_result.metrics.get("samples_total", 0)),
            "test_ic_mean": float(model_result.metrics.get("test_ic_mean", np.nan)),
        }
    )

    pred_dates = pd.DatetimeIndex(model_result.predictions.index.get_level_values("datetime")).unique().sort_values()
    bt_start_ts = pd.Timestamp(backtest_start) if backtest_start is not None else pred_dates.min()
    bt_end_ts = pd.Timestamp(backtest_end) if backtest_end is not None else pred_dates.max()
    if bt_start_ts > bt_end_ts:
        bt_start_ts, bt_end_ts = bt_end_ts, bt_start_ts

    backtest_pred = _filter_predictions_by_window(model_result.predictions, bt_start_ts, bt_end_ts)

    t_stage = datetime.now()
    backtest_result = run_long_short_backtest(
        backtest_pred,
        top_quantile=cfg.train.top_quantile,
        split="",
        benchmark_daily=bundle.benchmark,
    )
    logs.append(
        {
            "stage": "run_backtest",
            "seconds": (datetime.now() - t_stage).total_seconds(),
            "days": int(backtest_result.metrics.get("days", 0)),
            "sharpe": float(backtest_result.metrics.get("sharpe", np.nan)),
            "backtest_start": bt_start_ts.date().isoformat(),
            "backtest_end": bt_end_ts.date().isoformat(),
        }
    )

    evolution_result: EvolutionResult | None = None
    accepted_factors: list[dict[str, Any]] = []
    if evolution_cfg is not None and evolution_cfg.api_key.strip():
        attempt = 0
        t_stage = datetime.now()
        while True:
            attempt += 1
            evolution_result = run_evolution_round(
                panel=bundle.panel,
                alpha526_path=alpha526_path,
                api_key=evolution_cfg.api_key,
                num_candidates=evolution_cfg.num_candidates,
                top_k_accept=evolution_cfg.top_k_accept,
                label_horizon=cfg.train.label_horizon,
                top_quantile=cfg.train.top_quantile,
                model=evolution_cfg.model,
                temperature=evolution_cfg.temperature,
                output_dir=(project_root / cfg.io.output_dir).resolve(),
                history_feedback=evolution_cfg.history_feedback,
                persist_library_on_fallback=not evolution_cfg.require_glm_api,
            )
            if not evolution_cfg.require_glm_api or evolution_result.generation_mode == "glm_api":
                break
            if evolution_cfg.max_retries > 0 and attempt >= evolution_cfg.max_retries:
                break
            import time

            time.sleep(max(1, int(evolution_cfg.retry_wait_sec)))

        accepted_factors = evolution_result.accepted_factors
        logs.append(
            {
                "stage": "evolve_factors",
                "seconds": (datetime.now() - t_stage).total_seconds(),
                "generation_mode": evolution_result.generation_mode,
                "accepted_count": int(evolution_result.accepted_count),
            }
        )

    best_factor = accepted_factors[0] if accepted_factors else None
    effective_predict_asof = _resolve_asof_date(pred_dates, predict_asof_date)
    best_signal = pd.DataFrame(columns=["instrument", "value", "zscore", "rank_pct", "last_date"])
    if best_factor is not None:
        best_signal = _factor_latest_signal(bundle.panel, best_factor["expression"], asof_date=effective_predict_asof)
        best_factor["parsed_params"] = _extract_factor_params(best_factor["expression"])

    tomorrow_candidates = _tomorrow_candidates_from_factor(best_factor, best_signal, top_stock_k)
    model_candidates = _latest_model_candidates(
        model_result.predictions,
        asof_date=effective_predict_asof,
        top_k=top_stock_k,
    )

    runtime = (datetime.now() - t0).total_seconds()
    predicted_trade_date = (effective_predict_asof + BDay(1)).date().isoformat()
    return DashboardResult(
        data_mode=data_mode,
        panel=bundle.panel,
        factor_stats=factor_set.stats,
        model_result=model_result,
        backtest_result=backtest_result,
        evolution_result=evolution_result,
        accepted_factors=accepted_factors,
        best_factor=best_factor,
        best_factor_signal=best_signal,
        tomorrow_candidates=tomorrow_candidates,
        model_tomorrow_candidates=model_candidates,
        train_window={"train_start": str(cfg.train.train_start), "train_end": str(cfg.train.train_end)},
        backtest_window={"backtest_start": bt_start_ts.date().isoformat(), "backtest_end": bt_end_ts.date().isoformat()},
        predict_asof_date=effective_predict_asof.date().isoformat(),
        predicted_trade_date=predicted_trade_date,
        stage_logs=logs,
        runtime_sec=runtime,
        benchmark_daily=bundle.benchmark,
    )
