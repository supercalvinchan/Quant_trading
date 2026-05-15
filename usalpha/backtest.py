from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    daily: pd.DataFrame
    metrics: dict[str, Any]


def _safe_corr(pred: pd.Series, label: pd.Series) -> float:
    if pred.nunique(dropna=True) < 2 or label.nunique(dropna=True) < 2:
        return np.nan
    return float(pred.corr(label))


def run_long_short_backtest(
    predictions: pd.DataFrame,
    *,
    top_quantile: float = 0.2,
    split: str = "test",
    benchmark_daily: pd.DataFrame | None = None,
) -> BacktestResult:
    if not isinstance(predictions.index, pd.MultiIndex):
        raise ValueError("predictions must use MultiIndex (datetime, instrument)")

    df = predictions.copy()
    if split:
        df = df[df["split"] == split]

    rows: list[dict[str, Any]] = []
    q = float(top_quantile)
    if q <= 0.0 or q >= 0.5:
        raise ValueError("top_quantile should be in (0, 0.5)")

    for dt, g in df.groupby(level="datetime", sort=True):
        g = g[["pred", "label"]].dropna()
        if len(g) < 5:
            continue
        long_thr = g["pred"].quantile(1.0 - q)
        short_thr = g["pred"].quantile(q)

        long_bucket = g[g["pred"] >= long_thr]
        short_bucket = g[g["pred"] <= short_thr]
        if len(long_bucket) == 0 or len(short_bucket) == 0:
            continue

        long_ret = float(long_bucket["label"].mean())
        short_ret = float(short_bucket["label"].mean())
        ls_ret = long_ret - short_ret
        ic = _safe_corr(g["pred"], g["label"])

        long_stocks = [
            {"instrument": idx[-1], "pred": float(row["pred"]), "label": float(row["label"])}
            for idx, row in long_bucket.iterrows()
        ]
        short_stocks = [
            {"instrument": idx[-1], "pred": float(row["pred"]), "label": float(row["label"])}
            for idx, row in short_bucket.iterrows()
        ]

        rows.append(
            {
                "datetime": pd.Timestamp(dt),
                "long_ret": long_ret,
                "short_ret": short_ret,
                "ls_ret": ls_ret,
                "ic": ic,
                "n_long": int(len(long_bucket)),
                "n_short": int(len(short_bucket)),
                "long_stocks": long_stocks,
                "short_stocks": short_stocks,
            }
        )

    if len(rows) == 0:
        empty = pd.DataFrame(columns=["long_ret", "short_ret", "ls_ret", "ic", "n_long", "n_short", "cum_ls", "long_stocks", "short_stocks", "benchmark_ret", "cum_benchmark"])
        return BacktestResult(
            daily=empty,
            metrics={
                "days": 0,
                "annual_return": np.nan,
                "annual_volatility": np.nan,
                "sharpe": np.nan,
                "max_drawdown": np.nan,
                "ic_mean": np.nan,
                "ic_std": np.nan,
                "win_rate": np.nan,
            },
        )

    daily = pd.DataFrame(rows).set_index("datetime").sort_index()
    daily["cum_ls"] = (1.0 + daily["ls_ret"]).cumprod()

    # --- benchmark alignment ---
    if benchmark_daily is not None and "ret" in benchmark_daily.columns:
        bench_ret = benchmark_daily[["ret"]].copy()
        bench_ret.index = pd.to_datetime(bench_ret.index).tz_localize(None)
        bench_ret = bench_ret[~bench_ret.index.duplicated(keep="last")]
        daily["benchmark_ret"] = bench_ret.reindex(daily.index)["ret"]
        daily["cum_benchmark"] = (1.0 + daily["benchmark_ret"].fillna(0.0)).cumprod()
    else:
        daily["benchmark_ret"] = np.nan
        daily["cum_benchmark"] = np.nan

    rets = daily["ls_ret"].astype(float)
    ic_series = daily["ic"].astype(float)

    annual_factor = np.sqrt(252.0)
    ret_mean = float(rets.mean())
    ret_std = float(rets.std(ddof=0))

    final_cum = float(daily["cum_ls"].iloc[-1])
    if final_cum > 0:
        ann_ret = float(final_cum ** (252.0 / len(daily)) - 1.0)
    else:
        ann_ret = np.nan
    ann_vol = float(ret_std * annual_factor)
    sharpe = float((ret_mean / ret_std) * annual_factor) if ret_std > 0 else np.nan

    rolling_max = daily["cum_ls"].cummax()
    drawdown = daily["cum_ls"] / rolling_max - 1.0
    max_dd = float(drawdown.min())

    metrics = {
        "days": int(len(daily)),
        "annual_return": ann_ret,
        "annual_volatility": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "ic_mean": float(ic_series.mean()),
        "ic_std": float(ic_series.std(ddof=0)),
        "win_rate": float((rets > 0).mean()),
        "avg_n_long": float(daily["n_long"].mean()),
        "avg_n_short": float(daily["n_short"].mean()),
    }
    return BacktestResult(daily=daily, metrics=metrics)
