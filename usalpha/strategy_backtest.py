from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

from .data import MarketDataBundle
from .strategies import get_strategy
from .trading_methods import get_trading_method

try:
    import akshare as ak
except Exception:  # pylint: disable=broad-except
    ak = None


def build_data_by_symbol_from_bundle(bundle: MarketDataBundle) -> dict[str, pd.DataFrame]:
    panel = bundle.panel.sort_index()
    symbols = list(panel.columns.get_level_values(0).unique())
    out: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        frame = panel[symbol].copy()
        frame.columns = [str(col).lower() for col in frame.columns]
        out[str(symbol)] = frame.sort_index()
    return out


def _normalize_trade_unit(value: Any) -> int | None:
    if value is None:
        return None
    unit = int(value)
    if unit <= 0:
        raise ValueError("trade_unit must be > 0 or null")
    return unit


def _round_shares(shares: float, trade_unit: int | None) -> float:
    if trade_unit is None:
        return float(np.floor(max(shares, 0.0)))
    return float(np.floor(max(shares, 0.0) / trade_unit) * trade_unit)


def _safe_div(left: float, right: float) -> float:
    if abs(right) <= 1e-12:
        return 0.0
    return float(left / right)


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return float("nan")
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    return float(drawdown.min())


def _spearman_corr(left: pd.Series, right: pd.Series) -> float:
    pair = pd.concat([left.rename("left"), right.rename("right")], axis=1).dropna()
    if len(pair) < 2 or pair["left"].nunique() < 2 or pair["right"].nunique() < 2:
        return float("nan")
    ranked = pair.rank(method="average")
    return float(ranked["left"].corr(ranked["right"]))


def _compute_horizon_eval_rows(
    *,
    close_matrix: pd.DataFrame,
    score_series_by_date: dict[pd.Timestamp, pd.Series],
    horizons: list[int],
    topk: int,
) -> tuple[dict[int, pd.DataFrame], dict[int, float], dict[int, float], dict[int, float]]:
    horizon_rows: dict[int, list[dict[str, Any]]] = {h: [] for h in horizons}
    horizon_ic_mean: dict[int, float] = {}
    horizon_topk_mean: dict[int, float] = {}
    horizon_topk_excess_mean: dict[int, float] = {}

    date_index = pd.Index(close_matrix.index)
    for signal_date, scores in score_series_by_date.items():
        if signal_date not in date_index:
            continue
        signal_loc = int(date_index.get_loc(signal_date))
        current_close = close_matrix.loc[signal_date].reindex(scores.index)
        for horizon in horizons:
            future_loc = signal_loc + int(horizon)
            if future_loc >= len(date_index):
                continue
            future_date = pd.Timestamp(date_index[future_loc])
            future_close = close_matrix.loc[future_date].reindex(scores.index)
            future_ret = pd.to_numeric(future_close / current_close - 1.0, errors="coerce")
            pair = pd.concat([scores.rename("score"), future_ret.rename("future_ret")], axis=1).dropna()
            if len(pair) < 3 or pair["score"].nunique() < 2 or pair["future_ret"].nunique() < 2:
                continue
            ic_value = _spearman_corr(pair["score"], pair["future_ret"])
            ranked = pair.sort_values("score", ascending=False)
            top_slice = ranked.head(max(int(topk), 1))
            topk_ret = float(top_slice["future_ret"].mean()) if not top_slice.empty else float("nan")
            cross_mean = float(pair["future_ret"].mean()) if not pair.empty else float("nan")
            horizon_rows[horizon].append(
                {
                    "date": future_date,
                    "signal_date": signal_date,
                    "horizon": int(horizon),
                    "ic": ic_value,
                    "topk_future_return": topk_ret,
                    "cross_section_mean_return": cross_mean,
                    "topk_excess_return": topk_ret - cross_mean if np.isfinite(topk_ret) and np.isfinite(cross_mean) else float("nan"),
                    "coverage": int(len(pair)),
                }
            )

    horizon_daily: dict[int, pd.DataFrame] = {}
    for horizon in horizons:
        frame = pd.DataFrame(horizon_rows[horizon])
        horizon_daily[horizon] = frame
        if frame.empty:
            horizon_ic_mean[horizon] = float("nan")
            horizon_topk_mean[horizon] = float("nan")
            horizon_topk_excess_mean[horizon] = float("nan")
        else:
            horizon_ic_mean[horizon] = float(frame["ic"].mean())
            horizon_topk_mean[horizon] = float(frame["topk_future_return"].mean())
            horizon_topk_excess_mean[horizon] = float(frame["topk_excess_return"].mean())
    return horizon_daily, horizon_ic_mean, horizon_topk_mean, horizon_topk_excess_mean


def _build_price_matrices(data_by_symbol: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    fields = ["open", "high", "low", "close", "volume", "vwap"]
    out: dict[str, pd.DataFrame] = {}
    for field in fields:
        series_map: dict[str, pd.Series] = {}
        for symbol, frame in data_by_symbol.items():
            if field in frame.columns:
                series_map[str(symbol)] = pd.to_numeric(frame[field], errors="coerce")
        out[field] = pd.DataFrame(series_map).sort_index() if series_map else pd.DataFrame()
    return out


def _normalize_cn_symbol(value: str) -> str:
    raw = str(value or "").strip().upper()
    if raw.startswith(("SH", "SZ", "BJ")):
        raw = raw[2:]
    digits = "".join(ch for ch in raw if ch.isdigit())
    return digits.zfill(6) if digits else ""


def _fetch_cn_index_close_history(index_code: str) -> pd.Series:
    if ak is None:
        raise RuntimeError("dynamic CN topk requires akshare")
    symbol = _normalize_cn_symbol(index_code)
    errors: list[str] = []
    for fn_name, kwargs in [
        ("stock_zh_index_daily", {"symbol": f"sh{symbol}" if symbol == "000001" else symbol}),
        ("index_zh_a_hist", {"symbol": symbol, "period": "daily"}),
    ]:
        try:
            fn = getattr(ak, fn_name)
            raw = fn(**kwargs)
            if raw is None or raw.empty:
                continue
            frame = raw.copy()
            rename_map = {
                "date": "date",
                "日期": "date",
                "close": "close",
                "收盘": "close",
            }
            frame.columns = [rename_map.get(str(col), str(col)) for col in frame.columns]
            if "date" not in frame.columns or "close" not in frame.columns:
                continue
            frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None)
            frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
            frame = frame.dropna(subset=["date", "close"]).drop_duplicates(subset=["date"], keep="last")
            if frame.empty:
                continue
            return frame.set_index("date")["close"].sort_index()
        except Exception as exc:  # pylint: disable=broad-except
            errors.append(f"{fn_name}: {exc}")
    raise RuntimeError(f"failed to fetch CN index history for {symbol}: {'; '.join(errors)}")


def _get_signal_day_limit_up_holds(
    *,
    signal_date: pd.Timestamp,
    current_positions: list[str],
    close_matrix: pd.DataFrame,
    limit_threshold: float,
) -> set[str]:
    if signal_date not in close_matrix.index or not current_positions:
        return set()
    index_pos = close_matrix.index.get_loc(signal_date)
    if int(index_pos) <= 0:
        return set()
    today_row = close_matrix.loc[signal_date]
    prev_row = close_matrix.iloc[int(index_pos) - 1]
    out: set[str] = set()
    for inst in current_positions:
        close_price = pd.to_numeric(today_row.get(inst), errors="coerce")
        prev_close = pd.to_numeric(prev_row.get(inst), errors="coerce")
        if pd.isna(close_price) or pd.isna(prev_close) or float(prev_close) <= 0:
            continue
        if float(close_price) / float(prev_close) - 1.0 >= float(limit_threshold):
            out.add(str(inst).lower())
    return out


def _resolve_dynamic_topk_series(
    *,
    trading_method_type: str,
    trading_method_params: dict[str, Any],
    date_index: pd.Index,
) -> pd.Series | None:
    if str(trading_method_type).strip().lower() != "topk_dropout":
        return None
    if not bool(trading_method_params.get("dynamic_topk_enabled", False)):
        return None
    index_code = trading_method_params.get("dynamic_topk_index_code")
    if not index_code:
        return None
    ma_window = int(trading_method_params.get("dynamic_topk_ma_window", 10))
    raw_map = trading_method_params.get("dynamic_topk_map", []) or []
    if not raw_map:
        return None

    close = _fetch_cn_index_close_history(str(index_code)).reindex(pd.to_datetime(date_index)).ffill()
    ma = close.rolling(ma_window, min_periods=ma_window).mean()
    diff = close - ma
    out = pd.Series(index=pd.to_datetime(date_index), dtype=float)
    base_topk = int(trading_method_params["topk"])
    for date, value in diff.items():
        if pd.isna(value):
            out.loc[date] = float(base_topk)
            continue
        assigned = None
        for rule in raw_map:
            min_diff = rule.get("min_diff")
            max_diff = rule.get("max_diff")
            if min_diff is not None and float(value) < float(min_diff):
                continue
            if max_diff is not None and float(value) >= float(max_diff):
                continue
            assigned = int(rule["topk"])
            break
        out.loc[date] = float(assigned if assigned is not None else base_topk)
    return out.ffill().fillna(float(base_topk))


def _compute_limit_flags(
    *,
    close_price: float | None,
    prev_close_price: float | None,
    suspended: bool,
    limit_threshold: float,
    forbid_all_trade_at_limit: bool,
) -> tuple[bool, bool, str]:
    if suspended:
        return True, True, "suspended"
    if close_price is None or prev_close_price is None or prev_close_price <= 0:
        return False, False, "missing_prev_close"
    pct_change = close_price / prev_close_price - 1.0
    limit_buy = bool(pct_change >= limit_threshold)
    limit_sell = bool(pct_change <= -limit_threshold)
    if forbid_all_trade_at_limit and (limit_buy or limit_sell):
        return True, True, "limit_blocked"
    return limit_buy, limit_sell, "prev_close_change"


def _build_market_snapshot_for_day(
    *,
    execution_date: pd.Timestamp,
    price_matrices: dict[str, pd.DataFrame],
    params: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    close_matrix = price_matrices["close"]
    if execution_date not in close_matrix.index:
        return {}
    close_row = close_matrix.loc[execution_date]
    index_pos = close_matrix.index.get_loc(execution_date)
    prev_close_row = close_matrix.iloc[index_pos - 1] if index_pos > 0 else pd.Series(index=close_row.index, dtype=float)
    deal_field = str(params["deal_price"]).strip().lower()
    chosen_matrix = price_matrices.get(deal_field) if deal_field in price_matrices else None
    if chosen_matrix is None or chosen_matrix.empty or execution_date not in chosen_matrix.index:
        chosen_row = close_row
    else:
        chosen_row = chosen_matrix.loc[execution_date]
    volume_row = price_matrices["volume"].loc[execution_date] if execution_date in price_matrices["volume"].index else pd.Series(index=close_row.index, dtype=float)

    snapshots: dict[str, dict[str, Any]] = {}
    for symbol in close_row.index:
        close_price = pd.to_numeric(close_row.get(symbol), errors="coerce")
        prev_close = pd.to_numeric(prev_close_row.get(symbol), errors="coerce")
        deal_price = pd.to_numeric(chosen_row.get(symbol), errors="coerce")
        volume = pd.to_numeric(volume_row.get(symbol), errors="coerce")
        close_value = None if pd.isna(close_price) or float(close_price) <= 0 else float(close_price)
        prev_close_value = None if pd.isna(prev_close) or float(prev_close) <= 0 else float(prev_close)
        deal_value = None if pd.isna(deal_price) or float(deal_price) <= 0 else float(deal_price)
        suspended = deal_value is None
        limit_buy, limit_sell, limit_state = _compute_limit_flags(
            close_price=close_value,
            prev_close_price=prev_close_value,
            suspended=suspended,
            limit_threshold=float(params["limit_threshold"]),
            forbid_all_trade_at_limit=bool(params["forbid_all_trade_at_limit"]),
        )
        snapshots[str(symbol).lower()] = {
            "deal_price": deal_value,
            "close": close_value,
            "prev_close": prev_close_value,
            "volume": None if pd.isna(volume) or float(volume) < 0 else float(volume),
            "suspended": bool(suspended),
            "limit_buy": bool(limit_buy),
            "limit_sell": bool(limit_sell),
            "limit_state": limit_state,
        }
    return snapshots


def _mark_positions_to_market(positions: dict[str, dict[str, float]], market_data: dict[str, dict[str, Any]]) -> None:
    for inst, pos in positions.items():
        snapshot = market_data.get(inst)
        if not snapshot:
            continue
        close_price = snapshot.get("close")
        if close_price is not None and float(close_price) > 0:
            pos["current_price"] = float(close_price)


def _portfolio_value(positions: dict[str, dict[str, float]], cash: float) -> float:
    value = float(cash)
    for pos in positions.values():
        shares = float(pos.get("shares", 0.0))
        price = float(pos.get("current_price", 0.0))
        if shares > 0 and price > 0:
            value += shares * price
    return value


def _execute_rebalance_plan_for_day(
    *,
    signal_date: str,
    execution_date: str,
    positions: dict[str, dict[str, float]],
    cash: float,
    market_data: dict[str, dict[str, Any]],
    sell_orders: list[dict[str, Any]],
    buy_orders: list[dict[str, Any]],
    target_cash_ratio: float,
    execution_params: dict[str, Any],
) -> dict[str, Any]:
    open_cost = float(execution_params["open_cost"])
    close_cost = float(execution_params["close_cost"])
    min_cost = float(execution_params["min_cost"])
    impact_cost = float(execution_params["impact_cost"])
    trade_unit = _normalize_trade_unit(execution_params["trade_unit"])
    volume_limit_ratio = execution_params["volume_limit_ratio"]

    target_cash_ratio = min(max(float(target_cash_ratio), 0.0), 1.0)
    _mark_positions_to_market(positions, market_data)

    executed_trades: list[dict[str, Any]] = []
    failed_orders: list[dict[str, Any]] = []
    skipped_reasons: dict[str, int] = defaultdict(int)
    dealt_sell_amount: dict[str, float] = defaultdict(float)
    dealt_buy_amount: dict[str, float] = defaultdict(float)

    for order in sell_orders:
        inst = str(order.get("instrument", "")).strip().lower()
        ratio = float(order.get("sell_ratio", 0.0))
        pos = positions.get(inst)
        if pos is None:
            skipped_reasons["sell_not_held"] += 1
            failed_orders.append({"direction": "sell", "instrument": inst, "reason_code": "sell_not_held"})
            continue
        shares_holding = float(pos.get("shares", 0.0))
        if shares_holding <= 0:
            skipped_reasons["sell_zero_shares"] += 1
            failed_orders.append({"direction": "sell", "instrument": inst, "reason_code": "sell_zero_shares"})
            continue
        snapshot = market_data.get(inst)
        if not snapshot:
            skipped_reasons["sell_missing_market_data"] += 1
            failed_orders.append({"direction": "sell", "instrument": inst, "reason_code": "sell_missing_market_data"})
            continue
        if bool(snapshot.get("suspended", True)):
            skipped_reasons["sell_suspended"] += 1
            failed_orders.append({"direction": "sell", "instrument": inst, "reason_code": "sell_suspended"})
            continue
        if bool(snapshot.get("limit_sell", False)):
            skipped_reasons["sell_limit"] += 1
            failed_orders.append({"direction": "sell", "instrument": inst, "reason_code": "sell_limit"})
            continue
        trade_price = snapshot.get("deal_price")
        if trade_price is None or float(trade_price) <= 0:
            skipped_reasons["sell_invalid_price"] += 1
            failed_orders.append({"direction": "sell", "instrument": inst, "reason_code": "sell_invalid_price"})
            continue
        target_sell_amount = min(shares_holding, shares_holding * ratio)
        if volume_limit_ratio is not None and snapshot.get("volume") is not None:
            max_by_volume = float(snapshot["volume"]) * float(volume_limit_ratio) - dealt_sell_amount[inst]
            target_sell_amount = min(target_sell_amount, max(0.0, max_by_volume))
        if target_sell_amount < shares_holding:
            target_sell_amount = _round_shares(target_sell_amount, trade_unit)
        if target_sell_amount <= 0:
            skipped_reasons["sell_amount_zero_after_constraints"] += 1
            failed_orders.append({"direction": "sell", "instrument": inst, "reason_code": "sell_amount_zero_after_constraints"})
            continue
        trade_val = float(target_sell_amount) * float(trade_price)
        trade_cost = max(trade_val * (close_cost + impact_cost), min_cost)
        cash += trade_val - trade_cost
        dealt_sell_amount[inst] += float(target_sell_amount)
        remaining = shares_holding - float(target_sell_amount)
        cost_price = float(pos.get("cost_price", 0.0))
        realized_pnl = (float(trade_price) - cost_price) * float(target_sell_amount) - trade_cost
        if remaining <= 1e-9:
            positions.pop(inst, None)
        else:
            pos["shares"] = float(remaining)
            pos["current_price"] = float(snapshot.get("close") or trade_price)
        executed_trades.append(
            {
                "signal_date": signal_date,
                "execution_date": execution_date,
                "instrument": inst,
                "side": "sell",
                "shares": float(target_sell_amount),
                "price": float(trade_price),
                "turnover": trade_val,
                "trade_cost": trade_cost,
                "realized_pnl": realized_pnl,
            }
        )

    normalized_buy_orders: list[dict[str, Any]] = []
    total_weight = 0.0
    for order in buy_orders:
        inst = str(order.get("instrument", "")).strip().lower()
        weight = float(order.get("weight", 0.0))
        if not inst or weight <= 0:
            continue
        snapshot = market_data.get(inst)
        if not snapshot:
            skipped_reasons["buy_missing_market_data"] += 1
            failed_orders.append({"direction": "buy", "instrument": inst, "reason_code": "buy_missing_market_data"})
            continue
        if bool(snapshot.get("suspended", True)):
            skipped_reasons["buy_suspended"] += 1
            failed_orders.append({"direction": "buy", "instrument": inst, "reason_code": "buy_suspended"})
            continue
        if bool(snapshot.get("limit_buy", False)):
            skipped_reasons["buy_limit"] += 1
            failed_orders.append({"direction": "buy", "instrument": inst, "reason_code": "buy_limit"})
            continue
        trade_price = snapshot.get("deal_price")
        if trade_price is None or float(trade_price) <= 0:
            skipped_reasons["buy_invalid_price"] += 1
            failed_orders.append({"direction": "buy", "instrument": inst, "reason_code": "buy_invalid_price"})
            continue
        normalized_buy_orders.append({"instrument": inst, "weight": weight})
        total_weight += weight

    portfolio_value_before_buy = _portfolio_value(positions, cash)
    target_cash = portfolio_value_before_buy * target_cash_ratio
    buy_budget = max(0.0, cash - target_cash)
    for order in normalized_buy_orders:
        inst = order["instrument"]
        normalized_weight = _safe_div(float(order["weight"]), total_weight)
        alloc_budget = buy_budget * normalized_weight
        snapshot = market_data[inst]
        trade_price = float(snapshot["deal_price"])
        max_shares_by_budget = _round_shares(alloc_budget / (trade_price * (1.0 + open_cost + impact_cost)), trade_unit)
        if volume_limit_ratio is not None and snapshot.get("volume") is not None:
            max_by_volume = float(snapshot["volume"]) * float(volume_limit_ratio) - dealt_buy_amount[inst]
            max_shares_by_budget = min(max_shares_by_budget, _round_shares(max_by_volume, trade_unit))
        if max_shares_by_budget <= 0:
            skipped_reasons["buy_amount_zero_after_constraints"] += 1
            failed_orders.append({"direction": "buy", "instrument": inst, "reason_code": "buy_amount_zero_after_constraints"})
            continue
        trade_val = float(max_shares_by_budget) * trade_price
        trade_cost = max(trade_val * (open_cost + impact_cost), min_cost)
        total_cash_need = trade_val + trade_cost
        if total_cash_need > cash + 1e-9:
            max_shares_by_budget = _round_shares(cash / (trade_price * (1.0 + open_cost + impact_cost)), trade_unit)
            trade_val = float(max_shares_by_budget) * trade_price
            trade_cost = max(trade_val * (open_cost + impact_cost), min_cost) if trade_val > 0 else 0.0
            total_cash_need = trade_val + trade_cost
        if max_shares_by_budget <= 0 or total_cash_need > cash + 1e-9:
            skipped_reasons["buy_cash_insufficient"] += 1
            failed_orders.append({"direction": "buy", "instrument": inst, "reason_code": "buy_cash_insufficient"})
            continue
        cash -= total_cash_need
        dealt_buy_amount[inst] += float(max_shares_by_budget)
        existing = positions.get(inst)
        if existing is None:
            positions[inst] = {
                "shares": float(max_shares_by_budget),
                "cost_price": _safe_div(trade_val + trade_cost, float(max_shares_by_budget)),
                "current_price": float(snapshot.get("close") or trade_price),
            }
        else:
            prev_shares = float(existing.get("shares", 0.0))
            prev_cost = float(existing.get("cost_price", 0.0))
            new_shares = prev_shares + float(max_shares_by_budget)
            avg_cost = _safe_div(prev_shares * prev_cost + trade_val + trade_cost, new_shares)
            existing["shares"] = new_shares
            existing["cost_price"] = avg_cost
            existing["current_price"] = float(snapshot.get("close") or trade_price)
        executed_trades.append(
            {
                "signal_date": signal_date,
                "execution_date": execution_date,
                "instrument": inst,
                "side": "buy",
                "shares": float(max_shares_by_budget),
                "price": trade_price,
                "turnover": trade_val,
                "trade_cost": trade_cost,
                "realized_pnl": None,
            }
        )

    _mark_positions_to_market(positions, market_data)
    equity = _portfolio_value(positions, cash)
    return {
        "positions": positions,
        "cash": float(cash),
        "equity": float(equity),
        "trades": executed_trades,
        "requested_sell_orders": len(sell_orders),
        "requested_buy_orders": len(buy_orders),
        "executed_sell_orders": sum(1 for item in executed_trades if item["side"] == "sell"),
        "executed_buy_orders": sum(1 for item in executed_trades if item["side"] == "buy"),
        "failed_orders": failed_orders,
        "skipped_reasons": dict(skipped_reasons),
    }


def _apply_position_exit_rules(
    *,
    signal_date: str,
    execution_date: str,
    positions: dict[str, dict[str, float]],
    cash: float,
    market_data: dict[str, dict[str, Any]],
    execution_params: dict[str, Any],
) -> dict[str, Any]:
    take_profit_multiple = execution_params.get("take_profit_multiple")
    stop_loss_pct = execution_params.get("stop_loss_pct")
    if take_profit_multiple is None and stop_loss_pct is None:
        return {
            "positions": positions,
            "cash": float(cash),
            "equity": _portfolio_value(positions, cash),
            "trades": [],
            "failed_orders": [],
            "skipped_reasons": {},
        }

    sell_orders: list[dict[str, Any]] = []
    for inst, pos in positions.items():
        snapshot = market_data.get(inst)
        if not snapshot:
            continue
        trade_price = snapshot.get("deal_price")
        cost_price = float(pos.get("cost_price", 0.0))
        if trade_price is None or cost_price <= 0:
            continue
        should_sell = False
        if take_profit_multiple is not None and float(trade_price) >= cost_price * float(take_profit_multiple):
            should_sell = True
        if stop_loss_pct is not None and float(trade_price) < cost_price * (1.0 - float(stop_loss_pct)):
            should_sell = True
        if should_sell:
            sell_orders.append({"instrument": inst, "sell_ratio": 1.0})

    if not sell_orders:
        return {
            "positions": positions,
            "cash": float(cash),
            "equity": _portfolio_value(positions, cash),
            "trades": [],
            "failed_orders": [],
            "skipped_reasons": {},
        }
    return _execute_rebalance_plan_for_day(
        signal_date=signal_date,
        execution_date=execution_date,
        positions=positions,
        cash=cash,
        market_data=market_data,
        sell_orders=sell_orders,
        buy_orders=[],
        target_cash_ratio=0.0,
        execution_params=execution_params,
    )


def run_strategy_backtest(
    *,
    data_by_symbol: dict[str, pd.DataFrame],
    benchmark: pd.DataFrame | None,
    strategy_type: str,
    strategy_params: dict[str, Any],
    trading_method_type: str,
    trading_method_params: dict[str, Any],
    start: str,
    end: str,
    top_signal_count: int = 10,
) -> dict[str, Any]:
    if not data_by_symbol:
        raise ValueError("data_by_symbol cannot be empty")
    strategy = get_strategy(strategy_type)
    trading_method = get_trading_method(trading_method_type)
    strategy_runtime_params = dict(strategy_params or {})
    strategy_runtime_params["__benchmark__"] = benchmark
    score_matrix = strategy.generate_score_matrix(data_by_symbol, strategy_runtime_params)
    if score_matrix.empty:
        raise ValueError("strategy generated empty score matrix")

    price_matrices = _build_price_matrices(data_by_symbol)
    close_matrix = price_matrices["close"]
    open_matrix = price_matrices["open"]
    available_dates = close_matrix.index.intersection(score_matrix.index)
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    dates = list(available_dates[(available_dates >= start_ts) & (available_dates <= end_ts)])
    if len(dates) < 2:
        raise ValueError("backtest requires at least two trading days")

    resolved_trading_params = trading_method.resolve_params(trading_method_params)
    dynamic_topk_series = _resolve_dynamic_topk_series(
        trading_method_type=trading_method_type,
        trading_method_params=resolved_trading_params,
        date_index=pd.Index(dates),
    )
    positions: dict[str, dict[str, float]] = {}
    cash = float(resolved_trading_params["initial_capital"])
    daily_rows: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    execution_daily: list[dict[str, Any]] = []
    ic_rows: list[dict[str, Any]] = []
    score_series_by_date: dict[pd.Timestamp, pd.Series] = {}

    for idx in range(len(dates) - 1):
        signal_date = pd.Timestamp(dates[idx]).normalize()
        execution_date = pd.Timestamp(dates[idx + 1]).normalize()
        scores = pd.to_numeric(score_matrix.loc[signal_date], errors="coerce").dropna().sort_values(ascending=False)
        if not scores.empty:
            score_series_by_date[pd.Timestamp(signal_date)] = scores.copy()
        market_data = _build_market_snapshot_for_day(
            execution_date=execution_date,
            price_matrices=price_matrices,
            params=resolved_trading_params,
        )
        raw_signals = [
            {
                "instrument": str(inst).lower(),
                "score": float(score),
                "rank": rank,
                "action": "buy",
            }
            for rank, (inst, score) in enumerate(scores.items(), start=1)
        ]
        signals = raw_signals
        if bool(resolved_trading_params.get("exclude_suspended_candidates", False)):
            signals = [
                item
                for item in raw_signals
                if item["instrument"] in market_data
                and not bool(market_data[item["instrument"]].get("suspended", True))
                and not bool(market_data[item["instrument"]].get("limit_buy", False))
            ]
        plan = trading_method.generate_rebalance_plan(
            signals=signals,
            current_positions=list(positions.keys()),
            params={
                **resolved_trading_params,
                "effective_topk": int(dynamic_topk_series.loc[signal_date]) if dynamic_topk_series is not None and signal_date in dynamic_topk_series.index else int(resolved_trading_params["topk"]),
            },
            context={
                "signal_date": signal_date.date().isoformat(),
                "execution_date": execution_date.date().isoformat(),
            },
        )
        protected_holds = _get_signal_day_limit_up_holds(
            signal_date=signal_date,
            current_positions=list(positions.keys()),
            close_matrix=close_matrix,
            limit_threshold=float(resolved_trading_params["limit_threshold"]),
        ) if bool(resolved_trading_params.get("hold_limit_up_positions", False)) else set()
        if protected_holds:
            sell_instruments = [inst for inst in plan.get("sell_instruments", []) if str(inst).lower() not in protected_holds]
            plan["sell_instruments"] = sell_instruments
            plan["sell_ratios"] = {
                str(inst).lower(): float(value)
                for inst, value in plan.get("sell_ratios", {}).items()
                if str(inst).lower() in set(sell_instruments)
            }
        exit_summary = _apply_position_exit_rules(
            signal_date=signal_date.date().isoformat(),
            execution_date=execution_date.date().isoformat(),
            positions=positions,
            cash=cash,
            market_data=market_data,
            execution_params=resolved_trading_params,
        )
        positions = exit_summary["positions"]
        cash = float(exit_summary["cash"])
        trades.extend(exit_summary["trades"])
        summary = _execute_rebalance_plan_for_day(
            signal_date=signal_date.date().isoformat(),
            execution_date=execution_date.date().isoformat(),
            positions=positions,
            cash=cash,
            market_data=market_data,
            sell_orders=[
                {"instrument": inst, "sell_ratio": float(plan.get("sell_ratios", {}).get(inst, 1.0))}
                for inst in plan.get("sell_instruments", [])
            ],
            buy_orders=[
                {"instrument": inst, "weight": float(plan.get("buy_weights", {}).get(inst, 0.0))}
                for inst in plan.get("buy_instruments", [])
            ],
            target_cash_ratio=float(plan.get("target_cash_ratio", 0.0)),
            execution_params=resolved_trading_params,
        )
        positions = summary["positions"]
        cash = float(summary["cash"])
        trades.extend(summary["trades"])
        holdings = [
            {
                "symbol": symbol,
                "shares": int(round(pos.get("shares", 0.0))),
                "close": float(pos.get("current_price", 0.0)),
                "market_value": float(pos.get("shares", 0.0)) * float(pos.get("current_price", 0.0)),
                "cost_price": float(pos.get("cost_price", 0.0)),
            }
            for symbol, pos in sorted(positions.items())
            if float(pos.get("shares", 0.0)) > 0
        ]
        daily_rows.append(
            {
                "datetime": execution_date,
                "signal_date": signal_date,
                "equity": float(summary["equity"]),
                "cash": float(cash),
                "holdings": holdings,
                "signal_count": int(len(signals)),
                "requested_sell_orders": int(summary["requested_sell_orders"]),
                "requested_buy_orders": int(summary["requested_buy_orders"]),
                "executed_sell_orders": int(summary["executed_sell_orders"]),
                "executed_buy_orders": int(summary["executed_buy_orders"]),
                "effective_topk": int(dynamic_topk_series.loc[signal_date]) if dynamic_topk_series is not None and signal_date in dynamic_topk_series.index else int(resolved_trading_params["topk"]),
                "protected_limit_up_count": int(len(protected_holds)),
                "candidate_signal_count": int(len(signals)),
            }
        )
        execution_daily.append(
            {
                "signal_date": signal_date.date().isoformat(),
                "execution_date": execution_date.date().isoformat(),
                "effective_topk": int(dynamic_topk_series.loc[signal_date]) if dynamic_topk_series is not None and signal_date in dynamic_topk_series.index else int(resolved_trading_params["topk"]),
                "exit_trade_count": int(len(exit_summary["trades"])),
                "protected_limit_up_count": int(len(protected_holds)),
                "candidate_signal_count": int(len(signals)),
                "requested_sell_orders": int(summary["requested_sell_orders"]),
                "requested_buy_orders": int(summary["requested_buy_orders"]),
                "executed_sell_orders": int(summary["executed_sell_orders"]),
                "executed_buy_orders": int(summary["executed_buy_orders"]),
                "failed_order_count": int(len(summary["failed_orders"])),
                "skipped_reasons": summary["skipped_reasons"],
            }
        )
        for rank, (inst, score) in enumerate(scores.head(int(top_signal_count)).items(), start=1):
            selections.append(
                {
                    "date": signal_date,
                    "execution_date": execution_date,
                    "rank": rank,
                    "symbol": str(inst),
                    "score": float(score),
                }
            )
        next_close = close_matrix.loc[execution_date].reindex(scores.index)
        current_close = close_matrix.loc[signal_date].reindex(scores.index)
        next_ret = pd.to_numeric(next_close / current_close - 1.0, errors="coerce")
        ic_rows.append({"date": execution_date, "ic": _spearman_corr(scores, next_ret)})

    daily = pd.DataFrame(daily_rows).set_index("datetime").sort_index()
    if daily.empty:
        raise ValueError("backtest generated no daily rows")
    daily["daily_return"] = daily["equity"].pct_change().fillna(daily["equity"] / float(resolved_trading_params["initial_capital"]) - 1.0)
    daily["cum_strategy"] = daily["equity"] / float(resolved_trading_params["initial_capital"])

    benchmark_return = float("nan")
    if benchmark is not None and not benchmark.empty and "close" in benchmark.columns:
        benchmark_curve = benchmark.loc[(benchmark.index >= daily.index.min()) & (benchmark.index <= daily.index.max())].copy()
        benchmark_curve = benchmark_curve.reindex(daily.index).ffill().dropna(subset=["close"])
        if not benchmark_curve.empty:
            daily["benchmark_equity"] = benchmark_curve["close"] / float(benchmark_curve["close"].iloc[0]) * float(resolved_trading_params["initial_capital"])
            daily["cum_benchmark"] = daily["benchmark_equity"] / float(resolved_trading_params["initial_capital"])
            daily["benchmark_return"] = daily["benchmark_equity"].pct_change().fillna(0.0)
            benchmark_return = float(daily["cum_benchmark"].iloc[-1] - 1.0)

    ret = daily["daily_return"].dropna()
    sharpe = float(ret.mean() / ret.std(ddof=0) * np.sqrt(252.0)) if len(ret) > 1 and ret.std(ddof=0) > 0 else float("nan")
    ic_daily = pd.DataFrame(ic_rows)
    ic_mean = float(ic_daily["ic"].mean()) if not ic_daily.empty else float("nan")
    horizons = [1, 5, 10, 20]
    horizon_daily, horizon_ic_mean, horizon_topk_mean, horizon_topk_excess_mean = _compute_horizon_eval_rows(
        close_matrix=close_matrix,
        score_series_by_date=score_series_by_date,
        horizons=horizons,
        topk=int(top_signal_count),
    )

    skipped_totals: dict[str, int] = defaultdict(int)
    for item in execution_daily:
        for key, value in item["skipped_reasons"].items():
            skipped_totals[key] += int(value)

    metrics = {
        "final_equity": float(daily["equity"].iloc[-1]),
        "total_return": float(daily["cum_strategy"].iloc[-1] - 1.0),
        "benchmark_return": benchmark_return,
        "sharpe": sharpe,
        "ic": ic_mean,
        "ic_1d": horizon_ic_mean.get(1, float("nan")),
        "ic_5d": horizon_ic_mean.get(5, float("nan")),
        "ic_10d": horizon_ic_mean.get(10, float("nan")),
        "ic_20d": horizon_ic_mean.get(20, float("nan")),
        "topk_ret_1d": horizon_topk_mean.get(1, float("nan")),
        "topk_ret_5d": horizon_topk_mean.get(5, float("nan")),
        "topk_ret_10d": horizon_topk_mean.get(10, float("nan")),
        "topk_ret_20d": horizon_topk_mean.get(20, float("nan")),
        "topk_excess_1d": horizon_topk_excess_mean.get(1, float("nan")),
        "topk_excess_5d": horizon_topk_excess_mean.get(5, float("nan")),
        "topk_excess_10d": horizon_topk_excess_mean.get(10, float("nan")),
        "topk_excess_20d": horizon_topk_excess_mean.get(20, float("nan")),
        "max_drawdown": _max_drawdown(daily["equity"]),
        "trade_days": int(len(daily)),
    }

    latest_signal_date = pd.Timestamp(score_matrix.index.max()).normalize()
    tomorrow_trade_date = pd.Timestamp(latest_signal_date + BDay(1)).normalize()
    latest_scores = pd.to_numeric(score_matrix.loc[latest_signal_date], errors="coerce").dropna().sort_values(ascending=False)
    tomorrow_candidates = pd.DataFrame(
        [
            {
                "date": latest_signal_date,
                "trade_date": tomorrow_trade_date,
                "rank": rank,
                "symbol": str(inst),
                "score": float(score),
            }
            for rank, (inst, score) in enumerate(latest_scores.head(int(top_signal_count)).items(), start=1)
        ]
    )
    return {
        "strategy_type": str(strategy_type),
        "trading_method_type": str(trading_method_type),
        "daily": daily,
        "metrics": metrics,
        "trades": pd.DataFrame(trades),
        "selections": pd.DataFrame(selections),
        "tomorrow_candidates": tomorrow_candidates,
        "tomorrow_signal_date": latest_signal_date,
        "tomorrow_trade_date": tomorrow_trade_date,
        "ic_daily": ic_daily,
        "horizon_ic_daily": horizon_daily,
        "execution_diagnostics": {
            "daily": execution_daily,
            "totals": {
                "requested_sell_orders": int(sum(item["requested_sell_orders"] for item in execution_daily)),
                "requested_buy_orders": int(sum(item["requested_buy_orders"] for item in execution_daily)),
                "executed_sell_orders": int(sum(item["executed_sell_orders"] for item in execution_daily)),
                "executed_buy_orders": int(sum(item["executed_buy_orders"] for item in execution_daily)),
                "failed_order_count": int(sum(item["failed_order_count"] for item in execution_daily)),
                "skipped_reasons": dict(skipped_totals),
                "trade_count": int(len(trades)),
                "turnover_value": float(pd.DataFrame(trades)["turnover"].sum()) if trades else 0.0,
            },
        },
    }
