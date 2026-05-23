from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .base import BaseStrategy
from .small_cap_timing import (
    _normalize_cn_symbol,
    align_daily_series_to_frame_index,
    get_cn_symbol_name_map_cached,
    get_total_value_history_cached,
)


def _rank_pct_frame(frame: pd.DataFrame, *, ascending: bool = True) -> pd.DataFrame:
    return frame.rank(axis=1, pct=True, ascending=ascending)


class InstitutionalCrowdingStrategy(BaseStrategy):
    type_name = "institutional_crowding"

    def get_default_profile(self) -> dict[str, Any]:
        return {
            "min_total_value_yi": 200.0,
            "min_avg_amount": 300_000_000.0,
            "max_turnover_ratio": 0.08,
            "momentum_window": 60,
            "trend_window": 120,
            "turnover_window": 20,
            "vol_window": 20,
            "weights": {
                "cap": 0.22,
                "amount": 0.18,
                "low_turnover": 0.22,
                "momentum": 0.20,
                "trend": 0.10,
                "low_vol": 0.08,
            },
        }

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "min_total_value_yi": {"type": "number", "minimum": 0.0, "maximum": 1000000.0},
                "min_avg_amount": {"type": "number", "minimum": 0.0},
                "max_turnover_ratio": {"type": ["number", "null"], "minimum": 0.0, "maximum": 1.0},
                "momentum_window": {"type": "integer", "minimum": 5, "maximum": 250},
                "trend_window": {"type": "integer", "minimum": 10, "maximum": 250},
                "turnover_window": {"type": "integer", "minimum": 5, "maximum": 250},
                "vol_window": {"type": "integer", "minimum": 5, "maximum": 250},
                "exclude_st": {"type": "boolean"},
                "exclude_delisting": {"type": "boolean"},
            },
            "required": [],
            "additionalProperties": False,
        }

    def generate_score_matrix(
        self,
        data_by_symbol: dict[str, pd.DataFrame],
        parameters: dict[str, Any],
    ) -> pd.DataFrame:
        resolved = self.resolve_params(parameters)
        profile = self.get_default_profile()
        min_total_value_yi = float(resolved.get("min_total_value_yi", profile["min_total_value_yi"]))
        min_avg_amount = float(resolved.get("min_avg_amount", profile["min_avg_amount"]))
        max_turnover_ratio = resolved.get("max_turnover_ratio", profile["max_turnover_ratio"])
        momentum_window = int(resolved.get("momentum_window", profile["momentum_window"]))
        trend_window = int(resolved.get("trend_window", profile["trend_window"]))
        turnover_window = int(resolved.get("turnover_window", profile["turnover_window"]))
        vol_window = int(resolved.get("vol_window", profile["vol_window"]))
        weights = profile["weights"]
        exclude_st = bool(resolved.get("exclude_st", True))
        exclude_delisting = bool(resolved.get("exclude_delisting", True))
        symbol_name_map = get_cn_symbol_name_map_cached() if (exclude_st or exclude_delisting) else {}

        market_cap_map: dict[str, pd.Series] = {}
        avg_amount_map: dict[str, pd.Series] = {}
        turnover_ratio_map: dict[str, pd.Series] = {}
        momentum_map: dict[str, pd.Series] = {}
        trend_gap_map: dict[str, pd.Series] = {}
        volatility_map: dict[str, pd.Series] = {}
        supported_count = 0

        for symbol, frame in data_by_symbol.items():
            normalized = _normalize_cn_symbol(symbol)
            if len(normalized) != 6:
                continue
            name = str(symbol_name_map.get(normalized, "")).upper()
            if exclude_st and "ST" in name:
                continue
            if exclude_delisting and "退" in name:
                continue
            if frame.empty or "close" not in frame.columns or "amount" not in frame.columns:
                continue

            total_value = get_total_value_history_cached(normalized)
            aligned_market_cap = align_daily_series_to_frame_index(total_value, frame.index)
            if aligned_market_cap.empty:
                continue

            close = pd.to_numeric(frame["close"], errors="coerce")
            amount = pd.to_numeric(frame["amount"], errors="coerce")
            ret = close.pct_change()
            avg_amount = amount.rolling(turnover_window, min_periods=max(5, turnover_window // 2)).mean()
            turnover_ratio = avg_amount / (aligned_market_cap * 100_000_000.0)
            momentum = close / close.shift(momentum_window) - 1.0
            ma = close.rolling(trend_window, min_periods=max(10, trend_window // 2)).mean()
            trend_gap = close / ma - 1.0
            volatility = ret.rolling(vol_window, min_periods=max(5, vol_window // 2)).std()

            market_cap_map[str(symbol)] = aligned_market_cap.astype(float)
            avg_amount_map[str(symbol)] = avg_amount.astype(float)
            turnover_ratio_map[str(symbol)] = turnover_ratio.astype(float)
            momentum_map[str(symbol)] = momentum.astype(float)
            trend_gap_map[str(symbol)] = trend_gap.astype(float)
            volatility_map[str(symbol)] = volatility.astype(float)
            supported_count += 1

        if supported_count == 0:
            raise ValueError("institutional_crowding currently supports CN A-share symbols only")

        market_cap_frame = pd.DataFrame(market_cap_map).sort_index()
        avg_amount_frame = pd.DataFrame(avg_amount_map).reindex(market_cap_frame.index)
        turnover_ratio_frame = pd.DataFrame(turnover_ratio_map).reindex(market_cap_frame.index)
        momentum_frame = pd.DataFrame(momentum_map).reindex(market_cap_frame.index)
        trend_gap_frame = pd.DataFrame(trend_gap_map).reindex(market_cap_frame.index)
        volatility_frame = pd.DataFrame(volatility_map).reindex(market_cap_frame.index)

        valid_mask = market_cap_frame >= float(min_total_value_yi)
        if min_avg_amount > 0:
            valid_mask &= avg_amount_frame >= float(min_avg_amount)
        if max_turnover_ratio is not None:
            valid_mask &= turnover_ratio_frame <= float(max_turnover_ratio)

        cap_rank = _rank_pct_frame(market_cap_frame, ascending=True)
        amount_rank = _rank_pct_frame(avg_amount_frame, ascending=True)
        turnover_rank = _rank_pct_frame(turnover_ratio_frame, ascending=True)
        momentum_rank = _rank_pct_frame(momentum_frame, ascending=True)
        trend_rank = _rank_pct_frame(trend_gap_frame, ascending=True)
        vol_rank = _rank_pct_frame(volatility_frame, ascending=True)

        score = (
            float(weights["cap"]) * cap_rank
            + float(weights["amount"]) * amount_rank
            + float(weights["low_turnover"]) * (1.0 - turnover_rank)
            + float(weights["momentum"]) * momentum_rank
            + float(weights["trend"]) * trend_rank
            + float(weights["low_vol"]) * (1.0 - vol_rank)
        )
        score = score.where(valid_mask)
        return score.replace([np.inf, -np.inf], np.nan).sort_index()


class InstitutionalWhiteHorseStrategy(InstitutionalCrowdingStrategy):
    type_name = "institutional_white_horse"

    def get_default_profile(self) -> dict[str, Any]:
        return {
            "min_total_value_yi": 500.0,
            "min_avg_amount": 500_000_000.0,
            "max_turnover_ratio": 0.05,
            "momentum_window": 80,
            "trend_window": 150,
            "turnover_window": 20,
            "vol_window": 25,
            "weights": {
                "cap": 0.28,
                "amount": 0.18,
                "low_turnover": 0.24,
                "momentum": 0.12,
                "trend": 0.08,
                "low_vol": 0.10,
            },
        }


class InstitutionalGrowthStrategy(InstitutionalCrowdingStrategy):
    type_name = "institutional_growth"

    def get_default_profile(self) -> dict[str, Any]:
        return {
            "min_total_value_yi": 120.0,
            "min_avg_amount": 200_000_000.0,
            "max_turnover_ratio": 0.12,
            "momentum_window": 50,
            "trend_window": 90,
            "turnover_window": 20,
            "vol_window": 20,
            "weights": {
                "cap": 0.14,
                "amount": 0.18,
                "low_turnover": 0.16,
                "momentum": 0.26,
                "trend": 0.18,
                "low_vol": 0.08,
            },
        }
