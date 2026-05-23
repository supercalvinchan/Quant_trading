from __future__ import annotations

from typing import Any

from .base import BaseTradingMethod


def _to_unique_lower(items: list[str]) -> list[str]:
    seen = set()
    out: list[str] = []
    for item in items:
        value = str(item or "").strip().lower()
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _to_ranked_instruments_from_signals(signals: list[dict[str, Any]]) -> list[str]:
    if not isinstance(signals, list):
        raise ValueError("signals must be a list")
    seen = set()
    out: list[str] = []
    for idx, item in enumerate(signals):
        if not isinstance(item, dict):
            raise ValueError(f"signals[{idx}] must be an object")
        inst = str(item.get("instrument", "")).strip().lower()
        if not inst:
            raise ValueError(f"signals[{idx}].instrument must be non-empty")
        if inst in seen:
            continue
        seen.add(inst)
        out.append(inst)
    return out


class TopkDropoutMethod(BaseTradingMethod):
    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "topk": {"type": "integer", "minimum": 1, "maximum": 500},
                "n_drop": {"type": "integer", "minimum": 0, "maximum": 500},
                "initial_capital": {"type": "number", "minimum": 1000},
                "open_cost": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "close_cost": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "min_cost": {"type": "number", "minimum": 0.0},
                "deal_price": {"type": "string", "enum": ["open", "close", "vwap"]},
                "limit_threshold": {"type": "number", "minimum": 0.0, "maximum": 0.5},
                "impact_cost": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "trade_unit": {"type": ["integer", "null"], "minimum": 1},
                "volume_limit_ratio": {"type": ["number", "null"], "minimum": 0.0, "maximum": 1.0},
                "forbid_all_trade_at_limit": {"type": "boolean"},
                "dynamic_topk_enabled": {"type": "boolean"},
                "dynamic_topk_index_code": {"type": ["string", "null"], "minLength": 1},
                "dynamic_topk_ma_window": {"type": "integer", "minimum": 2, "maximum": 250},
                "dynamic_topk_map": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "min_diff": {"type": ["number", "null"]},
                            "max_diff": {"type": ["number", "null"]},
                            "topk": {"type": "integer", "minimum": 1, "maximum": 500},
                        },
                        "required": ["topk"],
                        "additionalProperties": False,
                    },
                },
                "take_profit_multiple": {"type": ["number", "null"], "minimum": 1.0, "maximum": 100.0},
                "stop_loss_pct": {"type": ["number", "null"], "minimum": 0.0, "maximum": 0.99},
                "hold_limit_up_positions": {"type": "boolean"},
                "exclude_suspended_candidates": {"type": "boolean"},
            },
            "required": [
                "topk",
                "n_drop",
                "initial_capital",
                "open_cost",
                "close_cost",
                "min_cost",
                "deal_price",
                "limit_threshold",
                "impact_cost",
                "trade_unit",
                "volume_limit_ratio",
                "forbid_all_trade_at_limit",
                "dynamic_topk_enabled",
                "dynamic_topk_index_code",
                "dynamic_topk_ma_window",
                "dynamic_topk_map",
                "take_profit_multiple",
                "stop_loss_pct",
                "hold_limit_up_positions",
                "exclude_suspended_candidates",
            ],
            "additionalProperties": False,
            "x_constraints": [
                {
                    "type": "compare",
                    "left": "n_drop",
                    "op": "<=",
                    "right": "topk",
                    "message": "trading_method.params.n_drop must be <= trading_method.params.topk",
                }
            ],
        }

    def generate_rebalance_plan(
        self,
        signals: list[dict[str, Any]],
        current_positions: list[str],
        params: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        _ = context
        override_topk = params.get("effective_topk")
        schema_params = {key: value for key, value in params.items() if key != "effective_topk"}
        resolved = self.resolve_params(schema_params)
        topk = int(override_topk) if override_topk is not None else int(resolved["topk"])
        n_drop = int(resolved["n_drop"])
        ranked = _to_ranked_instruments_from_signals(signals)
        held = _to_unique_lower(current_positions)

        rank_map = {inst: idx for idx, inst in enumerate(ranked)}
        last = sorted(held, key=lambda inst: rank_map.get(inst, 10**9))
        last_set = set(last)

        today_candidate_count = max(0, n_drop + topk - len(last))
        new_candidates = [inst for inst in ranked if inst not in last_set]
        today = new_candidates[:today_candidate_count]

        comb_set = last_set.union(today)
        comb = [inst for inst in ranked if inst in comb_set]
        drop_bottom_set = set(comb[-n_drop:] if n_drop > 0 else [])
        sell_list = [inst for inst in last if inst in drop_bottom_set]

        buy_slots = max(0, len(sell_list) + topk - len(last))
        buy_list = today[:buy_slots]
        buy_weights = {inst: 1.0 / len(buy_list) for inst in buy_list} if buy_list else {}

        return {
            "sell_instruments": sell_list,
            "sell_ratios": {inst: 1.0 for inst in sell_list},
            "buy_instruments": buy_list,
            "buy_weights": buy_weights,
            "target_cash_ratio": 0.0,
            "resolved_params": resolved,
            "meta": {
                "ranked_count": len(ranked),
                "held_count": len(held),
            },
        }
