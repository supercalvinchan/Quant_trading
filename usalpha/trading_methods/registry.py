from __future__ import annotations

from dataclasses import dataclass

from .base import BaseTradingMethod
from .topk_dropout import TopkDropoutMethod


@dataclass(frozen=True)
class TradingMethodSpec:
    type: str
    name: str
    description: str
    method_cls: type[BaseTradingMethod]


TRADING_METHOD_REGISTRY: dict[str, TradingMethodSpec] = {
    "topk_dropout": TradingMethodSpec(
        type="topk_dropout",
        name="TopK Dropout",
        description="持有 topk，按 n_drop 淘汰末位后等权补齐。",
        method_cls=TopkDropoutMethod,
    ),
}


def get_trading_method(trading_method_type: str) -> BaseTradingMethod:
    key = str(trading_method_type or "").strip().lower()
    if key not in TRADING_METHOD_REGISTRY:
        raise ValueError(f"unknown trading_method_type: {trading_method_type}")
    return TRADING_METHOD_REGISTRY[key].method_cls()


def list_trading_method_metadata() -> list[dict[str, str]]:
    return [
        {
            "type": spec.type,
            "name": spec.name,
            "description": spec.description,
        }
        for spec in TRADING_METHOD_REGISTRY.values()
    ]
