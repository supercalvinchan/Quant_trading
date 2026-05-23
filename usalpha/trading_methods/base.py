from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from usalpha.runtime_schema import validate_against_schema


class BaseTradingMethod(ABC):
    @abstractmethod
    def get_schema(self) -> dict[str, Any]:
        pass

    def resolve_params(self, raw_params: dict[str, Any] | None) -> dict[str, Any]:
        if raw_params is None or not isinstance(raw_params, dict):
            raise ValueError("trading_method.params must be an object")
        return validate_against_schema(
            raw_params,
            self.get_schema(),
            source_name="trading_method.params",
            allow_unknown_fields=False,
        )

    @abstractmethod
    def generate_rebalance_plan(
        self,
        signals: list[dict[str, Any]],
        current_positions: list[str],
        params: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        pass
