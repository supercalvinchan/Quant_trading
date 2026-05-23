from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from usalpha.runtime_schema import validate_against_schema


class BaseStrategy(ABC):
    type_name: str = ""

    @abstractmethod
    def get_schema(self) -> dict[str, Any]:
        pass

    @abstractmethod
    def generate_score_matrix(
        self,
        data_by_symbol: dict[str, pd.DataFrame],
        parameters: dict[str, Any],
    ) -> pd.DataFrame:
        pass

    def resolve_params(self, raw_params: dict[str, Any] | None) -> dict[str, Any]:
        if raw_params is None:
            raw_params = {}
        if not isinstance(raw_params, dict):
            raise ValueError("strategy.parameters must be an object")
        return validate_against_schema(
            raw_params,
            self.get_schema(),
            source_name="strategy.parameters",
            allow_unknown_fields=False,
        )
