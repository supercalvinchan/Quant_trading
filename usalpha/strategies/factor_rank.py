from __future__ import annotations

from typing import Any

import pandas as pd

from usalpha.factors import _eval_base_expression

from .base import BaseStrategy


class FactorRankStrategy(BaseStrategy):
    type_name = "factor_rank"

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "minLength": 1,
                    "pattern": ".*\\S.*",
                }
            },
            "required": ["expression"],
            "additionalProperties": False,
        }

    def generate_score_matrix(
        self,
        data_by_symbol: dict[str, pd.DataFrame],
        parameters: dict[str, Any],
    ) -> pd.DataFrame:
        resolved = self.resolve_params(parameters)
        expression = str(resolved["expression"]).strip()
        per_symbol: dict[str, pd.Series] = {}
        for symbol, frame in data_by_symbol.items():
            if frame.empty:
                continue
            prepared = frame.copy()
            prepared.columns = [str(col).lower() for col in prepared.columns]
            series = _eval_base_expression(expression, prepared)
            if len(series) > 0:
                per_symbol[str(symbol)] = pd.to_numeric(series, errors="coerce")
        if not per_symbol:
            return pd.DataFrame()
        return pd.DataFrame(per_symbol).sort_index()
