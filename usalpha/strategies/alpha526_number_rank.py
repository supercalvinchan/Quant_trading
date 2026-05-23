from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from usalpha.config import USAlphaConfig
from usalpha.factors import compute_single_alpha_factor, load_alpha526_catalog

from .base import BaseStrategy


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _alpha526_path() -> Path:
    return USAlphaConfig().resolve_alpha526_path(_project_root())


def get_alpha526_factor_meta(factor_number: int) -> dict[str, Any] | None:
    catalog = load_alpha526_catalog(str(_alpha526_path()))
    idx = int(factor_number) - 1
    if idx < 0 or idx >= len(catalog):
        return None
    return catalog[idx]


def _build_panel_from_data_by_symbol(data_by_symbol: dict[str, pd.DataFrame]) -> pd.DataFrame:
    normalized: dict[str, pd.DataFrame] = {}
    all_index = pd.Index([])
    all_fields: set[str] = set()
    for symbol, frame in data_by_symbol.items():
        if frame is None or frame.empty:
            continue
        out = frame.copy()
        out.index = pd.to_datetime(out.index).tz_localize(None)
        out = out.sort_index()
        out.columns = [str(col).lower() for col in out.columns]
        normalized[str(symbol)] = out
        all_index = all_index.union(out.index)
        all_fields.update(out.columns)
    if not normalized:
        return pd.DataFrame()
    field_order = [field for field in ["open", "high", "low", "close", "volume", "vwap", "amount", "ret"] if field in all_fields]
    wide_parts: list[pd.DataFrame] = []
    keys: list[str] = []
    for symbol, frame in normalized.items():
        part = frame.reindex(all_index)
        part = part.reindex(columns=field_order)
        wide_parts.append(part)
        keys.append(symbol)
    panel = pd.concat(wide_parts, axis=1, keys=keys)
    panel.index = pd.to_datetime(panel.index).tz_localize(None)
    panel = panel.sort_index()
    return panel


class Alpha526NumberRankStrategy(BaseStrategy):
    type_name = "alpha526_number_rank"

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "factor_number": {"type": "integer", "minimum": 1, "maximum": 526},
            },
            "required": ["factor_number"],
            "additionalProperties": False,
        }

    def generate_score_matrix(
        self,
        data_by_symbol: dict[str, pd.DataFrame],
        parameters: dict[str, Any],
    ) -> pd.DataFrame:
        cleaned = {k: v for k, v in (parameters or {}).items() if not str(k).startswith("__")}
        resolved = self.resolve_params(cleaned)
        benchmark_df = parameters.get("__benchmark__")
        panel = _build_panel_from_data_by_symbol(data_by_symbol)
        if panel.empty:
            return pd.DataFrame()
        _, wide = compute_single_alpha_factor(
            panel,
            benchmark_df if isinstance(benchmark_df, pd.DataFrame) else pd.DataFrame(),
            alpha526_path=_alpha526_path(),
            factor_number=int(resolved["factor_number"]),
        )
        return wide.sort_index()
