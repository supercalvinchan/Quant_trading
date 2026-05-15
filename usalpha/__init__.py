"""USalpha: minimal US-equity factor mining pipeline."""

from __future__ import annotations

from typing import Any

from .config import USAlphaConfig


def run_pipeline(config: USAlphaConfig | dict[str, Any] | None = None) -> dict[str, Any]:
    from .pipeline import run_pipeline as _run_pipeline

    return _run_pipeline(config)


__all__ = ["USAlphaConfig", "run_pipeline"]
