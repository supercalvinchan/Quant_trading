#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


def _check_file(path: Path) -> tuple[bool, str]:
    return path.exists(), str(path)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    os.environ["PYTHONPATH"] = str(root) + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")
    required = [
        root / "usalpha" / "alpha526_specs.json",
        root / "scripts" / "run_usalpha.py",
        root / "scripts" / "run_llm_factor_round.py",
        root / "apps" / "factor_dashboard.py",
    ]

    failed = False
    print(f"repo_root={root}")
    for path in required:
        ok, label = _check_file(path)
        print(("OK   " if ok else "MISS "), label)
        failed = failed or not ok

    from usalpha.config import USAlphaConfig

    cfg = USAlphaConfig()
    resolved = cfg.resolve_alpha526_path(root)
    print(f"resolved_alpha526_path={resolved}")
    if not resolved.exists():
        print("MISS resolved alpha526 path does not exist")
        failed = True

    if failed:
        raise SystemExit(1)
    print("portable_setup=ok")


if __name__ == "__main__":
    main()
