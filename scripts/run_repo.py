#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _base_env(root: Path) -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(root) if not existing else f"{root}{os.pathsep}{existing}"
    return env


def main() -> None:
    parser = argparse.ArgumentParser(description="Run USalpha entry points from any folder name")
    parser.add_argument(
        "target",
        choices=["pipeline", "llm-round", "dashboard"],
        help="Which built-in entry point to launch",
    )
    parser.add_argument("args", nargs=argparse.REMAINDER, help="Arguments passed through to the target")
    ns = parser.parse_args()

    root = _repo_root()
    env = _base_env(root)

    if ns.target == "pipeline":
        cmd = [sys.executable, str(root / "scripts" / "run_usalpha.py"), *ns.args]
    elif ns.target == "llm-round":
        cmd = [sys.executable, str(root / "scripts" / "run_llm_factor_round.py"), *ns.args]
    else:
        cmd = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(root / "apps" / "factor_dashboard.py"),
            "--server.port",
            "8501",
            "--server.address",
            "0.0.0.0",
            *ns.args,
        ]

    raise SystemExit(subprocess.call(cmd, cwd=root, env=env))


if __name__ == "__main__":
    main()

