from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import random
import re
from typing import Any

import numpy as np
import pandas as pd

from .backtest import run_long_short_backtest
from .factors import _eval_base_expression, parse_alpha526_specs
from .llm_client import GLMClientConfig, glm_chat_completion


ALLOWED_CHARS_RE = re.compile(r"^[A-Za-z0-9_\$\(\)\+\-\*\/,\.\s:<>=!&\|\?]+$")
FORBIDDEN_TOKENS = ("__", "import", "lambda", "eval", "exec", "os.", "sys.")


@dataclass
class CandidateFactor:
    name: str
    expression: str
    reason: str = ""


@dataclass
class EvolutionResult:
    run_dir: str
    prompt_path: str
    raw_response_path: str
    generation_mode: str
    candidate_count: int
    valid_count: int
    accepted_count: int
    accepted_factors: list[dict[str, Any]]


def _stack_frame(frame: pd.DataFrame) -> pd.Series:
    try:
        return frame.stack(future_stack=True)
    except TypeError:
        return frame.stack(dropna=False)


def _normalize_expr(expr: str) -> str:
    return re.sub(r"\s+", "", expr.strip()).lower()


def _sanitize_name(text: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_]+", "_", text.upper()).strip("_")
    return s[:64] if s else "LLM_FACTOR"


def _extract_json_block(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z0-9]*\n", "", stripped)
        stripped = re.sub(r"\n```$", "", stripped)
    if stripped.startswith("{") or stripped.startswith("["):
        return stripped

    first_obj = stripped.find("{")
    first_arr = stripped.find("[")
    starts = [x for x in (first_obj, first_arr) if x >= 0]
    if not starts:
        raise ValueError("no JSON object found in LLM output")

    start = min(starts)
    candidate = stripped[start:]
    if candidate.startswith("{"):
        end = candidate.rfind("}")
        if end >= 0:
            return candidate[: end + 1]
    if candidate.startswith("["):
        end = candidate.rfind("]")
        if end >= 0:
            return candidate[: end + 1]
    raise ValueError("failed to locate complete JSON block")


def _parse_candidates(raw_text: str) -> list[CandidateFactor]:
    block = _extract_json_block(raw_text)
    payload = json.loads(block)

    items: list[dict[str, Any]]
    if isinstance(payload, dict):
        if "hypotheses" in payload and isinstance(payload["hypotheses"], list):
            items = payload["hypotheses"]
        elif "candidates" in payload and isinstance(payload["candidates"], list):
            items = payload["candidates"]
        else:
            items = []
    elif isinstance(payload, list):
        items = payload
    else:
        items = []

    out: list[CandidateFactor] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        expr = str(item.get("expression", "")).strip()
        if not expr:
            continue
        name = str(item.get("name", "")).strip() or _sanitize_name(expr[:40])
        reason = str(item.get("reason", "")).strip()
        out.append(CandidateFactor(name=name, expression=expr, reason=reason))
    return out


def _is_valid_expression(expr: str, sample_df: pd.DataFrame) -> tuple[bool, str]:
    clean = expr.strip()
    if not clean:
        return False, "empty"
    if len(clean) > 256:
        return False, "too_long"
    if not ALLOWED_CHARS_RE.match(clean):
        return False, "invalid_chars"

    low = clean.lower()
    for token in FORBIDDEN_TOKENS:
        if token in low:
            return False, f"forbidden_token:{token}"

    try:
        value = _eval_base_expression(clean, sample_df)
        if not isinstance(value, pd.Series):
            return False, "not_series"
    except Exception as exc:  # pylint: disable=broad-except
        return False, f"eval_error:{exc}"
    return True, "ok"


def _build_prompt(
    base_examples: list[str],
    existing_exprs: list[str],
    num_candidates: int,
    history_feedback: str | None = None,
) -> str:
    example_text = "\n".join(f"- {x}" for x in base_examples)
    dedup_text = "\n".join(f"- {x}" for x in existing_exprs[:80])
    feedback = history_feedback or "无历史反馈，本轮先追求多样性与可计算性。"

    return f"""
你是量化研究员，生成美股日频因子表达式（base 风格），目标是可计算、低冗余。

仅允许字段：$open, $high, $low, $close, $vwap, $volume
仅允许函数：Ref, Mean, Std, Slope, Rsquare, Resi, Max, Min, Quantile, Greater, Less
允许运算符：+ - * / ()
禁止使用任何其他函数、模块、变量。

历史反馈：
{feedback}

已有因子表达式（避免重复）：
{dedup_text}

参考表达式风格：
{example_text}

请输出 {num_candidates} 个候选，严格输出 JSON，不要加解释文字。
JSON schema:
{{
  "hypotheses": [
    {{"name": "简短英文名", "expression": "表达式", "reason": "一句话"}}
  ]
}}
""".strip()


def _local_mutation_candidates(num_candidates: int) -> list[CandidateFactor]:
    rng = random.Random(20260420)
    windows = [2, 3, 5, 8, 10, 13, 20, 30, 40, 60]

    templates = [
        "Ref($close, {n})/$close",
        "($close-Ref($close, {n}))/Ref($close, {n})",
        "($open-Ref($close, {n}))/Ref($close, {n})",
        "($high-$low)/(Ref($close, {n})+1e-12)",
        "Slope($close, {n})/$close",
        "Slope($vwap, {n})/$close",
        "Rsquare($close, {n})",
        "Resi($close, {n})/$close",
        "Mean($close, {n})/$close",
        "Std($close, {n})/$close",
        "(Quantile($close, {n}, 0.8)-Quantile($close, {n}, 0.2))/($close+1e-12)",
        "(Mean($close, {n})-Mean($close, {m}))/$close",
        "(Std($close, {n})-Std($close, {m}))/$close",
        "(Mean($volume, {n})-Mean($volume, {m}))/(Mean($volume, {m})+1e-12)",
        "(Greater($open,$close)-Less($open,$close))/($close+1e-12)",
    ]

    out: list[CandidateFactor] = []
    for idx in range(num_candidates * 3):
        tpl = templates[idx % len(templates)]
        n = rng.choice(windows)
        m = rng.choice([x for x in windows if x != n])
        expr = tpl.format(n=n, m=m)
        out.append(
            CandidateFactor(
                name=f"LOCAL_MUT_{idx+1:03d}",
                expression=expr,
                reason="local_fallback_mutation",
            )
        )
        if len(out) >= num_candidates:
            break
    return out


def _evaluate_expression(
    expr: str,
    panel: pd.DataFrame,
    label_horizon: int,
    top_quantile: float,
) -> dict[str, Any]:
    tickers = list(panel.columns.get_level_values(0).unique())
    dates = panel.index

    per_ticker: dict[str, pd.Series] = {}
    for ticker in tickers:
        frame = panel[ticker].copy()
        frame.columns = [str(c).lower() for c in frame.columns]
        per_ticker[ticker] = _eval_base_expression(expr, frame)

    pred_wide = pd.DataFrame(per_ticker, index=dates)
    pred_s = _stack_frame(pred_wide).rename("pred")
    pred_s.index = pred_s.index.set_names(["datetime", "instrument"])

    close = panel.xs("close", axis=1, level=1).sort_index()
    label_wide = close.shift(-int(label_horizon)) / close - 1.0
    label_s = _stack_frame(label_wide).rename("label")
    label_s.index = label_s.index.set_names(["datetime", "instrument"])

    df = pred_s.to_frame().join(label_s, how="inner").dropna()
    if len(df) == 0:
        return {
            "samples": 0,
            "coverage": 0.0,
            "ic_mean": np.nan,
            "ic_std": np.nan,
            "sharpe": np.nan,
            "annual_return": np.nan,
            "max_drawdown": np.nan,
            "score": -1e9,
        }

    def _daily_ic(g: pd.DataFrame) -> float:
        if g["pred"].nunique(dropna=True) < 2 or g["label"].nunique(dropna=True) < 2:
            return np.nan
        return float(g["pred"].corr(g["label"]))

    daily_ic = df.groupby(level="datetime", sort=True).apply(_daily_ic)

    bt_input = df.copy()
    bt_input["split"] = "all"
    bt = run_long_short_backtest(bt_input, top_quantile=top_quantile, split="")

    ic_mean = float(daily_ic.mean()) if len(daily_ic) else np.nan
    ic_std = float(daily_ic.std(ddof=0)) if len(daily_ic) else np.nan
    sharpe = float(bt.metrics.get("sharpe", np.nan))
    ann_ret = float(bt.metrics.get("annual_return", np.nan))
    max_dd = float(bt.metrics.get("max_drawdown", np.nan))

    ic_term = 0.0 if np.isnan(ic_mean) else ic_mean * 120.0
    sharpe_term = 0.0 if np.isnan(sharpe) else sharpe
    score = ic_term + sharpe_term

    return {
        "samples": int(len(df)),
        "coverage": float(len(df) / len(label_s.dropna())) if len(label_s.dropna()) > 0 else 0.0,
        "ic_mean": ic_mean,
        "ic_std": ic_std,
        "sharpe": sharpe,
        "annual_return": ann_ret,
        "max_drawdown": max_dd,
        "score": float(score),
    }


def _append_library(library_path: Path, accepted: list[dict[str, Any]]) -> None:
    if library_path.exists():
        existing = json.loads(library_path.read_text(encoding="utf-8"))
        if not isinstance(existing, list):
            existing = []
    else:
        existing = []

    existing_expr = {_normalize_expr(str(x.get("expression", ""))) for x in existing if isinstance(x, dict)}
    merged = list(existing)
    for item in accepted:
        expr_key = _normalize_expr(str(item.get("expression", "")))
        if expr_key in existing_expr:
            continue
        merged.append(item)
        existing_expr.add(expr_key)

    library_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")


def run_evolution_round(
    *,
    panel: pd.DataFrame,
    alpha526_path: Path,
    api_key: str,
    num_candidates: int = 72,
    top_k_accept: int = 12,
    label_horizon: int = 1,
    top_quantile: float = 0.2,
    model: str = "glm-5",
    temperature: float = 0.8,
    output_dir: Path,
    history_feedback: str | None = None,
    persist_library_on_fallback: bool = True,
) -> EvolutionResult:
    specs = parse_alpha526_specs(alpha526_path)
    base_specs = [s for s in specs if s.category == "base"]

    existing_exprs = [s.expression for s in specs]
    base_examples = [s.expression for s in base_specs[:24]]
    prompt = _build_prompt(base_examples, existing_exprs, num_candidates, history_feedback)

    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("evolution_round_%Y%m%d_%H%M%S")
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    prompt_path = run_dir / "prompt.txt"
    prompt_path.write_text(prompt, encoding="utf-8")

    messages = [
        {"role": "system", "content": "You are a quantitative factor mining assistant."},
        {"role": "user", "content": prompt},
    ]

    generation_mode = "glm_api"
    raw_resp_path = run_dir / "llm_raw_response.json"
    try:
        llm_result = glm_chat_completion(
            messages=messages,
            api_key=api_key,
            cfg=GLMClientConfig(model=model, temperature=temperature),
        )
        raw_resp_path.write_text(json.dumps(llm_result["response"], ensure_ascii=False, indent=2), encoding="utf-8")
        raw_text = llm_result.get("content", "")
        (run_dir / "llm_content.txt").write_text(raw_text, encoding="utf-8")
        parsed = _parse_candidates(raw_text)
    except Exception as exc:  # pylint: disable=broad-except
        generation_mode = "local_fallback"
        (run_dir / "llm_error.txt").write_text(str(exc), encoding="utf-8")
        parsed = _local_mutation_candidates(num_candidates)
        raw_resp_path.write_text(
            json.dumps({"fallback": True, "error": str(exc)}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    parsed_rows = [
        {"name": x.name, "expression": x.expression, "reason": x.reason}
        for x in parsed
    ]
    (run_dir / "candidates_parsed.json").write_text(
        json.dumps(parsed_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    sample_ticker = panel.columns.get_level_values(0).unique()[0]
    sample_df = panel[sample_ticker].copy()
    sample_df.columns = [str(c).lower() for c in sample_df.columns]

    existing_set = {_normalize_expr(x) for x in existing_exprs}
    seen: set[str] = set()

    valid_rows: list[dict[str, Any]] = []
    for idx, cand in enumerate(parsed, start=1):
        expr_key = _normalize_expr(cand.expression)
        if expr_key in seen or expr_key in existing_set:
            continue
        ok, reason = _is_valid_expression(cand.expression, sample_df)
        if not ok:
            continue
        seen.add(expr_key)

        metrics = _evaluate_expression(
            cand.expression,
            panel=panel,
            label_horizon=label_horizon,
            top_quantile=top_quantile,
        )

        valid_rows.append(
            {
                "proposal_rank": idx,
                "name": cand.name,
                "expression": cand.expression,
                "reason": cand.reason,
                **metrics,
            }
        )

    scored = pd.DataFrame(valid_rows)
    if len(scored) > 0:
        scored = scored.sort_values(["score", "ic_mean", "sharpe"], ascending=False)
    scored.to_csv(run_dir / "candidates_scored.csv", index=False)

    accepted: list[dict[str, Any]] = []
    if len(scored) > 0:
        selected = scored.head(int(top_k_accept)).reset_index(drop=True)
        for i, row in selected.iterrows():
            accepted.append(
                {
                    "name": f"US_LLM_R1_{i + 1:03d}",
                    "category": "llm_base",
                    "expression": str(row["expression"]),
                    "source": model if generation_mode == "glm_api" else generation_mode,
                    "reason": str(row.get("reason", "")),
                    "metrics": {
                        "score": float(row.get("score", np.nan)),
                        "ic_mean": float(row.get("ic_mean", np.nan)),
                        "sharpe": float(row.get("sharpe", np.nan)),
                        "annual_return": float(row.get("annual_return", np.nan)),
                        "max_drawdown": float(row.get("max_drawdown", np.nan)),
                    },
                }
            )

    accepted_path = run_dir / "accepted_factors.json"
    accepted_path.write_text(json.dumps(accepted, ensure_ascii=False, indent=2), encoding="utf-8")

    library_path = Path(__file__).resolve().parent / "llm_factor_library.json"
    should_persist = generation_mode == "glm_api" or persist_library_on_fallback
    if should_persist:
        _append_library(library_path, accepted)

    return EvolutionResult(
        run_dir=str(run_dir),
        prompt_path=str(prompt_path),
        raw_response_path=str(raw_resp_path),
        generation_mode=generation_mode,
        candidate_count=len(parsed),
        valid_count=len(scored),
        accepted_count=len(accepted),
        accepted_factors=accepted,
    )
