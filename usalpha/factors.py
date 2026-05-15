from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd


ALPHA526_RE = re.compile(
    r"UnifiedAlphaSpec\(name='(?P<name>[^']+)',\s*category='(?P<category>[^']+)',\s*expression='(?P<expr>(?:\\'|[^'])*)'",
    re.M,
)


@dataclass(frozen=True)
class FactorSpec:
    name: str
    category: str
    expression: str


@dataclass
class FactorSet:
    features: pd.DataFrame
    stats: dict[str, Any]


# -------------------------
# Common utilities
# -------------------------


def parse_alpha526_specs(alpha526_path: Path) -> list[FactorSpec]:
    if alpha526_path.suffix.lower() == ".json":
        payload = json.loads(alpha526_path.read_text(encoding="utf-8"))
        items = payload.get("factors", payload) if isinstance(payload, dict) else payload
        if not isinstance(items, list):
            raise ValueError(f"alpha526 JSON parse failed, expected list got {type(items).__name__}")
        specs = [
            FactorSpec(
                name=str(item["name"]),
                category=str(item["category"]),
                expression=str(item["expression"]),
            )
            for item in items
            if isinstance(item, dict)
        ]
        if len(specs) != 526:
            raise ValueError(f"alpha526 JSON parse failed, expected 526 got {len(specs)}")
        return specs

    text = alpha526_path.read_text(encoding="utf-8")
    specs: list[FactorSpec] = []
    for match in ALPHA526_RE.finditer(text):
        specs.append(
            FactorSpec(
                name=match.group("name"),
                category=match.group("category"),
                expression=match.group("expr").replace("\\'", "'"),
            )
        )
    if len(specs) != 526:
        raise ValueError(f"alpha526 parse failed, expected 526 got {len(specs)}")
    return specs


def _stack_frame(frame: pd.DataFrame) -> pd.Series:
    try:
        return frame.stack(future_stack=True)
    except TypeError:
        return frame.stack(dropna=False)


def _coerce_frame(value: Any, template: pd.DataFrame) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.reindex(index=template.index, columns=template.columns)
    if isinstance(value, pd.Series):
        if value.index.equals(template.index):
            return pd.DataFrame({col: value for col in template.columns}, index=template.index)
        if value.index.equals(template.columns):
            return pd.DataFrame([value.reindex(template.columns)] * len(template.index), index=template.index)
    return pd.DataFrame(value, index=template.index, columns=template.columns)


def _broadcast(value: Any, template: pd.DataFrame) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.reindex(index=template.index, columns=template.columns)
    if isinstance(value, pd.Series):
        if value.index.equals(template.index):
            return pd.DataFrame({col: value for col in template.columns}, index=template.index)
        if value.index.equals(template.columns):
            return pd.DataFrame([value.reindex(template.columns)] * len(template.index), index=template.index)
    return pd.DataFrame(value, index=template.index, columns=template.columns)


def _floor_window(value: float) -> int:
    return max(int(np.floor(float(value))), 1)


def _rolling_apply(frame: pd.DataFrame, window: float, func) -> pd.DataFrame:
    return frame.rolling(_floor_window(window), min_periods=1).apply(func, raw=True)


def _corr(frame_x: pd.DataFrame, frame_y: pd.DataFrame, window: float) -> pd.DataFrame:
    w = _floor_window(window)
    return pd.concat(
        {column: frame_x[column].rolling(w, min_periods=1).corr(frame_y[column]) for column in frame_x.columns},
        axis=1,
    )


def _cov(frame_x: pd.DataFrame, frame_y: pd.DataFrame, window: float) -> pd.DataFrame:
    w = _floor_window(window)
    return pd.concat(
        {column: frame_x[column].rolling(w, min_periods=1).cov(frame_y[column]) for column in frame_x.columns},
        axis=1,
    )


# -------------------------
# Base factor evaluator
# -------------------------


def _roll_slope(series: pd.Series, window: int) -> pd.Series:
    w = max(int(window), 1)

    def _slope(values: np.ndarray) -> float:
        mask = ~np.isnan(values)
        if mask.sum() < 2:
            return np.nan
        y = values[mask]
        x = np.arange(len(values), dtype=float)[mask]
        x_center = x - x.mean()
        den = float((x_center * x_center).sum())
        if den <= 0:
            return np.nan
        y_center = y - y.mean()
        return float((x_center * y_center).sum() / den)

    return series.rolling(w, min_periods=1).apply(_slope, raw=True)


def _roll_rsquare(series: pd.Series, window: int) -> pd.Series:
    w = max(int(window), 1)

    def _rsq(values: np.ndarray) -> float:
        mask = ~np.isnan(values)
        if mask.sum() < 2:
            return np.nan
        y = values[mask]
        x = np.arange(len(values), dtype=float)[mask]
        x_center = x - x.mean()
        y_center = y - y.mean()
        den_x = float((x_center * x_center).sum())
        den_y = float((y_center * y_center).sum())
        if den_x <= 0 or den_y <= 0:
            return np.nan
        corr = float((x_center * y_center).sum() / np.sqrt(den_x * den_y))
        return corr * corr

    return series.rolling(w, min_periods=1).apply(_rsq, raw=True)


def _roll_residual(series: pd.Series, window: int) -> pd.Series:
    w = max(int(window), 1)

    def _resi(values: np.ndarray) -> float:
        mask = ~np.isnan(values)
        if mask.sum() < 2:
            return np.nan
        y = values[mask]
        x = np.arange(len(values), dtype=float)[mask]
        x_design = np.column_stack([np.ones(len(x)), x])
        beta, *_ = np.linalg.lstsq(x_design, y, rcond=None)
        pred = x_design @ beta
        return float(y[-1] - pred[-1])

    return series.rolling(w, min_periods=1).apply(_resi, raw=True)


def _series_binary(left: Any, right: Any, op) -> pd.Series:
    if isinstance(left, pd.Series) and isinstance(right, pd.Series):
        idx = left.index.union(right.index)
        return pd.Series(op(left.reindex(idx).to_numpy(), right.reindex(idx).to_numpy()), index=idx)
    if isinstance(left, pd.Series):
        return pd.Series(op(left.to_numpy(), right), index=left.index)
    if isinstance(right, pd.Series):
        return pd.Series(op(left, right.to_numpy()), index=right.index)
    return pd.Series(op(left, right))


def _eval_base_expression(expr: str, instrument_df: pd.DataFrame) -> pd.Series:
    open_s = instrument_df["open"]
    close_s = instrument_df["close"]
    high_s = instrument_df["high"]
    low_s = instrument_df["low"]
    vwap_s = instrument_df["vwap"]
    volume_s = instrument_df["volume"]

    def Ref(s: pd.Series, n: int) -> pd.Series:
        return s.shift(int(n))

    def Mean(s: pd.Series, n: int) -> pd.Series:
        return s.rolling(max(int(n), 1), min_periods=1).mean()

    def Std(s: pd.Series, n: int) -> pd.Series:
        return s.rolling(max(int(n), 1), min_periods=1).std()

    def Slope(s: pd.Series, n: int) -> pd.Series:
        return _roll_slope(s, int(n))

    def Rsquare(s: pd.Series, n: int) -> pd.Series:
        return _roll_rsquare(s, int(n))

    def Resi(s: pd.Series, n: int) -> pd.Series:
        return _roll_residual(s, int(n))

    def Max(s: pd.Series, n: int) -> pd.Series:
        return s.rolling(max(int(n), 1), min_periods=1).max()

    def Min(s: pd.Series, n: int) -> pd.Series:
        return s.rolling(max(int(n), 1), min_periods=1).min()

    def Quantile(s: pd.Series, n: int, q: float) -> pd.Series:
        return s.rolling(max(int(n), 1), min_periods=1).quantile(float(q))

    def Greater(a: Any, b: Any) -> pd.Series:
        return _series_binary(a, b, np.maximum)

    def Less(a: Any, b: Any) -> pd.Series:
        return _series_binary(a, b, np.minimum)

    prepared = (
        expr.replace("$open", "open")
        .replace("$close", "close")
        .replace("$high", "high")
        .replace("$low", "low")
        .replace("$vwap", "vwap")
        .replace("$volume", "volume")
    )

    env: dict[str, Any] = {
        "open": open_s,
        "close": close_s,
        "high": high_s,
        "low": low_s,
        "vwap": vwap_s,
        "volume": volume_s,
        "Ref": Ref,
        "Mean": Mean,
        "Std": Std,
        "Slope": Slope,
        "Rsquare": Rsquare,
        "Resi": Resi,
        "Max": Max,
        "Min": Min,
        "Quantile": Quantile,
        "Greater": Greater,
        "Less": Less,
        "np": np,
    }
    value = eval(prepared, {"__builtins__": {}}, env)  # noqa: S307
    if isinstance(value, pd.Series):
        return value.reindex(open_s.index)
    return pd.Series(value, index=open_s.index)


# -------------------------
# Alpha101 evaluator (local fallback)
# -------------------------


def _strip_outer_parens(expr: str) -> str:
    expr = expr.strip()
    while expr.startswith("(") and expr.endswith(")"):
        depth = 0
        balanced = True
        for idx, char in enumerate(expr):
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0 and idx != len(expr) - 1:
                    balanced = False
                    break
        if not balanced:
            break
        expr = expr[1:-1].strip()
    return expr


def _convert_ternary(expr: str) -> str:
    expr = _strip_outer_parens(expr)
    depth = 0
    qpos = None
    for idx, char in enumerate(expr):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == "?" and depth == 0:
            qpos = idx
            break
    if qpos is None:
        return expr

    depth = 0
    nested = 0
    cpos = None
    for idx in range(qpos + 1, len(expr)):
        char = expr[idx]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == "?" and depth == 0:
            nested += 1
        elif char == ":" and depth == 0:
            if nested == 0:
                cpos = idx
                break
            nested -= 1
    if cpos is None:
        raise ValueError(f"Failed to parse ternary expression: {expr}")

    cond = _convert_ternary(expr[:qpos].strip())
    left = _convert_ternary(expr[qpos + 1 : cpos].strip())
    right = _convert_ternary(expr[cpos + 1 :].strip())
    return f"where({cond}, {left}, {right})"


def _compile_alpha101_formula(formula: str) -> str:
    expr = _convert_ternary(formula)
    expr = expr.replace("^", "**").replace("||", "|").replace("&&", "&")
    expr = re.sub(r"\bIndClass\.subindustry\b", "'subindustry'", expr, flags=re.I)
    expr = re.sub(r"\bIndClass\.industry\b", "'industry'", expr, flags=re.I)
    expr = re.sub(r"\bIndClass\.sector\b", "'sector'", expr, flags=re.I)

    replacements = {
        "Ts_ArgMax": "ts_argmax",
        "Ts_ArgMin": "ts_argmin",
        "Ts_Rank": "ts_rank",
        "ts_rank": "ts_rank",
        "ts_argmax": "ts_argmax",
        "ts_argmin": "ts_argmin",
        "ts_min": "ts_min",
        "ts_max": "ts_max",
        "SignedPower": "signedpower",
        "IndNeutralize": "indneutralize",
        "indneutralize": "indneutralize",
        "Log": "log",
        "Sign": "sign",
    }
    for source, target in replacements.items():
        expr = re.sub(rf"\b{source}\b", target, expr, flags=re.I)
    return expr


def _safe_last_rank(values: np.ndarray) -> float:
    if np.isnan(values[-1]):
        return np.nan
    valid = values[~np.isnan(values)]
    if valid.size == 0:
        return np.nan
    return pd.Series(valid).rank(pct=True).iloc[-1]


def _safe_argmax(values: np.ndarray) -> float:
    if np.all(np.isnan(values)):
        return np.nan
    return float(np.nanargmax(values) + 1)


def _safe_argmin(values: np.ndarray) -> float:
    if np.all(np.isnan(values)):
        return np.nan
    return float(np.nanargmin(values) + 1)


class _Alpha101Environment:
    def __init__(self, raw_frames: dict[str, pd.DataFrame], groups: dict[str, pd.Series]):
        self.raw_frames = raw_frames
        self.groups = groups
        self.template = next(iter(raw_frames.values()))

    def rank(self, value):
        frame = _broadcast(value, self.template)
        return frame.rank(axis=1, pct=True)

    def delay(self, value, days):
        frame = _broadcast(value, self.template)
        return frame.shift(_floor_window(days))

    def delta(self, value, days):
        return _broadcast(value, self.template) - self.delay(value, days)

    def correlation(self, left, right, days):
        return _corr(_broadcast(left, self.template), _broadcast(right, self.template), days)

    def covariance(self, left, right, days):
        return _cov(_broadcast(left, self.template), _broadcast(right, self.template), days)

    def scale(self, value, a=1):
        frame = _broadcast(value, self.template)
        denom = frame.abs().sum(axis=1).replace(0, np.nan)
        return frame.div(denom, axis=0) * a

    def signedpower(self, value, exponent):
        frame = _broadcast(value, self.template)
        exp_frame = _broadcast(exponent, frame)
        return np.sign(frame) * np.power(np.abs(frame), exp_frame)

    def decay_linear(self, value, days):
        frame = _broadcast(value, self.template)

        def _weighted(values: np.ndarray) -> float:
            valid = ~np.isnan(values)
            if not valid.any():
                return np.nan
            weights = np.arange(1, len(values) + 1, dtype=float)[valid]
            weights /= weights.sum()
            return float(np.dot(values[valid], weights))

        return _rolling_apply(frame, days, _weighted)

    def indneutralize(self, value, group_key):
        frame = _broadcast(value, self.template)
        groups = self.groups[str(group_key)]
        neutral = frame.T.groupby(groups).transform(lambda grp: grp - grp.mean())
        return neutral.T

    def ts_min(self, value, days):
        return _broadcast(value, self.template).rolling(_floor_window(days), min_periods=1).min()

    def ts_max(self, value, days):
        return _broadcast(value, self.template).rolling(_floor_window(days), min_periods=1).max()

    def ts_argmax(self, value, days):
        return _rolling_apply(_broadcast(value, self.template), days, _safe_argmax)

    def ts_argmin(self, value, days):
        return _rolling_apply(_broadcast(value, self.template), days, _safe_argmin)

    def ts_rank(self, value, days):
        return _rolling_apply(_broadcast(value, self.template), days, _safe_last_rank)

    def minimum(self, left, right):
        frame_left = _broadcast(left, self.template)
        if isinstance(right, pd.DataFrame):
            return pd.DataFrame(
                np.minimum(frame_left.to_numpy(), right.to_numpy()),
                index=frame_left.index,
                columns=frame_left.columns,
            )
        if np.isscalar(right):
            return frame_left.rolling(_floor_window(right), min_periods=1).min()
        frame_right = _broadcast(right, frame_left)
        return pd.DataFrame(
            np.minimum(frame_left.to_numpy(), frame_right.to_numpy()),
            index=frame_left.index,
            columns=frame_left.columns,
        )

    def maximum(self, left, right):
        frame_left = _broadcast(left, self.template)
        if isinstance(right, pd.DataFrame):
            return pd.DataFrame(
                np.maximum(frame_left.to_numpy(), right.to_numpy()),
                index=frame_left.index,
                columns=frame_left.columns,
            )
        if np.isscalar(right):
            return frame_left.rolling(_floor_window(right), min_periods=1).max()
        frame_right = _broadcast(right, frame_left)
        return pd.DataFrame(
            np.maximum(frame_left.to_numpy(), frame_right.to_numpy()),
            index=frame_left.index,
            columns=frame_left.columns,
        )

    def ts_sum(self, value, days):
        return _broadcast(value, self.template).rolling(_floor_window(days), min_periods=1).sum()

    def product(self, value, days):
        return _rolling_apply(_broadcast(value, self.template), days, np.nanprod)

    def stddev(self, value, days):
        return _broadcast(value, self.template).rolling(_floor_window(days), min_periods=1).std()

    def abs(self, value):
        return _broadcast(value, self.template).abs()

    def log(self, value):
        return np.log(_broadcast(value, self.template).clip(lower=1e-12))

    def sign(self, value):
        return np.sign(_broadcast(value, self.template))

    def where(self, condition, left, right):
        cond = _broadcast(condition, self.template)
        left_frame = _broadcast(left, cond)
        right_frame = _broadcast(right, cond)
        return left_frame.where(cond, other=right_frame)

    def build_eval_env(self) -> dict[str, object]:
        env: dict[str, object] = {
            "rank": self.rank,
            "delay": self.delay,
            "correlation": self.correlation,
            "covariance": self.covariance,
            "scale": self.scale,
            "delta": self.delta,
            "signedpower": self.signedpower,
            "decay_linear": self.decay_linear,
            "indneutralize": self.indneutralize,
            "ts_min": self.ts_min,
            "ts_max": self.ts_max,
            "ts_argmax": self.ts_argmax,
            "ts_argmin": self.ts_argmin,
            "ts_rank": self.ts_rank,
            "min": self.minimum,
            "max": self.maximum,
            "sum": self.ts_sum,
            "product": self.product,
            "stddev": self.stddev,
            "abs": self.abs,
            "log": self.log,
            "sign": self.sign,
            "where": self.where,
            "open": self.raw_frames["open"],
            "close": self.raw_frames["close"],
            "high": self.raw_frames["high"],
            "low": self.raw_frames["low"],
            "vwap": self.raw_frames["vwap"],
            "volume": self.raw_frames["volume"],
            "returns": self.raw_frames["returns"],
            "cap": self.raw_frames["amount"],
        }
        for window in (5, 10, 15, 20, 30, 40, 50, 60, 81, 120, 150, 180):
            env[f"adv{window}"] = self.ts_sum(self.raw_frames["volume"], window) / window
        return env


# -------------------------
# Alpha191 evaluator (local fallback)
# -------------------------


SPECIAL_ALPHA191_FORMULAS: dict[int, str] = {
    3: "SUM(where(CLOSE=DELAY(CLOSE,1),0,CLOSE-where(CLOSE>DELAY(CLOSE,1),MIN(LOW,DELAY(CLOSE,1)),MAX(HIGH,DELAY(CLOSE,1)))),6)",
    10: "RANK(MAX(where(RET<0,STD(RET,20),CLOSE)^2,5))",
    23: "SMA(where(CLOSE>DELAY(CLOSE,1),STD(CLOSE,20),0),20,1)/(SMA(where(CLOSE>DELAY(CLOSE,1),STD(CLOSE,20),0),20,1)+SMA(where(CLOSE<=DELAY(CLOSE,1),STD(CLOSE,20),0),20,1))*100",
    28: "3*SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1)-2*SMA(SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1),3,1)",
    30: "WMA((REGRESI(CLOSE/DELAY(CLOSE,1)-1,MKT,SMB,HML,60))^2,20)",
    36: "RANK(SUM(CORR(RANK(VOLUME), RANK(VWAP), 6), 2))",
    40: "SUM(where(CLOSE>DELAY(CLOSE,1),VOLUME,0),26)/SUM(where(CLOSE<=DELAY(CLOSE,1),VOLUME,0),26)*100",
    43: "SUM(where(CLOSE>DELAY(CLOSE,1),VOLUME,where(CLOSE<DELAY(CLOSE,1),-VOLUME,0)),6)",
    49: "SUM(where((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1)),0,MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)/(SUM(where((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1)),0,MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)+SUM(where((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1)),0,MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12))",
    50: "SUM(where((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1)),0,MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)/(SUM(where((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1)),0,MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)+SUM(where((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1)),0,MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12))-SUM(where((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1)),0,MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)/(SUM(where((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1)),0,MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)+SUM(where((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1)),0,MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12))",
    51: "SUM(where((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1)),0,MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)/(SUM(where((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1)),0,MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)+SUM(where((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1)),0,MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12))",
    52: "SUM(MAX(0,HIGH-DELAY((HIGH+LOW+CLOSE)/3,1)),26)/SUM(MAX(0,DELAY((HIGH+LOW+CLOSE)/3,1)-LOW),26)*100",
    54: "(-1 * RANK((STD(ABS(CLOSE - OPEN), 5) + (CLOSE - OPEN)) + CORR(CLOSE, OPEN,10)))",
    55: "0",
    59: "SUM(where(CLOSE=DELAY(CLOSE,1),0,CLOSE-where(CLOSE>DELAY(CLOSE,1),MIN(LOW,DELAY(CLOSE,1)),MAX(HIGH,DELAY(CLOSE,1)))),20)",
    69: "(SUM(DTM,20)>SUM(DBM,20)?(SUM(DTM,20)-SUM(DBM,20))/SUM(DTM,20):(SUM(DTM,20)==SUM(DBM,20)?0:(SUM(DTM,20)-SUM(DBM,20))/SUM(DBM,20)))",
    75: "COUNT((CLOSE>OPEN) & (BENCHMARKINDEXCLOSE<BENCHMARKINDEXOPEN),50)/COUNT(BENCHMARKINDEXCLOSE<BENCHMARKINDEXOPEN,50)",
    83: "(-1 * RANK(COVARIANCE(RANK(HIGH), RANK(VOLUME), 5)))",
    84: "SUM(where(CLOSE>DELAY(CLOSE,1),VOLUME,where(CLOSE<DELAY(CLOSE,1),-VOLUME,0)),20)",
    93: "SUM(where(OPEN>=DELAY(OPEN,1),0,MAX((OPEN-LOW),(OPEN-DELAY(OPEN,1)))),20)",
    94: "SUM(where(CLOSE>DELAY(CLOSE,1),VOLUME,where(CLOSE<DELAY(CLOSE,1),-VOLUME,0)),30)",
    98: "(((DELTA((SUM(CLOSE, 100) / 100), 100) / DELAY(CLOSE, 100)) <= 0.05) ? (-1 * (CLOSE - TSMIN(CLOSE, 100))) : (-1 * DELTA(CLOSE, 3)))",
    99: "(-1 * RANK(COVARIANCE(RANK(CLOSE), RANK(VOLUME), 5)))",
    111: "SMA(VOLUME*((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW),11,2)-SMA(VOLUME*((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW),4,2)",
    112: "(SUM(where(CLOSE-DELAY(CLOSE,1)>0,CLOSE-DELAY(CLOSE,1),0),12)-SUM(where(CLOSE-DELAY(CLOSE,1)<0,ABS(CLOSE-DELAY(CLOSE,1)),0),12))/(SUM(where(CLOSE-DELAY(CLOSE,1)>0,CLOSE-DELAY(CLOSE,1),0),12)+SUM(where(CLOSE-DELAY(CLOSE,1)<0,ABS(CLOSE-DELAY(CLOSE,1)),0),12))*100",
    128: "100-(100/(1+SUM(where((HIGH+LOW+CLOSE)/3>DELAY((HIGH+LOW+CLOSE)/3,1),(HIGH+LOW+CLOSE)/3*VOLUME,0),14)/SUM(where((HIGH+LOW+CLOSE)/3<DELAY((HIGH+LOW+CLOSE)/3,1),(HIGH+LOW+CLOSE)/3*VOLUME,0),14)))",
    129: "SUM(where(CLOSE-DELAY(CLOSE,1)<0,ABS(CLOSE-DELAY(CLOSE,1)),0),12)",
    131: "(RANK(DELTA(VWAP, 1))^TSRANK(CORR(CLOSE,MEAN(VOLUME,50),18),18))",
    137: "0",
    159: "((CLOSE-SUM(MIN(LOW,DELAY(CLOSE,1)),6))/SUM(MAX(HIGH,DELAY(CLOSE,1))-MIN(LOW,DELAY(CLOSE,1)),6)*12*24+(CLOSE-SUM(MIN(LOW,DELAY(CLOSE,1)),12))/SUM(MAX(HIGH,DELAY(CLOSE,1))-MIN(LOW,DELAY(CLOSE,1)),12)*6*24+(CLOSE-SUM(MIN(LOW,DELAY(CLOSE,1)),24))/SUM(MAX(HIGH,DELAY(CLOSE,1))-MIN(LOW,DELAY(CLOSE,1)),24)*6*12)*100/(6*12+6*24+12*24)",
    160: "SMA(where(CLOSE<=DELAY(CLOSE,1),STD(CLOSE,20),0),20,1)",
    164: "SMA((where(CLOSE>DELAY(CLOSE,1),1/(CLOSE-DELAY(CLOSE,1)),1)-MIN(where(CLOSE>DELAY(CLOSE,1),1/(CLOSE-DELAY(CLOSE,1)),1),12))/(HIGH-LOW)*100,13,2)",
    165: "(MAX(SUMAC(CLOSE-MEAN(CLOSE,48)))-MIN(SUMAC(CLOSE-MEAN(CLOSE,48))))/STD(CLOSE,48)",
    166: "0",
    167: "SUM(where(CLOSE-DELAY(CLOSE,1)>0,CLOSE-DELAY(CLOSE,1),0),12)",
    172: "0",
    174: "SMA(where(CLOSE>DELAY(CLOSE,1),STD(CLOSE,20),0),20,1)",
    180: "((MEAN(VOLUME,20) < VOLUME) ? (((-1 * TSRANK(ABS(DELTA(CLOSE, 7)), 60)) * SIGN(DELTA(CLOSE, 7)))) : (-1 * VOLUME))",
    181: "SUM((((CLOSE/DELAY(CLOSE,1)-1)-MEAN((CLOSE/DELAY(CLOSE,1)-1),20))-(BENCHMARKINDEXCLOSE-MEAN(BENCHMARKINDEXCLOSE,20)))^2,20)/SUM((BENCHMARKINDEXCLOSE-MEAN(BENCHMARKINDEXCLOSE,20))^3,20)",
    182: "COUNT(((CLOSE>OPEN) & (BENCHMARKINDEXCLOSE>BENCHMARKINDEXOPEN)) | ((CLOSE<OPEN) & (BENCHMARKINDEXCLOSE<BENCHMARKINDEXOPEN)),20)/20",
    183: "(MAX(SUMAC(CLOSE-MEAN(CLOSE,24)))-MIN(SUMAC(CLOSE-MEAN(CLOSE,24))))/STD(CLOSE,24)",
    186: "0",
    187: "SUM(where(OPEN<=DELAY(OPEN,1),0,MAX((HIGH-OPEN),(OPEN-DELAY(OPEN,1)))),20)",
    188: "((HIGH-LOW-SMA(HIGH-LOW,11,2))/SMA(HIGH-LOW,11,2))*100",
    190: "0",
}


def _normalize_alpha191_formula(number: int, formula: str) -> str:
    formula = SPECIAL_ALPHA191_FORMULAS.get(number, formula)
    formula = " ".join(formula.split())
    formula = (
        formula.replace("？", "?")
        .replace("：", ":")
        .replace("，", ",")
        .replace("；", "")
        .replace(";", "")
        .replace("–", "-")
        .replace("—", "-")
        .replace("。", "")
        .replace("./", "/")
        .replace(".*", "*")
        .replace("COVIANCE", "COVARIANCE")
        .replace("DELAT", "DELTA")
        .replace("HGIH", "HIGH")
        .replace("BANCHMARKINDEX", "BENCHMARKINDEX")
        .replace("SMEAN", "SMA")
        .replace("STD(CLOSE:20)", "STD(CLOSE,20)")
        .replace("REGBETA(CLOSE,SEQUENCE,20)", "REGBETA(CLOSE, SEQUENCE(20), 20)")
    )
    formula = re.sub(r"\bOR\b", "|", formula, flags=re.I)
    formula = re.sub(r"\bAND\b", "&", formula, flags=re.I)
    formula = formula.replace("||", "|").replace("&&", "&")
    formula = re.sub(r"(?<![<>=!])=(?!=)", "==", formula)
    return formula


def _compile_alpha191_formula(number: int, formula: str) -> str:
    expr = _normalize_alpha191_formula(number, formula)
    expr = _convert_ternary(expr)
    expr = expr.replace("^", "**")

    replacements = {
        "CORR": "correlation",
        "COVARIANCE": "covariance",
        "COUNT": "count",
        "DELAY": "delay",
        "DELTA": "delta",
        "DECAYLINEAR": "decay_linear",
        "FILTER": "filter_values",
        "HIGHDAY": "highday",
        "LOWDAY": "lowday",
        "LOG": "log",
        "ABS": "abs",
        "SIGN": "sign",
        "MEAN": "mean",
        "MA": "mean",
        "MAX": "maximum",
        "MIN": "minimum",
        "PROD": "product",
        "RANK": "rank",
        "REGBETA": "regbeta",
        "REGRESI": "regresi",
        "SEQUENCE": "sequence",
        "SMA": "sma",
        "STD": "stddev",
        "SUMIF": "sumif",
        "SUMAC": "sumac",
        "SUM": "ts_sum",
        "TSMAX": "ts_max",
        "TSMIN": "ts_min",
        "TSRANK": "ts_rank",
        "WMA": "wma",
    }
    for source, target in replacements.items():
        expr = re.sub(rf"\b{source}\b", target, expr, flags=re.I)

    variable_replacements = {
        "OPEN": "open",
        "CLOSE": "close",
        "HIGH": "high",
        "LOW": "low",
        "VWAP": "vwap",
        "VOLUME": "volume",
        "AMOUNT": "amount",
        "RET": "ret",
        "BENCHMARKINDEXOPEN": "benchmarkindexopen",
        "BENCHMARKINDEXCLOSE": "benchmarkindexclose",
        "MKT": "mkt",
        "SMB": "smb",
        "HML": "hml",
        "DTM": "dtm",
        "DBM": "dbm",
        "TR": "tr",
        "HD": "hd",
        "LD": "ld",
        "L": "low",
    }
    for source, target in variable_replacements.items():
        expr = re.sub(rf"\b{source}\b", target, expr, flags=re.I)
    return expr


def _safe_rank_last(values: np.ndarray) -> float:
    if np.isnan(values[-1]):
        return np.nan
    valid = values[~np.isnan(values)]
    if valid.size == 0:
        return np.nan
    return pd.Series(valid).rank(pct=True).iloc[-1]


def _safe_argmax_recent(values: np.ndarray) -> float:
    if np.all(np.isnan(values)):
        return np.nan
    return float(len(values) - np.nanargmax(values))


def _safe_argmin_recent(values: np.ndarray) -> float:
    if np.all(np.isnan(values)):
        return np.nan
    return float(len(values) - np.nanargmin(values))


def _linear_weighted(values: np.ndarray) -> float:
    valid = ~np.isnan(values)
    if not valid.any():
        return np.nan
    weights = np.arange(1, len(values) + 1, dtype=float)[valid]
    weights /= weights.sum()
    return float(np.dot(values[valid], weights))


def _std_last_residual(y: np.ndarray, factors: np.ndarray) -> float:
    mask = ~np.isnan(y)
    if factors.ndim == 1:
        mask &= ~np.isnan(factors)
        factors = factors[:, None]
    else:
        mask &= ~np.isnan(factors).any(axis=1)
    if mask.sum() < max(factors.shape[1] + 1, 3):
        return np.nan
    yv = y[mask]
    xv = factors[mask]
    xv = np.column_stack([np.ones(len(xv)), xv])
    beta, *_ = np.linalg.lstsq(xv, yv, rcond=None)
    resid = yv - xv @ beta
    return float(resid[-1])


def _beta(y: np.ndarray, x: np.ndarray) -> float:
    mask = ~np.isnan(y) & ~np.isnan(x)
    if mask.sum() < 2:
        return np.nan
    xv = x[mask]
    yv = y[mask]
    var = np.var(xv)
    if np.isclose(var, 0.0):
        return np.nan
    cov = np.cov(xv, yv, ddof=0)[0, 1]
    return float(cov / var)


class _SequenceToken:
    pass


class _Alpha191Environment:
    def __init__(self, raw_frames: dict[str, pd.DataFrame], benchmark: dict[str, pd.DataFrame]):
        self.raw_frames = raw_frames
        self.benchmark = benchmark
        self.template = next(iter(raw_frames.values()))
        zero = pd.DataFrame(0.0, index=self.template.index, columns=self.template.columns)
        self.sequence_token = _SequenceToken()
        prev_open = self.delay(raw_frames["open"], 1)
        prev_close = self.delay(raw_frames["close"], 1)
        prev_low = self.delay(raw_frames["low"], 1)
        prev_high = self.delay(raw_frames["high"], 1)
        self.derived = {
            "hd": raw_frames["high"] - prev_high,
            "ld": prev_low - raw_frames["low"],
            "dtm": self.where(
                raw_frames["open"] <= prev_open,
                0.0,
                self.maximum(raw_frames["high"] - raw_frames["open"], raw_frames["open"] - prev_open),
            ),
            "dbm": self.where(
                raw_frames["open"] >= prev_open,
                0.0,
                self.maximum(raw_frames["open"] - raw_frames["low"], raw_frames["open"] - prev_open),
            ),
            "tr": self.maximum(
                self.maximum(raw_frames["high"] - raw_frames["low"], (raw_frames["high"] - prev_close).abs()),
                (raw_frames["low"] - prev_close).abs(),
            ),
            "mkt": benchmark["ret"],
            "smb": zero,
            "hml": zero,
        }

    def rank(self, value):
        return _broadcast(value, self.template).rank(axis=1, pct=True)

    def delay(self, value, days):
        return _broadcast(value, self.template).shift(_floor_window(days))

    def delta(self, value, days):
        frame = _broadcast(value, self.template)
        return frame - self.delay(frame, days)

    def correlation(self, left, right, days=6):
        return _corr(_broadcast(left, self.template), _broadcast(right, self.template), days)

    def covariance(self, left, right, days):
        return _cov(_broadcast(left, self.template), _broadcast(right, self.template), days)

    def ts_sum(self, value, days):
        return _broadcast(value, self.template).rolling(_floor_window(days), min_periods=1).sum()

    def mean(self, value, days=20):
        return _broadcast(value, self.template).rolling(_floor_window(days), min_periods=1).mean()

    def sma(self, value, days, weight=1):
        frame = _broadcast(value, self.template)
        alpha = float(weight) / float(days)
        return frame.ewm(alpha=alpha, adjust=False, min_periods=1).mean()

    def wma(self, value, days):
        return _rolling_apply(_broadcast(value, self.template), days, _linear_weighted)

    def stddev(self, value, days):
        return _broadcast(value, self.template).rolling(_floor_window(days), min_periods=1).std()

    def ts_max(self, value, days):
        return _broadcast(value, self.template).rolling(_floor_window(days), min_periods=1).max()

    def ts_min(self, value, days):
        return _broadcast(value, self.template).rolling(_floor_window(days), min_periods=1).min()

    def maximum(self, left, right=None):
        frame_left = _broadcast(left, self.template)
        if right is None:
            return frame_left.expanding(min_periods=1).max()
        if np.isscalar(right):
            return frame_left.rolling(_floor_window(right), min_periods=1).max()
        frame_right = _broadcast(right, frame_left)
        return pd.DataFrame(
            np.maximum(frame_left.to_numpy(), frame_right.to_numpy()),
            index=frame_left.index,
            columns=frame_left.columns,
        )

    def minimum(self, left, right=None):
        frame_left = _broadcast(left, self.template)
        if right is None:
            return frame_left.expanding(min_periods=1).min()
        if np.isscalar(right):
            return frame_left.rolling(_floor_window(right), min_periods=1).min()
        frame_right = _broadcast(right, frame_left)
        return pd.DataFrame(
            np.minimum(frame_left.to_numpy(), frame_right.to_numpy()),
            index=frame_left.index,
            columns=frame_left.columns,
        )

    def ts_rank(self, value, days):
        return _rolling_apply(_broadcast(value, self.template), days, _safe_rank_last)

    def highday(self, value, days):
        return _rolling_apply(_broadcast(value, self.template), days, _safe_argmax_recent)

    def lowday(self, value, days):
        return _rolling_apply(_broadcast(value, self.template), days, _safe_argmin_recent)

    def count(self, condition, days):
        frame = _broadcast(condition, self.template)
        return frame.astype(float).rolling(_floor_window(days), min_periods=1).sum()

    def sumif(self, value, days, condition):
        frame = _broadcast(value, self.template)
        cond = _broadcast(condition, frame)
        masked = frame.where(cond, other=0.0)
        return masked.rolling(_floor_window(days), min_periods=1).sum()

    def product(self, value, days):
        return _rolling_apply(_broadcast(value, self.template), days, np.nanprod)

    def log(self, value):
        return np.log(_broadcast(value, self.template).clip(lower=1e-12))

    def abs(self, value):
        return _broadcast(value, self.template).abs()

    def sign(self, value):
        return np.sign(_broadcast(value, self.template))

    def decay_linear(self, value, days):
        return _rolling_apply(_broadcast(value, self.template), days, _linear_weighted)

    def filter_values(self, value, condition):
        frame = _broadcast(value, self.template)
        cond = _broadcast(condition, frame)
        return frame.where(cond)

    def sumac(self, value):
        return _broadcast(value, self.template).cumsum()

    def regbeta(self, y, x, days=None):
        frame_y = _broadcast(y, self.template)
        window = _floor_window(days) if days is not None else None
        if isinstance(x, _SequenceToken):
            if window is None:
                raise ValueError("SEQUENCE requires explicit window")
            seq = np.arange(1, window + 1, dtype=float)
            return _rolling_apply(frame_y, window, lambda values: _beta(values, seq))
        if isinstance(x, tuple):
            seq = np.asarray(x, dtype=float)
            window = len(seq)
            return _rolling_apply(frame_y, window, lambda values: _beta(values, seq[-len(values) :]))
        frame_x = _broadcast(x, frame_y)
        if window is None:
            raise ValueError("regbeta requires window")
        result = {}
        for column in frame_y.columns:
            result[column] = frame_y[column].rolling(window, min_periods=2).apply(
                lambda values, col=column: _beta(values, frame_x[col].loc[values.index].to_numpy()),
                raw=False,
            )
        return pd.concat(result, axis=1)

    def regresi(self, y, *args):
        if len(args) < 2:
            raise ValueError("regresi expects factors and window")
        *factors, days = args
        window = _floor_window(days)
        frame_y = _broadcast(y, self.template)
        factor_frames = [_broadcast(factor, self.template) for factor in factors]
        result = {}
        for column in frame_y.columns:
            y_series = frame_y[column]
            x_stack = np.column_stack([factor[column].to_numpy() for factor in factor_frames])
            values = []
            for idx in range(len(y_series)):
                start = max(0, idx - window + 1)
                y_window = y_series.iloc[start : idx + 1].to_numpy()
                x_window = x_stack[start : idx + 1]
                values.append(_std_last_residual(y_window, x_window))
            result[column] = pd.Series(values, index=frame_y.index)
        return pd.concat(result, axis=1)

    def where(self, condition, left, right):
        cond = _broadcast(condition, self.template)
        left_frame = _broadcast(left, cond)
        right_frame = _broadcast(right, cond)
        return left_frame.where(cond, other=right_frame)

    def sequence(self, days=None):
        if days is None:
            return self.sequence_token
        return tuple(float(i) for i in range(1, _floor_window(days) + 1))

    def alpha143(self):
        close = self.raw_frames["close"]
        prev_close = self.delay(close, 1)
        ret = close / prev_close.replace(0, np.nan)
        gated = self.where(close > prev_close, ret, 1.0)
        return gated.cumprod()

    def build_eval_env(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "delay": self.delay,
            "delta": self.delta,
            "correlation": self.correlation,
            "covariance": self.covariance,
            "ts_sum": self.ts_sum,
            "mean": self.mean,
            "sma": self.sma,
            "wma": self.wma,
            "stddev": self.stddev,
            "ts_max": self.ts_max,
            "ts_min": self.ts_min,
            "maximum": self.maximum,
            "minimum": self.minimum,
            "ts_rank": self.ts_rank,
            "highday": self.highday,
            "lowday": self.lowday,
            "count": self.count,
            "sumif": self.sumif,
            "product": self.product,
            "log": self.log,
            "abs": self.abs,
            "sign": self.sign,
            "decay_linear": self.decay_linear,
            "filter_values": self.filter_values,
            "sumac": self.sumac,
            "regbeta": self.regbeta,
            "regresi": self.regresi,
            "where": self.where,
            "sequence": self.sequence,
            "open": self.raw_frames["open"],
            "close": self.raw_frames["close"],
            "high": self.raw_frames["high"],
            "low": self.raw_frames["low"],
            "vwap": self.raw_frames["vwap"],
            "volume": self.raw_frames["volume"],
            "amount": self.raw_frames["amount"],
            "ret": self.raw_frames["ret"],
            "benchmarkindexopen": self.benchmark["open"],
            "benchmarkindexclose": self.benchmark["close"],
            "mkt": self.derived["mkt"],
            "smb": self.derived["smb"],
            "hml": self.derived["hml"],
            "dtm": self.derived["dtm"],
            "dbm": self.derived["dbm"],
            "tr": self.derived["tr"],
            "hd": self.derived["hd"],
            "ld": self.derived["ld"],
        }


# -------------------------
# Orchestration
# -------------------------


def _build_raw_frames(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    def field(name: str) -> pd.DataFrame:
        return panel.xs(name, axis=1, level=1).sort_index()

    close = field("close")
    return {
        "open": field("open"),
        "close": close,
        "high": field("high"),
        "low": field("low"),
        "vwap": field("vwap"),
        "volume": field("volume"),
        "amount": field("amount"),
        "returns": close.pct_change(),
        "ret": close.pct_change(),
    }


def _build_alpha101_groups(columns: pd.Index) -> dict[str, pd.Series]:
    instruments = pd.Index(columns.astype(str), name="instrument")
    sector = instruments.to_series(index=instruments).str.slice(0, 1)
    industry = instruments.to_series(index=instruments).str.slice(0, 3)
    subindustry = instruments.to_series(index=instruments).str.slice(0, 4)
    return {"sector": sector, "industry": industry, "subindustry": subindustry}


def _compute_base_wide(panel: pd.DataFrame, base_specs: list[FactorSpec]) -> dict[str, pd.DataFrame]:
    tickers = list(panel.columns.get_level_values(0).unique())
    dates = panel.index
    outputs: dict[str, pd.DataFrame] = {}

    instrument_frames: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        frame = panel[ticker].copy()
        frame.columns = [str(c).lower() for c in frame.columns]
        instrument_frames[ticker] = frame

    for spec in base_specs:
        cols: dict[str, pd.Series] = {}
        for ticker in tickers:
            value = _eval_base_expression(spec.expression, instrument_frames[ticker])
            cols[ticker] = value
        outputs[spec.name] = pd.DataFrame(cols, index=dates)
    return outputs


def _compute_alpha101_wide(
    raw_frames: dict[str, pd.DataFrame],
    alpha101_specs: list[FactorSpec],
) -> dict[str, pd.DataFrame]:
    template = raw_frames["close"]
    groups = _build_alpha101_groups(template.columns)

    compile_fn = _compile_alpha101_formula
    env = _Alpha101Environment(raw_frames=raw_frames, groups=groups).build_eval_env()
    print("[USalpha] alpha101 using local evaluator")

    outputs: dict[str, pd.DataFrame] = {}
    for spec in alpha101_specs:
        try:
            compiled = compile_fn(spec.expression)
            value = eval(compiled, {"__builtins__": {}}, env)  # noqa: S307
            frame = _coerce_frame(value, template)
            outputs[spec.name] = frame.replace([np.inf, -np.inf], np.nan)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[USalpha] alpha101 factor failed: {spec.name}, err={exc}")
            outputs[spec.name] = pd.DataFrame(np.nan, index=template.index, columns=template.columns)
    return outputs


def _parse_alpha191_number(name: str) -> int:
    match = re.search(r"(\d{3})$", name)
    if not match:
        match = re.search(r"(\d{1,3})", name)
    if not match:
        raise ValueError(f"cannot parse alpha191 number from {name}")
    return int(match.group(1))


def _build_benchmark_frames(benchmark_df: pd.DataFrame, template: pd.DataFrame) -> dict[str, pd.DataFrame]:
    index = template.index
    columns = template.columns

    bench = benchmark_df.copy()
    bench.index = pd.to_datetime(bench.index).tz_localize(None)
    bench = bench.sort_index()

    open_series = bench["open"].reindex(index).ffill().bfill()
    close_series = bench["close"].reindex(index).ffill().bfill()

    open_frame = pd.DataFrame({col: open_series for col in columns}, index=index)
    close_frame = pd.DataFrame({col: close_series for col in columns}, index=index)
    ret_frame = close_frame.pct_change()
    return {"open": open_frame, "close": close_frame, "ret": ret_frame}


def _compute_alpha191_wide(
    raw_frames: dict[str, pd.DataFrame],
    benchmark_df: pd.DataFrame,
    alpha191_specs: list[FactorSpec],
) -> dict[str, pd.DataFrame]:
    template = raw_frames["close"]
    benchmark_frames = _build_benchmark_frames(benchmark_df, template)

    compile_fn = _compile_alpha191_formula
    env_obj: Any = _Alpha191Environment(raw_frames=raw_frames, benchmark=benchmark_frames)
    eval_env = env_obj.build_eval_env()
    print("[USalpha] alpha191 using local evaluator")

    outputs: dict[str, pd.DataFrame] = {}
    for spec in alpha191_specs:
        number = _parse_alpha191_number(spec.name)
        try:
            if number == 143:
                value = env_obj.alpha143()
            else:
                compiled = compile_fn(number, spec.expression)
                value = eval(compiled, {"__builtins__": {}}, eval_env)  # noqa: S307
            frame = _coerce_frame(value, template)
            outputs[spec.name] = frame.replace([np.inf, -np.inf], np.nan)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[USalpha] alpha191 factor failed: {spec.name}, err={exc}")
            outputs[spec.name] = pd.DataFrame(np.nan, index=template.index, columns=template.columns)
    return outputs


def compute_526_factors(
    panel: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    *,
    alpha526_path: Path,
) -> FactorSet:
    specs = parse_alpha526_specs(alpha526_path)
    base_specs = [s for s in specs if s.category == "base"]
    alpha101_specs = [s for s in specs if s.category == "alpha101"]
    alpha191_specs = [s for s in specs if s.category == "alpha191"]

    raw_frames = _build_raw_frames(panel)

    print("[USalpha] computing base factors...")
    base_wide = _compute_base_wide(panel, base_specs)

    print("[USalpha] computing alpha101 factors...")
    alpha101_wide = _compute_alpha101_wide(raw_frames, alpha101_specs)

    print("[USalpha] computing alpha191 factors...")
    alpha191_wide = _compute_alpha191_wide(raw_frames, benchmark_df, alpha191_specs)

    all_wide: dict[str, pd.DataFrame] = {}
    all_wide.update(base_wide)
    all_wide.update(alpha101_wide)
    all_wide.update(alpha191_wide)

    ordered_names = [s.name for s in specs]
    stacked = []
    for name in ordered_names:
        frame = all_wide.get(name)
        if frame is None:
            frame = pd.DataFrame(np.nan, index=panel.index, columns=raw_frames["close"].columns)
        series = _stack_frame(frame).rename(name)
        stacked.append(series)

    features = pd.concat(stacked, axis=1)
    features.index = features.index.set_names(["datetime", "instrument"])
    features = features.sort_index()

    stats = {
        "factor_count": len(ordered_names),
        "base_count": len(base_specs),
        "alpha101_count": len(alpha101_specs),
        "alpha191_count": len(alpha191_specs),
        "feature_shape": [int(features.shape[0]), int(features.shape[1])],
    }
    return FactorSet(features=features, stats=stats)
