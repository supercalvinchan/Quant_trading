from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .base import BaseStrategy


def _add_macd(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    close = out["close"].astype(float)
    ema12 = close.ewm(span=12, adjust=False, min_periods=1).mean()
    ema26 = close.ewm(span=26, adjust=False, min_periods=1).mean()
    out["macd"] = ema12 - ema26
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False, min_periods=1).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]
    return out


def _add_kdj(frame: pd.DataFrame, n: int = 9) -> pd.DataFrame:
    out = frame.copy()
    low_n = out["low"].rolling(n, min_periods=1).min()
    high_n = out["high"].rolling(n, min_periods=1).max()
    rsv = (out["close"] - low_n) / (high_n - low_n).replace(0, np.nan) * 100.0
    out["kdj_k"] = rsv.ewm(alpha=1 / 3, adjust=False, min_periods=1).mean().fillna(50.0)
    out["kdj_d"] = out["kdj_k"].ewm(alpha=1 / 3, adjust=False, min_periods=1).mean().fillna(50.0)
    out["kdj_j"] = 3 * out["kdj_k"] - 2 * out["kdj_d"]
    return out


def _add_rsi(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    delta = out["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    for window in (6, 12, 24):
        avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=1).mean()
        avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=1).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        out[f"rsi_{window}"] = (100 - 100 / (1 + rs)).fillna(100.0)
    return out


def _add_dmi(frame: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    out = frame.copy()
    high = out["high"]
    low = out["low"]
    close = out["close"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=out.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=out.index)
    tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / n, adjust=False, min_periods=1).mean().replace(0, np.nan)
    out["dmi_pdi"] = 100 * plus_dm.ewm(alpha=1 / n, adjust=False, min_periods=1).mean() / atr
    out["dmi_mdi"] = 100 * minus_dm.ewm(alpha=1 / n, adjust=False, min_periods=1).mean() / atr
    dx = ((out["dmi_pdi"] - out["dmi_mdi"]).abs() / (out["dmi_pdi"] + out["dmi_mdi"]).replace(0, np.nan)) * 100
    out["dmi_adx"] = dx.ewm(alpha=1 / n, adjust=False, min_periods=1).mean()
    return out


def _add_wr(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for window in (10, 20):
        high_n = out["high"].rolling(window, min_periods=1).max()
        low_n = out["low"].rolling(window, min_periods=1).min()
        out[f"wr_{window}"] = (high_n - out["close"]) / (high_n - low_n).replace(0, np.nan) * 100.0
    return out


def technical_score_series(frame: pd.DataFrame) -> pd.Series:
    if frame.empty or len(frame) < 2:
        return pd.Series(dtype=float)

    required = ["open", "high", "low", "close", "volume"]
    data = frame.copy()
    data = data[[col for col in required if col in data.columns]].copy()
    if len(data.columns) < len(required):
        return pd.Series(dtype=float)

    data = _add_wr(_add_dmi(_add_rsi(_add_kdj(_add_macd(data)))))
    close = data["close"].astype(float)
    prev = data.shift(1)

    macd_score = pd.Series(np.where(data["macd"] > data["macd_signal"], 35.0, -35.0), index=data.index)
    macd_score += np.where(
        (prev["macd"] <= prev["macd_signal"]) & (data["macd"] > data["macd_signal"]),
        45.0,
        np.where((prev["macd"] >= prev["macd_signal"]) & (data["macd"] < data["macd_signal"]), -45.0, 0.0),
    )
    macd_score += np.where(data["macd_hist"] > prev["macd_hist"], 20.0, -20.0)

    kdj_score = (50.0 - (data["kdj_k"].astype(float) - 50.0).abs()) * 0.4
    kdj_score += np.where(data["kdj_k"] > data["kdj_d"], 25.0, -25.0)
    kdj_score += np.where(data["kdj_j"] < 20, 35.0, np.where(data["kdj_j"] > 80, -35.0, 0.0))

    rsi6 = data["rsi_6"].astype(float)
    rsi_score = pd.Series(np.where(rsi6 < 30, 70.0, np.where(rsi6 > 70, -70.0, (50.0 - rsi6) * 1.2)), index=data.index)
    rsi_score += np.where(data["rsi_6"] > prev["rsi_6"], 15.0, -15.0)

    dmi_score = pd.Series(np.where(data["dmi_pdi"] > data["dmi_mdi"], 30.0, -30.0), index=data.index)
    adx = data["dmi_adx"].astype(float).clip(upper=40.0).fillna(0.0)
    dmi_score += np.where(data["dmi_pdi"] > data["dmi_mdi"], adx, -adx)

    wr = data["wr_10"].astype(float)
    wr_score = pd.Series(np.where(wr > 80, 70.0, np.where(wr < 20, -70.0, (50.0 - wr) * -1.2)), index=data.index)

    ma5 = close.rolling(5, min_periods=1).mean()
    ma10 = close.rolling(10, min_periods=1).mean()
    ma20 = close.rolling(20, min_periods=1).mean()
    ma120 = close.rolling(120, min_periods=1).mean()
    ma_score = pd.Series(0.0, index=data.index)
    ma_score += np.where(close > ma5, 25.0, -25.0)
    ma_score += np.where(ma5 > ma10, 25.0, -25.0)
    ma_score += np.where(ma10 > ma20, 25.0, -25.0)
    ma_score += np.where(close > ma120, 25.0, -25.0)

    score = pd.concat(
        [
            macd_score.clip(-100, 100),
            kdj_score.clip(-100, 100),
            rsi_score.clip(-100, 100),
            dmi_score.clip(-100, 100),
            wr_score.clip(-100, 100),
            ma_score.clip(-100, 100),
        ],
        axis=1,
    ).mean(axis=1)
    score.iloc[:1] = np.nan
    return score.replace([np.inf, -np.inf], np.nan)


class TechnicalScoreStrategy(BaseStrategy):
    type_name = "technical_score"

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        }

    def generate_score_matrix(
        self,
        data_by_symbol: dict[str, pd.DataFrame],
        parameters: dict[str, Any],
    ) -> pd.DataFrame:
        _ = self.resolve_params(parameters)
        per_symbol: dict[str, pd.Series] = {}
        for symbol, frame in data_by_symbol.items():
            series = technical_score_series(frame)
            if len(series) > 0:
                per_symbol[str(symbol)] = series.astype(float)
        if not per_symbol:
            return pd.DataFrame()
        return pd.DataFrame(per_symbol).sort_index()
