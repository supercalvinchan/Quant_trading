from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .config import TrainConfig


@dataclass
class RidgeLikeModel:
    coef_: np.ndarray
    intercept_: float
    feature_means_: np.ndarray
    feature_stds_: np.ndarray

    def predict(self, x: np.ndarray) -> np.ndarray:
        x_std = (x - self.feature_means_) / self.feature_stds_
        return x_std @ self.coef_ + self.intercept_


@dataclass
class ModelResult:
    model: RidgeLikeModel
    predictions: pd.DataFrame
    metrics: dict[str, Any]
    feature_columns: list[str]


def _stack_frame(frame: pd.DataFrame) -> pd.Series:
    try:
        return frame.stack(future_stack=True)
    except TypeError:
        return frame.stack(dropna=False)


def _daily_ic(df: pd.DataFrame) -> pd.Series:
    def _corr(g: pd.DataFrame) -> float:
        if g["pred"].nunique(dropna=True) < 2 or g["label"].nunique(dropna=True) < 2:
            return np.nan
        return float(g["pred"].corr(g["label"]))

    return df.groupby(level="datetime", sort=True).apply(_corr)


def _sanitize_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.replace([np.inf, -np.inf], np.nan)
    out = out.fillna(0.0)
    return out.astype(np.float32)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _fit_ridge_closed_form(x: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> RidgeLikeModel:
    if x.ndim != 2:
        raise ValueError(f"x must be 2D, got shape={x.shape}")
    if y.ndim != 1:
        y = y.reshape(-1)
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"x/y row mismatch: {x.shape[0]} vs {y.shape[0]}")
    if x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError("empty training matrix")

    # Keep only rows with finite labels; non-finite feature values are converted to 0.
    row_mask = np.isfinite(y)
    if not np.any(row_mask):
        raise ValueError("no finite labels for model fitting")
    x = x[row_mask]
    y = y[row_mask]
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    means = x.mean(axis=0)
    stds = x.std(axis=0)
    means = np.nan_to_num(means, nan=0.0, posinf=0.0, neginf=0.0)
    stds = np.nan_to_num(stds, nan=1.0, posinf=1.0, neginf=1.0)
    stds = np.where(stds < 1e-12, 1.0, stds)

    x_std = (x - means) / stds
    x_std = np.nan_to_num(x_std, nan=0.0, posinf=0.0, neginf=0.0)
    y_mean = float(y.mean())
    y_centered = y - y_mean
    y_centered = np.nan_to_num(y_centered, nan=0.0, posinf=0.0, neginf=0.0)

    n_features = x_std.shape[1]
    gram = x_std.T @ x_std
    rhs = x_std.T @ y_centered

    gram = np.nan_to_num(gram, nan=0.0, posinf=0.0, neginf=0.0)
    rhs = np.nan_to_num(rhs, nan=0.0, posinf=0.0, neginf=0.0)

    base_alpha = float(alpha)
    if not np.isfinite(base_alpha) or base_alpha <= 0:
        base_alpha = 1e-3

    coef = None
    for scale in (1.0, 10.0, 100.0, 1000.0):
        reg = (base_alpha * scale) * np.eye(n_features, dtype=np.float64)
        system = gram + reg
        system = np.nan_to_num(system, nan=0.0, posinf=0.0, neginf=0.0)
        try:
            coef = np.linalg.solve(system, rhs)
            break
        except np.linalg.LinAlgError:
            continue

    if coef is None:
        reg = (base_alpha * 1000.0) * np.eye(n_features, dtype=np.float64)
        system = np.nan_to_num(gram + reg, nan=0.0, posinf=0.0, neginf=0.0)
        try:
            coef = np.linalg.pinv(system, rcond=1e-8) @ rhs
        except np.linalg.LinAlgError:
            # Final fallback: least squares on stabilized system.
            coef = np.linalg.lstsq(system, rhs, rcond=None)[0]

    coef = np.nan_to_num(coef, nan=0.0, posinf=0.0, neginf=0.0)
    intercept = y_mean
    return RidgeLikeModel(
        coef_=coef,
        intercept_=intercept,
        feature_means_=means,
        feature_stds_=stds,
    )


def train_factor_model(
    features: pd.DataFrame,
    panel: pd.DataFrame,
    train_config: TrainConfig,
) -> ModelResult:
    close = panel.xs("close", axis=1, level=1).sort_index()
    horizon = int(train_config.label_horizon)
    forward_ret = close.shift(-horizon) / close - 1.0

    label = _stack_frame(forward_ret).rename("label")
    label.index = label.index.set_names(["datetime", "instrument"])

    data = features.join(label, how="left")
    data = data.dropna(subset=["label"]).sort_index()
    if len(data) == 0:
        raise RuntimeError("no labeled samples after join; please expand data window")
    dt_index = data.index.get_level_values("datetime")

    train_start_ts = pd.Timestamp(train_config.train_start)
    train_end_ts = pd.Timestamp(train_config.train_end)
    in_sample_mask = dt_index >= train_start_ts
    data = data.loc[in_sample_mask]

    feature_cols = list(features.columns)
    dt_index = data.index.get_level_values("datetime")
    train_mask = (dt_index >= train_start_ts) & (dt_index <= train_end_ts)

    train_part = data.loc[train_mask, feature_cols]
    valid_cols = [col for col in feature_cols if train_part[col].notna().any()]
    used_zero_fill_fallback = False
    if len(valid_cols) == 0:
        # Short windows may make all rolling factors NaN; fallback to zero-filled subset
        # so workflow can still complete and report metrics.
        valid_cols = feature_cols[: min(16, len(feature_cols))]
        data.loc[:, valid_cols] = data.loc[:, valid_cols].fillna(0.0)
        used_zero_fill_fallback = True

    x_all_df = _sanitize_features(data[valid_cols])
    y_all = data["label"].astype(np.float64)

    x_train = x_all_df.loc[train_mask].to_numpy(dtype=np.float64)
    y_train = y_all.loc[train_mask].to_numpy(dtype=np.float64)
    used_all_data_as_train = False
    if x_train.shape[0] == 0:
        x_train = x_all_df.to_numpy(dtype=np.float64)
        y_train = y_all.to_numpy(dtype=np.float64)
        train_mask = np.ones(len(x_all_df), dtype=bool)
        used_all_data_as_train = True

    # Keep alpha tied to feature count so regularization scales stably.
    alpha = max(1e-3, 0.05 * x_train.shape[1])
    model = _fit_ridge_closed_form(x_train, y_train, alpha=alpha)

    x_all = x_all_df.to_numpy(dtype=np.float64)
    pred_all = pd.Series(model.predict(x_all), index=x_all_df.index, name="pred")

    pred_df = pd.DataFrame({"pred": pred_all, "label": y_all})
    pred_df["split"] = np.where(train_mask, "train", "test")

    train_df = pred_df[pred_df["split"] == "train"]
    test_df = pred_df[pred_df["split"] == "test"]

    train_ic = float(_daily_ic(train_df).mean()) if len(train_df) > 0 else np.nan
    test_ic = float(_daily_ic(test_df).mean()) if len(test_df) > 0 else np.nan

    metrics: dict[str, Any] = {
        "samples_total": int(len(pred_df)),
        "samples_train": int(len(train_df)),
        "samples_test": int(len(test_df)),
        "feature_count": int(len(valid_cols)),
        "ridge_alpha": float(alpha),
        "train_ic_mean": train_ic,
        "test_ic_mean": test_ic,
        "train_rmse": _rmse(train_df["label"].to_numpy(), train_df["pred"].to_numpy()),
        "test_rmse": _rmse(test_df["label"].to_numpy(), test_df["pred"].to_numpy()),
        "used_zero_fill_fallback": bool(used_zero_fill_fallback),
        "used_all_data_as_train": bool(used_all_data_as_train),
    }

    return ModelResult(model=model, predictions=pred_df, metrics=metrics, feature_columns=valid_cols)
