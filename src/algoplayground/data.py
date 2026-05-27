"""OHLCV CSV loading and sliding-window dataset for sequence models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

OHLCV_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")
FEATURE_COLUMNS: tuple[str, ...] = OHLCV_COLUMNS
TARGET_COLUMN: str = "close"


@dataclass(frozen=True)
class Normalization:
    """Per-feature mean/std used to standardise inputs and targets."""

    feature_mean: np.ndarray
    feature_std: np.ndarray
    target_mean: float
    target_std: float

    @classmethod
    def fit(cls, features: np.ndarray, targets: np.ndarray) -> Normalization:
        feature_std = features.std(axis=0)
        feature_std[feature_std == 0.0] = 1.0
        target_std = float(targets.std()) or 1.0
        return cls(
            feature_mean=features.mean(axis=0),
            feature_std=feature_std,
            target_mean=float(targets.mean()),
            target_std=target_std,
        )

    def transform_features(self, x: np.ndarray) -> np.ndarray:
        return (x - self.feature_mean) / self.feature_std

    def transform_target(self, y: np.ndarray) -> np.ndarray:
        return (y - self.target_mean) / self.target_std

    def invert_target(self, y: np.ndarray) -> np.ndarray:
        return y * self.target_std + self.target_mean


def load_ohlcv_csv(path: str | Path) -> pd.DataFrame:
    """Load an OHLCV CSV and return it sorted by timestamp.

    The CSV must contain columns: timestamp, open, high, low, close, volume.
    Timestamps such as ``"2025-05-23 13:38:00+00"`` are parsed as UTC-aware.
    """
    df = pd.read_csv(path)
    expected = {"timestamp", *OHLCV_COLUMNS}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df[list(OHLCV_COLUMNS)] = df[list(OHLCV_COLUMNS)].astype(np.float32)
    return df


class OHLCVWindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Sliding-window dataset over OHLCV bars.

    Each sample is a window of ``window_size`` bars of OHLCV features and the
    target is the close price ``horizon`` bars after the end of the window.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        window_size: int = 32,
        horizon: int = 1,
        normalization: Normalization | None = None,
    ) -> None:
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        if len(frame) < window_size + horizon:
            raise ValueError(
                f"Need at least {window_size + horizon} rows, got {len(frame)}"
            )

        self.window_size = window_size
        self.horizon = horizon

        features = frame[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float32)
        targets = frame[TARGET_COLUMN].to_numpy(dtype=np.float32)

        if normalization is None:
            fit_end = len(features) - horizon
            normalization = Normalization.fit(features[:fit_end], targets[:fit_end])
        self.normalization = normalization

        self._features = normalization.transform_features(features).astype(np.float32)
        self._targets = normalization.transform_target(targets).astype(np.float32)

    @property
    def n_features(self) -> int:
        return len(FEATURE_COLUMNS)

    def __len__(self) -> int:
        return len(self._features) - self.window_size - self.horizon + 1

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        start = index
        end = index + self.window_size
        x = self._features[start:end]
        y = self._targets[end + self.horizon - 1]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)


def train_val_split(
    frame: pd.DataFrame, val_fraction: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split that preserves time ordering."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1)")
    split = int(len(frame) * (1.0 - val_fraction))
    return frame.iloc[:split].reset_index(drop=True), frame.iloc[split:].reset_index(drop=True)


def train_val_holdout_split(
    frame: pd.DataFrame,
    val_fraction: float = 0.15,
    holdout_fraction: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological 3-way split: train / val / holdout, all in time order."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1)")
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be in (0, 1)")
    if val_fraction + holdout_fraction >= 1.0:
        raise ValueError("val_fraction + holdout_fraction must be < 1")
    n = len(frame)
    train_end = int(n * (1.0 - val_fraction - holdout_fraction))
    val_end = int(n * (1.0 - holdout_fraction))
    return (
        frame.iloc[:train_end].reset_index(drop=True),
        frame.iloc[train_end:val_end].reset_index(drop=True),
        frame.iloc[val_end:].reset_index(drop=True),
    )