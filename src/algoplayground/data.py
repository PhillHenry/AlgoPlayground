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
    df[list(OHLCV_COLUMNS)] = df[list(OHLCV_COLUMNS)].diff()
    df = df.iloc[1:].reset_index(drop=True)
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


@dataclass(frozen=True)
class WalkForwardFold:
    """One walk-forward training/validation pair (in chronological order)."""

    train: pd.DataFrame
    val: pd.DataFrame


def walk_forward_folds(
    frame: pd.DataFrame,
    *,
    holdout_fraction: float = 0.15,
    val_chunk_size: int = 60,
    initial_train_size: int | None = None,
    step_size: int | None = None,
    expanding: bool = True,
) -> tuple[list[WalkForwardFold], pd.DataFrame]:
    """Build walk-forward train/val folds plus a final chronological holdout.

    The last ``holdout_fraction`` of ``frame`` is reserved as the holdout. The
    remaining pre-holdout range is sliced into overlapping ``(train, val)``
    folds: fold 0 trains on the first ``initial_train_size`` rows and
    validates on the next ``val_chunk_size`` rows; subsequent folds advance
    the val window by ``step_size`` rows and extend (``expanding=True``) or
    slide (``expanding=False``, fixed-size rolling window of
    ``initial_train_size`` rows) the train window accordingly.

    Defaults:
      ``initial_train_size`` → ``val_chunk_size`` (smallest meaningful fold).
      ``step_size``          → ``val_chunk_size`` (non-overlapping val sets).
    """
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be in (0, 1)")
    if val_chunk_size < 1:
        raise ValueError("val_chunk_size must be >= 1")
    if initial_train_size is None:
        initial_train_size = val_chunk_size
    if initial_train_size < 1:
        raise ValueError("initial_train_size must be >= 1")
    if step_size is None:
        step_size = val_chunk_size
    if step_size < 1:
        raise ValueError("step_size must be >= 1")

    n = len(frame)
    holdout_start = int(n * (1.0 - holdout_fraction))
    pre_holdout = frame.iloc[:holdout_start]
    holdout = frame.iloc[holdout_start:].reset_index(drop=True)

    folds: list[WalkForwardFold] = []
    train_end = initial_train_size
    while train_end + val_chunk_size <= len(pre_holdout):
        val_end = train_end + val_chunk_size
        if expanding:
            train_part = pre_holdout.iloc[:train_end]
        else:
            train_part = pre_holdout.iloc[
                max(0, train_end - initial_train_size) : train_end
            ]
        val_part = pre_holdout.iloc[train_end:val_end]
        folds.append(
            WalkForwardFold(
                train=train_part.reset_index(drop=True),
                val=val_part.reset_index(drop=True),
            )
        )
        train_end += step_size

    if not folds:
        raise ValueError(
            f"No walk-forward folds produced: pre-holdout has {len(pre_holdout)} "
            f"rows but initial_train_size + val_chunk_size = "
            f"{initial_train_size + val_chunk_size}."
        )
    return folds, holdout


def train_val_holdout_split(
    frame: pd.DataFrame,
    holdout_fraction: float = 0.15,
    chunk_size: int = 1440,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Reserve the final ``holdout_fraction`` of rows as a chronological holdout.

    The pre-holdout range is sliced into contiguous blocks of ``chunk_size``
    rows and the blocks are dealt out in alternation: block 0 → train,
    block 1 → val, block 2 → train, block 3 → val, … so both sets cover the
    full pre-holdout time range. A short trailing block (if the pre-holdout
    length is not a multiple of ``chunk_size``) is assigned to whichever set
    is next in the rotation.
    """
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be in (0, 1)")
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")

    n = len(frame)
    holdout_start = int(n * (1.0 - holdout_fraction))
    holdout = frame.iloc[holdout_start:].reset_index(drop=True)

    train_parts: list[pd.DataFrame] = []
    val_parts: list[pd.DataFrame] = []
    for i, start in enumerate(range(0, holdout_start, chunk_size)):
        end = min(start + chunk_size, holdout_start)
        chunk = frame.iloc[start:end]
        (train_parts if i % 2 == 0 else val_parts).append(chunk)

    empty = frame.iloc[0:0]
    train = pd.concat(train_parts, ignore_index=True) if train_parts else empty
    val = pd.concat(val_parts, ignore_index=True) if val_parts else empty
    return train, val, holdout