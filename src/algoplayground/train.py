"""Training loop for sequence regressors over OHLCV windows."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 30
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    early_stopping_patience: int = 5
    device: str = "cpu"


@dataclass(frozen=True)
class TrainResult:
    best_val_loss: float
    best_epoch: int
    history: list[tuple[float, float]]  # (train_loss, val_loss) per epoch


PruneCallback = Callable[[int, float], None]


def train_model(
    model: nn.Module,
    train_dataset: Dataset[tuple[torch.Tensor, torch.Tensor]],
    val_dataset: Dataset[tuple[torch.Tensor, torch.Tensor]],
    config: TrainConfig = TrainConfig(),
    on_epoch_end: PruneCallback | None = None,
) -> TrainResult:
    """Train ``model`` with Adam and early stopping on val MSE.

    ``on_epoch_end`` is invoked after each validation pass with
    ``(epoch, val_loss)`` and may raise to signal a pruned trial.
    """
    device = torch.device(config.device)
    model.to(device)

    train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=False
    )
    val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False, drop_last=False
    )

    optimiser = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    loss_fn = nn.MSELoss()

    best_val = math.inf
    best_epoch = -1
    epochs_without_improvement = 0
    history: list[tuple[float, float]] = []

    for epoch in range(config.epochs):
        train_loss = _run_epoch(model, train_loader, loss_fn, device, optimiser)
        val_loss = _run_epoch(model, val_loader, loss_fn, device, optimiser=None)
        history.append((train_loss, val_loss))

        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        logger.debug(
            "Epoch %d: train_loss=%.6f val_loss=%.6f%s",
            epoch,
            train_loss,
            val_loss,
            " (new best)" if improved else "",
        )

        if on_epoch_end is not None:
            on_epoch_end(epoch, val_loss)

        if epochs_without_improvement >= config.early_stopping_patience:
            logger.info(
                "Early stopping at epoch %d (best epoch %d, best val_loss=%.6f)",
                epoch,
                best_epoch,
                best_val,
            )
            break

    return TrainResult(best_val_loss=best_val, best_epoch=best_epoch, history=history)


def evaluate_model(
    model: nn.Module,
    dataset: Dataset[tuple[torch.Tensor, torch.Tensor]],
    batch_size: int = 64,
    device: str = "cpu",
) -> float:
    """Compute mean MSE loss of ``model`` over ``dataset`` (no gradient updates)."""
    device_t = torch.device(device)
    model.to(device_t)
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, drop_last=False
    )
    return _run_epoch(model, loader, nn.MSELoss(), device_t, optimiser=None)


def predict_dataset(
    model: nn.Module,
    dataset: Dataset[tuple[torch.Tensor, torch.Tensor]],
    batch_size: int = 64,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Run ``model`` over ``dataset`` and return ``(predictions, targets)``."""
    device_t = torch.device(device)
    model.to(device_t)
    model.eval()
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, drop_last=False
    )
    preds: list[np.ndarray] = []
    tgts: list[np.ndarray] = []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device_t)
            preds.append(model(inputs).detach().cpu().numpy())
            tgts.append(targets.detach().cpu().numpy())
    return np.concatenate(preds), np.concatenate(tgts)


def _run_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    loss_fn: nn.Module,
    device: torch.device,
    optimiser: torch.optim.Optimizer | None,
) -> float:
    is_train = optimiser is not None
    model.train(is_train)

    total_loss = 0.0
    total_samples = 0
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            predictions = model(inputs)
            loss = loss_fn(predictions, targets)
            if is_train:
                assert optimiser is not None
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
    return total_loss / max(total_samples, 1)