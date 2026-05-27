"""Optuna-driven hyperparameter search over LSTM and CNN regressors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import optuna
import pandas as pd

from .data import OHLCVWindowDataset, train_val_split
from .models import CNNConfig, CNNRegressor, LSTMConfig, LSTMRegressor
from .train import TrainConfig, train_model

ModelKind = Literal["lstm", "cnn", "auto"]


@dataclass(frozen=True)
class TuneConfig:
    n_trials: int = 25
    max_epochs: int = 30
    val_fraction: float = 0.2
    model_kind: ModelKind = "auto"
    device: str = "cpu"
    seed: int = 42


def _suggest_lstm(trial: optuna.trial.Trial, n_features: int) -> LSTMRegressor:
    return LSTMRegressor(
        LSTMConfig(
            n_features=n_features,
            hidden_size=trial.suggest_categorical("hidden_size", [32, 64, 128]),
            num_layers=trial.suggest_int("num_layers", 1, 3),
            dropout=trial.suggest_float("dropout", 0.0, 0.5),
        )
    )


def _suggest_cnn(trial: optuna.trial.Trial, n_features: int) -> CNNRegressor:
    depth = trial.suggest_int("cnn_depth", 1, 3)
    base = trial.suggest_categorical("cnn_base_channels", [16, 32, 64])
    channels = tuple(base * (2**i) for i in range(depth))
    return CNNRegressor(
        CNNConfig(
            n_features=n_features,
            channels=channels,
            kernel_size=trial.suggest_categorical("kernel_size", [3, 5, 7]),
            dropout=trial.suggest_float("dropout", 0.0, 0.5),
        )
    )


def tune(frame: pd.DataFrame, config: TuneConfig = TuneConfig()) -> optuna.Study:
    """Run an Optuna study and return the completed study object.

    The best hyperparameters are available via ``study.best_params`` and the
    best validation MSE via ``study.best_value``.
    """
    train_frame, val_frame = train_val_split(frame, val_fraction=config.val_fraction)

    def objective(trial: optuna.trial.Trial) -> float:
        window_size = trial.suggest_int("window_size", 8, 64, step=8)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-8, 1e-3, log=True)

        train_ds = OHLCVWindowDataset(train_frame, window_size=window_size)
        val_ds = OHLCVWindowDataset(
            val_frame, window_size=window_size, normalization=train_ds.normalization
        )

        kind: ModelKind = config.model_kind
        if kind == "auto":
            kind = trial.suggest_categorical("model_kind", ["lstm", "cnn"])  # type: ignore[assignment]

        if kind == "lstm":
            model = _suggest_lstm(trial, train_ds.n_features)
        else:
            model = _suggest_cnn(trial, train_ds.n_features)

        train_config = TrainConfig(
            epochs=config.max_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            device=config.device,
        )

        def report(epoch: int, val_loss: float) -> None:
            trial.report(val_loss, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        result = train_model(model, train_ds, val_ds, train_config, on_epoch_end=report)
        return result.best_val_loss

    sampler = optuna.samplers.TPESampler(seed=config.seed)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=3)
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=config.n_trials)
    return study