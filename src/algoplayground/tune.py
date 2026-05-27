"""Optuna-driven hyperparameter search over LSTM and CNN regressors."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import optuna
import pandas as pd

from .data import OHLCVWindowDataset, load_ohlcv_csv, train_val_split
from .models import CNNConfig, CNNRegressor, LSTMConfig, LSTMRegressor
from .train import TrainConfig, train_model

logger = logging.getLogger(__name__)

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
    logger.info(
        "Split data into %d train rows and %d val rows (val_fraction=%.2f)",
        len(train_frame),
        len(val_frame),
        config.val_fraction,
    )

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

        logger.info(
            "Trial %d starting: model=%s window=%d batch=%d lr=%.2e wd=%.2e",
            trial.number,
            kind,
            window_size,
            batch_size,
            learning_rate,
            weight_decay,
        )

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
                logger.info("Trial %d pruned at epoch %d (val_loss=%.6f)", trial.number, epoch, val_loss)
                raise optuna.TrialPruned()

        result = train_model(model, train_ds, val_ds, train_config, on_epoch_end=report)
        logger.info(
            "Trial %d finished: best_val_loss=%.6f at epoch %d",
            trial.number,
            result.best_val_loss,
            result.best_epoch,
        )
        return result.best_val_loss

    sampler = optuna.samplers.TPESampler(seed=config.seed)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=3)
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
    logger.info(
        "Starting Optuna study: n_trials=%d max_epochs=%d model_kind=%s device=%s",
        config.n_trials,
        config.max_epochs,
        config.model_kind,
        config.device,
    )
    study.optimize(objective, n_trials=config.n_trials)
    logger.info(
        "Study complete: best_value=%.6f best_trial=%d",
        study.best_value,
        study.best_trial.number,
    )
    return study


def main(
    csv_path: str,
    n_trials: int = 25,
    max_epochs: int = 30,
    val_fraction: float = 0.2,
    model_kind: ModelKind = "auto",
    device: str = "cpu",
    seed: int = 42,
) -> optuna.Study:
    """Load OHLCV data from ``csv_path`` and run an Optuna study."""
    logger.info("Loading OHLCV CSV from %s", csv_path)
    frame = load_ohlcv_csv(csv_path)
    logger.info(
        "Loaded %d rows spanning %s to %s",
        len(frame),
        frame["timestamp"].iloc[0],
        frame["timestamp"].iloc[-1],
    )
    study = tune(
        frame,
        TuneConfig(
            n_trials=n_trials,
            max_epochs=max_epochs,
            val_fraction=val_fraction,
            model_kind=model_kind,
            device=device,
            seed=seed,
        ),
    )
    logger.info("Best validation MSE: %.6f", study.best_value)
    logger.info("Best params:")
    for name, value in study.best_params.items():
        logger.info("  %s: %s", name, value)
    return study


if __name__ == "__main__":
    """
    Typical Output:
        Best validation MSE: 0.000373
        Best params:
          window_size: 24
          batch_size: 32
          learning_rate: 0.0002051338263087451
          weight_decay: 6.02521573620385e-08
          model_kind: cnn
          cnn_depth: 2
          cnn_base_channels: 64
          kernel_size: 3
          dropout: 0.09170225492671691
    """
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Train and tune an OHLCV sequence regressor from a CSV file."
    )
    parser.add_argument("csv_path", help="Path to an OHLCV CSV file.")
    parser.add_argument("--n-trials", type=int, default=25)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument(
        "--model-kind",
        choices=["lstm", "cnn", "auto"],
        default="auto",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    main(
        csv_path=args.csv_path,
        n_trials=args.n_trials,
        max_epochs=args.max_epochs,
        val_fraction=args.val_fraction,
        model_kind=args.model_kind,
        device=args.device,
        seed=args.seed,
    )