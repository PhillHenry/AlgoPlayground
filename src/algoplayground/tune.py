"""Optuna-driven hyperparameter search over LSTM and CNN regressors."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
import optuna
import pandas as pd

from .data import (
    OHLCVWindowDataset,
    WalkForwardFold,
    load_ohlcv_csv,
    merge_ohlcv_feeds,
    walk_forward_folds,
)
from .models import CNNConfig, CNNRegressor, LSTMConfig, LSTMRegressor
from .train import TrainConfig, evaluate_model, predict_dataset, train_model

logger = logging.getLogger(__name__)

ModelKind = Literal["lstm", "cnn", "auto"]


@dataclass(frozen=True)
class TuneConfig:
    n_trials: int = 25
    max_epochs: int = 30
    holdout_fraction: float = 0.15
    chunk_size: int = 60
    initial_train_size: int | None = None
    step_size: int | None = None
    expanding: bool = False
    model_kind: ModelKind = "auto"
    device: str = "cpu"
    seed: int = 42
    feature_columns: tuple[str, ...] | None = None
    target_column: str | None = None


@dataclass(frozen=True)
class TuneResult:
    study: optuna.Study
    holdout_loss: float
    holdout_sum_diff: float
    holdout_mape: float


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


def _build_from_params(
    params: dict,
    n_features: int,
    fallback_model_kind: ModelKind,
):
    kind = params.get("model_kind", fallback_model_kind)
    if kind == "lstm":
        return LSTMRegressor(
            LSTMConfig(
                n_features=n_features,
                hidden_size=params["hidden_size"],
                num_layers=params["num_layers"],
                dropout=params["dropout"],
            )
        )
    depth = params["cnn_depth"]
    base = params["cnn_base_channels"]
    channels = tuple(base * (2**i) for i in range(depth))
    return CNNRegressor(
        CNNConfig(
            n_features=n_features,
            channels=channels,
            kernel_size=params["kernel_size"],
            dropout=params["dropout"],
        )
    )


def tune(frame: pd.DataFrame, config: TuneConfig = TuneConfig()) -> TuneResult:
    """Run an Optuna study under walk-forward validation, score best on holdout.

    The pre-holdout range is split into walk-forward folds (see
    :func:`walk_forward_folds`). Each Optuna trial trains a fresh model on
    every fold and the trial's objective is the **mean** of the per-fold best
    val MSEs. After tuning, the best params are refit on the last fold's
    train (early-stopped on its val) and scored once on the holdout.
    """
    folds, holdout_frame = walk_forward_folds(
        frame,
        holdout_fraction=config.holdout_fraction,
        val_chunk_size=config.chunk_size,
        initial_train_size=config.initial_train_size,
        step_size=config.step_size,
        expanding=config.expanding,
    )
    logger.info(
        "Built %d walk-forward folds (val_chunk_size=%d, initial_train_size=%s, "
        "step_size=%s, expanding=%s) + %d holdout rows",
        len(folds),
        config.chunk_size,
        config.initial_train_size if config.initial_train_size is not None else config.chunk_size,
        config.step_size if config.step_size is not None else config.chunk_size,
        config.expanding,
        len(holdout_frame),
    )
    for i, fold in enumerate(folds):
        logger.info(
            "  fold %d: train=%d val=%d", i, len(fold.train), len(fold.val)
        )

    def objective(trial: optuna.trial.Trial) -> float:
        window_size = trial.suggest_int("window_size", 8, 64, step=8)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-8, 1e-3, log=True)

        kind: ModelKind = config.model_kind
        if kind == "auto":
            kind = trial.suggest_categorical("model_kind", ["lstm", "cnn"])  # type: ignore[assignment]

        # Suggest model-specific hyperparams ONCE per trial (Optuna disallows
        # re-suggesting the same name), then reuse them to build a fresh model
        # for each walk-forward fold.
        if kind == "lstm":
            hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128])
            num_layers = trial.suggest_int("num_layers", 1, 3)
            dropout = trial.suggest_float("dropout", 0.0, 0.5)
        else:
            cnn_depth = trial.suggest_int("cnn_depth", 1, 3)
            cnn_base_channels = trial.suggest_categorical("cnn_base_channels", [16, 32, 64])
            kernel_size = trial.suggest_categorical("kernel_size", [3, 5, 7])
            dropout = trial.suggest_float("dropout", 0.0, 0.5)

        logger.info(
            "Trial %d starting: model=%s window=%d batch=%d lr=%.2e wd=%.2e (%d folds)",
            trial.number,
            kind,
            window_size,
            batch_size,
            learning_rate,
            weight_decay,
            len(folds),
        )

        train_config = TrainConfig(
            epochs=config.max_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            device=config.device,
        )

        fold_losses: list[float] = []
        for fold_idx, fold in enumerate(folds):
            train_ds = OHLCVWindowDataset(
                fold.train,
                window_size=window_size,
                feature_columns=config.feature_columns,
                target_column=config.target_column,
            )
            val_ds = OHLCVWindowDataset(
                fold.val,
                window_size=window_size,
                normalization=train_ds.normalization,
                feature_columns=config.feature_columns,
                target_column=config.target_column,
            )

            if kind == "lstm":
                model = LSTMRegressor(
                    LSTMConfig(
                        n_features=train_ds.n_features,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        dropout=dropout,
                    )
                )
            else:
                channels = tuple(cnn_base_channels * (2**i) for i in range(cnn_depth))
                model = CNNRegressor(
                    CNNConfig(
                        n_features=train_ds.n_features,
                        channels=channels,
                        kernel_size=kernel_size,
                        dropout=dropout,
                    )
                )

            result = train_model(model, train_ds, val_ds, train_config)
            fold_losses.append(result.best_val_loss)
            logger.info(
                "  trial %d fold %d/%d: val_loss=%.6f (best epoch %d)",
                trial.number,
                fold_idx,
                len(folds) - 1,
                result.best_val_loss,
                result.best_epoch,
            )

            running_mean = float(np.mean(fold_losses))
            trial.report(running_mean, step=fold_idx)
            if trial.should_prune():
                logger.info(
                    "Trial %d pruned after fold %d (running mean val_loss=%.6f)",
                    trial.number,
                    fold_idx,
                    running_mean,
                )
                raise optuna.TrialPruned()

        mean_loss = float(np.mean(fold_losses))
        logger.info(
            "Trial %d finished: mean_val_loss=%.6f over %d folds",
            trial.number,
            mean_loss,
            len(folds),
        )
        return mean_loss

    sampler = optuna.samplers.TPESampler(seed=config.seed)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=1)
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

    holdout_loss, holdout_sum_diff, holdout_mape = _evaluate_best_on_holdout(
        study,
        folds,
        holdout_frame,
        config,
    )
    return TuneResult(
        study=study,
        holdout_loss=holdout_loss,
        holdout_sum_diff=holdout_sum_diff,
        holdout_mape=holdout_mape,
    )


def _evaluate_best_on_holdout(
    study: optuna.Study,
    folds: list[WalkForwardFold],
    holdout_frame: pd.DataFrame,
    config: TuneConfig,
) -> tuple[float, float, float]:
    """Refit best params on the last walk-forward fold and score on holdout.

    Returns ``(mse, sum_of_signed_differences, mape_percent)``. The sum and
    MAPE are computed in the original target units (i.e. after inverting the
    normalization). MAPE is expressed as a percentage; samples with a zero
    actual target are excluded from the MAPE denominator.
    """
    params = study.best_params
    window_size = params["window_size"]
    last_fold = folds[-1]

    train_ds = OHLCVWindowDataset(
        last_fold.train,
        window_size=window_size,
        feature_columns=config.feature_columns,
        target_column=config.target_column,
    )
    val_ds = OHLCVWindowDataset(
        last_fold.val,
        window_size=window_size,
        normalization=train_ds.normalization,
        feature_columns=config.feature_columns,
        target_column=config.target_column,
    )
    holdout_ds = OHLCVWindowDataset(
        holdout_frame,
        window_size=window_size,
        normalization=train_ds.normalization,
        feature_columns=config.feature_columns,
        target_column=config.target_column,
    )

    model = _build_from_params(params, train_ds.n_features, config.model_kind)
    train_config = TrainConfig(
        epochs=config.max_epochs,
        batch_size=params["batch_size"],
        learning_rate=params["learning_rate"],
        weight_decay=params["weight_decay"],
        device=config.device,
    )

    logger.info(
        "Refitting best model on last walk-forward fold (%d train / %d val rows) "
        "for final holdout evaluation",
        len(last_fold.train),
        len(last_fold.val),
    )
    train_model(model, train_ds, val_ds, train_config)

    holdout_loss = evaluate_model(
        model,
        holdout_ds,
        batch_size=params["batch_size"],
        device=config.device,
    )
    preds_norm, targets_norm = predict_dataset(
        model,
        holdout_ds,
        batch_size=params["batch_size"],
        device=config.device,
    )
    preds = holdout_ds.normalization.invert_target(preds_norm)
    targets = holdout_ds.normalization.invert_target(targets_norm)
    sum_diff = float(np.sum(preds - targets))

    nonzero = targets != 0
    if nonzero.any():
        mape = float(
            np.mean(np.abs((preds[nonzero] - targets[nonzero]) / targets[nonzero]))
            * 100.0
        )
    else:
        mape = float("nan")

    logger.info(
        "Holdout MSE: %.6f (over %d windows, target lags features by %d bar)",
        holdout_loss,
        len(holdout_ds),
        holdout_ds.horizon,
    )
    logger.info(
        "Holdout sum of (prediction - actual) in target units: %.6f "
        "(mean per-sample: %.6f)",
        sum_diff,
        sum_diff / max(len(holdout_ds), 1),
    )
    logger.info(
        "Holdout MAPE: %.4f%% (over %d non-zero targets of %d)",
        mape,
        int(nonzero.sum()),
        len(holdout_ds),
    )
    return holdout_loss, sum_diff, mape


def main(
    csv_path: str,
    csv_path_2: str,
    n_trials: int = 25,
    max_epochs: int = 30,
    holdout_fraction: float = 0.15,
    chunk_size: int = 60,
    model_kind: ModelKind = "auto",
    device: str = "cpu",
    seed: int = 42,
) -> TuneResult:
    """Load two OHLCV feeds, align them on timestamp, tune, and report error.

    Both CSVs are loaded and inner-joined on ``timestamp`` so every training
    and evaluation sample carries OHLCV features from both feeds at the same
    instant. The prediction target is the first (primary) feed's next close.
    """
    logger.info("Loading OHLCV CSVs from %s and %s", csv_path, csv_path_2)
    frame_a = load_ohlcv_csv(csv_path)
    frame_b = load_ohlcv_csv(csv_path_2)
    frame, feature_columns, target_column = merge_ohlcv_feeds(frame_a, frame_b)
    logger.info(
        "Merged feeds: %d primary + %d secondary rows -> %d aligned rows "
        "spanning %s to %s (%d features, target=%s)",
        len(frame_a),
        len(frame_b),
        len(frame),
        frame["timestamp"].iloc[0],
        frame["timestamp"].iloc[-1],
        len(feature_columns),
        target_column,
    )
    result = tune(
        frame,
        TuneConfig(
            n_trials=n_trials,
            max_epochs=max_epochs,
            holdout_fraction=holdout_fraction,
            chunk_size=chunk_size,
            model_kind=model_kind,
            device=device,
            seed=seed,
            feature_columns=tuple(feature_columns),
            target_column=target_column,
        ),
    )
    logger.info("Best validation MSE:        %.6f", result.study.best_value)
    logger.info("Holdout MSE:                %.6f", result.holdout_loss)
    logger.info("Holdout Σ(pred - actual):   %.6f", result.holdout_sum_diff)
    logger.info("Holdout MAPE:               %.4f%%", result.holdout_mape)
    logger.info("Best params:")
    for name, value in result.study.best_params.items():
        logger.info("  %s: %s", name, value)
    return result


if __name__ == "__main__":
    """
    Run with something like:
    ./.venv/bin/python -m algoplayground.tune  /home/henryp/Downloads/googl_dataset_London-Strategic-Edge.csv /home/henryp/Downloads/aapl_dataset_London-Strategic-Edge.csv --n-trials 10 --max-epochs 30 --model-kind auto --device cuda --model-kind lstm
    
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
    parser.add_argument(
        "csv_path",
        help="Path to the primary OHLCV CSV file (its next close is the target).",
    )
    parser.add_argument(
        "csv_path_2",
        help="Path to the secondary OHLCV CSV file (aligned on timestamp).",
    )
    parser.add_argument("--n-trials", type=int, default=25)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--holdout-fraction", type=float, default=0.15)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1440,
        help="Size of alternating train/val blocks within the non-holdout range.",
    )
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
        csv_path_2=args.csv_path_2,
        n_trials=args.n_trials,
        max_epochs=args.max_epochs,
        holdout_fraction=args.holdout_fraction,
        chunk_size=args.chunk_size,
        model_kind=args.model_kind,
        device=args.device,
        seed=args.seed,
    )