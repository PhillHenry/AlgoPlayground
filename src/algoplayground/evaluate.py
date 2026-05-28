"""Train one model from explicit hyperparameters and score it on the holdout."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from .data import OHLCVWindowDataset, load_ohlcv_csv, train_val_holdout_split
from .models import CNNConfig, CNNRegressor, LSTMConfig, LSTMRegressor
from .train import TrainConfig, evaluate_model, predict_dataset, train_model

logger = logging.getLogger(__name__)

ModelKind = Literal["lstm", "cnn"]


@dataclass(frozen=True)
class HoldoutMetrics:
    mse: float
    sum_diff: float
    mape: float
    n_samples: int


def _compute_holdout_metrics(
    model,
    holdout_ds: OHLCVWindowDataset,
    batch_size: int,
    device: str,
) -> HoldoutMetrics:
    mse = evaluate_model(model, holdout_ds, batch_size=batch_size, device=device)
    preds_norm, targets_norm = predict_dataset(
        model, holdout_ds, batch_size=batch_size, device=device
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
    return HoldoutMetrics(
        mse=mse, sum_diff=sum_diff, mape=mape, n_samples=len(holdout_ds)
    )


def evaluate_on_holdout(
    frame: pd.DataFrame,
    *,
    model_kind: ModelKind,
    window_size: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    dropout: float,
    hidden_size: int | None = None,
    num_layers: int | None = None,
    cnn_depth: int | None = None,
    cnn_base_channels: int | None = None,
    kernel_size: int | None = None,
    holdout_fraction: float = 0.15,
    chunk_size: int = 60,
    max_epochs: int = 30,
    early_stopping_patience: int = 5,
    device: str = "cpu",
) -> HoldoutMetrics:
    """Split ``frame``, build the requested model, train, and score on the holdout."""
    train_frame, val_frame, holdout_frame = train_val_holdout_split(
        frame, holdout_fraction=holdout_fraction, chunk_size=chunk_size
    )
    logger.info(
        "Split data into %d train / %d val / %d holdout rows "
        "(holdout=%.2f, alternating chunk_size=%d)",
        len(train_frame),
        len(val_frame),
        len(holdout_frame),
        holdout_fraction,
        chunk_size,
    )

    train_ds = OHLCVWindowDataset(train_frame, window_size=window_size)
    val_ds = OHLCVWindowDataset(
        val_frame, window_size=window_size, normalization=train_ds.normalization
    )
    holdout_ds = OHLCVWindowDataset(
        holdout_frame, window_size=window_size, normalization=train_ds.normalization
    )

    if model_kind == "lstm":
        if hidden_size is None or num_layers is None:
            raise ValueError("LSTM requires hidden_size and num_layers")
        model = LSTMRegressor(
            LSTMConfig(
                n_features=train_ds.n_features,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
            )
        )
    elif model_kind == "cnn":
        if cnn_depth is None or cnn_base_channels is None or kernel_size is None:
            raise ValueError(
                "CNN requires cnn_depth, cnn_base_channels and kernel_size"
            )
        channels = tuple(cnn_base_channels * (2**i) for i in range(cnn_depth))
        model = CNNRegressor(
            CNNConfig(
                n_features=train_ds.n_features,
                channels=channels,
                kernel_size=kernel_size,
                dropout=dropout,
            )
        )
    else:
        raise ValueError(f"Unknown model_kind: {model_kind}")

    train_config = TrainConfig(
        epochs=max_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        early_stopping_patience=early_stopping_patience,
        device=device,
    )
    logger.info(
        "Training %s: window=%d batch=%d lr=%.2e wd=%.2e dropout=%.3f epochs=%d",
        model_kind,
        window_size,
        batch_size,
        learning_rate,
        weight_decay,
        dropout,
        max_epochs,
    )
    train_model(model, train_ds, val_ds, train_config)

    metrics = _compute_holdout_metrics(model, holdout_ds, batch_size, device)
    logger.info(
        "Holdout MSE:              %.6f (over %d windows, target lags features by %d bar)",
        metrics.mse,
        metrics.n_samples,
        holdout_ds.horizon,
    )
    logger.info(
        "Holdout Σ(pred - actual): %.6f (mean per-sample: %.6f)",
        metrics.sum_diff,
        metrics.sum_diff / max(metrics.n_samples, 1),
    )
    logger.info("Holdout MAPE:             %.4f%%", metrics.mape)
    return metrics


_REQUIRED_COMMON_KEYS = (
    "model_kind",
    "window_size",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "dropout",
)
_REQUIRED_LSTM_KEYS = ("hidden_size", "num_layers")
_REQUIRED_CNN_KEYS = ("cnn_depth", "cnn_base_channels", "kernel_size")


def _load_params(path: str) -> dict:
    """Load and validate a model-params JSON file.

    Expected shape (LSTM):
        {"model_kind": "lstm", "window_size": int, "batch_size": int,
         "learning_rate": float, "weight_decay": float, "dropout": float,
         "hidden_size": int, "num_layers": int}

    Expected shape (CNN):
        {"model_kind": "cnn", "window_size": int, "batch_size": int,
         "learning_rate": float, "weight_decay": float, "dropout": float,
         "cnn_depth": int, "cnn_base_channels": int, "kernel_size": int}
    """
    import json
    from pathlib import Path

    with Path(path).open() as fh:
        parsed = json.load(fh)
    if not isinstance(parsed, dict):
        raise ValueError(f"Params JSON must be an object, got {type(parsed).__name__}")

    missing = [k for k in _REQUIRED_COMMON_KEYS if k not in parsed]
    if missing:
        raise ValueError(f"Params JSON missing required keys: {missing}")

    kind = parsed["model_kind"]
    if kind == "lstm":
        required = _REQUIRED_LSTM_KEYS
    elif kind == "cnn":
        required = _REQUIRED_CNN_KEYS
    else:
        raise ValueError(f"Unknown model_kind: {kind!r} (expected 'lstm' or 'cnn')")
    missing = [k for k in required if k not in parsed]
    if missing:
        raise ValueError(f"Params JSON missing {kind} keys: {missing}")

    return parsed


if __name__ == "__main__":
    """
    Run with something like:
    ./.venv/bin/python -m algoplayground.evaluate  /home/henryp/Downloads/googl_dataset_London-Strategic-Edge.csv --max-epochs 22 --device cuda  /home/henryp/Code/Python/MyCode/AlgoPlayground/lstm.json 
    """
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description=(
            "Train a single model whose hyperparameters are loaded from a JSON "
            "file and report its holdout error on the given OHLCV CSV."
        )
    )
    parser.add_argument("csv_path", help="Path to an OHLCV CSV file.")
    parser.add_argument(
        "params_path",
        help=(
            "Path to a JSON file with model parameters. Required keys: "
            "model_kind, window_size, batch_size, learning_rate, weight_decay, "
            "dropout, plus (lstm) hidden_size, num_layers OR (cnn) cnn_depth, "
            "cnn_base_channels, kernel_size."
        ),
    )

    parser.add_argument("--holdout-fraction", type=float, default=0.15)
    parser.add_argument("--chunk-size", type=int, default=60)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--device", default="cpu")

    args = parser.parse_args()

    try:
        params = _load_params(args.params_path)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    logger.info("Loaded model params from %s: %s", args.params_path, params)
    logger.info("Loading OHLCV CSV from %s", args.csv_path)
    frame = load_ohlcv_csv(args.csv_path)
    logger.info(
        "Loaded %d rows spanning %s to %s",
        len(frame),
        frame["timestamp"].iloc[0],
        frame["timestamp"].iloc[-1],
    )

    evaluate_on_holdout(
        frame,
        model_kind=params["model_kind"],
        window_size=params["window_size"],
        batch_size=params["batch_size"],
        learning_rate=params["learning_rate"],
        weight_decay=params["weight_decay"],
        dropout=params["dropout"],
        hidden_size=params.get("hidden_size"),
        num_layers=params.get("num_layers"),
        cnn_depth=params.get("cnn_depth"),
        cnn_base_channels=params.get("cnn_base_channels"),
        kernel_size=params.get("kernel_size"),
        holdout_fraction=args.holdout_fraction,
        chunk_size=args.chunk_size,
        max_epochs=args.max_epochs,
        early_stopping_patience=args.early_stopping_patience,
        device=args.device,
    )