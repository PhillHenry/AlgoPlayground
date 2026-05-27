"""Sequence models for next-close regression on OHLCV windows."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class LSTMConfig:
    n_features: int
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.1


@dataclass(frozen=True)
class CNNConfig:
    n_features: int
    channels: tuple[int, ...] = (32, 64)
    kernel_size: int = 3
    dropout: float = 0.1


class LSTMRegressor(nn.Module):
    """Stacked LSTM whose final hidden state feeds a linear head."""

    def __init__(self, config: LSTMConfig) -> None:
        super().__init__()
        self.config = config
        self.lstm = nn.LSTM(
            input_size=config.n_features,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features)
        output, _ = self.lstm(x)
        last_step = output[:, -1, :]
        return self.head(last_step).squeeze(-1)


class CNNRegressor(nn.Module):
    """1D CNN over time with global average pooling."""

    def __init__(self, config: CNNConfig) -> None:
        super().__init__()
        if not config.channels:
            raise ValueError("channels must not be empty")
        self.config = config

        layers: list[nn.Module] = []
        in_channels = config.n_features
        padding = config.kernel_size // 2
        for out_channels in config.channels:
            layers.extend(
                [
                    nn.Conv1d(in_channels, out_channels, config.kernel_size, padding=padding),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Dropout(config.dropout),
                ]
            )
            in_channels = out_channels
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(in_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features) -> conv expects (batch, channels, seq_len)
        x = x.transpose(1, 2)
        x = self.features(x)
        x = self.pool(x).squeeze(-1)
        return self.head(x).squeeze(-1)