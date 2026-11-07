from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseMonteCarloNet(nn.Module):
    def __init__(
        self,
        n_criteria: int,
        neurons: int | None = None,
        layers: int = 3,
        expansion_factor: int = 4,
        dropout_rate: float = 0.2,
        negative_slope: float = 0.01,
        use_batch_norm: bool = True,
    ):
        super(DenseMonteCarloNet, self).__init__()
        self.n = n_criteria
        self.input_dim = n_criteria * (n_criteria - 1) // 2
        self.negative_slope = negative_slope
        self.num_layers = layers

        if layers < 1:
            raise ValueError("Number of layers must be at least 1.")

        if neurons is None:
            neurons = self.input_dim * expansion_factor

        if neurons < 1:
            raise ValueError("Number of neurons must be at least 1.")

        self.neurons = neurons

        hidden_dims: list[int] = []
        current_dim = neurons
        for i in range(layers):
            if i == 0:
                hidden_dims.append(current_dim)
            else:
                current_dim = max(1, current_dim // 2)
                hidden_dims.append(current_dim)

        self.hidden_dims = hidden_dims

        layer_modules: list[nn.Module] = []
        prev_dim = self.input_dim

        for h_dim in hidden_dims:
            layer_modules.append(nn.Linear(prev_dim, h_dim))
            if use_batch_norm:
                layer_modules.append(nn.BatchNorm1d(h_dim))
            layer_modules.append(nn.LeakyReLU(negative_slope=negative_slope))
            layer_modules.append(nn.Dropout(dropout_rate))
            prev_dim = h_dim

        self.feature_extractor = nn.Sequential(*layer_modules)
        self.head = nn.Linear(prev_dim, n_criteria)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight,
                    a=self.negative_slope,
                    mode="fan_in",
                    nonlinearity="leaky_relu",
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        logits = self.head(features)
        return logits

    def predict_weights(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)


DenseMonteCarloAHPNet = DenseMonteCarloNet
