from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseMonteCarloNet(nn.Module):
    def __init__(
        self,
        n_criteria: int,
        hidden_dims: list[int] | None = None,
        expansion_factor: int = 4,
        dropout_rate: float = 0.2,
        negative_slope: float = 0.01,
        use_batch_norm: bool = True,
    ):
        super(DenseMonteCarloNet, self).__init__()
        self.n = n_criteria
        self.input_dim = n_criteria * (n_criteria - 1) // 2
        self.negative_slope = negative_slope

        if hidden_dims is None:
            first_dim = self.input_dim * expansion_factor
            second_dim = self.input_dim * 2
            third_dim = max(self.input_dim, n_criteria)
            hidden_dims = [first_dim, second_dim, third_dim]

        layers: list[nn.Module] = []
        prev_dim = self.input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.LeakyReLU(negative_slope=negative_slope))
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = h_dim

        self.feature_extractor = nn.Sequential(*layers)
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
