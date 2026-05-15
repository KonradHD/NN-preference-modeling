import torch
import torch.nn as nn

class DeepAHPNet(nn.Module):
    def __init__(self, n_criteria: int, hidden_dims: list = [128, 64, 32], dropout_rate: float = 0.2):
        super(DeepAHPNet, self).__init__()
        self.n = n_criteria
        self.input_dim = n_criteria * (n_criteria - 1) // 2
        

        layers = []
        prev_dim = self.input_dim
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = h_dim
            
        self.feature_extractor = nn.Sequential(*layers)
        self.head = nn.Linear(prev_dim, n_criteria)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        logits = self.head(features)
        
        return logits