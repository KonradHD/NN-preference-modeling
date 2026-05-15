import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout_rate: float = 0.2):
        super(DenseResidualBlock, self).__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout_rate)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.fc1(x)
        out = self.ln1(out)
        out = F.gelu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.ln2(out)
        
        return F.gelu(out + residual)


class ResDeepAHPNet(nn.Module):
    def __init__(self, n_criteria: int, hidden_dim: int = 128, num_blocks: int = 4, dropout_rate: float = 0.2):
        super(ResDeepAHPNet, self).__init__()
        self.n = n_criteria
        self.input_dim = n_criteria * (n_criteria - 1) // 2
        self.input_projection = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        blocks = []
        for _ in range(num_blocks):
            blocks.append(DenseResidualBlock(hidden_dim, dropout_rate))
        
        self.feature_extractor = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, n_criteria)
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        x = self.feature_extractor(x)
        logits = self.head(x)
        
        return logits