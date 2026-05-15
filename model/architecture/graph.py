import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAHPConv(nn.Module):
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1):
        super(GraphAHPConv, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.layer_norm = nn.LayerNorm(out_features)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
        support = self.linear(x)
        out = torch.bmm(adj_matrix, support)
        out = self.layer_norm(out)
        out = F.gelu(out)
        out = self.dropout(out)
        return out


class GraphAHPNet(nn.Module):
    def __init__(self, n_criteria: int, hidden_dim: int = 64, num_layers: int = 3):
        super(GraphAHPNet, self).__init__()
        self.n = n_criteria
        self.initial_node_features = nn.Parameter(torch.randn(1, n_criteria, hidden_dim))
        self.gcn_layers = nn.ModuleList()

        for _ in range(num_layers):
            self.gcn_layers.append(GraphAHPConv(hidden_dim, hidden_dim))
            
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )


    def forward(self, adj_matrix: torch.Tensor) -> torch.Tensor:
        batch_size = adj_matrix.size(0)
        x = self.initial_node_features.expand(batch_size, -1, -1)
        
        for gcn in self.gcn_layers:
            x = gcn(x, adj_matrix)
            
        logits = self.head(x)
        return logits.squeeze(-1)