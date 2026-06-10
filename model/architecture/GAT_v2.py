import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

class GATv2(nn.Module):
    def __init__(self, n_criteria, hidden_dim=64, num_heads=4, num_layers=3, dropout=0.1):
        super(GATv2, self).__init__()
        self.n = n_criteria
        self.initial_node_features = nn.Parameter(torch.randn(1, n_criteria, hidden_dim))
        self.conv_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for _ in range(num_layers):
            self.conv_layers.append(
                GATv2Conv(
                    in_channels=hidden_dim, 
                    out_channels=hidden_dim // num_heads, 
                    heads=num_heads, 
                    concat=True,
                    edge_dim=1,
                    dropout=dropout
                )
            )
            self.norms.append(nn.LayerNorm(hidden_dim))
            
        self.dropout_layer = nn.Dropout(dropout)
        
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )


    def forward(self, dense_adj_matrices):
        batch_size = dense_adj_matrices.size(0)
        device = dense_adj_matrices.device
        
        x = self.initial_node_features.expand(batch_size, -1, -1).reshape(batch_size * self.n, -1)
        
        edge_indices = []
        edge_attrs = []
        
        for b in range(batch_size):
            matrix = dense_adj_matrices[b]
            i, j = torch.where(matrix > 0)

            edge_index = torch.stack([i, j]) + (b * self.n)            
            edge_attr = torch.log(matrix[i, j] + 1e-8).unsqueeze(-1)
            edge_indices.append(edge_index)
            edge_attrs.append(edge_attr)
            
        batch_edge_index = torch.cat(edge_indices, dim=1)
        batch_edge_attr = torch.cat(edge_attrs, dim=0)

        for conv, norm in zip(self.conv_layers, self.norms):
            residual = x
            
            x = conv(x, batch_edge_index, edge_attr=batch_edge_attr)
            x = norm(x)
            x = F.gelu(x)
            x = self.dropout_layer(x)
            x = x + residual

        logits = self.head(x)
        return logits.view(batch_size, self.n)