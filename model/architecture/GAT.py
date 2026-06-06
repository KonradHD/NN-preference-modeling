import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GAT(nn.Module):
    def __init__(self, n_criteria, in_channels=1, hidden_dim=64, num_heads=4, dropout=0.1):
        super().__init__()
        self.n = n_criteria
        self.initial_node_features = torch.nn.Parameter(torch.randn(1, n_criteria, in_channels))
        
        self.conv1 = GATConv(
            in_channels=in_channels, 
            out_channels=hidden_dim, 
            heads=num_heads, 
            dropout=dropout,
            edge_dim=1
        )
        
        self.conv2 = GATConv(
            in_channels=hidden_dim * num_heads, 
            out_channels=1,
            heads=1, 
            concat=False, 
            dropout=dropout,
            edge_dim=1
        )
        self.dropout = dropout

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
        
        x = self.conv1(x, batch_edge_index, edge_attr=batch_edge_attr)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        logits = self.conv2(x, batch_edge_index, edge_attr=batch_edge_attr)
        
        return logits.view(batch_size, self.n)