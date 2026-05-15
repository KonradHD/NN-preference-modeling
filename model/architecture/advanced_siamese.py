import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, dropout_rate=0.1):
        super(ResidualBlock, self).__init__()
        expanded_dim = hidden_dim * 2
        
        self.fc1 = nn.Linear(hidden_dim, expanded_dim)
        self.ln1 = nn.LayerNorm(expanded_dim)
        self.fc2 = nn.Linear(expanded_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)


    def forward(self, x):
        residual = x
        
        out = self.fc1(x)
        out = self.ln1(out)
        out = F.gelu(out)
        out = self.dropout(out)        
        out = self.fc2(out)
        out = self.ln2(out)
        
        return F.gelu(out + residual)


class AdvancedAHPEncoder(nn.Module):
    def __init__(self, n_criteria, hidden_dim=128, num_blocks=2, dropout_rate=0.1):
        super(AdvancedAHPEncoder, self).__init__()
        self.n = n_criteria
        input_dim = n_criteria * n_criteria
        
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        blocks = []
        for _ in range(num_blocks):
            blocks.append(ResidualBlock(hidden_dim, dropout_rate))
        self.feature_extractor = nn.Sequential(*blocks)
        
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, n_criteria)
        )


    def forward(self, x):
        x = x.view(x.size(0), -1)
        
        x = self.input_projection(x)
        x = self.feature_extractor(x)
        logits = self.head(x)
        
        return logits


class AdvancedSiameseModel(nn.Module):
    def __init__(self, n_criteria):
        super(AdvancedSiameseModel, self).__init__()
        self.encoder = AdvancedAHPEncoder(n_criteria, hidden_dim=128, num_blocks=2)

    def forward(self, m1, m2):
        logits1 = self.encoder(m1)
        logits2 = self.encoder(m2)
        return logits1, logits2