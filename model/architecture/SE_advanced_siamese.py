import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        attention_weights = self.fc(x)
        return x * attention_weights


class AdvancedResidualBlock(nn.Module):
    def __init__(self, hidden_dim, dropout_rate=0.1):
        super(AdvancedResidualBlock, self).__init__()
        expanded_dim = hidden_dim * 2
        
        self.fc1 = nn.Linear(hidden_dim, expanded_dim)
        self.ln1 = nn.LayerNorm(expanded_dim)
        
        self.fc2 = nn.Linear(expanded_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        
        self.se_block = SEBlock(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        residual = x
        
        out = self.fc1(x)
        out = self.ln1(out)
        out = F.gelu(out)
        out = self.dropout(out)        
        
        out = self.fc2(out)
        out = self.ln2(out)
        
        out = self.se_block(out)
        
        return F.gelu(out + residual)


class AdvancedAHPEncoder(nn.Module):
    def __init__(self, n_criteria, hidden_dim=128, num_blocks=3, dropout_rate=0.1):
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
            blocks.append(AdvancedResidualBlock(hidden_dim, dropout_rate))
        self.feature_extractor = nn.Sequential(*blocks)
        
        self.priority_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, n_criteria)
        )
        
        self.cr_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        
        x = self.input_projection(x)
        features = self.feature_extractor(x)
        
        logits = self.priority_head(features)
        predicted_cr = self.cr_head(features)
        
        return logits, predicted_cr


class SEAdvancedSiameseModel(nn.Module):
    def __init__(self, n_criteria, hidden_dim=128, num_blocks=3):
        super(SEAdvancedSiameseModel, self).__init__()
        self.encoder = AdvancedAHPEncoder(n_criteria, hidden_dim=hidden_dim, num_blocks=num_blocks)

    def forward(self, m1, m2):
        logits1, cr1 = self.encoder(m1)
        logits2, cr2 = self.encoder(m2)
        
        return logits1, logits2