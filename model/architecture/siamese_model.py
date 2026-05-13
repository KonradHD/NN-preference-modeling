import torch.nn as nn
import torch.nn.functional as F

class AHPEncoder(nn.Module):
    def __init__(self, n_criteria):
        super(AHPEncoder, self).__init__()
        self.n = n_criteria
        input_dim = n_criteria * n_criteria
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2),
            
            nn.Linear(32, n_criteria)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        logits = self.net(x)
        return logits


class SiameseAHPModel(nn.Module):
    def __init__(self, n_criteria):
        super(SiameseAHPModel, self).__init__()
        self.encoder = AHPEncoder(n_criteria)

    def forward(self, m1, m2):
        logits1 = self.encoder(m1)
        logits2 = self.encoder(m2)
        
        return logits1, logits2
