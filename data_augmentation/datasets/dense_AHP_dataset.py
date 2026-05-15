import torch
from torch.utils.data import Dataset
import numpy as np

class DenseAHPDataset(Dataset):
    def __init__(self, matrices: np.ndarray, weights: np.ndarray, augment: bool = False):
        self.matrices = torch.tensor(matrices, dtype=torch.float32)
        self.targets = torch.tensor(weights, dtype=torch.float32)
        self.augment = augment


    def __len__(self):
        return len(self.matrices)


    def __getitem__(self, idx):
        matrix = self.matrices[idx]
        weights = self.targets[idx]
        
        if self.augment:
            n = matrix.shape[0]
            perm = torch.randperm(n)
            matrix = matrix[perm, :][:, perm]
            weights = weights[perm]

        matrix = torch.log(matrix + 1e-8)
        
        return (matrix.unsqueeze(0), weights)