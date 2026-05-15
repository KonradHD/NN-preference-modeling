import torch
from torch.utils.data import Dataset
import numpy as np

class GraphAHPDataset(Dataset):
    def __init__(self, matrices: np.ndarray, weights: np.ndarray, augment: bool = False):
        self.matrices = torch.tensor(matrices, dtype=torch.float32)
        self.targets = torch.tensor(weights, dtype=torch.float32)
        self.augment = augment

        n_criteria = self.matrices.shape[1]
        self.triu_i, self.triu_j = torch.triu_indices(n_criteria, n_criteria, offset=1)


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

        return matrix, weights