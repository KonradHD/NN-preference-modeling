from torch.utils.data import Dataset
import numpy as np
import torch


class PairwiseAHPDataset(Dataset):
    def __init__(self, matrices: np.ndarray, weights: np.ndarray):
        if matrices.shape[0] != weights.shape[0]:
           raise ValueError("Target weights shape does not match with number of criterias in matrices.")
        
        self.weights = torch.tensor(weights, dtype=torch.float32)
        self.matrices = torch.tensor(matrices, dtype=torch.float32)
        self._num_matrices = matrices.shape[0]
        self._num_criteria = matrices.shape[2]
        self._pairs = []
        
        for m in range(self._num_matrices):
            for i in range(self._num_criteria):
                for j in range(self._num_criteria):
                    if i != j:
                        self._pairs.append((m, i, j))


    def __len__(self) -> int:
        return len(self._pairs)


    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # TODO: 
        m, i, j = self._pairs[idx]
        matrix = self.matrices[m]

        row_i = torch.tensor(matrix[i, :], dtype=torch.float32)
        row_j = torch.tensor(matrix[j, :], dtype=torch.float32)
        target = self.weights[idx]

        return row_i, row_j, target