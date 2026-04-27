from torch.utils.data import Dataset
import numpy as np
import torch


class PairwiseAHPDataset(Dataset):
    def __init__(self, matrices: np.ndarray):
        self.matrices = matrices
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
        m, i, j = self.pairs[idx]
        matrix = self.matrices[m]

        row_i = torch.tensor(matrix[i, :], dtype=torch.float32)
        row_j = torch.tensor(matrix[j, :], dtype=torch.float32)

        a_ij = matrix[i, j]
        target_val = np.log(a_ij)
        target = torch.tensor([target_val], dtype=torch.float32)

        return row_i, row_j, target