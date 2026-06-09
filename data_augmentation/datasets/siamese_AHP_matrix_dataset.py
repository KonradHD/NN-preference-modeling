import torch
from torch.utils.data import Dataset

class SiameseAHPMatrixDataset(Dataset):
    def __init__(self, base_matrices, comparison_matrices, weights):
        if base_matrices.shape != comparison_matrices.shape:
           raise Exception("Matrices shape does not match")
        
        if base_matrices.shape[0] != weights.shape[0]:
           raise ValueError("Target weights shape does not match with number of criterias in matrices.")
        
        self.base_matrices = torch.tensor(base_matrices, dtype=torch.float32)
        self.comparison_matrices = torch.tensor(comparison_matrices, dtype=torch.float32)
        self.targets = torch.tensor(weights, dtype=torch.float32)


    def __len__(self):
        return len(self.base_matrices)

    # TODO: generowanie macierzy porównawczych w locie
    # TODO: wymagana jest permutacja wierszy i kolumn 
    def __getitem__(self, idx):
        m1 = self.base_matrices[idx]
        m2 = self.comparison_matrices[idx]
        weights = self.targets[idx]
        m1 = torch.log(m1 + 1e-8)
        m2 = torch.log(m2 + 1e-8)
        
        return (m1.unsqueeze(0), m2.unsqueeze(0), weights)