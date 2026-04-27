import os 
import numpy as np
from data_augmentation.generator.matrices_generator import MatricesGenerator

class CoherentMatricesGenerator(MatricesGenerator):
    def __init__(self, num_matrices: int, num_criteria: int, dir: str | None = None):
        super().__init__(num_matrices, num_criteria, dir)
        self._final_dir = os.path.join(self._dir, "coherent")
        os.makedirs(self._final_dir, exist_ok=True)
        
        self.matrices = None
        self.target_weights = None


    def generate(self) -> tuple[np.ndarray, np.ndarray]:
        # Losowanie wag (unikamy bliskich zera, rozkład jednostajny)
        raw_weights = np.random.uniform(0.1, 1.0, size=(self.num_matrices, self.num_criteria))
        
        # Normalizacja wierszami, aby sumy wag wynosiły 1
        self.target_weights = raw_weights / raw_weights.sum(axis=1, keepdims=True)
        
        # Inicjalizacja pustego tensora macierzy (N_macierzy x N_kryteriów x N_kryteriów)
        self.matrices = np.zeros((self.num_matrices, self.num_criteria, self.num_criteria))
        
        # Wypełnianie macierzy: a_ij = w_i / w_j
        for i in range(self.num_criteria):
            for j in range(self.num_criteria):
                self.matrices[:, i, j] = self.target_weights[:, i] / self.target_weights[:, j]
                
        return self.matrices, self.target_weights


    def save(self, prefix: str = ""):
        if self.matrices is None or self.target_weights is None:
            raise ValueError("Najpierw wywołaj metodę generate()!")
            
        matrices_path = os.path.join(self._final_dir, f"{prefix}matrices.npy")
        weights_path = os.path.join(self._final_dir, f"{prefix}weights.npy")
        
        np.save(matrices_path, self.matrices)
        np.save(weights_path, self.target_weights)
        print(f"Zapisano spójne macierze w: {self._final_dir}")