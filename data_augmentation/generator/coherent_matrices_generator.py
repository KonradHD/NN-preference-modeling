import os 
import numpy as np
from data_augmentation.generator.matrices_generator import MatricesGenerator

class CoherentMatricesGenerator(MatricesGenerator):
    def __init__(self, num_matrices: int, num_criteria: int, dir: str | None = None):
        super().__init__(num_matrices, num_criteria, dir)
        self._coherent_dir = os.path.join(self._dir, "coherent")
        os.makedirs(self._coherent_dir, exist_ok=True)
        
        self.matrices = None
        self.target_weights = None


    def generate_uniform(self) -> tuple[np.ndarray, np.ndarray]:
        raw_weights = np.random.uniform(0.1, 1.0, size=(self.num_matrices, self.num_criteria))
        self.target_weights = raw_weights / raw_weights.sum(axis=1, keepdims=True)

        self.matrices = np.zeros((self.num_matrices, self.num_criteria, self.num_criteria))

        for i in range(self.num_criteria):
            for j in range(self.num_criteria):
                self.matrices[:, i, j] = self.target_weights[:, i] / self.target_weights[:, j]
                
        return self.matrices, self.target_weights


    def generate_dirichlet(self, alpha: list[float]) -> tuple[np.ndarray, np.ndarray]:

        if len(alpha) != self.num_criteria:
            raise ValueError(f"Alpha vectors have to match criteria number: {self.num_criteria}")

        self.target_weights = np.random.dirichlet(np.array(alpha), size=self.num_matrices)
        self.matrices = np.zeros((self.num_matrices, self.num_criteria, self.num_criteria))

        for i in range(self.num_criteria):
            for j in range(self.num_criteria):
                self.matrices[:, i, j] = self.target_weights[:, i] / self.target_weights[:, j]
                
        return self.matrices, self.target_weights


    def save_state(self, is_uniform: bool, prefix: str = ""):
        if self.matrices is None or self.target_weights is None:
            raise ValueError("You have to generate matrices and weights first")
        
        saving_path = ""
        if is_uniform:
            saving_path = os.path.join(self._coherent_dir, "uniform")
        else: 
            saving_path = os.path.join(self._coherent_dir, "dirichlet")

        os.makedirs(saving_path, exist_ok=True)
        matrices_path = os.path.join(saving_path, f"{prefix}matrices.npy")
        weights_path = os.path.join(saving_path, f"{prefix}weights.npy")
        
        np.save(matrices_path, self.matrices)
        np.save(weights_path, self.target_weights)
        print(f"Coherent matrices and weights have been saved in: {saving_path}")