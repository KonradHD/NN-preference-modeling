import numpy as np
import os
from data_augmentation.generator.matrices_generator import MatricesGenerator

class NoisyMatricesGenerator(MatricesGenerator):
    def __init__(self, num_matrices: int, num_criteria: int, coherence_rate: float, dir: str | None = None):
        if not (0 <= coherence_rate < 1):
            raise ValueError(f"Coherence rate must be in range of [0, 1), there is: {coherence_rate}")
        
        super().__init__(num_matrices, num_criteria, dir)
        self.coherence_rate = max(0.0, min(1.0, coherence_rate)) 
        self._final_dir = os.path.join(self._dir, "noisy")
        os.makedirs(self._final_dir, exist_ok=True)
        
        self.matrices = None
        self.target_weights = None


    def generate(self) -> tuple[np.ndarray, np.ndarray]:
        raw_weights = np.random.uniform(0.1, 1.0, size=(self.num_matrices, self.num_criteria))
        self.target_weights = raw_weights / raw_weights.sum(axis=1, keepdims=True)
        
        self.matrices = np.zeros((self.num_matrices, self.num_criteria, self.num_criteria))
        noise_std = 1.0 - self.coherence_rate 
        
        for n in range(self.num_matrices):
            for i in range(self.num_criteria):
                for j in range(self.num_criteria):
                    if i == j:
                        self.matrices[n, i, j] = 1.0
                    elif i < j:
                        ideal_val = self.target_weights[n, i] / self.target_weights[n, j]
                        
                        noise_factor = np.random.lognormal(mean=0.0, sigma=noise_std)
                        noised_val = ideal_val * noise_factor
                        
                        self.matrices[n, i, j] = noised_val
                    else:
                        self.matrices[n, i, j] = 1.0 / self.matrices[n, j, i]
                        
        return self.matrices, self.target_weights


    def save(self, prefix: str = "") -> None:
        if self.matrices is None or self.target_weights is None:
            raise ValueError("Najpierw wywołaj metodę generate()!")
            
        coherence_rate_str = f"{self.coherence_rate:.2f}".replace(".", "")
        matrices_path = os.path.join(self._final_dir, f"{prefix}matrices_c{coherence_rate_str}.npy")
        weights_path = os.path.join(self._final_dir, f"{prefix}weights_c{coherence_rate_str}.npy")
        
        np.save(matrices_path, self.matrices)
        np.save(weights_path, self.target_weights)
        print(f"Zapisano zaszumione macierze w: {self._final_dir}")


    def generate_from_coherent(self, coherent_matrices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if coherent_matrices.ndim == 2:
            coherent_matrices = np.expand_dims(coherent_matrices, axis=0)
            
        num_matrices = coherent_matrices.shape[0]
        num_criteria = coherent_matrices.shape[1]

        self._num_matrices = num_matrices
        self._num_criteria = num_criteria
        first_column = coherent_matrices[:, :, 0]
        self.target_weights = first_column / np.sum(first_column, axis=1, keepdims=True)
        
        self.matrices = np.zeros_like(coherent_matrices)
        noise_std = 1.0 - self.coherence_rate
        
        for n in range(num_matrices):
            for i in range(num_criteria):
                for j in range(num_criteria):
                    if i == j:
                        self.matrices[n, i, j] = 1.0 
                    elif i < j:
                        ideal_val = coherent_matrices[n, i, j]
                        noise_factor = np.random.lognormal(mean=0.0, sigma=noise_std)
                        
                        self.matrices[n, i, j] = ideal_val * noise_factor
                    else:
                        self.matrices[n, i, j] = 1.0 / self.matrices[n, j, i]
                        
        return self.matrices, self.target_weights
    