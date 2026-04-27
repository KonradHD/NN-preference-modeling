import numpy as np
import os

from data_augmentation.generator.matrices_generator import MatricesGenerator


class TargetCRMatricesGenerator(MatricesGenerator):
    def __init__(self, num_matrices: int, num_criteria: int, target_cr: float, tolerance: float = 0.005, base_dir: str | None = None):
        if not (0 <= target_cr < 1):
            raise ValueError(f"Coherence rate must be in range of [0, 1), there is: {target_cr}")
        
        super().__init__(num_matrices, num_criteria, base_dir)
        self._final_dir = os.path.join(self._dir, "noisy")
        os.makedirs(self._final_dir, exist_ok=True)

        self.target_cr = max(0.0, min(1.0, target_cr)) 
        self.tolerance = tolerance
        
        self.matrices = None
        self.target_weights = None
        self._ri_dict = {3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}


    def _calculate_cr(self, matrix: np.ndarray) -> float:
        """Szybka funkcja pomocnicza do liczenia CR wewnątrz pętli generującej."""
        n = matrix.shape[0]
        eigenvalues = np.linalg.eigvals(matrix)
        lambda_max = np.max(np.real(eigenvalues))
        
        if n > 2:
            ci = (lambda_max - n) / (n - 1)
            ri = self._ri_dict.get(n, 1.49)
            return max(0.0, ci / ri)
        return 0.0


    def generate(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Generuje zbiór macierzy adaptacyjnie dobierając szum, aby trafić w target_cr.
        """
        self.matrices = np.zeros((self.num_matrices, self.num_criteria, self.num_criteria))
        self.target_weights = np.zeros((self.num_matrices, self.num_criteria))
        
        print(f"Rozpoczynam generowanie {self.num_matrices} macierzy celując w CR = {self.target_cr} (±{self.tolerance})...")
        
        for n in range(self.num_matrices):
            raw_weights = np.random.uniform(0.1, 1.0, size=self.num_criteria)
            ideal_weights = raw_weights / np.sum(raw_weights)
            self.target_weights[n] = ideal_weights
            
            ideal_matrix = np.zeros((self.num_criteria, self.num_criteria))
            for i in range(self.num_criteria):
                for j in range(self.num_criteria):
                    ideal_matrix[i, j] = ideal_weights[i] / ideal_weights[j]

            if self.target_cr == 0.0:
                self.matrices[n] = ideal_matrix
                continue

            current_sigma = 0.3
            learning_rate = 0.05
            
            while True:
                noisy_matrix = np.zeros_like(ideal_matrix)
                
                for i in range(self.num_criteria):
                    for j in range(self.num_criteria):
                        if i == j:
                            noisy_matrix[i, j] = 1.0
                        elif i < j:
                            noise_factor = np.random.lognormal(mean=0.0, sigma=current_sigma)
                            noisy_matrix[i, j] = ideal_matrix[i, j] * noise_factor
                        else:
                            noisy_matrix[i, j] = 1.0 / noisy_matrix[j, i]
                
                current_cr = self._calculate_cr(noisy_matrix)

                if abs(current_cr - self.target_cr) <= self.tolerance:
                    self.matrices[n] = noisy_matrix
                    break
                elif current_cr < self.target_cr:
                    current_sigma += learning_rate
                else:
                    current_sigma = max(0.01, current_sigma - (learning_rate * 0.5))

        print("Generowanie zakończone pomyślnie!")
        return self.matrices, self.target_weights
    

    def save(self, prefix: str = "") -> None:
        if self.matrices is None or self.target_weights is None:
            raise ValueError("Najpierw wywołaj metodę generate()!")
            
        target_cr_str = f"{self.target_cr:.2f}".replace(".", "")
        matrices_path = os.path.join(self._final_dir, f"{prefix}matrices_c{target_cr_str}.npy")
        weights_path = os.path.join(self._final_dir, f"{prefix}weights_c{target_cr_str}.npy")
        
        np.save(matrices_path, self.matrices)
        np.save(weights_path, self.target_weights)
        print(f"Zapisano zaszumione macierze w: {self._final_dir}")