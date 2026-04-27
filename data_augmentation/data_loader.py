import os
import numpy as np


class DataLoader():
    def __init__(self, base_dir: str | None = None):
        self.base_dir = base_dir
        if self.base_dir is None: 
            self.base_dir = os.path.join("data", "synthetic_data")
        
        self._coherent_dir = os.path.join(self.base_dir, "coherent")
        self._noisy_dir = os.path.join(self.base_dir, "noisy")
        self._noisy_cr_dir = os.path.join(self.base_dir, "noisy_cr")
        

    def load_coherent_matrices(self, is_uniform: bool) -> tuple[np.ndarray, np.ndarray]:

        coherent_saving_path = ""
        if is_uniform:
            coherent_saving_path = os.path.join(self._coherent_dir, "uniform")
        else: 
            coherent_saving_path = os.path.join(self._coherent_dir, "dirichlet")

        matrices_path = os.path.join(coherent_saving_path, "matrices.npy")
        weights_path = os.path.join(coherent_saving_path, "weights.npy")

        if not os.path.exists(matrices_path) or not os.path.exists(weights_path):
            raise FileNotFoundError(f"File {coherent_saving_path} does not exist. Generate coherent matrices using CoherentMatricesGenerator.")

        matrices = np.load(matrices_path)
        weights = np.load(weights_path)
        
        print(f"Pomyślnie wczytano {len(matrices)} spójnych macierzy.")
        return matrices, weights

    
    def load_noisy_matrices(self, coherence_rate: float, is_uniform: bool) -> tuple[np.ndarray, np.ndarray]:
        if not (0.0 <= coherence_rate <= 1.0):
            raise ValueError(f"Coherence rate must be in range of [0, 1), given: {coherence_rate}")
        
        noisy_saving_path = ""
        if is_uniform:
            noisy_saving_path = os.path.join(self._noisy_dir, "uniform")
        else: 
            noisy_saving_path = os.path.join(self._noisy_dir, "dirichlet")

        c_str = f"{coherence_rate:.2f}".replace(".", "")
        matrices_path = os.path.join(noisy_saving_path, f"matrices_coh{c_str}.npy")
        weights_path = os.path.join(noisy_saving_path, f"weights_coh{c_str}.npy")
        
        if not os.path.exists(matrices_path) or not os.path.exists(weights_path):
            raise FileNotFoundError(f"Files not found with coherence_rate={coherence_rate} in {noisy_saving_path}.")

        matrices = np.load(matrices_path)
        weights = np.load(weights_path)
        
        print(f"Successfuly uploaded {len(matrices)} noised matrices (level: c{c_str}).")
        return matrices, weights
    

    def load_noisy_cr_matrices(self, consistency_ratio: float, is_uniform: bool) -> tuple[np.ndarray, np.ndarray]:
        if not (0.0 <= consistency_ratio <= 1.0):
            raise ValueError(f"consistency ratio must be in range of [0, 1), given: {consistency_ratio}")
        
        noisy_saving_path = ""
        if is_uniform:
            noisy_saving_path = os.path.join(self._noisy_cr_dir, "uniform")
        else: 
            noisy_saving_path = os.path.join(self._noisy_cr_dir, "dirichlet")

        c_str = f"{consistency_ratio:.2f}".replace(".", "")
        matrices_path = os.path.join(noisy_saving_path, f"matrices_cr{c_str}.npy")
        weights_path = os.path.join(noisy_saving_path, f"weights_cr{c_str}.npy")
        
        if not os.path.exists(matrices_path) or not os.path.exists(weights_path):
            raise FileNotFoundError(f"Files not found with consistency_ratio={consistency_ratio} in {noisy_saving_path}.")

        matrices = np.load(matrices_path)
        weights = np.load(weights_path)
        
        print(f"Successfuly uploaded {len(matrices)} noised matrices (level: c{c_str}).")
        return matrices, weights


