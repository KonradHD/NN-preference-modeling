import os
import numpy as np


class DataLoader():
    def __init__(self, base_dir: str | None = None):
        self.base_dir = base_dir
        if self.base_dir is None: 
            self.base_dir = os.path.join("data", "synthetic_data")
        
        self._coherent_dir = os.path.join(self.base_dir, "coherent")
        self._noisy_dir = os.path.join(self.base_dir, "noisy")
        

    def load_coherent_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        matrices_path = os.path.join(self._coherent_dir, "matrices.npy")
        weights_path = os.path.join(self._coherent_dir, "weights.npy")

        if not os.path.exists(matrices_path) or not os.path.exists(weights_path):
            raise FileNotFoundError(f"Nie znaleziono plików w {self._coherent_dir}. Upewnij się, że wywołałeś CoherentMatricesGenerator.")

        matrices = np.load(matrices_path)
        weights = np.load(weights_path)
        
        print(f"Pomyślnie wczytano {len(matrices)} spójnych macierzy.")
        return matrices, weights

    
    def load_noisy_matrices(self, coherence_rate: float) -> tuple[np.ndarray, np.ndarray]:
        if not (0.0 <= coherence_rate <= 1.0):
            raise ValueError(f"Parametr coherence_rate musi być w przedziale [0, 1]. Otrzymano: {coherence_rate}")

        c_str = f"{coherence_rate:.2f}".replace(".", "")
        matrices_path = os.path.join(self._noisy_dir, f"matrices_c{c_str}.npy")
        weights_path = os.path.join(self._noisy_dir, f"weights_c{c_str}.npy")
        
        if not os.path.exists(matrices_path) or not os.path.exists(weights_path):
            raise FileNotFoundError(f"Nie znaleziono plików dla coherence_rate={coherence_rate} w {self._noisy_dir}.")

        matrices = np.load(matrices_path)
        weights = np.load(weights_path)
        
        print(f"Pomyślnie wczytano {len(matrices)} zaszumionych macierzy (poziom: c{c_str}).")
        return matrices, weights


