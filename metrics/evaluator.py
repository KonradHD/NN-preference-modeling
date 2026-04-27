import numpy as np
from scipy.stats import kendalltau

class Evaluator:
    """
    Klasa odpowiedzialna za wyliczanie metryk błędu pomiędzy prawdziwymi wagami (Ground Truth), 
    a wagami wyestymowanymi przez dowolny algorytm (EVM lub Sieć Neuronową).
    """
    
    @staticmethod
    def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Oblicza błąd MAE (Mean Absolute Error) dla pojedynczego wektora wag.
        Mówi nam, o ile punktów algorytm pomylił się w samej wartości ułamkowej.
        """
        return float(np.mean(np.abs(y_true - y_pred)))

    @staticmethod
    def calculate_kendall_tau(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Oblicza korelację rang Kendalla (Kendall's Tau) dla pojedynczego wektora wag.
        Mówi nam, jak dobrze zachowana została kolejność (ranking) kryteriów.
        Zakres: 1.0 (idealna kolejność), 0.0 (losowa), -1.0 (odwrotna).
        """
        tau, p_value = kendalltau(y_true, y_pred)
    
        if np.isnan(tau):
            return 0.0
            
        return float(tau)

    def evaluate_batch(self, y_true_batch: np.ndarray, y_pred_batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Przetwarza cały zbiór (batch) wektorów na raz.
        
        Parametry:
        - y_true_batch: Docelowe wagi z generatora (N_macierzy, N_kryteriów)
        - y_pred_batch: Wagi obliczone przez algorytm (N_macierzy, N_kryteriów)
        
        Zwraca:
        - mae_scores: Wektor błędów MAE (N_macierzy,)
        - tau_scores: Wektor korelacji Kendalla (N_macierzy,)
        """
        if y_true_batch.shape != y_pred_batch.shape:
            raise ValueError(f"Niezgodność wymiarów! Ground Truth: {y_true_batch.shape}, Predykcje: {y_pred_batch.shape}")

        num_samples = y_true_batch.shape[0]
        mae_scores = np.zeros(num_samples)
        tau_scores = np.zeros(num_samples)

        for i in range(num_samples):
            mae_scores[i] = self.calculate_mae(y_true_batch[i], y_pred_batch[i])
            tau_scores[i] = self.calculate_kendall_tau(y_true_batch[i], y_pred_batch[i])

        return mae_scores, tau_scores