import numpy as np

class CRCalculator:
    def __init__(self):
        self._ri_dict = {3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
    
    
    def saatys_index(self, matrix: np.ndarray) -> float:
        n = matrix.shape[0]
        eigenvalues = np.linalg.eigvals(matrix)
        lambda_max = np.max(np.real(eigenvalues))
        
        if n > 2:
            ci = (lambda_max - n) / (n - 1)
            ri = self._ri_dict.get(n, 1.49)
            return max(0.0, ci / ri)
        return 0.0


    def upper_triangle_saatys_index(self, matrix: np.ndarray) -> float:
        n = matrix.shape[0]
        reconstructed_matrix = np.eye(n)
        row_idx, col_idx = np.triu_indices(n, k=1)
        upper_elements = matrix[row_idx, col_idx]
        
        reconstructed_matrix[row_idx, col_idx] = upper_elements
        reconstructed_matrix[col_idx, row_idx] = 1.0 / upper_elements
        
        eigenvalues = np.linalg.eigvals(reconstructed_matrix)
        lambda_max = np.max(np.real(eigenvalues))
        
        if n > 2:
            ci = (lambda_max - n) / (n - 1)
            ri = self._ri_dict.get(n, 1.49)
            return max(0.0, ci / ri)
        return 0.0


    def koczkodaj_index(self, matrix: np.ndarray) -> float:
        n = matrix.shape[0]
        if n < 3:
            return 0.0
        
        max_ki = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    val1 = matrix[i, j] * matrix[j, k] / matrix[i, k]
                    val2 = matrix[i, k] / (matrix[i, j] * matrix[j, k])
                    ki = 1.0 - min(val1, val2)
                    if ki > max_ki:
                        max_ki = ki
        return float(max_ki)


    def upper_triangle_koczkodaj_index(self, matrix: np.ndarray) -> float:
        n = matrix.shape[0]
        if n < 3:
            return 0.0
            
        reconstructed_matrix = np.eye(n)
        row_idx, col_idx = np.triu_indices(n, k=1)
        upper_elements = matrix[row_idx, col_idx]
        
        reconstructed_matrix[row_idx, col_idx] = upper_elements
        reconstructed_matrix[col_idx, row_idx] = 1.0 / upper_elements
        
        return self.koczkodaj_index(reconstructed_matrix)


    def geometric_consistency_index(self, matrix: np.ndarray) -> float:
        n = matrix.shape[0]
        if n < 3:
            return 0.0
        
        log_matrix = np.log(matrix)
        log_w = np.mean(log_matrix, axis=1)
        
        log_w_diff = log_w[:, np.newaxis] - log_w[np.newaxis, :]
        error_matrix = log_matrix - log_w_diff
        
        row_idx, col_idx = np.triu_indices(n, k=1)
        gci_sum = np.sum(error_matrix[row_idx, col_idx] ** 2)
        
        return float((2.0 / ((n - 1) * (n - 2))) * gci_sum)


    def upper_triangle_geometric_consistency_index(self, matrix: np.ndarray) -> float:
        n = matrix.shape[0]
        if n < 3:
            return 0.0
            
        reconstructed_matrix = np.eye(n)
        row_idx, col_idx = np.triu_indices(n, k=1)
        upper_elements = matrix[row_idx, col_idx]
        
        reconstructed_matrix[row_idx, col_idx] = upper_elements
        reconstructed_matrix[col_idx, row_idx] = 1.0 / upper_elements
        
        return self.geometric_consistency_index(reconstructed_matrix)