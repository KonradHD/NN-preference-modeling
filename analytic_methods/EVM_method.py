import numpy as np

class EVM():
    def __init__(self):
        self.ri_dict = {
            1: 0.00, 
            2: 0.00, 
            3: 0.58, 
            4: 0.90, 
            5: 1.12,
            6: 1.24, 
            7: 1.32, 
            8: 1.41, 
            9: 1.45, 
            10: 1.49
        }

    def compute_single(self, matrix: np.ndarray) -> tuple[np.ndarray, float]:
        n = matrix.shape[0]
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        
        real_eigenvalues = np.real(eigenvalues)
        max_index = np.argmax(real_eigenvalues)
        lambda_max = real_eigenvalues[max_index]

        principal_eigenvector = np.real(eigenvectors[:, max_index])
        
        if np.sum(principal_eigenvector) < 0:
            principal_eigenvector = -principal_eigenvector
        
        weights = principal_eigenvector / np.sum(principal_eigenvector)
        if n > 2:
            ci = (lambda_max - n) / (n - 1)
            ri = self.ri_dict.get(n, 1.49)
            cr = ci / ri
        else:
            cr = 0.0 
            
        return weights, cr

    def compute_batch(self, matrices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        num_matrices = matrices.shape[0]
        num_criteria = matrices.shape[1]
        
        all_weights = np.zeros((num_matrices, num_criteria))
        all_crs = np.zeros(num_matrices)
        
        for i in range(num_matrices):
            weights, cr = self.compute_single(matrices[i])
            all_weights[i] = weights
            all_crs[i] = cr
            
        return all_weights, all_crs