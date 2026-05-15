import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
import  matplotlib.colors as mcolors

class MatricesVisualizator():
    def __init__(self, figsize: tuple[int, int] = (8, 6)):
        self.figsize = figsize
        sns.set_theme(style="whitegrid")

    def plot_matrix(self, matrix: np.ndarray, title: str = "AHP Matrix") -> None:
        plt.figure(figsize=self.figsize)
    
        sns.heatmap(
            matrix, 
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=1.0, # kluczowe dla AHP
            linewidths=0.5, 
            cbar_kws={'label': 'Mark value (a_ij)'}
        )
        
        plt.title(title, fontsize=14, pad=15)
        plt.xlabel("Criterium j", fontsize=12)
        plt.ylabel("Criterium i", fontsize=12)
        plt.show()


    def compare_matrices(self, clean_matrix: np.ndarray, noisy_matrix: np.ndarray, coherence_rate: float = None) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        sns.heatmap(
            clean_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=1.0, 
            ax=axes[0], cbar=False, linewidths=0.5
        )
        axes[0].set_title("Coherent Matrix", fontsize=13)
        axes[0].set_xlabel("Criterium j")
        axes[0].set_ylabel("Criterium i")
        
        title_noisy = "Noisy Matrix"
        if coherence_rate is not None:
            title_noisy += f" (coherence={coherence_rate})"
            
        sns.heatmap(
            noisy_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=1.0, 
            ax=axes[1], linewidths=0.5, cbar_kws={'label': 'a_ij'}
        )
        axes[1].set_title(title_noisy, fontsize=13)
        axes[1].set_xlabel("Criterium j")
        
        plt.tight_layout()
        plt.show()



    def noisy_matrices_comparison(self, noisy_matrices: list[np.ndarray], consistency_ratio: list[float], title: str = "Noisy matrices comparison"):
        num_matrices = len(noisy_matrices)
        
        if num_matrices == 0:
            print("No matrices to display")
            return
            
        if num_matrices != len(consistency_ratio):
            raise ValueError(f"Data incompatibility! Given {num_matrices} matrices and {len(consistency_ratio)} CR values.")


        custom_cmap = mcolors.LinearSegmentedColormap.from_list("gray_to_red", ["lightgray", "red"])

        all_error_matrices = []
        global_max_error = 0.0
        
        for matrix in noisy_matrices:
            n = matrix.shape[0]
            log_matrix = np.log(matrix)
            error_matrix = np.zeros_like(matrix)
            
            for row in range(n):
                for col in range(n):
                    if row != col:
                        errors = []
                        for k in range(n):
                            expected_val = log_matrix[row, k] + log_matrix[k, col]
                            actual_val = log_matrix[row, col]
                            errors.append(abs(actual_val - expected_val))
                        error_matrix[row, col] = np.mean(errors)
                        
            all_error_matrices.append(error_matrix)
            global_max_error = max(global_max_error, np.max(error_matrix))

        global_max_error = max(global_max_error, 0.01)
        cols = min(3, num_matrices)
        rows = math.ceil(num_matrices / cols)

        fig_width = max(self.figsize[0], 5 * cols)
        fig_height = max(self.figsize[1], 4.5 * rows)
        
        fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
        
        if num_matrices == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        fig.suptitle(title, fontsize=16, fontweight='bold')

        for i in range(num_matrices):
            ax = axes[i]
            matrix = noisy_matrices[i]
            error_matrix = all_error_matrices[i]
            
            sns.heatmap(
                error_matrix, 
                annot=matrix, 
                fmt=".2f", 
                cmap=custom_cmap, 
                vmin=0.0,
                vmax=global_max_error,
                ax=ax, 
                linewidths=0.5,
                cbar_kws={'label': 'Log-consistency error', 'shrink': 0.8}
            )
            
            ax.set_title(f"cr={consistency_ratio[i]:.4f}", fontsize=13, pad=10)
            ax.set_xlabel("Criterium j", fontsize=10)
            if i % cols == 0: 
                ax.set_ylabel("Criterium i", fontsize=10)

        for j in range(num_matrices, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout(rect=[0, 0.0, 1, 0.95])
        plt.show()