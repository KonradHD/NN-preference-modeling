import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
import  matplotlib.colors as mcolors
import pandas as pd

class MatricesVisualizator():
    def __init__(self, figsize: tuple[int, int] = (8, 6)):
        self.figsize = figsize
        sns.set_theme(style="whitegrid")

    def plot_matrix(self, matrix: np.ndarray, title: str = "Macierz Porównań AHP") -> None:
        plt.figure(figsize=self.figsize)
    
        sns.heatmap(
            matrix, 
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=1.0,          # kluczowe dla AHP!
            linewidths=0.5, 
            cbar_kws={'label': 'Wartość oceny (a_ij)'}
        )
        
        plt.title(title, fontsize=14, pad=15)
        plt.xlabel("Kryterium j", fontsize=12)
        plt.ylabel("Kryterium i", fontsize=12)
        plt.show()

    def plot_weights(self, weights: np.ndarray, title: str = "Docelowe Wagi Priorytetów") -> None:
        plt.figure(figsize=(self.figsize[0], self.figsize[1] // 2))
        
        criteria_labels = [f"Kryt {i}" for i in range(len(weights))]
        ax = sns.barplot(x=criteria_labels, y=weights, palette="viridis")
    
        for i, val in enumerate(weights):
            ax.text(i, val + 0.02, f"{val:.2f}", ha='center', fontsize=10)
            
        plt.title(title, fontsize=14, pad=15)
        plt.ylabel("Waga (0.0 - 1.0)", fontsize=12)
        plt.ylim(0, 1.0)
        plt.show()


    def plot_weights_comparison(self, weights: np.ndarray, pred_weights: np.ndarray, consistency_ratio: float, title: str = "Weights Comparison") -> None:
        plt.figure(figsize=(max(self.figsize[0], 8), self.figsize[1] // 2 + 1))
        ax = plt.gca()

        x = np.arange(len(weights))  
        width = 0.35
        rects1 = ax.bar(x - width/2, weights, width, label='Wagi Docelowe (Ground Truth)', color='#2ca02c', alpha=0.85)
        rects2 = ax.bar(x + width/2, pred_weights, width, label='Wagi Przewidziane (Sieć)', color='#1f77b4', alpha=0.85)

        ax.set_ylabel('Wartość wagi (0.0 - 1.0)', fontsize=12)
        ax.set_title(f"{title}, cr={consistency_ratio:.4f}", fontsize=14, pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels([f"Kryt {i}" for i in range(len(weights))], fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=11, loc='upper right')

        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.2f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 4),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9, fontweight='bold')

        autolabel(rects1)
        autolabel(rects2)

        plt.tight_layout()
        plt.show()


    def compare_matrices(self, clean_matrix: np.ndarray, noisy_matrix: np.ndarray, coherence_rate: float = None) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        sns.heatmap(
            clean_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=1.0, 
            ax=axes[0], cbar=False, linewidths=0.5
        )
        axes[0].set_title("Macierz Idealna (Spójna)", fontsize=13)
        axes[0].set_xlabel("Kryterium j")
        axes[0].set_ylabel("Kryterium i")
        
        title_noisy = "Macierz Zaszumiona"
        if coherence_rate is not None:
            title_noisy += f" (coherence={coherence_rate})"
            
        sns.heatmap(
            noisy_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=1.0, 
            ax=axes[1], linewidths=0.5, cbar_kws={'label': 'a_ij'}
        )
        axes[1].set_title(title_noisy, fontsize=13)
        axes[1].set_xlabel("Kryterium j")
        
        plt.tight_layout()
        plt.show()


    def plot_weights_comparison(self, n_weights: list[np.ndarray], consistency_ratio: float, title: str = "Weights Comparison") -> None:
        num_models = len(n_weights)
        if num_models == 0:
            print("Brak danych do narysowania.")
            return

        num_criteria = len(n_weights[0])
        x = np.arange(num_criteria)
        
        total_width = 0.8
        width = total_width / num_models

        plt.figure(figsize=(max(self.figsize[0], 10), self.figsize[1] // 2 + 1))
        ax = plt.gca()

        if labels is None or len(labels) != num_models:
            labels = [f"Model {i+1}" for i in range(num_models)]
        colors = sns.color_palette("deep", num_models)
        rects_list = []

        for i, weights in enumerate(n_weights):
            offset = (i - num_models / 2 + 0.5) * width
            
            rects = ax.bar(x + offset, weights, width, label=labels[i], color=colors[i], alpha=0.85)
            rects_list.append(rects)

        ax.set_ylabel('Wartość wagi (0.0 - 1.0)', fontsize=12)

        full_title = f"{title}\n(Wskaźnik spójności CR macierzy wejściowej: {consistency_ratio:.3f})"
        ax.set_title(full_title, fontsize=14, pad=15)
        
        ax.set_xticks(x)
        ax.set_xticklabels([f"Kryt {i}" for i in range(num_criteria)], fontsize=11)
        ax.set_ylim(0, 1.1) 
        ax.legend(fontsize=10, loc='best')

        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                rotation = 45 if num_models > 3 else 0 
                
                ax.annotate(f'{height:.2f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 4),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9, rotation=rotation)

        for rects in rects_list:
            autolabel(rects)

        plt.tight_layout()
        plt.show()


    def noisy_matrices_comparison(self, noisy_matrices: list[np.ndarray], consistency_ratio: list[float], title: str = "Noisy matrices comparison"):
        num_matrices = len(noisy_matrices)
        
        if num_matrices == 0:
            print("Brak macierzy do wyświetlenia.")
            return
            
        if num_matrices != len(consistency_ratio):
            raise ValueError(f"Niezgodność danych! Przekazano {num_matrices} macierzy i {len(consistency_ratio)} wartości CR.")


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
                cbar_kws={'label': 'Błąd log-spójności', 'shrink': 0.8}
            )
            
            ax.set_title(f"cr={consistency_ratio[i]:.4f}", fontsize=13, pad=10)
            ax.set_xlabel("Kryterium j", fontsize=10)
            if i % cols == 0: 
                ax.set_ylabel("Kryterium i", fontsize=10)

        for j in range(num_matrices, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout(rect=[0, 0.0, 1, 0.95])
        plt.show()


    def plot_evm_degradation(self, cr_values: np.ndarray, mae_errors: np.ndarray, kendall_taus: np.ndarray) -> None:
        df = pd.DataFrame({
            'CR': cr_values,
            'Błąd Estymacji (MAE)': mae_errors,
            'Korelacja Rang (Kendall Tau)': kendall_taus
        })

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        sns.scatterplot(data=df, x='CR', y='Błąd Estymacji (MAE)', alpha=0.4, color='royalblue', ax=axes[0])
        sns.regplot(data=df, x='CR', y='Błąd Estymacji (MAE)', scatter=False, color='darkblue', ax=axes[0], line_kws={"linewidth": 2})

        axes[0].axvline(0.1, color='red', linestyle='--', linewidth=1.5, label='Próg Saaty\'ego (CR=0.1)')
        axes[0].set_title('Wzrost błędu bezwzględnego wag (MAE)', fontsize=14, pad=10)
        axes[0].set_xlabel('Współczynnik Niespójności (CR)', fontsize=12)
        axes[0].set_ylabel('Błąd MAE', fontsize=12)
        axes[0].legend()

        df['CR_binned'] = df['CR'].round(2)
        
        sns.lineplot(data=df, x='CR_binned', y='Korelacja Rang (Kendall Tau)', color='forestgreen', linewidth=2, ax=axes[1])
        axes[1].axvline(0.1, color='red', linestyle='--', linewidth=1.5)
        
        axes[1].set_title('Degradacja rankingu kryteriów (Kendall\'s Tau)', fontsize=14, pad=10)
        axes[1].set_xlabel('Współczynnik Niespójności (CR)', fontsize=12)
        axes[1].set_ylabel('Kendall Tau (1.0 = ideał)', fontsize=12)

        axes[1].set_ylim(0.0, 1.05) 

        plt.suptitle("Analiza Odporności Klasycznego Algorytmu EVM na Szum Zniekształcający", fontsize=16, fontweight='bold', y=1.05)
        plt.tight_layout()
        plt.show()