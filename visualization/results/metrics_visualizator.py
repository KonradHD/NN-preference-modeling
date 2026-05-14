import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import numpy as np
import pandas as pd

class MetricsVisualizator():
    def __init__(self, figsize: tuple[int, int] = (8, 6)):
        self.figsize = figsize
        sns.set_theme(style="whitegrid")


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


    def plot_comparison(self, cr_values: np.ndarray, mae_errors_analytic: np.ndarray, 
                        kendall_taus_analytic: np.ndarray, mae_errors_nn: np.ndarray, 
                        kendall_taus_nn: np.ndarray, nn_name: str) -> None:
        data_analytic = pd.DataFrame({
            'CR': cr_values,
            'MAE': mae_errors_analytic,
            'Kendall Tau': kendall_taus_analytic,
            'Method': 'Analytic (EVM)'
        })
        
        data_nn = pd.DataFrame({
            'CR': cr_values,
            'MAE': mae_errors_nn,
            'Kendall Tau': kendall_taus_nn,
            'Method': nn_name
        })
        
        df = pd.concat([data_analytic, data_nn], axis=0)
        df['CR_binned'] = df['CR'].round(2)

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        sns.lineplot(data=df, x='CR_binned', y='MAE', hue='Method', palette=['royalblue', 'orange'], ax=axes[0], linewidth=2.5)
        axes[0].axvline(0.1, color='red', linestyle='--', alpha=0.6, label="Saaty's Threshold")
        axes[0].set_title('MAE Comparison\n', fontsize=14)
        axes[0].set_xlabel('Consistency Ratio (CR)')
        axes[0].set_ylabel('Mean Absolute Error (MAE)')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        sns.lineplot(data=df, x='CR_binned', y='Kendall Tau', hue='Method', palette=['forestgreen', 'darkred'], ax=axes[1], linewidth=2.5)
        axes[1].axvline(0.1, color='red', linestyle='--', alpha=0.6)
        axes[1].set_title('Ranking Stability (Kendall Tau)', fontsize=14)
        axes[1].set_xlabel('Consistency Ratio (CR)')
        axes[1].set_ylabel('Rank Correlation')
        axes[1].set_ylim(0, 1.05)
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        plt.suptitle("Stability comparison: Classic algorithm vs Neural Network", fontsize=18, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.show()

    
    def display_loss(self, history: dict[str, list]) -> None:
        if not history or 'train' not in history or not history['train']:
            print("Błąd: Brak danych treningowych do wyświetlenia na wykresie.")
            return

        epochs = range(1, len(history['train']) + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, history['train'], label='Strata Treningowa (Train Loss)', 
                color='#1f77b4', linewidth=2.5)

        if 'valid' in history and history['valid']:
            plt.plot(epochs, history['valid'], label='Strata Walidacyjna (Valid Loss)', 
                    color='#ff7f0e', linewidth=2.5, linestyle='--')

        plt.title('Krzywe Uczenia Sieci SiameseAHP', fontsize=14, pad=15)
        plt.xlabel('Epoka', fontsize=12)
        plt.ylabel('Wartość Straty (Loss)', fontsize=12)

        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        plt.grid(True, linestyle=':', alpha=0.7)
        plt.legend(fontsize=11, loc='upper right')
        plt.tight_layout()
        plt.show()