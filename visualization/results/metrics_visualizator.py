import matplotlib.pyplot as plt
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
                        kendall_taus_nn: np.ndarray):
        data_analytic = pd.DataFrame({
            'CR': cr_values,
            'MAE': mae_errors_analytic,
            'Kendall Tau': kendall_taus_analytic,
            'Metoda': 'Analityczna (EVM)'
        })
        
        data_nn = pd.DataFrame({
            'CR': cr_values,
            'MAE': mae_errors_nn,
            'Kendall Tau': kendall_taus_nn,
            'Metoda': 'Sieć Neuronowa'
        })
        
        df = pd.concat([data_analytic, data_nn], axis=0)
        df['CR_binned'] = df['CR'].round(2)

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        sns.lineplot(data=df, x='CR_binned', y='MAE', hue='Metoda', palette=['royalblue', 'orange'], ax=axes[0], linewidth=2.5)
        axes[0].axvline(0.1, color='red', linestyle='--', alpha=0.6, label="Próg Saaty'ego")
        axes[0].set_title('Porównanie błędów MAE\n(Im niżej, tym lepiej)', fontsize=14)
        axes[0].set_xlabel('Współczynnik Niespójności (CR)')
        axes[0].set_ylabel('Błąd Średni Bezwzględny (MAE)')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        sns.lineplot(data=df, x='CR_binned', y='Kendall Tau', hue='Metoda', palette=['forestgreen', 'darkred'], ax=axes[1], linewidth=2.5)
        axes[1].axvline(0.1, color='red', linestyle='--', alpha=0.6)
        axes[1].set_title('Stabilność Rankingu (Kendall Tau)\n(Im wyżej, tym lepiej)', fontsize=14)
        axes[1].set_xlabel('Współczynnik Niespójności (CR)')
        axes[1].set_ylabel('Korelacja Rang')
        axes[1].set_ylim(0, 1.05)
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        plt.suptitle("Porównanie Odporności: Algorytm Klasyczny vs Sieć Neuronowa", fontsize=18, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.show()