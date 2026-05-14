import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class WeightsVisualizator():
    def __init__(self, figsize: tuple[int, int] = (8, 6)):
        self.figsize = figsize
        sns.set_theme(style="whitegrid")


    def plot_comparison_pred(self, weights: np.ndarray, pred_weights: np.ndarray, consistency_ratio: float, title: str = "Weights Comparison") -> None:
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


    def plot_comparison_all(self, weights: np.ndarray, pred_weights: np.ndarray, analytic_weights: np.ndarray,
                         consistency_ratio: float, title: str = "Weights Comparison"):
        plt.figure(figsize=(max(self.figsize[0], 10), self.figsize[1] // 2 + 1))
        ax = plt.gca()

        x = np.arange(len(weights))  
        width = 0.25

        rects1 = ax.bar(x - width, weights, width, label='Wagi Docelowe (Ground Truth)', color='#2ca02c', alpha=0.85)
        rects2 = ax.bar(x, pred_weights, width, label='Wagi Sieci (NN)', color='#1f77b4', alpha=0.85)
        rects3 = ax.bar(x + width, analytic_weights, width, label='Klasyczne AHP (EVM)', color='#d62728', alpha=0.85)

        ax.set_ylabel('Wartość wagi (0.0 - 1.0)', fontsize=12)
        ax.set_title(f"{title} (CR = {consistency_ratio:.4f})", fontsize=14, pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels([f"Kryt {i+1}" for i in range(len(weights))], fontsize=11)
        ax.set_ylim(0, 1.15)
        
        ax.legend(fontsize=11, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=3)

        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                if height > 0.01:
                    ax.annotate(f'{height:.2f}',
                                xy=(rect.get_x() + rect.get_width() / 2, height),
                                xytext=(0, 4),
                                textcoords="offset points",
                                ha='center', va='bottom', fontsize=8, fontweight='bold', rotation=90)

        autolabel(rects1)
        autolabel(rects2)
        autolabel(rects3)

        plt.tight_layout()
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