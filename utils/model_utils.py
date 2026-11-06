import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kendalltau
import torch.optim as optim

from model.architecture.dense_monte_carlo import DenseMonteCarloNet
from model.loss_function.dense_monte_carlo.custom_dense_MC_loss import CustomDenseMCLoss
from model.model_loader import ModelLoader
from model.trainers.dense_monte_carlo_trainer import DenseMonteCarloTrainer
from visualization.results.metrics_visualizator import MetricsVisualizator
from utils.loss_function_parameters import LossParameters, LossParametersMapper

def criterion_to_str(cop: float, fit: float, stab: float, margin: float, 
                    p: int, cop_mode: str, stab_mode: str, normalize_initial_loss: bool) -> str:
    cop_str = str(int(round(cop * 100)))
    fit_str = str(int(round(fit * 100)))
    stab_str = str(int(round(stab * 100)))
    margin_str = str(int(round(margin * 100)))
    cop_letter = str(cop_mode[0])
    stab_letter = str(stab_mode[0])
    name = f"cop{cop_str}{cop_letter}_fit{fit_str}_stab{stab_str}{stab_letter}_margin{margin_str}_p{p}"
    if normalize_initial_loss:
        name += "_initnorm"
    return name


def str_to_criterion(criterion_str: str) -> tuple[float, float, float, float, int, str, str, bool]:
    parts = criterion_str.split("_")
    cop = int(parts[0][3:]) / 100
    fit = int(parts[1][3:]) / 100
    stab = int(parts[2][3:]) / 100
    margin = int(parts[3][6:]) / 100
    p = int(parts[4][1:])
    cop_mode = parts[0][5:]
    stab_mode = parts[2][5:]
    normalize_initial_loss = "initnorm" in criterion_str
    return cop, fit, stab, margin, p, cop_mode, stab_mode, normalize_initial_loss


def compare_results_strata(solver, evaluator, matrices, weights, predicted_priority_weights, nn_name):
    all_cr = []
    all_mae_evm = []
    all_kendall_taus_evm = []
    all_mae_nn = []
    all_kendall_taus_nn = []
    all_top1_evm = []
    all_top1_nn = []
    all_spearman_evm = []
    all_spearman_nn = []

    for stratum_name, test_matrices in matrices.items():
        weights_vectors = weights[stratum_name]
        pred_weights_vectors = predicted_priority_weights[stratum_name]

        evm_weights_vectors, cr_values = solver.compute_batch(test_matrices)
        mae_evm, kendall_taus_evm, top1_evm, spearman_evm = evaluator.evaluate_batch(weights_vectors, evm_weights_vectors)
        mae_nn, kendall_taus_nn, top1_nn, spearman_nn = evaluator.evaluate_batch(weights_vectors, pred_weights_vectors)

        mean_cr = float(np.mean(cr_values))
        mean_mae_evm = float(np.mean(mae_evm))
        mean_tau_evm = float(np.mean(kendall_taus_evm))
        mean_spearman_evm = float(np.mean(spearman_evm))
        mean_top1_evm = float(np.mean(top1_evm))
        mean_spearman_evm = float(np.mean(spearman_evm))
        mean_mae_nn = float(np.mean(mae_nn))
        mean_tau_nn = float(np.mean(kendall_taus_nn))
        mean_top1_nn = float(np.mean(top1_nn))
        mean_spearman_nn = float(np.mean(spearman_nn))

        all_cr.append(mean_cr)
        all_mae_evm.append(mean_mae_evm)
        all_kendall_taus_evm.append(mean_tau_evm)
        all_mae_nn.append(mean_mae_nn)
        all_kendall_taus_nn.append(mean_tau_nn)
        all_top1_evm.append(mean_top1_evm)
        all_top1_nn.append(mean_top1_nn)
        all_spearman_evm.append(mean_spearman_evm)
        all_spearman_nn.append(mean_spearman_nn)

        # print(f"Stratum '{stratum_name}': Mean CR = {mean_cr:.4f}")
        # print(f"  MAE: EVM = {mean_mae_evm:.4f} | NN = {mean_mae_nn:.4f}")
        # print(f"  Kendall Tau: EVM = {mean_tau_evm:.4f} | NN = {mean_tau_nn:.4f}\n")

    final_cr = np.array(all_cr)
    final_mae_evm = np.array(all_mae_evm)
    final_kendall_taus_evm = np.array(all_kendall_taus_evm)
    final_mae_nn = np.array(all_mae_nn)
    final_kendall_taus_nn = np.array(all_kendall_taus_nn)
    final_top1_evm = np.array(all_top1_evm)
    final_top1_nn = np.array(all_top1_nn)
    final_spearman_evm = np.array(all_spearman_evm)
    final_spearman_nn = np.array(all_spearman_nn)

    # print("--- GLOBAL EXPERIMENT RESULTS ---")
    # print(f"Global mean CR: {np.mean(final_cr):.4f}")
    # print(f"Global mean MAE: EVM = {np.mean(final_mae_evm):.4f}, NN = {np.mean(final_mae_nn):.4f}")
    # print(f"Global mean Kendall Tau: EVM = {np.mean(final_kendall_taus_evm):.4f}, NN = {np.mean(final_kendall_taus_nn):.4f}")

    viz = MetricsVisualizator()
    viz.plot_comparison(final_cr, final_mae_evm, final_kendall_taus_evm, final_mae_nn, final_kendall_taus_nn, nn_name)
    # viz.plot_top1_spearman(final_cr, final_top1_evm, final_top1_nn, final_spearman_evm, final_spearman_nn, nn_name)

def compare_mean_mae_models(
        true_weights: dict[str, np.ndarray], 
        predicted_priority_weights: dict[str, list[np.ndarray]],
        nn_names: list[str],
        legend_title: str = "Metody obliczania wag"
    ) -> pd.DataFrame:
    strata = ["low_inconsistency", "moderate_inconsistency", "high_inconsistency"]
    strata_pol = ['Niska niespójność', "Umiarkowana niespójność", "Wysoka niespójność"]
    records = []

    for stratum, stratum_pol in zip(strata, strata_pol):
        if stratum not in true_weights or stratum not in predicted_priority_weights:
            continue
        y_true = np.asarray(true_weights[stratum])
        model_preds = predicted_priority_weights[stratum]

        for idx, name in enumerate(nn_names):
            if isinstance(model_preds, list):
                y_pred = np.asarray(model_preds[idx])
            else:
                y_pred = np.asarray(model_preds[name])

            if y_true.ndim != y_pred.ndim:
                y_pred = np.squeeze(y_pred)
                y_true_squeezed = np.squeeze(y_true)
            else:
                y_true_squeezed = y_true

            mae = float(np.mean(np.abs(y_true_squeezed - y_pred)))
            records.append({
                "Stratum": stratum_pol,
                "Model": name,
                "Mean MAE": mae
            })

    df = pd.DataFrame(records)

    sns.set_theme(style="whitegrid")
    fig_width = max(12, int(len(nn_names) * 0.9))
    plt.figure(figsize=(fig_width, 6))
    palette = sns.color_palette("muted", len(nn_names))
    ax = sns.barplot(
        data=df, 
        x="Stratum", 
        y="Mean MAE", 
        hue="Model", 
        palette=palette
    )

    label_rot = 90 if len(nn_names) > 6 else 0
    font_sz = 8 if len(nn_names) > 6 else 9
    for container in ax.containers:
        ax.bar_label(container, fmt="%.4f", padding=3, fontsize=font_sz, rotation=label_rot)

    plt.title("Porównanie średniego błędu MAE modeli dla poziomu niespójności", fontsize=14, pad=15)
    plt.xlabel("Poziom niespójności", fontsize=12)
    plt.ylabel("Średnie MAE", fontsize=12)
    max_val = df["Mean MAE"].max() if not df.empty else 1.0
    y_mult = 1.55 if len(nn_names) > 6 else 1.45
    plt.ylim(0, max_val * y_mult)
    plt.legend(title=legend_title, loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.show()

    return df


def compare_kendal_tau_models(
        true_weights: dict[str, np.ndarray], 
        predicted_priority_weights: dict[str, list[np.ndarray]],
        nn_names: list[str],
        legend_title: str = "Metody obliczania wag"
    ) -> pd.DataFrame:
    strata = ["low_inconsistency", "moderate_inconsistency", "high_inconsistency"]
    strata_pol = ['Niska niespójność', "Umiarkowana niespójność", "Wysoka niespójność"]
    records = []

    for stratum, stratum_pol in zip(strata, strata_pol):
        if stratum not in true_weights or stratum not in predicted_priority_weights:
            continue
        y_true = np.asarray(true_weights[stratum])
        model_preds = predicted_priority_weights[stratum]

        for idx, name in enumerate(nn_names):
            if isinstance(model_preds, list):
                y_pred = np.asarray(model_preds[idx])
            else:
                y_pred = np.asarray(model_preds[name])

            if y_true.ndim != y_pred.ndim:
                y_pred = np.squeeze(y_pred)
                y_true_squeezed = np.squeeze(y_true)
            else:
                y_true_squeezed = y_true

            taus = []
            for j in range(len(y_true_squeezed)):
                tau, _ = kendalltau(y_true_squeezed[j], y_pred[j])
                if np.isnan(tau):
                    taus.append(0.0)
                else:
                    taus.append(float(tau))

            mean_tau = float(np.mean(taus)) if taus else 0.0
            records.append({
                "Stratum": stratum_pol,
                "Model": name,
                "Kendall Tau": mean_tau
            })

    df = pd.DataFrame(records)

    sns.set_theme(style="whitegrid")
    fig_width = max(12, int(len(nn_names) * 0.9))
    plt.figure(figsize=(fig_width, 6))
    palette = sns.color_palette("muted", len(nn_names))
    ax = sns.barplot(
        data=df, 
        x="Stratum", 
        y="Kendall Tau", 
        hue="Model", 
        palette=palette
    )

    label_rot = 90 if len(nn_names) > 6 else 0
    font_sz = 8 if len(nn_names) > 6 else 9
    for container in ax.containers:
        ax.bar_label(container, fmt="%.4f", padding=3, fontsize=font_sz, rotation=label_rot)

    plt.title("Porównanie korelacji rang Kendalla modeli dla poziomów niespójności", fontsize=14, pad=15)
    plt.xlabel("Poziom niespójności", fontsize=14)
    plt.ylabel("Kendall Tau", fontsize=14)
    plt.ylim(0, 1)
    plt.legend(title=legend_title, loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.show()

    return df


def compare_results_strata_cr(
    solver, 
    evaluator, 
    matrices: dict[str, np.ndarray], 
    weights: dict[str, np.ndarray], 
    predicted_priority_weights: dict[str, np.ndarray], 
    nn_name: str,
    bin_size: float = 0.02
) -> None:
    num_bins = int(round(0.4 / bin_size)) + 1
    decimals = max(2, len(str(bin_size).split(".")[-1]))
    cr_keys = [round(i * bin_size, decimals) for i in range(num_bins)]

    binned_matrices: dict[float, list[np.ndarray]] = {k: [] for k in cr_keys}
    binned_weights: dict[float, list[np.ndarray]] = {k: [] for k in cr_keys}
    binned_pred_weights: dict[float, list[np.ndarray]] = {k: [] for k in cr_keys}

    for stratum_name, test_matrices in matrices.items():
        weights_vectors = np.asarray(weights[stratum_name])
        pred_weights_vectors = np.asarray(predicted_priority_weights[stratum_name])

        if weights_vectors.ndim != pred_weights_vectors.ndim:
            pred_weights_vectors = np.squeeze(pred_weights_vectors)
            weights_vectors = np.squeeze(weights_vectors)

        for i in range(len(test_matrices)):
            matrix = test_matrices[i]
            if hasattr(solver, "compute_single"):
                _, cr = solver.compute_single(matrix)
            else:
                _, cr_arr = solver.compute_batch(matrix[np.newaxis, ...])
                cr = float(cr_arr[0])

            cr_bin = round(round(cr / bin_size) * bin_size, decimals)
            if 0.0 <= cr_bin <= 0.4 and cr_bin in binned_matrices:
                binned_matrices[cr_bin].append(matrix)
                binned_weights[cr_bin].append(weights_vectors[i])
                binned_pred_weights[cr_bin].append(pred_weights_vectors[i])

    final_cr = []
    final_mae_evm = []
    final_kendall_taus_evm = []
    final_mae_nn = []
    final_kendall_taus_nn = []
    final_top1_evm = []
    final_top1_nn = []
    final_spearman_evm = []
    final_spearman_nn = []

    for cr_val in cr_keys:
        mats = binned_matrices[cr_val]
        if not mats:
            continue

        mats_arr = np.asarray(mats)
        true_w_arr = np.asarray(binned_weights[cr_val])
        pred_w_arr = np.asarray(binned_pred_weights[cr_val])

        evm_w_arr, _ = solver.compute_batch(mats_arr)
        mae_evm, tau_evm, top1_evm, spearman_evm = evaluator.evaluate_batch(true_w_arr, evm_w_arr)
        mae_nn, tau_nn, top1_nn, spearman_nn = evaluator.evaluate_batch(true_w_arr, pred_w_arr)

        final_cr.append(cr_val)
        final_mae_evm.append(float(np.mean(mae_evm)))
        final_kendall_taus_evm.append(float(np.mean(tau_evm)))
        final_mae_nn.append(float(np.mean(mae_nn)))
        final_kendall_taus_nn.append(float(np.mean(tau_nn)))
        final_top1_evm.append(float(np.mean(top1_evm)))
        final_top1_nn.append(float(np.mean(top1_nn)))
        final_spearman_evm.append(float(np.mean(spearman_evm)))
        final_spearman_nn.append(float(np.mean(spearman_nn)))

    if final_cr:
        viz = MetricsVisualizator()
        viz.plot_comparison(
            np.asarray(final_cr),
            np.asarray(final_mae_evm),
            np.asarray(final_kendall_taus_evm),
            np.asarray(final_mae_nn),
            np.asarray(final_kendall_taus_nn),
            nn_name
        )
        # viz.plot_top1_spearman(
        #     np.asarray(final_cr),
        #     np.asarray(final_top1_evm),
        #     np.asarray(final_top1_nn),
        #     np.asarray(final_spearman_evm),
        #     np.asarray(final_spearman_nn),
        #     nn_name
        # )


def collect_grid_search_results(
    evaluator,
    true_weights: dict[str, np.ndarray],
    parameters: list[list],
    predicted_priority_weights: dict[str, list[np.ndarray]],
    model_names: list[str]
) -> pd.DataFrame:
    strata = list(true_weights.keys())
    records = []

    for model_idx, (param, name) in enumerate(zip(parameters, model_names)):
        cop = param[0]
        fit = param[1]
        stab = param[2]
        p = param[3]
        margin = param[4]
        normalize = param[5]
        cop_mode = param[6]
        stab_mode = param[7]

        for stratum in strata:
            if stratum not in true_weights:
                continue

            y_true = np.asarray(true_weights[stratum])
            preds_for_stratum = predicted_priority_weights[stratum]

            if isinstance(preds_for_stratum, list):
                y_pred = np.asarray(preds_for_stratum[model_idx])
            else:
                y_pred = np.asarray(preds_for_stratum[name])

            if y_true.ndim != y_pred.ndim:
                y_pred = np.squeeze(y_pred)
                y_true = np.squeeze(y_true)

            mae_scores, tau_scores, top1_scores, spearman_scores = evaluator.evaluate_batch(y_true, y_pred)

            records.append({
                "model_name": name,
                "stratum": stratum,
                LossParameters.COP_WEIGHT: cop,
                LossParameters.FIT_WEIGHT: fit,
                LossParameters.STAB_WEIGHT: stab,
                LossParameters.P: p,
                LossParameters.MARGIN: margin,
                LossParameters.NORMALIZE: normalize,
                LossParameters.COP_MODE: cop_mode,
                LossParameters.STAB_MODE: stab_mode,
                "mae": float(np.mean(mae_scores)),
                "kendall_tau": float(np.mean(tau_scores)),
                "top1": float(np.mean(top1_scores)),
                "spearman": float(np.mean(spearman_scores))
            })

    return pd.DataFrame(records)


def _metric_props(value_metric: str) -> tuple[str, str]:
    if value_metric == "mae":
        return "MAE", "viridis_r"
    if value_metric == "kendall_tau":
        return "Kendall Tau", "viridis"
    if value_metric == "top1":
        return "Top-1 Hit Rate", "viridis"
    if value_metric == "spearman":
        return "Spearman Rank Correlation", "viridis"
    return value_metric.upper(), "viridis"


def plot_grid_search_heatmap(
    grid_results_df: pd.DataFrame,
    value_metric: str,
    stratum_name: str,
    loss_param1: str = LossParameters.COP_WEIGHT,
    loss_param2: str = LossParameters.FIT_WEIGHT
) -> None:
    df_stratum = grid_results_df[grid_results_df["stratum"] == stratum_name]
    stratum_translate = {
        "low_inconsistency": "niskim poziomie niespójności",
        "moderate_inconsistency": "umiarkowanym poziomie niespójności",
        "high_inconsistency": "wysokim poziomie niespójności"
    }

    if df_stratum.empty:
        print(f"Brak danych dla warstwy: {stratum_name}")
        return

    try:
        pivot_table = df_stratum.pivot(
            index=loss_param1,
            columns=loss_param2,
            values=value_metric
        )
    except Exception:
        pivot_table = df_stratum.groupby([loss_param1, loss_param2])[value_metric].mean().unstack(loss_param2)

    metric_label, cmap = _metric_props(value_metric)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        pivot_table,
        annot=True,
        fmt=".4f",
        cmap=cmap,
        linewidths=0.5,
        linecolor="white"
    )

    title = f'Wartości {metric_label} dla zbioru o {stratum_translate.get(stratum_name)}' if value_metric == "kendall_tau" else f'Błędy {metric_label} dla zbioru o {stratum_translate.get(stratum_name)}' 
    plt.title(title, fontsize=13)
    plt.xlabel(LossParametersMapper.get(loss_param2, loss_param2), fontsize=11)
    plt.ylabel(LossParametersMapper.get(loss_param1, loss_param1), fontsize=11)
    plt.tight_layout()
    plt.show()


def plot_mean_results_heatmap(
    grid_results_df: pd.DataFrame,
    value_metric: str,
    loss_param1: str = LossParameters.COP_WEIGHT,
    loss_param2: str = LossParameters.FIT_WEIGHT
) -> None:
    pivot_table = (
        grid_results_df
        .groupby([loss_param1, loss_param2])[value_metric]
        .mean()
        .unstack(loss_param2)
    )

    metric_label, cmap = _metric_props(value_metric)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        pivot_table,
        annot=True,
        fmt=".4f",
        cmap=cmap,
        linewidths=0.5,
        linecolor="white"
    )
    plt.title(f"Średnia błędów {metric_label} ze wszystkich poziomów niespójności", fontsize=13)
    plt.xlabel(LossParametersMapper.get(loss_param2, loss_param2), fontsize=11)
    plt.ylabel(LossParametersMapper.get(loss_param1, loss_param1), fontsize=11)
    plt.tight_layout()
    plt.show()


def plot_mean_results_heatmap_3_params(
    grid_results_df: pd.DataFrame,
    value_metric: str,
    loss_param1: str = LossParameters.COP_WEIGHT,
    loss_param2: str = LossParameters.FIT_WEIGHT,
    loss_param3: str = LossParameters.STAB_WEIGHT
) -> None:
    param3_values = sorted(grid_results_df[loss_param3].unique())
    n_cols = len(param3_values)

    metric_label, cmap = _metric_props(value_metric)

    all_vals = grid_results_df[value_metric]
    vmin, vmax = all_vals.min(), all_vals.max()

    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5), sharey=True)

    if n_cols == 1:
        axes = [axes]

    for ax, p3_val in zip(axes, param3_values):
        subset = grid_results_df[grid_results_df[loss_param3] == p3_val]

        pivot_table = (
            subset
            .groupby([loss_param1, loss_param2])[value_metric]
            .mean()
            .unstack(loss_param2)
        )

        sns.heatmap(
            pivot_table,
            annot=True,
            fmt=".4f",
            cmap=cmap,
            linewidths=0.5,
            linecolor="white",
            ax=ax,
            vmin=vmin,
            vmax=vmax,
            cbar=(ax == axes[-1])
        )

        p3_label = LossParametersMapper.get(loss_param3, loss_param3)
        ax.set_title(f"{p3_label} = {p3_val}", fontsize=12)
        ax.set_xlabel(LossParametersMapper.get(loss_param2, loss_param2), fontsize=11)
        ax.set_ylabel(LossParametersMapper.get(loss_param1, loss_param1) if ax == axes[0] else "", fontsize=11)

    fig.suptitle(f"{metric_label} — średnia po wszystkich warstwach niespójności", fontsize=14)
    plt.tight_layout()
    plt.show()

def get_predictor(device, cop, fit, stab, margin, p, cop_mode, stab_mode, normalize, lr, criteria_num):
    criterion_str = criterion_to_str(cop, fit, stab, margin, p, cop_mode, stab_mode, normalize)
    model_loader = ModelLoader()
    empty_model = DenseMonteCarloNet(n_criteria=criteria_num)
    model, history = model_loader.load_dense_MC(empty_model, criteria_num, criterion_str)
    criterion = CustomDenseMCLoss(n_criteria=criteria_num, lambda_cop=cop, lambda_fit=fit, 
        lambda_stab=stab, p=p, normalize_initial_loss=normalize, margin=margin,
        cop_mode=cop_mode, stab_mode=stab_mode).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    trainer = DenseMonteCarloTrainer(model, optimizer, criterion, device)

    return trainer, criterion_str