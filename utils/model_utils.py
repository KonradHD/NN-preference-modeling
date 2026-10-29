import numpy as np
from visualization.results.metrics_visualizator import MetricsVisualizator

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

    for stratum_name, test_matrices in matrices.items():
        weights_vectors = weights[stratum_name]
        pred_weights_vectors = predicted_priority_weights[stratum_name]

        evm_weights_vectors, cr_values = solver.compute_batch(test_matrices)
        mae_evm, kendall_taus_evm = evaluator.evaluate_batch(weights_vectors, evm_weights_vectors)
        mae_nn, kendall_taus_nn = evaluator.evaluate_batch(weights_vectors, pred_weights_vectors)

        mean_cr = float(np.mean(cr_values))
        mean_mae_evm = float(np.mean(mae_evm))
        mean_tau_evm = float(np.mean(kendall_taus_evm))
        mean_mae_nn = float(np.mean(mae_nn))
        mean_tau_nn = float(np.mean(kendall_taus_nn))

        all_cr.append(mean_cr)
        all_mae_evm.append(mean_mae_evm)
        all_kendall_taus_evm.append(mean_tau_evm)
        all_mae_nn.append(mean_mae_nn)
        all_kendall_taus_nn.append(mean_tau_nn)

        # print(f"Stratum '{stratum_name}': Mean CR = {mean_cr:.4f}")
        # print(f"  MAE: EVM = {mean_mae_evm:.4f} | NN = {mean_mae_nn:.4f}")
        # print(f"  Kendall Tau: EVM = {mean_tau_evm:.4f} | NN = {mean_tau_nn:.4f}\n")

    final_cr = np.array(all_cr)
    final_mae_evm = np.array(all_mae_evm)
    final_kendall_taus_evm = np.array(all_kendall_taus_evm)
    final_mae_nn = np.array(all_mae_nn)
    final_kendall_taus_nn = np.array(all_kendall_taus_nn)

    # print("--- GLOBAL EXPERIMENT RESULTS ---")
    # print(f"Global mean CR: {np.mean(final_cr):.4f}")
    # print(f"Global mean MAE: EVM = {np.mean(final_mae_evm):.4f}, NN = {np.mean(final_mae_nn):.4f}")
    # print(f"Global mean Kendall Tau: EVM = {np.mean(final_kendall_taus_evm):.4f}, NN = {np.mean(final_kendall_taus_nn):.4f}")

    viz = MetricsVisualizator()
    viz.plot_comparison(final_cr, final_mae_evm, final_kendall_taus_evm, final_mae_nn, final_kendall_taus_nn, nn_name)