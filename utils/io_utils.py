import os 
import torch
import json
import numpy as np

from data_augmentation.matrices_loader import MatricesLoader
from utils.phase import Phase
from data_augmentation.generator.target_cr_generator import TargetCRMatricesGenerator
from data_augmentation.datasets.siamese_AHP_matrix_dataset import SiameseAHPMatrixDataset
from data_augmentation.datasets.dense_AHP_dataset import DenseAHPDataset
from data_augmentation.datasets.graph_AHP_dataset import GraphAHPDataset


def load_model_weights_history(model: torch.nn.Module, model_type_name: str, loss_param: str, name: str, 
                               device: torch.device) -> tuple[torch.nn.Module, dict[str, list]]:
    base_dir = "models"
    dir_path = os.path.join(base_dir, model_type_name, loss_param)
    model_path = os.path.join(dir_path, f"model_{name}.pth")
    history_path = os.path.join(dir_path, f"history_{name}.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    if not os.path.exists(history_path):
        raise FileNotFoundError(f"History file not found: {history_path}")

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    with open(history_path, "r", encoding="utf-8") as file:
        history = json.load(file)
    
    print(f"History and model: {name} was loaded from {dir_path}")
    return model, history


def load_test_matrices(loader: MatricesLoader, consistency_rates: list[float], criteria_num: int, matrices_num: int, is_uniform: bool = True):
    all_matrices = {}
    all_weights = {}

    for cr in consistency_rates:
        matrices, weights = loader.load_noisy_cr_matrices(Phase.TEST, cr, is_uniform, criteria_num, matrices_num)
        all_matrices[cr] = matrices
        all_weights[cr] = weights

    print(f"{len(list(all_matrices.keys())) * matrices_num} matrices and weights with different cr was successfully uploaded")
    return all_matrices, all_weights


def load_train_dataset_siamese(loader: MatricesLoader, consistency_rates: list[float], criteria_num: int, matrices_num: int, is_uniform: bool = True):
    all_matrices = []
    all_comparison_matrices = []
    all_weights = []

    for cr in consistency_rates:
        matrices, weights = loader.load_noisy_cr_matrices(Phase.TRAIN, cr, is_uniform, criteria_num, matrices_num)
        generator = TargetCRMatricesGenerator(matrices.shape[0], matrices.shape[1], cr)
        comparison_matrices = generator.from_weights(weights)

        all_matrices.append(matrices)
        all_comparison_matrices.append(comparison_matrices)
        all_weights.append(weights)
    
    final_matrices = np.concatenate(all_matrices, axis=0)
    final_weights = np.concatenate(all_weights, axis=0)
    final_comparison_matrices = np.concatenate(all_comparison_matrices, axis=0)
    dataset = SiameseAHPMatrixDataset(final_matrices, final_comparison_matrices, final_weights)

    print(f"Train dataset was successfully created and {len(dataset)} matrices was uploaded")
    return dataset


def load_train_dataset_dnn(loader: MatricesLoader, consistency_rates: list[float], criteria_num: int, matrices_num: int, is_uniform: bool = True):
    all_matrices = []
    all_weights = []

    for cr in consistency_rates:
        matrices, weights = loader.load_noisy_cr_matrices(Phase.TRAIN, cr, is_uniform, criteria_num, matrices_num)

        all_matrices.append(matrices)
        all_weights.append(weights)
    
    final_matrices = np.concatenate(all_matrices, axis=0)
    final_weights = np.concatenate(all_weights, axis=0)
    dataset = DenseAHPDataset(final_matrices, final_weights, augment=True)

    print(f"Train dataset was successfully created and {len(dataset)} matrices was uploaded")
    return dataset


def load_valid_dataset_siamese(loader: MatricesLoader, consistency_rates: list[float], criteria_num: int, matrices_num: int, is_uniform: bool = True):
    all_matrices = []
    all_comparison_matrices = []
    all_weights = []

    for cr in consistency_rates:
        matrices, weights = loader.load_noisy_cr_matrices(Phase.VALIDATION, cr, is_uniform, criteria_num, matrices_num)
        generator = TargetCRMatricesGenerator(matrices.shape[0], matrices.shape[1], cr)
        comparison_matrices = generator.from_weights(weights)

        all_matrices.append(matrices)
        all_comparison_matrices.append(comparison_matrices)
        all_weights.append(weights)
    
    final_matrices = np.concatenate(all_matrices, axis=0)
    final_weights = np.concatenate(all_weights, axis=0)
    final_comparison_matrices = np.concatenate(all_comparison_matrices, axis=0)
    dataset = SiameseAHPMatrixDataset(final_matrices, final_comparison_matrices, final_weights)

    print(f"Validation dataset was successfully created and {len(dataset)} matrices was uploaded")
    return dataset


def load_valid_dataset_dnn(loader: MatricesLoader, consistency_rates: list[float], criteria_num: int, matrices_num: int, is_uniform: bool = True):
    all_matrices = []
    all_weights = []

    for cr in consistency_rates:
        matrices, weights = loader.load_noisy_cr_matrices(Phase.VALIDATION, cr, is_uniform, criteria_num, matrices_num)

        all_matrices.append(matrices)
        all_weights.append(weights)
    
    final_matrices = np.concatenate(all_matrices, axis=0)
    final_weights = np.concatenate(all_weights, axis=0)
    dataset = DenseAHPDataset(final_matrices, final_weights, augment=False)

    print(f"Validation dataset was successfully created and {len(dataset)} matrices was uploaded")
    return dataset


def load_train_dataset_graph(loader: MatricesLoader, consistency_rates: list[float], criteria_num: int, matrices_num: int, is_uniform: bool = True):
    all_matrices = []
    all_weights = []

    for cr in consistency_rates:
        matrices, weights = loader.load_noisy_cr_matrices(Phase.TRAIN, cr, is_uniform, criteria_num, matrices_num)

        all_matrices.append(matrices)
        all_weights.append(weights)
    
    final_matrices = np.concatenate(all_matrices, axis=0)
    final_weights = np.concatenate(all_weights, axis=0)
    dataset = GraphAHPDataset(final_matrices, final_weights, augment=True)

    print(f"Train dataset was successfully created and {len(dataset)} matrices was uploaded")
    return dataset


def load_valid_dataset_graph(loader: MatricesLoader, consistency_rates: list[float], criteria_num: int, matrices_num: int, is_uniform: bool = True):
    all_matrices = []
    all_weights = []

    for cr in consistency_rates:
        matrices, weights = loader.load_noisy_cr_matrices(Phase.VALIDATION, cr, is_uniform, criteria_num, matrices_num)

        all_matrices.append(matrices)
        all_weights.append(weights)
    
    final_matrices = np.concatenate(all_matrices, axis=0)
    final_weights = np.concatenate(all_weights, axis=0)
    dataset = GraphAHPDataset(final_matrices, final_weights, augment=False)

    print(f"Validation dataset was successfully created and {len(dataset)} matrices was uploaded")
    return dataset