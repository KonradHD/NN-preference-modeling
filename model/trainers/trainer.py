from PIL import TiffImagePlugin
from abc import ABC
from abc import abstractmethod 
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import copy
import numpy as np
import os 
import json
import time

from model.loss_function.dense_monte_carlo.MC_base_loss import MCBaseLoss


class Trainer(ABC):
    def __init__(self, model: torch.nn.Module, optimizer: torch.optim,
                    criterion: MCBaseLoss, device: torch.device):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.history = {
            'train_loss': [], 
            'valid_loss': [], 
            'valid_mae': [],
            'valid_mse': [],
            'train_mse': [], 
            'train_mae': [],
            'train_cop': [], 
            'train_rec': [], 
            'train_stab': [], 
            'train_fit': []
        }
        self._model_type: str = "unknown"


    @abstractmethod
    def _train_epoch(self, train_dataloader: DataLoader, valid_dataloader: DataLoader) -> tuple[float, float, float]:
        raise NotImplementedError()
    

    def train(self, train_dataloader: DataLoader, valid_dataloader: DataLoader, epochs: int = 500, 
              early_stopping: bool = True, es_patience: int = 15, reduce_lr_on_plateau: bool = True, rlr_patience: int = 5) -> tuple[torch.nn.Module, dict[str, float]]:

        best_valid_loss = float('inf')
        best_model_state = None
        epochs_no_improve = 0
        self.history["epochs"] = epochs
        self.history["early_stopping"] = early_stopping
        self.history["es_patience"] = es_patience
        self.history["reduce_lr_on_plateau"] = reduce_lr_on_plateau
        self.history["rlr_patience"] = rlr_patience
        self.history["criterion"] = str(self.criterion)
        self.history["learning_rates"] = []
        self.history.update(self.criterion.get_config())

        start_time = time.time()

        if reduce_lr_on_plateau:
            scheduler = ReduceLROnPlateau(optimizer=self.optimizer, mode="min", factor=0.5, patience=rlr_patience)
            current_lr = self.optimizer.param_groups[0]["lr"]

        for epoch in range(epochs):
            avg_train_loss, avg_valid_loss, avg_valid_mae = self._train_epoch(train_dataloader, valid_dataloader)

            if reduce_lr_on_plateau:
                scheduler.step(avg_valid_loss)

                new_lr = self.optimizer.param_groups[0]['lr']
                if new_lr < current_lr:
                    print(f"Epoch number {epoch+1}: Learning Rate was reduced to {new_lr:.6f}")
                    current_lr = new_lr
            
            self.history["learning_rates"].append(self.optimizer.param_groups[0]['lr'])

            if avg_valid_loss < best_valid_loss:
                best_valid_loss = avg_valid_loss
                best_model_state = copy.deepcopy(self.model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if (epoch + 1) % 10 == 0:
                print(f"Epoch number [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.6f} | Valid Loss: {avg_valid_loss:.6f} | Valid MAE: {avg_valid_mae:.4f}")

            if early_stopping and epochs_no_improve >= es_patience:
                self.history["epochs"] = epoch + 1
                print(f"\nEARLY STOPPING in epoch {epoch + 1}.")
                break

        end_time = time.time()
        self.history['train_time'] = end_time - start_time

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            self.history["best_valid_loss"] = best_valid_loss
            print(f"Wages are restored to the best valid loss: {best_valid_loss:.6f}")

        return self.model, self.history


    @abstractmethod
    def predict_single(self, matrix: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
    

    @abstractmethod
    def predict_batch(self, matrix: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
    

    def save_state(self, name: str, base_dir: str = "models"):
        dir_path = os.path.join(base_dir, self._model_type, str(self.criterion))

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        model_path = os.path.join(dir_path, f"model_{name}.pth")
        history_path = os.path.join(dir_path, f"history_{name}.json")

        torch.save(self.model.state_dict(), model_path)
        with open(history_path, "w", encoding="utf-8") as file:
            json.dump(self.history, file, indent=4)
            
        print(f"History and model: {name} was saved in {dir_path}")


    def save_state_criteria(self, criteria_num: int, base_dir: str = "models"):
        dir_path = os.path.join(base_dir, self._model_type, f"criteria{criteria_num}")

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        criterion_str = str(self.criterion)
        model_path = os.path.join(dir_path, "model", f"{criterion_str}.pth")
        history_path = os.path.join(dir_path, "history", f"{criterion_str}.json")

        torch.save(self.model.state_dict(), model_path)
        with open(history_path, "w", encoding="utf-8") as file:
            json.dump(self.history, file, indent=4)
            
        print(f"History and model: {criterion_str} was saved in {dir_path}")

    def save_state_model(
        self,
        params: dict | None = None,
        model_str: str | None = None,
        model_type: str = "dense_hiperparameters",
        base_dir: str = "models",
    ) -> str:
        return save_state_model(
            model=self.model,
            history=self.history,
            params=params,
            model_str=model_str,
            model_type=model_type,
            base_dir=base_dir,
        )


def get_hyperparameter_model_str(
    params: dict | None = None,
    neurons: int | None = None,
    layers: int | None = None,
    expansion_factor: int | None = None,
    dropout_rate: float | None = None,
    negative_slope: float | None = None,
    learning_rate: float | None = None,
    batch_size: int | None = None,
    batch_norm: bool | None = None,
    **kwargs,
) -> str:
    p = params.copy() if params is not None else {}
    p.update(kwargs)

    neu = neurons if neurons is not None else p.get("neurons")
    lay = layers if layers is not None else p.get("layers")
    exp = expansion_factor if expansion_factor is not None else p.get("expansion_factor")
    drop = dropout_rate if dropout_rate is not None else p.get("dropout_rate")
    slope = negative_slope if negative_slope is not None else p.get("negative_slope")
    lr = learning_rate if learning_rate is not None else p.get("learning_rate")
    bs = batch_size if batch_size is not None else p.get("batch_size")
    bn = batch_norm if batch_norm is not None else p.get("batch_norm")

    return f"neu{neu}_lay{lay}_exp{exp}_drop{int(drop*10000)}_slope{int(slope*10000)}_lr{int(lr*10000)}_bs{bs}_bn{bn}"


def save_state_model(
    model: torch.nn.Module,
    history: dict,
    params: dict | None = None,
    model_str: str | None = None,
    model_type: str = "dense_hiperparameters",
    base_dir: str = "models",
) -> str:
    dir_path = os.path.join(base_dir, model_type)
    model_dir = os.path.join(dir_path, "model")
    history_dir = os.path.join(dir_path, "history")

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(history_dir, exist_ok=True)

    if model_str is None:
        p = params if params is not None else history.get("params", history)
        model_str = get_hyperparameter_model_str(p)

    model_path = os.path.join(model_dir, f"{model_str}.pth")
    history_path = os.path.join(history_dir, f"{model_str}.json")

    torch.save(model.state_dict(), model_path)
    with open(history_path, "w", encoding="utf-8") as file:
        json.dump(history, file, indent=4)

    print(f"History and model: {model_str} was saved in {dir_path}")
    return model_str


def load_model(
    model: torch.nn.Module | None = None,
    params: dict | None = None,
    model_str: str | None = None,
    model_type: str = "dense_hiperparameters",
    base_dir: str = "models",
    device: torch.device | str | None = None,
    n_criteria: int | None = None,
    **kwargs,
) -> tuple[torch.nn.Module, dict]:
    target_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dir_path = os.path.join(base_dir, model_type)

    if model_str is None:
        p = params if params is not None else kwargs
        model_str = get_hyperparameter_model_str(p)

    model_path = os.path.join(dir_path, "model", f"{model_str}.pth")
    history_path = os.path.join(dir_path, "history", f"{model_str}.json")

    if not os.path.exists(model_path) or not os.path.exists(history_path):
        raise FileNotFoundError(f"There are no files in {dir_path} for {model_str}")

    with open(history_path, "r", encoding="utf-8") as file:
        history = json.load(file)

    if model is None:
        from model.architecture.dense_monte_carlo import DenseMonteCarloNet
        p = params if params is not None else kwargs
        crit_num = n_criteria or history.get("n_criteria", p.get("n_criteria", 5))
        bn = p.get("batch_norm")
        if bn is None:
            bn = p.get("use_batch_norm", True)
        model = DenseMonteCarloNet(
            n_criteria=crit_num,
            neurons=p.get("neurons"),
            layers=p.get("layers", 3),
            expansion_factor=p.get("expansion_factor", 4),
            dropout_rate=p.get("dropout_rate", 0.2),
            negative_slope=p.get("negative_slope", 0.01),
            use_batch_norm=bn,
        )

    model.load_state_dict(torch.load(model_path, map_location=target_device, weights_only=True))
    model.to(target_device)

    print(f"History and model: {model_str} loaded from {dir_path}")
    return model, history