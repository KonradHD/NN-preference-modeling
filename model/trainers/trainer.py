from abc import ABC
from abc import abstractmethod 
import torch
from torch.utils.data import DataLoader
import copy
import numpy as np
import os 
import json

from model.architecture.advanced_siamese import AdvancedSiameseModel
from model.architecture.basic_siamese import BasicSiameseModel
from model.loss_function.siamese.siamese_base_loss import SiameseBaseLoss


class Trainer(ABC):
    def __init__(self, model: torch.nn.Module, optimizer: torch.optim,
                    criterion: SiameseBaseLoss, device: torch.device):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.history = {'train_loss': [], 'valid_loss': [], 'valid_mae': []}
        self._model_type: str = "unknown"


    @abstractmethod
    def _train_epoch(self, train_dataloader: DataLoader, valid_dataloader: DataLoader) -> tuple[float, float, float]:
        raise NotImplementedError()
    

    def train(self, train_dataloader: DataLoader, valid_dataloader: DataLoader, epochs: int = 500, 
              early_stopping: bool = True, patience: int = 15) -> tuple[torch.nn.Module, dict[str, float]]:
        
        best_valid_loss = float('inf')
        best_model_state = None
        epochs_no_improve = 0

        for epoch in range(epochs):
            avg_train_loss, avg_valid_loss, avg_valid_mae = self._train_epoch(train_dataloader, valid_dataloader)

            if avg_valid_loss < best_valid_loss:
                best_valid_loss = avg_valid_loss
                best_model_state = copy.deepcopy(self.model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if (epoch + 1) % 10 == 0:
                print(f"Epoch number [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.6f} | Valid Loss: {avg_valid_loss:.6f} | Valid MAE: {avg_valid_mae:.4f}")

            if early_stopping and epochs_no_improve >= patience:
                print(f"\nEARLY STOPPING in epoch {epoch + 1}.")
                break

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
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