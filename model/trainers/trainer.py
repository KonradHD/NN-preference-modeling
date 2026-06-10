from abc import ABC
from abc import abstractmethod 
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import copy
import numpy as np
import os 
import json

from model.architecture.siamese_residual_block import AdvancedSiameseModel
from model.architecture.basic_siamese import BasicSiameseModel
from model.loss_function.siamese.siamese_base_loss import SiameseBaseLoss


class Trainer(ABC):
    def __init__(self, model: torch.nn.Module, optimizer: torch.optim,
                    criterion: SiameseBaseLoss, device: torch.device):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.history = {'train_loss': [], 'valid_loss': [], 'valid_mae': [], 'train_cop': [], 'train_rec': [], 'train_stab': []}
        self._model_type: str = "unknown"


    @abstractmethod
    def _train_epoch(self, train_dataloader: DataLoader, valid_dataloader: DataLoader) -> tuple[float, float, float]:
        raise NotImplementedError()
    

    def train(self, train_dataloader: DataLoader, valid_dataloader: DataLoader, epochs: int = 500, 
              early_stopping: bool = True, es_patience: int = 15, reduce_lr_on_plateau: bool = True, rlr_patience: int = 5) -> tuple[torch.nn.Module, dict[str, float]]:
        
        best_valid_loss = float('inf')
        best_model_state = None
        epochs_no_improve = 0

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

            if avg_valid_loss < best_valid_loss:
                best_valid_loss = avg_valid_loss
                best_model_state = copy.deepcopy(self.model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if (epoch + 1) % 10 == 0:
                print(f"Epoch number [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.6f} | Valid Loss: {avg_valid_loss:.6f} | Valid MAE: {avg_valid_mae:.4f}")

            if early_stopping and epochs_no_improve >= es_patience:
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