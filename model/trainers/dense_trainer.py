import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

from model.architecture.dense import DeepAHPNet
from model.architecture.basic_siamese import BasicSiameseModel
from model.loss_function.dense.custom_dense_loss import CustomDenseLoss
from model.trainers.trainer import Trainer

class DenseTrainer(Trainer):
    def __init__(self, model: DeepAHPNet, optimizer: torch.optim,
                    criterion: CustomDenseLoss, device: torch.device):
        super().__init__(DenseTrainer, model, optimizer, criterion, device)
        self._model_type = "dense"


    def _train_epoch(self, train_dataloader: DataLoader, valid_dataloader: DataLoader) -> tuple[float, float, float]:
        self.model.train()
        train_epoch_loss = 0.0
        
        for batch_idx, (matrix_batch, weights_batch) in enumerate(train_dataloader):
            matrix_batch = matrix_batch.to(self.device)
            weights_batch = weights_batch.to(self.device)

            logits = self.model(matrix_batch)
            loss = self.criterion(logits, weights_batch, matrix_batch)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            train_epoch_loss += loss.item()

        avg_train_loss = train_epoch_loss / len(train_dataloader)
        self.history['train_loss'].append(avg_train_loss)

        self.model.eval()
        valid_epoch_loss = 0.0
        valid_epoch_mae = 0.0
        
        with torch.no_grad():
            for matrix_batch, weights_batch in valid_dataloader:
                matrix_batch = matrix_batch.to(self.device)
                weights_batch = weights_batch.to(self.device)

                logits = self.model(matrix_batch)
                
                val_loss = self.criterion(logits, weights_batch, matrix_batch)
                valid_epoch_loss += val_loss.item()

                pred_weights = F.softmax(logits, dim=1)
                mae = torch.abs(pred_weights - weights_batch).mean()
                valid_epoch_mae += mae.item()
                
        avg_valid_loss = valid_epoch_loss / len(valid_dataloader)
        avg_valid_mae = valid_epoch_mae / len(valid_dataloader)
        self.history['valid_loss'].append(avg_valid_loss)
        self.history["valid_mae"].append(avg_valid_mae)

        return avg_train_loss, avg_valid_loss, avg_valid_mae


    def predict_single(self, matrix: np.ndarray) -> np.ndarray:
        self.model.eval()

        tensor_matrix = torch.tensor(matrix, dtype=torch.float32).to(self.device)
        tensor_log = torch.log(tensor_matrix + 1e-8)

        n_criteria = matrix.shape[0]
        triu_i, triu_j = torch.triu_indices(n_criteria, n_criteria, offset=1)
        upper_triangle = tensor_log[triu_i, triu_j]

        tensor_input = upper_triangle.unsqueeze(0)
        with torch.no_grad():
            logits = self.model(tensor_input)
            weights = F.softmax(logits, dim=1)

        final_weights = weights.squeeze(0).cpu().numpy()
        return final_weights


    def predict_batch(self, matrices: np.ndarray) -> np.ndarray:
        self.model.eval()

        batch_size, n_criteria, _ = matrices.shape
        triu_i, triu_j = torch.triu_indices(n_criteria, n_criteria, offset=1)

        tensor_matrices = torch.tensor(matrices, dtype=torch.float32).to(self.device)
        tensor_log = torch.log(tensor_matrices + 1e-8)
        tensor_input = tensor_log[:, triu_i, triu_j]

        with torch.no_grad():
            logits = self.model(tensor_input)
            weights = F.softmax(logits, dim=1)

        final_weights_batch = weights.cpu().numpy()
        return final_weights_batch