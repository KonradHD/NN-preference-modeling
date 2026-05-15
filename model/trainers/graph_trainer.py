import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

from model.architecture.graph import GraphAHPNet
from model.architecture.basic_siamese import BasicSiameseModel
from model.loss_function.graph.custom_graph_loss import CustomGraphLoss
from model.trainers.trainer import Trainer

class GraphTrainer(Trainer):
    def __init__(self, model: GraphAHPNet, optimizer: torch.optim,
                    criterion: CustomGraphLoss, device: torch.device):
        super().__init__(model, optimizer, criterion, device)
        self._model_type = "graph"


    def _train_epoch(self, train_dataloader: DataLoader, valid_dataloader: DataLoader) -> tuple[float, float, float]:
        self.model.train()
        train_epoch_loss = 0.0
        
        n_criteria = self.model.n
        triu_i, triu_j = torch.triu_indices(n_criteria, n_criteria, offset=1, device=self.device)
        
        for batch_idx, (raw_matrix_batch, weights_batch) in enumerate(train_dataloader):
            raw_matrix_batch = raw_matrix_batch.to(self.device)
            weights_batch = weights_batch.to(self.device)

            row_sums = raw_matrix_batch.sum(dim=2, keepdim=True)
            gnn_matrix_batch = raw_matrix_batch / row_sums
            
            log_matrix = torch.log(raw_matrix_batch + 1e-8)
            loss_matrix = log_matrix[:, triu_i, triu_j]

            logits = self.model(gnn_matrix_batch)
            loss = self.criterion(logits, weights_batch, loss_matrix)
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
            for raw_matrix_batch, weights_batch in valid_dataloader:
                raw_matrix_batch = raw_matrix_batch.to(self.device)
                weights_batch = weights_batch.to(self.device)

                row_sums = raw_matrix_batch.sum(dim=2, keepdim=True)
                gnn_matrix_batch = raw_matrix_batch / row_sums
                
                log_matrix = torch.log(raw_matrix_batch + 1e-8)
                loss_matrix = log_matrix[:, triu_i, triu_j]

                logits = self.model(gnn_matrix_batch)
                
                val_loss = self.criterion(logits, weights_batch, loss_matrix)
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
        row_sums = tensor_matrix.sum(dim=1, keepdim=True)
        norm_matrix = tensor_matrix / row_sums
        tensor_input = norm_matrix.unsqueeze(0)
        
        with torch.no_grad():
            logits = self.model(tensor_input)
            weights = F.softmax(logits, dim=1)

        final_weights = weights.squeeze(0).cpu().numpy()
        return final_weights


    def predict_batch(self, matrices: np.ndarray) -> np.ndarray:
        self.model.eval()

        tensor_matrices = torch.tensor(matrices, dtype=torch.float32).to(self.device)
        row_sums = tensor_matrices.sum(dim=2, keepdim=True)
        norm_matrices = tensor_matrices / row_sums

        with torch.no_grad():
            logits = self.model(norm_matrices)
            weights = F.softmax(logits, dim=1)

        final_weights_batch = weights.cpu().numpy()
        return final_weights_batch