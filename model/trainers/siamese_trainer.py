import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

from model.architecture.advanced_siamese import AdvancedSiameseModel
from model.architecture.basic_siamese import BasicSiameseModel
from model.loss_function.siamese.siamese_base_loss import SiameseBaseLoss
from model.trainers.trainer import Trainer

class SiameseTrainer(Trainer):
    def __init__(self, model: AdvancedSiameseModel | BasicSiameseModel, optimizer: torch.optim,
                    criterion: SiameseBaseLoss, device: torch.device):
        super().__init__(model, optimizer, criterion, device)
        self._model_type = "siamese"
        self.history['train_cons'] = []


    def _train_epoch(self, train_dataloader: DataLoader, valid_dataloader: DataLoader) -> tuple[float, float, float]:
        self.model.train()
        train_epoch_loss = 0.0
        train_epoch_cop = 0.0
        train_epoch_rec = 0.0
        train_epoch_stab = 0.0
        train_epoch_cons = 0.0
        
        for batch_idx, (m1_batch, m2_batch, weights_batch) in enumerate(train_dataloader):
            m1_batch = m1_batch.to(self.device)
            m2_batch = m2_batch.to(self.device)
            weights_batch = weights_batch.to(self.device)

            logits1, logits2 = self.model(m1_batch, m2_batch)
            losses = self.criterion(logits1, logits2, weights_batch, m1_batch.squeeze(1), m2_batch.squeeze(1))
            cop, rec, stab, cons = losses
            total_loss = sum(losses)

            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            train_epoch_loss += total_loss.item()
            train_epoch_cop += cop.item()
            train_epoch_rec += rec.item()
            train_epoch_stab += stab.item()
            train_epoch_cons += cons.item()

        avg_train_loss = train_epoch_loss / len(train_dataloader)
        avg_train_cop = train_epoch_cop / len(train_dataloader)
        avg_train_rec = train_epoch_rec / len(train_dataloader)
        avg_train_stab = train_epoch_stab / len(train_dataloader)
        avg_train_cons = train_epoch_cons / len(train_dataloader)
        self.history['train_loss'].append(avg_train_loss)
        self.history['train_cop'].append(avg_train_cop)
        self.history['train_rec'].append(avg_train_rec)
        self.history['train_stab'].append(avg_train_stab)
        self.history['train_cons'].append(avg_train_cons)

        self.model.eval()
        valid_epoch_loss = 0.0
        valid_epoch_mae = 0.0
        
        with torch.no_grad():
            for m1_batch, m2_batch, weights_batch in valid_dataloader:
                m1_batch = m1_batch.to(self.device)
                m2_batch = m2_batch.to(self.device)
                weights_batch = weights_batch.to(self.device)

                logits1, logits2 = self.model(m1_batch, m2_batch)
                
                losses = self.criterion(logits1, logits2, weights_batch, m1_batch.squeeze(1), m2_batch.squeeze(1))
                val_loss = sum(losses)
                valid_epoch_loss += val_loss.item()

                pred_weights = F.softmax(logits1, dim=1)
                mae = torch.abs(pred_weights - weights_batch).mean()
                valid_epoch_mae += mae.item()
                
        avg_valid_loss = valid_epoch_loss / len(valid_dataloader)
        avg_valid_mae = valid_epoch_mae / len(valid_dataloader)
        self.history['valid_loss'].append(avg_valid_loss)
        self.history['valid_mae'].append(avg_valid_mae)

        return avg_train_loss, avg_valid_loss, avg_valid_mae
    

    def predict_single(self, matrix: np.ndarray) -> np.ndarray:
        self.model.eval()

        tensor_matrix = torch.tensor(matrix, dtype=torch.float32).to(self.device)
        tensor_log = torch.log(tensor_matrix + 1e-8)
        tensor_input = tensor_log.unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            logits = self.model.encoder(tensor_input)
            weights = F.softmax(logits, dim=1)

        final_weights = weights.squeeze(0).cpu().numpy()
        return final_weights


    def predict_batch(self, matrices: np.ndarray) -> np.ndarray:
        self.model.eval()

        tensor_matrices = torch.tensor(matrices, dtype=torch.float32).to(self.device)
        tensor_log = torch.log(tensor_matrices + 1e-8)
        tensor_input = tensor_log.unsqueeze(1)

        with torch.no_grad():
            logits = self.model.encoder(tensor_input)
            weights = F.softmax(logits, dim=1)

        final_weights_batch = weights.cpu().numpy()
        return final_weights_batch