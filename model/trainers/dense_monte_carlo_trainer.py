from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

from model.loss_function.dense_monte_carlo.MC_base_loss import MCBaseLoss
from model.trainers.trainer import Trainer


class DenseMonteCarloTrainer(Trainer):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: MCBaseLoss,
        device: torch.device,
    ):
        super().__init__(model, optimizer, criterion, device)
        self._model_type = "dense_monte_carlo"

    def init_loss_normalization(
        self,
        train_dataloader: DataLoader,
        max_batches: int = 50,
    ) -> dict[str, float]:
        if not getattr(self.criterion, "normalize_initial_loss", False):
            return {}

        self.model.eval()
        total_cop = 0.0
        total_fit = 0.0
        total_stab = 0.0
        batch_count = 0

        with torch.no_grad():
            for idx, (matrix_batch, weights_batch) in enumerate(train_dataloader):
                if idx >= max_batches:
                    break
                matrix_batch = matrix_batch.to(self.device)

                logits = self.model(matrix_batch)
                raw_fit = self.criterion.fit_part(logits, matrix_batch).item()
                raw_cop = self.criterion.COP_part(logits, matrix_batch, self.criterion.margin).item()
                raw_stab = self.criterion.stability_part(logits).item()

                total_cop += raw_cop
                total_fit += raw_fit
                total_stab += raw_stab
                batch_count += 1

        if batch_count > 0:
            init_cop = total_cop / batch_count
            init_fit = total_fit / batch_count
            init_stab = total_stab / batch_count
            self.criterion.set_initial_losses(init_cop, init_fit, init_stab)
            self.history["initial_losses"] = {
                "cop": init_cop,
                "fit": init_fit,
                "stab": init_stab,
            }
            print(f"Initial loss normalization computed over {batch_count} batches:")
            print(f"  COP_0 = {init_cop:.6f} | FIT_0 = {init_fit:.6f} | STAB_0 = {init_stab:.6f}")
            return self.history["initial_losses"]
        return {}

    def train(
        self,
        train_dataloader: DataLoader,
        valid_dataloader: DataLoader,
        epochs: int = 500,
        early_stopping: bool = True,
        es_patience: int = 15,
        reduce_lr_on_plateau: bool = True,
        rlr_patience: int = 5,
    ) -> tuple[torch.nn.Module, dict[str, float]]:
        if getattr(self.criterion, "normalize_initial_loss", False) and not getattr(self.criterion, "has_initial_losses", False):
            self.init_loss_normalization(train_dataloader)
        return super().train(
            train_dataloader=train_dataloader,
            valid_dataloader=valid_dataloader,
            epochs=epochs,
            early_stopping=early_stopping,
            es_patience=es_patience,
            reduce_lr_on_plateau=reduce_lr_on_plateau,
            rlr_patience=rlr_patience,
        )

    def _train_epoch(
        self,
        train_dataloader: DataLoader,
        valid_dataloader: DataLoader,
    ) -> tuple[float, float, float]:
        self.model.train()
        train_epoch_loss = 0.0
        train_epoch_cop = 0.0
        train_epoch_fit = 0.0
        train_epoch_stab = 0.0

        for batch_idx, (matrix_batch, weights_batch) in enumerate(train_dataloader):
            matrix_batch = matrix_batch.to(self.device)
            weights_batch = weights_batch.to(self.device)

            logits = self.model(matrix_batch)
            losses = self.criterion(logits, weights_batch, matrix_batch)

            if isinstance(losses, (tuple, list)):
                cop, fit, stab = losses
                total_loss = sum(losses)
                train_epoch_cop += cop.item()
                train_epoch_fit += fit.item()
                train_epoch_stab += stab.item()
            else:
                total_loss = losses

            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            train_epoch_loss += total_loss.item()

        avg_train_loss = train_epoch_loss / len(train_dataloader)
        avg_train_cop = train_epoch_cop / len(train_dataloader)
        avg_train_fit = train_epoch_fit / len(train_dataloader)
        avg_train_stab = train_epoch_stab / len(train_dataloader)

        self.history["train_loss"].append(avg_train_loss)
        self.history["train_cop"].append(avg_train_cop)
        self.history["train_fit"].append(avg_train_fit)
        self.history["train_stab"].append(avg_train_stab)

        self.model.eval()
        valid_epoch_loss = 0.0
        valid_epoch_mae = 0.0

        with torch.no_grad():
            for matrix_batch, weights_batch in valid_dataloader:
                matrix_batch = matrix_batch.to(self.device)
                weights_batch = weights_batch.to(self.device)

                logits = self.model(matrix_batch)
                val_losses = self.criterion(logits, weights_batch, matrix_batch)

                if isinstance(val_losses, (tuple, list)):
                    val_loss = sum(val_losses)
                else:
                    val_loss = val_losses

                valid_epoch_loss += val_loss.item()

                pred_weights = F.softmax(logits, dim=-1)
                mae = torch.abs(pred_weights - weights_batch).mean()
                valid_epoch_mae += mae.item()

        avg_valid_loss = valid_epoch_loss / len(valid_dataloader)
        avg_valid_mae = valid_epoch_mae / len(valid_dataloader)

        self.history["valid_loss"].append(avg_valid_loss)
        self.history["valid_mae"].append(avg_valid_mae)

        return avg_train_loss, avg_valid_loss, avg_valid_mae

    def predict_single(self, matrix: np.ndarray) -> np.ndarray:
        self.model.eval()

        if matrix.ndim == 1:
            tensor_input = torch.tensor(matrix, dtype=torch.float32).unsqueeze(0).to(self.device)
        elif matrix.ndim == 2:
            n_criteria = matrix.shape[0]
            triu_i, triu_j = torch.triu_indices(n_criteria, n_criteria, offset=1)
            tensor_matrix = torch.tensor(matrix, dtype=torch.float32).to(self.device)
            tensor_log = torch.log(tensor_matrix + 1e-8)
            upper_triangle = tensor_log[triu_i, triu_j]
            tensor_input = upper_triangle.unsqueeze(0)
        else:
            raise ValueError(f"Expected 1D or 2D array, got shape {matrix.shape}")

        with torch.no_grad():
            logits = self.model(tensor_input)
            weights = F.softmax(logits, dim=-1)

        return weights.squeeze(0).cpu().numpy()

    def predict_batch(self, matrices: np.ndarray) -> np.ndarray:
        self.model.eval()

        if matrices.ndim == 2:
            tensor_input = torch.tensor(matrices, dtype=torch.float32).to(self.device)
        elif matrices.ndim == 3:
            _, n_criteria, _ = matrices.shape
            triu_i, triu_j = torch.triu_indices(n_criteria, n_criteria, offset=1)
            tensor_matrices = torch.tensor(matrices, dtype=torch.float32).to(self.device)
            tensor_log = torch.log(tensor_matrices + 1e-8)
            tensor_input = tensor_log[:, triu_i, triu_j]
        else:
            raise ValueError(f"Expected 2D or 3D array, got shape {matrices.shape}")

        with torch.no_grad():
            logits = self.model(tensor_input)
            weights = F.softmax(logits, dim=-1)

        return weights.cpu().numpy()

    def evaluate(self, dataloader: DataLoader) -> tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        total_mae = 0.0

        with torch.no_grad():
            for matrix_batch, weights_batch in dataloader:
                matrix_batch = matrix_batch.to(self.device)
                weights_batch = weights_batch.to(self.device)

                logits = self.model(matrix_batch)
                losses = self.criterion(logits, weights_batch, matrix_batch)

                if isinstance(losses, (tuple, list)):
                    loss = sum(losses)
                else:
                    loss = losses

                total_loss += loss.item()
                pred_weights = F.softmax(logits, dim=-1)
                mae = torch.abs(pred_weights - weights_batch).mean()
                total_mae += mae.item()

        avg_loss = total_loss / len(dataloader)
        avg_mae = total_mae / len(dataloader)
        return avg_loss, avg_mae


DenseMCTrainer = DenseMonteCarloTrainer
