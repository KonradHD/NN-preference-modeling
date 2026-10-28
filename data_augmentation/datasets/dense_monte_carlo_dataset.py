from __future__ import annotations

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

from utils.phase import Phase
from data_augmentation.monte_carlo_loader import MonteCarloLoader


class DenseMonteCarloDataset(Dataset):
    def __init__(
        self,
        num_criteria: int | None = None,
        phase: Phase | str | None = None,
        matrices: np.ndarray | torch.Tensor | None = None,
        weights: np.ndarray | torch.Tensor | None = None,
        strata: list[str] | None = None,
        max_per_stratum: int | None = None,
        alpha: float | None = None,
        sigma: float | None = None,
        base_dir: str | None = None,
        augment: bool = False,
    ):
        if matrices is not None and weights is not None:
            if isinstance(matrices, np.ndarray):
                self.matrices = torch.from_numpy(matrices).float()
            else:
                self.matrices = matrices.float()

            if isinstance(weights, np.ndarray):
                self.targets = torch.from_numpy(weights).float()
            else:
                self.targets = weights.float()

            if self.matrices.dim() != 3 or self.matrices.shape[1] != self.matrices.shape[2]:
                raise ValueError("matrices must be of shape (N, n, n)")
            if self.targets.dim() != 2 or self.matrices.shape[0] != self.targets.shape[0]:
                raise ValueError("weights shape must match matrices count")
            if self.matrices.shape[1] != self.targets.shape[1]:
                raise ValueError("weights criteria count must match matrix dimensions")

            self.num_criteria = self.matrices.shape[1]
        else:
            if num_criteria is None or phase is None:
                raise ValueError("Either provide (matrices, weights) or both (num_criteria, phase).")

            if isinstance(phase, str):
                phase_normalized = phase.lower().strip()
                if phase_normalized in ("val", "valid", "validation"):
                    phase = Phase.VALIDATION
                elif phase_normalized == "train":
                    phase = Phase.TRAIN
                elif phase_normalized == "test":
                    phase = Phase.TEST
                else:
                    phase = Phase(phase)

            loader = MonteCarloLoader(base_dir=base_dir)
            loaded_matrices, loaded_weights = loader.load(
                phase=phase,
                num_criteria=num_criteria,
                strata=strata,
                max_per_stratum=max_per_stratum,
                alpha=alpha,
                sigma=sigma,
            )

            self.matrices = torch.from_numpy(loaded_matrices).float()
            self.targets = torch.from_numpy(loaded_weights).float()
            self.num_criteria = num_criteria

        self.weights = self.targets
        self.augment = augment
        self.triu_i, self.triu_j = torch.triu_indices(self.num_criteria, self.num_criteria, offset=1)

    def __len__(self) -> int:
        return len(self.matrices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        matrix = self.matrices[idx]
        weights = self.targets[idx]

        if self.augment:
            n = matrix.shape[0]
            perm = torch.randperm(n)
            matrix = matrix[perm, :][:, perm]
            weights = weights[perm]

        matrix = torch.log(matrix + 1e-8)
        upper_triangle = matrix[self.triu_i, self.triu_j]
        return upper_triangle, weights

    def get_raw(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.matrices[idx], self.targets[idx]

    def get_dataloader(
        self,
        batch_size: int = 1024,
        shuffle: bool | None = None,
        **kwargs,
    ) -> DataLoader:
        if shuffle is None:
            shuffle = self.augment
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, **kwargs)
