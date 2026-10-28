from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.loss_function.dense_monte_carlo.MC_base_loss import MCBaseLoss


class CustomDenseMCLoss(MCBaseLoss):
    def __init__(
        self,
        n_criteria: int,
        lambda_cop: float = 2.0,
        lambda_fit: float = 3.0,
        lambda_stab: float = 0.1,
        margin: float = 0.05,
        p: int = 1,
        cop_mode: str = "margin",
        stab_mode: str = "kullback",
        entropy_threshold: float | None = None,
        normalize_initial_loss: bool = False,
    ):
        super(CustomDenseMCLoss, self).__init__(lambda_cop, lambda_fit, lambda_stab)
        self.n_criteria = n_criteria
        self.lambda_fit = lambda_fit
        self.margin = margin
        self.p = p
        self.cop_mode = cop_mode
        self.stab_mode = stab_mode
        self.entropy_threshold = entropy_threshold
        self.normalize_initial_loss = normalize_initial_loss

        i, j = torch.triu_indices(n_criteria, n_criteria, offset=1)
        self.register_buffer("triu_i", i)
        self.register_buffer("triu_j", j)

        self.register_buffer("initial_cop", torch.tensor(1.0))
        self.register_buffer("initial_fit", torch.tensor(1.0))
        self.register_buffer("initial_stab", torch.tensor(1.0))
        self.has_initial_losses = False

    def set_initial_losses(self, cop_0: float, fit_0: float, stab_0: float) -> None:
        self.initial_cop.fill_(max(float(cop_0), 1e-8))
        self.initial_fit.fill_(max(float(fit_0), 1e-8))
        self.initial_stab.fill_(max(float(stab_0), 1e-8))
        self.has_initial_losses = True

    def reset_initial_losses(self) -> None:
        self.initial_cop.fill_(1.0)
        self.initial_fit.fill_(1.0)
        self.initial_stab.fill_(1.0)
        self.has_initial_losses = False

    def _extract_upper_triangle(self, matrices: torch.Tensor) -> torch.Tensor:
        if matrices.dim() == 3:
            return torch.log(matrices + 1e-8)[:, self.triu_i, self.triu_j]
        elif matrices.dim() == 2:
            if matrices.shape[1] == len(self.triu_i):
                return matrices
            elif matrices.shape[0] == matrices.shape[1] == self.n_criteria:
                return torch.log(matrices + 1e-8)[self.triu_i, self.triu_j].unsqueeze(0)
            else:
                return matrices
        elif matrices.dim() == 1:
            return matrices.unsqueeze(0)
        return matrices

    def COP_part(
        self,
        logits: torch.Tensor,
        matrices: torch.Tensor,
        margin: float | None = None,
    ) -> torch.Tensor:
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)

        ln_a = self._extract_upper_triangle(matrices)
        logits_i = logits[:, self.triu_i]
        logits_j = logits[:, self.triu_j]
        diffs = logits_i - logits_j

        mask_i_better = (ln_a > 1e-8).float()
        mask_j_better = (ln_a < -1e-8).float()
        valid_pairs_count = mask_i_better.sum() + mask_j_better.sum()

        if self.cop_mode == "bradley":
            cost_i = F.softplus(-diffs) * mask_i_better
            cost_j = F.softplus(diffs) * mask_j_better
            if valid_pairs_count > 0:
                return (cost_i + cost_j).sum() / valid_pairs_count
            return (logits * 0.0).sum()

        if self.cop_mode == "margin":
            m = margin if margin is not None else self.margin
            hinge_i = torch.clamp(-diffs + m, min=0.0) * mask_i_better
            hinge_j = torch.clamp(diffs + m, min=0.0) * mask_j_better
            weighted_loss = hinge_i + hinge_j

            if valid_pairs_count > 0:
                return weighted_loss.sum() / valid_pairs_count
            return (logits * 0.0).sum()

    def fit_part(
        self,
        logits: torch.Tensor,
        matrices: torch.Tensor,
    ) -> torch.Tensor:
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)

        ln_a = self._extract_upper_triangle(matrices)
        logits_i = logits[:, self.triu_i]
        logits_j = logits[:, self.triu_j]
        diffs = logits_i - logits_j
        errors = diffs - ln_a

        if self.p == 1:
            return torch.abs(errors).mean()
        elif self.p == 2:
            return torch.pow(errors, 2).mean()
        return torch.pow(torch.abs(errors), self.p).mean()


    def stability_part(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)

        weights = F.softmax(logits, dim=-1)
        log_weights = F.log_softmax(logits, dim=-1)
        log_n = math.log(self.n_criteria)

        if self.stab_mode == "kullback":
            kl_per_sample = (weights * (log_weights + log_n)).sum(dim=-1)
        elif self.stab_mode == "entropy": # implement
            kl_per_sample = -(weights * log_weights).sum(dim=-1)

        if self.entropy_threshold is not None:
            entropy_per_sample = -(weights * log_weights).sum(dim=-1)
            mask = (entropy_per_sample < self.entropy_threshold).float()
            mask_sum = mask.sum()
            if mask_sum > 0:
                return (kl_per_sample * mask).sum() / mask_sum
            return (logits * 0.0).sum()

        return kl_per_sample.mean()

    def forward(
        self,
        logits: torch.Tensor,
        weights: torch.Tensor,
        matrices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        fit_loss = self.fit_part(logits, matrices)
        cop_loss = self.COP_part(logits, matrices, self.margin)
        stab_loss = self.stability_part(logits)

        if self.normalize_initial_loss:
            if not self.has_initial_losses:
                self.set_initial_losses(
                    cop_0=cop_loss.detach().item(),
                    fit_0=fit_loss.detach().item(),
                    stab_0=stab_loss.detach().item(),
                )
            fit_loss = fit_loss / self.initial_fit
            cop_loss = cop_loss / self.initial_cop
            stab_loss = stab_loss / self.initial_stab

        weighted_cop = self.lambda_cop * cop_loss
        weighted_fit = self.lambda_fit * fit_loss
        weighted_stab = self.lambda_stab * stab_loss

        return (weighted_cop, weighted_fit, weighted_stab)

    def __str__(self) -> str:
        cop_str = str(int(round(self.lambda_cop * 100)))
        fit_str = str(int(round(self.lambda_fit * 100)))
        stab_str = str(int(round(self.lambda_stab * 100)))
        margin_str = str(int(round(self.margin * 100)))
        cop_letter = str(self.cop_mode[0])
        stab_letter = str(self.stab_mode[0])
        name = f"cop{cop_str}{cop_letter}_fit{fit_str}_stab{stab_str}{stab_letter}_margin{margin_str}_p{self.p}"
        if self.normalize_initial_loss:
            name += "_initnorm"
        return name

    def get_config(self) -> dict:
        return {
            "lambda_cop": self.lambda_cop,
            "lambda_fit": self.lambda_fit,
            "lambda_stab": self.lambda_stab,
            "margin": self.margin,
            "p": self.p,
            "cop_mode": self.cop_mode,
            "stab_mode": self.stab_mode,
            "entropy_threshold": self.entropy_threshold,
            "normalize_initial_loss": self.normalize_initial_loss,
        }


CustomDenseMonteCarloLoss = CustomDenseMCLoss
