import torch
import math 
import torch.nn as nn
import torch.nn.functional as F

from model.loss_function.graph.graph_base_loss import GraphBaseLoss


class COPOptimizedGraphLoss(GraphBaseLoss):

    def __init__(self, n_criteria: int, lambda_cop=2.0, lambda_rec=3, lambda_stab=0.5):
        super(COPOptimizedGraphLoss, self).__init__(lambda_cop, lambda_rec, lambda_stab)
        self.mse = nn.MSELoss()
        i, j = torch.triu_indices(n_criteria, n_criteria, offset=1)
        self.register_buffer('triu_i', i)
        self.register_buffer('triu_j', j)


    def COP_part(self, logits: torch.Tensor, weights: torch.Tensor, margin=0.5):
        logits_i = logits[:, self.triu_i]
        logits_j = logits[:, self.triu_j]
        pred_diffs = logits_i - logits_j 
        
        weights_i = weights[:, self.triu_i]
        weights_j = weights[:, self.triu_j]
        ideal_diffs = weights_i - weights_j
        
        mask_i_better = (ideal_diffs > 0).float()
        mask_j_better = (ideal_diffs < 0).float()
        
        hinge_i = torch.clamp(-pred_diffs + margin, min=0.0) * mask_i_better
        hinge_j = torch.clamp(pred_diffs + margin, min=0.0) * mask_j_better
        
        weighted_loss = hinge_i + hinge_j
        valid_pairs_count = mask_i_better.sum() + mask_j_better.sum()
        if valid_pairs_count > 0:
            return weighted_loss.sum() / valid_pairs_count
        else:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

    
    def reconstruction_part(self, logits: torch.Tensor, weights: torch.Tensor):
        pred_weights = F.softmax(logits, dim=1)
        return self.mse(pred_weights, weights)
    

    def stability_part(self, logits: torch.Tensor):
        n = logits.shape[1]
        weights = F.softmax(logits, dim=1)
        log_weights = F.log_softmax(logits, dim=1)
        log_n = math.log(n)
        
        # ln(w) + ln(n)) ->  w * ln(w * n)
        kl = weights * (log_weights + log_n)
        return kl.sum(dim=1).mean()


    def forward(self, logits: torch.Tensor, weights: torch.Tensor, matrices: torch.Tensor):
        COP_loss = self.COP_part(logits, matrices)
        rec_loss = self.reconstruction_part(logits, weights)
        stab_loss = self.stability_part(logits)

        total_loss =  (
            COP_loss * self.lambda_cop +
            rec_loss * self.lambda_rec + 
            stab_loss * self.lambda_stab
        )
        return (self.lambda_cop * COP_loss, self.lambda_rec * rec_loss, self.lambda_stab * stab_loss)


    def __str__(self):
        cop_str = str(int(round(self.lambda_cop * 100)))
        rec_str = str(int(round(self.lambda_rec * 100)))
        stab_str = str(int(round(self.lambda_stab * 100)))
        return f"cop{cop_str}_rec{rec_str}_stab{stab_str}"