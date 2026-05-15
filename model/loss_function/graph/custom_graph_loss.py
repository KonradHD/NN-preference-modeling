import torch
import math 
import torch.nn as nn
import torch.nn.functional as F

from model.loss_function.graph.graph_base_loss import GraphBaseLoss


class CustomGraphLoss(GraphBaseLoss):

    def __init__(self, lambda_cop=2.0, lambda_rec=3, lambda_stab=0.5):
        super(CustomGraphLoss, self).__init__(lambda_cop, lambda_rec, lambda_stab)
        self.mse = nn.MSELoss()


    def COP_part(self, logits: torch.Tensor, matrices: torch.Tensor, margin: float = 0.5):
        score_diffs = logits.unsqueeze(2) - logits.unsqueeze(1)
        mask = (matrices > 1).float()
        
        hinge = torch.clamp(-(score_diffs) + margin, min=0.0)
        weighted_loss = hinge * mask
        
        valid_pairs_count = mask.sum()
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

        return (
            COP_loss * self.lambda_cop +
            rec_loss * self.lambda_rec + 
            stab_loss * self.lambda_stab
        )
    

    def __str__(self):
        cop_str = str(int(round(self.lambda_cop * 100)))
        rec_str = str(int(round(self.lambda_rec * 100)))
        stab_str = str(int(round(self.lambda_stab * 100)))
        return f"cop{cop_str}_rec{rec_str}_stab{stab_str}"