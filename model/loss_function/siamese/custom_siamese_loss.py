import torch
import torch.nn as nn
import torch.nn.functional as F
from model.loss_function.siamese.siamese_base_loss import SiameseBaseLoss
import math

class CustomSiameseLoss(SiameseBaseLoss):
    def __init__(self, lambda_cop=2.0, lambda_rec=5.0, lambda_stab=0.05, lambda_cons=0.5):
        super(CustomSiameseLoss, self).__init__(lambda_cop, lambda_rec, lambda_stab, lambda_cons)
        self.mse = nn.MSELoss()


    def COP_part(self, logits: torch.Tensor, matrices: torch.Tensor, margin=0.5):
        score_diffs = logits.unsqueeze(2) - logits.unsqueeze(1)
        mask = (matrices > 1).float()
        
        hinge = torch.clamp(-(score_diffs) + margin, min=0.0)
        weighted_loss = hinge * mask
        
        valid_pairs_count = mask.sum()
        if valid_pairs_count > 0:
            return weighted_loss.sum() / valid_pairs_count
        else:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)


    def stability_part(self, logits1: torch.Tensor, logits2: torch.Tensor):
        n = logits1.shape[1]
        weights1 = F.softmax(logits1, dim=1)
        log_weights1 = F.log_softmax(logits1, dim=1)
        
        weights2 = F.softmax(logits2, dim=1)
        log_weights2 = F.log_softmax(logits2, dim=1)
        
        log_n = math.log(n)
        
        # ln(w) + ln(n)) ->  w * ln(w * n)
        kl1 = weights1 * (log_weights1 + log_n)
        kl2 = weights2 * (log_weights2 + log_n)
        
        loss_kl1 = kl1.sum(dim=1).mean()
        loss_kl2 = kl2.sum(dim=1).mean()
        
        return (loss_kl1 + loss_kl2) / 2


    # TODO: reconstruction powinien sprawdzać dopasowanie wektora wag do macierzy, a to powinno być supervised_part
    def reconstruction_part(self, logits1: torch.Tensor, logits2: torch.Tensor, weights: torch.Tensor):
        pred_weights1 = F.softmax(logits1, dim=1)
        pred_weights2 = F.softmax(logits2, dim=1)

        mse1 = self.mse(pred_weights1, weights)
        mse2 = self.mse(pred_weights2, weights)

        return (mse1 + mse2) / 2.0


    def consistency_part(self, logits1: torch.Tensor, logits2: torch.Tensor):
        return self.mse(logits1, logits2)


    def forward(self, logits1: torch.Tensor, logits2: torch.Tensor, 
                weights: torch.Tensor, base_matrices: torch.Tensor, comparison_matrices: torch.Tensor):
        loss_cop1 = self.COP_part(logits1, base_matrices)
        loss_cop2 = self.COP_part(logits2, comparison_matrices)
        loss_cop = (loss_cop1 + loss_cop2) / 2
        
        loss_rec = self.reconstruction_part(logits1, logits2, weights)
        loss_stab = self.stability_part(logits1, logits2)
        loss_cons = self.consistency_part(logits1, logits2)
        
        total_loss = (self.lambda_cop * loss_cop + 
                      self.lambda_rec * loss_rec + 
                      self.lambda_stab * loss_stab + 
                      self.lambda_cons * loss_cons)
        
        return (self.lambda_cop * loss_cop, self.lambda_rec * loss_rec, self.lambda_stab * loss_stab, self.lambda_cons * loss_cons)


    def __str__(self):
        cop_str = str(int(round(self.lambda_cop * 100)))
        rec_str = str(int(round(self.lambda_rec * 100)))
        stab_str = str(int(round(self.lambda_stab * 100)))
        cons_str = str(int(round(self.lambda_cons * 100)))
        return f"cop{cop_str}_rec{rec_str}_stab{stab_str}_cons{cons_str}"