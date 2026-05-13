import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class SiameseBaseLoss(nn.Module, ABC):

    def __init__(self, lambda_cop, lambda_rec, lambda_stab, lambda_cons):
        super(SiameseBaseLoss, self).__init__()
        self.lambda_cop = lambda_cop
        self.lambda_rec = lambda_rec
        self.lambda_stab = lambda_stab
        self.lambda_cons = lambda_cons


    @abstractmethod
    def forward(self, logits1: torch.Tensor, logits2: torch.Tensor, 
                weights: torch.Tensor, base_matrices: torch.Tensor, comparison_matrices: torch.Tensor):
        raise NotImplementedError()


    @abstractmethod
    def COP_part(self, logits: torch.Tensor, matrix: torch.Tensor, margin: float):
        raise NotImplementedError()
    

    @abstractmethod
    def reconstruction_part(self, logits1: torch.Tensor, logits2: torch.Tensor, weights: torch.Tensor):
        raise NotImplementedError()
    

    @abstractmethod
    def stability_part(self, logits1: torch.Tensor, logits2: torch.Tensor):
        raise NotImplementedError()
    
    @abstractmethod
    def consistency_part(self, logits1: torch.Tensor, logits2: torch.Tensor):
        raise NotImplementedError()