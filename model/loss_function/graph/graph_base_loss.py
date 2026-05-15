import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class GraphBaseLoss(nn.Module, ABC):

    def __init__(self, lambda_cop, lambda_rec, lambda_stab):
        super(GraphBaseLoss, self).__init__()
        self.lambda_cop = lambda_cop
        self.lambda_rec = lambda_rec
        self.lambda_stab = lambda_stab


    @abstractmethod
    def forward(self, logits: torch.Tensor, weights: torch.Tensor, matrix: torch.Tensor):
        raise NotImplementedError()


    @abstractmethod
    def COP_part(self, logits: torch.Tensor, weights: torch.Tensor, margin: float):
        raise NotImplementedError()
    

    @abstractmethod
    def reconstruction_part(self, logits: torch.Tensor, weights: torch.Tensor):
        raise NotImplementedError()
    

    @abstractmethod
    def stability_part(self, logits: torch.Tensor):
        raise NotImplementedError()
    

    @abstractmethod
    def __str__(self):
        raise NotImplementedError()
