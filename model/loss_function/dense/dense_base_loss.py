import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class DenseBaseLoss(nn.Module, ABC):

    def __init__(self):
        super(DenseBaseLoss, self).__init__()


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