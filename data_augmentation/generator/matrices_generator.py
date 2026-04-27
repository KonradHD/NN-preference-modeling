from abc import ABC
import os
from abc import abstractmethod
import numpy as np


class MatricesGenerator(ABC):
    def __init__(self, num_matrices: int, num_criteria: int, dir: str | None = None):
        self.num_matrices = num_matrices
        self.num_criteria = num_criteria
        self._dir = dir
        if self._dir is None:
            self._dir = os.path.join('data', "synthetic_data")
        
        os.makedirs(self._dir, exist_ok=True)


    @abstractmethod
    def generate_uniform(self) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError()
    
    
    @abstractmethod
    def generate_dirichlet(self, alpha: list[float]) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError()
    

    @abstractmethod
    def save_state(self, is_uniform: bool, prefix: str = ""):
        raise NotImplementedError()