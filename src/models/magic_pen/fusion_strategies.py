from abc import ABC, abstractmethod

import hydra
from omegaconf import DictConfig
from .strategies import FusionStrategyModule
import torch.nn as nn

class FusionStrategyModule(nn.Module, ABC):
    @abstractmethod
    def forward(self, x):
        pass

    @classmethod
    def create(cls, config: DictConfig):
        return hydra.utils.instantiate(config)


class DiffFusionModule(FusionStrategyModule):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        # Implementation for difference fusion
        pass

class ConcatFusionModule(FusionStrategyModule):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        # Implementation for concatenation fusion
        pass

class CrossAttentionFusionModule(FusionStrategyModule):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        # Initialize cross-attention layers

    def forward(self, x):
        # Implementation for cross-attention fusion
        pass