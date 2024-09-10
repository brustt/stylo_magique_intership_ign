from abc import ABC, abstractmethod
from models.magic_pen.fusion_strategies import ConcatFusionModule, CrossAttentionFusionModule, DiffFusionModule
from models.magic_pen.tuner_strategies import AdapterModule, FinetuneModule, LoraModule, ProbingModule
import torch.nn as nn
from omegaconf import DictConfig
from hydra.utils import instantiate

FUSION_STRATEGY_MAP = {
    'diff': DiffFusionModule,
    'concat': ConcatFusionModule,
    'cross_attention': CrossAttentionFusionModule,
}

TUNER_STRATEGY_MAP = {
    'lora': LoraModule,
    'adapter': AdapterModule,
    'probing': ProbingModule,
    'finetune': FinetuneModule,
}
class FusionStrategyModule(nn.Module, ABC):
    @abstractmethod
    def forward(self, x):
        pass

    @classmethod
    def create(cls, config: DictConfig):
        return instantiate(config)

class TunerStrategyModule(nn.Module, ABC):
    @abstractmethod
    def forward(self, x):
        pass

    @classmethod
    def create(cls, config: DictConfig):
        return instantiate(config)