from abc import ABC, abstractmethod
import math
from typing import List
import hydra
from omegaconf import DictConfig
import torch
from .strategies import TunerStrategyModule
import torch.nn as nn

class TunerStrategyModule(nn.Module, ABC):
    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def freeze_weights(self, model):
        pass

    @abstractmethod
    def post_init(self, model):
        pass

    @classmethod
    def create(cls, config: DictConfig):
        return hydra.utils.instantiate(config)
    
class LoraModule(TunerStrategyModule):
    def __init__(self, r: int, alpha: float, target_modules: List[str] = ['qkv', 'proj']):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.target_modules = target_modules
        self.lora_layers = nn.ModuleDict()

    def create_lora_layers(self, model: nn.Module):
        for name, module in model.named_modules():
            if any(target in name for target in self.target_modules):
                if isinstance(module, nn.Linear):
                    in_features, out_features = module.in_features, module.out_features
                    self.lora_layers[name] = nn.Sequential(
                        nn.Linear(in_features, self.r, bias=False),
                        nn.Linear(self.r, out_features, bias=False)
                    )
                    # Initialize LoRA layers
                    nn.init.kaiming_uniform_(self.lora_layers[name][0].weight, a=math.sqrt(5))
                    nn.init.zeros_(self.lora_layers[name][1].weight)

    def forward(self, x: torch.Tensor, module_name: str) -> torch.Tensor:
        if module_name not in self.lora_layers:
            raise ValueError(f"LoRA layer for '{module_name}' not found. "
                             f"Available modules: {list(self.lora_layers.keys())}")
        return self.lora_layers[module_name](x) * (self.alpha / self.r)

    def freeze_weights(self, model: nn.Module):
        for name, param in model.named_parameters():
            if not any(target in name for target in self.target_modules):
                param.requires_grad_(False)

    def post_init(self, model: nn.Module):
        self.create_lora_layers(model)

class AdapterModule(TunerStrategyModule):
    def __init__(self, dim: int, adapter_dim: int):
        super().__init__()
        self.dim = dim
        self.adapter_dim = adapter_dim
        self.down_proj = nn.Linear(dim, adapter_dim)
        self.act = nn.GELU()
        self.up_proj = nn.Linear(adapter_dim, dim)
        self.scale = nn.Parameter(torch.ones(dim))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=1e-3)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.down_proj(x)
        x = self.act(x)
        x = self.up_proj(x)
        return residual + (x * self.scale)

    def freeze_weights(self, model: nn.Module):
        for name, param in model.named_parameters():
            if 'adapter' not in name:
                param.requires_grad_(False)

    def post_init(self, model: nn.Module):
        pass 

class ProbingModule(TunerStrategyModule):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        # Initialize probing layers

    def forward(self, x):
        # Implementation for Probing
        pass

    def freeze_weights(self, model: nn.Module):
        for param in model.bitemporal_transformer_blocks.parameters():
            param.requires_grad_(False)

    def post_init(self, model: nn.Module):
        # Any post-initialization steps for probing
        pass

class FinetuneModule(TunerStrategyModule):
    def __init__(self):
        super().__init__()
        # No additional parameters for fine-tuning

    def forward(self, x):
        # Implementation for Fine-tuning (usually just passes through)
        return x