from abc import ABC, abstractmethod
import math
from typing import List
import hydra
from models.lestylonet.blocks import Attention, BitemporalTransformerBlock
from omegaconf import DictConfig
import torch
import torch.nn as nn

class TunerStrategyModule(nn.Module, ABC):
    @abstractmethod
    def forward(self, x):
        pass

    def freeze_weights(self, model: nn.Module):
        for name, param in model.bitemporal_transformer_blocks.named_parameters():
            if not any(target in name for target in self.target_modules):
                param.requires_grad_(False)

    @abstractmethod
    def post_init(self, model):
        pass

    @classmethod
    def create(cls, config: DictConfig):
        return hydra.utils.instantiate(config)
    
class LoraModule(TunerStrategyModule):
    def __init__(self, r: int, alpha: float, target_modules: List[str] = ['q', 'k', 'v']):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.target_modules = target_modules
        self.lora_layers = nn.ModuleDict()

    def create_lora_layers(self, model: nn.Module):
        for layer_id, module in enumerate(model.bitemporal_transformer_blocks):
            if isinstance(module.attn, Attention):
                for target in self.target_modules:
                    if hasattr(module.attn, target):
                        linear_module = getattr(module.attn, target)
                        if isinstance(linear_module, nn.Linear):
                            in_features, out_features = linear_module.in_features, linear_module.out_features
                            layer_id = f"{layer_id}.{target}"
                            self.lora_layers[layer_id] = nn.Sequential(
                                nn.Linear(in_features, self.r, bias=False),
                                nn.Linear(self.r, out_features, bias=False)
                            )
                            nn.init.kaiming_uniform_(self.lora_layers[layer_id][0].weight, a=math.sqrt(5))
                            nn.init.zeros_(self.lora_layers[layer_id][1].weight)

    def forward(self, x: torch.Tensor, target: str, layer_id: int) -> torch.Tensor:
        if target not in self.target_modules:
            raise ValueError(f"Target {target} not found in target modules")
        layer_key = f"{layer_id}.{target}"
        if layer_key not in self.lora_layers:
            raise ValueError(f"Layer {layer_key} not found in lora layers")
        return self.lora_layers[layer_key](x) * (self.alpha / self.r)

    def post_init(self, model: nn.Module):
        self.create_lora_layers(model)


class AdapterModule(TunerStrategyModule):
    def __init__(self, dim: int, adapter_dim: int, target_modules: List[str] = ['mlp']):
        super().__init__()
        self.dim = dim
        self.adapter_dim = adapter_dim
        self.target_modules = target_modules
        self.adapter_layers = nn.ModuleDict()
        self.scales = nn.ParameterDict()

    def create_adapter_layers(self, model: nn.Module):
        for layer_id, module in enumerate(model.bitemporal_transformer_blocks):
            for target in self.target_modules:
                if hasattr(module, target):
                    linear_module = getattr(module, target)
                    layer_key = f"{layer_id}.{target}"
                    in_features, out_features = linear_module.in_features, linear_module.out_features
                    self.adapter_layers[layer_key] = nn.Sequential(
                        nn.Linear(in_features, self.adapter_dim),
                        nn.GELU(),
                        nn.Linear(self.adapter_dim, out_features)
                    )
                    self.adapter_layers[layer_key].apply(self._init_weights)
                    self.scales[layer_key] = nn.Parameter(torch.ones(out_features))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=1e-3)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, target: str, layer_id: str) -> torch.Tensor:
        if target not in self.target_modules:
            raise ValueError(f"Target {target} not found in target modules")
        layer_key = f"{layer_id}.{target}"
        if layer_key not in self.adapter_layers:
            raise ValueError(f"Layer {layer_key} not found in adapter layers")
        return x + self.scales[layer_key] * self.adapter_layers[layer_key](x)

    def post_init(self, model: nn.Module):
        self.create_adapter_layers(model)


class ProbingModule(TunerStrategyModule):
    def __init__(self):
        super().__init__()
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