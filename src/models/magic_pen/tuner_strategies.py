import math
from typing import List
import torch
from .strategies import TunerStrategyModule
import torch.nn as nn


class LoraModule(TunerStrategyModule):
    def __init__(self, r: int, alpha: float, target_modules: List[str] = ['qkv']):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.target_modules = target_modules
        self.lora_layers = nn.ModuleDict()

    def create_lora_layers(self, model: nn.Module):
        for name, module in model.named_modules():
            # TODO: target "qkv" layers => assign directly ?
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
        if module_name in self.lora_layers:
            return self.lora_layers[module_name](x) * (self.alpha / self.r)
        return torch.zeros_like(x)

    def freeze_weights(self, model: nn.Module):
        for name, param in model.named_parameters():
            if not any(target in name for target in self.target_modules):
                param.requires_grad_(False)

    def post_init(self, model: nn.Module):
        self.create_lora_layers(model)

class AdapterModule(TunerStrategyModule):
    def __init__(self, dim: int, mlp_ratio: float):
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        # Initialize adapter layers

    def forward(self, x):
        # Implementation for Adapter
        pass

    def freeze_weights(self, model: nn.Module):
        for name, param in model.bitemporal_transformer_blocks.named_parameters():
            if 'adapter' not in name:
                param.requires_grad_(False)

    def post_init(self, model: nn.Module):
        # Any post-initialization steps for adapter
        model.bitemporal_transformer_blocks.init_adapter_layers()
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