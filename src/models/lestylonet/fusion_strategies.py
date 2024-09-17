from abc import ABC, abstractmethod

import hydra
from models.lestylonet.blocks import CrossMaskAttention, MLPBlock
import numpy as np
from omegaconf import DictConfig
import torch
import torch.nn as nn

class FusionStrategyModule(nn.Module, ABC):
    @abstractmethod
    def forward(self, x):
        pass

    @classmethod
    def create(cls, config: DictConfig):
        return hydra.utils.instantiate(config)


class DiffFusion(FusionStrategyModule):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        return x1 - x2

class ConcatFusion(FusionStrategyModule):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        return torch.cat([x1, x2], dim=1)

class CrossAttentionFusion(FusionStrategyModule):
    def __init__(self,
            dim, 
            mlp_ratio=4., 
            qkv_bias=False, 
            drop=0., 
            attn_drop=0.,
            num_patches=4096):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)

        # downscale x 2 spatial dims
        self.down_layer = nn.Conv2d(dim, dim, kernel_size=2, stride=2)

        self.attn = CrossMaskAttention(dim, 
                                      qkv_bias=qkv_bias, 
                                      attn_drop=attn_drop, 
                                      proj_drop=drop, 
                                      num_patches=num_patches,
                                      )
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=mlp_hidden_dim)
        # get back spatial dim
        self.up_layer = nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2)


    def forward(self, x1, x2):
        x = torch.stack([x1, x2], axis=2)
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)
        # downscale & flat spatial dimensions
        x = self.down_layer(x).reshape(B, T, C, -1).permute(0, 3, 1, 2)

        # flat spatial dimensions
        x = self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        # get reduc spatial dims  - h == w
        emb_dim = int(np.sqrt(x.shape[-2]))
        x = x.permute(0, 2, 1).reshape(B, C, emb_dim, emb_dim)
        x = self.up_layer(x)
        return x

class BitemporalEmbeddingFusion(nn.Module):
    def __init__(self, config: DictConfig, fusion_strategy: nn.Module):
        super().__init__()
        self.fusion_strategy = fusion_strategy

    def forward(self, x):
        # Bitemporal embedding fusion forward pass
        return self.fusion_strategy(x)

