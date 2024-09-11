from typing import Optional, Tuple
from models.commons.utils import window_partition, window_unpartition
from models.magic_pen.factories import create_attention, create_mlp
from models.magic_pen.strategies import TunerStrategyModule
import torch.nn as nn
from omegaconf import DictConfig


class EncoderNeck(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        # Initialize encoder neck layers

    def forward(self, x):
        # Encoder neck forward pass
        pass

class PromptEncoder(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        # Initialize prompt encoder layers

    def forward(self, x):
        # Prompt encoder forward pass
        pass

class ImagePromptFusion(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        # Initialize image-prompt fusion layers

    def forward(self, image_embed, prompt_embed):
        # Image-prompt fusion forward pass
        pass

class BitemporalEmbeddingFusion(nn.Module):
    def __init__(self, config: DictConfig, fusion_strategy: nn.Module):
        super().__init__()
        self.fusion_strategy = fusion_strategy

    def forward(self, x):
        # Bitemporal embedding fusion forward pass
        return self.fusion_strategy(x)

class IouScoreHead(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        # Initialize IOU score head layers

    def forward(self, x):
        # IOU score head forward pass
        pass

class MaskHead(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        # Initialize mask head layers

    def forward(self, x):
        # Mask head forward pass
        pass