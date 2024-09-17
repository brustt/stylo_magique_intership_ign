import torch
import torch.nn as nn
from omegaconf import DictConfig
from models.commons.utils import window_partition, window_unpartition
from models.lestylonet.blocks import LayerNorm2d

class NeckModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class SamNeck(NeckModule):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(in_channels, out_channels)
        self.neck = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(out_channels),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.neck(x)
