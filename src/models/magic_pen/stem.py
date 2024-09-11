from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn as nn

@dataclass
class StemModule(nn.Module, ABC):
    img_size: int = 1024
    patch_size: int = 16
    in_chans: int = 3
    embed_dim: int = 768
    use_abs_pos: bool = True
    use_rel_pos: bool = False
    
    @abstractmethod
    def forward(self, x):
        pass


class SAMStem(StemModule):
    def __init__(self):
        super().__init__()
        self.patch_embed = PatchEmbed(
            kernel_size=(self.patch_size, self.patch_size),
            stride=(self.patch_size, self.patch_size),
            in_chans=self.in_chans,
            embed_dim=self.embed_dim,
        )
        self.pos_embed: Optional[nn.Parameter] = None
        if self.use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            self.pos_embed = nn.Parameter(
                torch.zeros(
                    1, self.img_size // self.patch_size, self.img_size // self.patch_size, self.embed_dim
                )
            )

    def forward(self, x):
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        return x

class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (16, 16),
        stride: Tuple[int, int] = (16, 16),
        padding: Tuple[int, int] = (0, 0),
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
        """
        super().__init__()

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)
        return x