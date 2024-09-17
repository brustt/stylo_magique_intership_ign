import math
from models.commons.scale_attention import _attention_rel_h_rel_w
from models.commons.utils import add_decomposed_rel_pos, window_partition, window_unpartition
from models.lestylonet.factories import create_attention, create_mlp
from models.lestylonet.tuner_strategies import AdapterModule, LoraModule, TunerStrategyModule
import torch
import torch.nn as nn
from typing import Optional, Tuple, Type



"""TRANSFORMER"""
class BitemporalTransformerBlock(nn.Module):
    def __init__(self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        input_size: Optional[Tuple[int, int]] = None,
        tuner_strategy: Optional[TunerStrategyModule] = None,
        layer_id: str = None,

        ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size,
            tuner_strategy=tuner_strategy,
            layer_id=layer_id
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = create_mlp(
            dim=dim,
            mlp_ratio=mlp_ratio,
            act_layer=nn.GELU,
            tuner_strategy=tuner_strategy
        )
        self.window_size = window_size
        self.tuner_strategy = tuner_strategy
        self.layer_id = layer_id

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        pass


"""TRANSFORMER BLOCKS"""
class Attention(nn.Module):
    """Multi-head Attention block with optional LoRA."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: Optional[Tuple[int, int]] = None,
        tuner_strategy: Optional[TunerStrategyModule] = None,
        layer_id: str = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert input_size is not None, "Input size must be provided if using relative positional encoding."
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))
            if rel_pos_zero_init:
                nn.init.zeros_(self.rel_pos_h)
                nn.init.zeros_(self.rel_pos_w)

        self.tuner_strategy = tuner_strategy
        self.layer_id = layer_id

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        N = H * W

        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        if isinstance(self.tuner_strategy, LoraModule):
            q = q + self.tuner_strategy(x, 'q', self.layer_id)
            k = k + self.tuner_strategy(x, 'k', self.layer_id)
            v = v + self.tuner_strategy(x, 'v', self.layer_id)

        q = q.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        k = k.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        v = v.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))

        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, H, W, -1)
        x = self.proj(x)

        return x
    

class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))
    
class AdapterMLP(nn.Module):
    def __init__(
        self,
        dim: int,
        mlp_ratio: float,
        act_layer: Type[nn.Module],
        adapter_module: AdapterModule
    ):
        super().__init__()
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)
        self.adapter_module = adapter_module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        return self.adapter_module(x)

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class CrossMaskAttention(nn.Module):
    """
    Inspired from https://arxiv.org/pdf/2312.04869
    """
    def __init__(
            self, 
            dim, 
            qkv_bias=False, 
            attn_drop=0., 
            proj_drop=0., 
            num_patches=4096):
        super().__init__()

        self.num_patches= num_patches
        self.scale = dim ** -0.5
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)

        self.q_learned = nn.Parameter(torch.zeros(1, self.num_patches, dim))

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):

        B, N, T, C = x.shape
        q = self.q_learned.expand(B, -1, -1).unsqueeze(2)
        k = self.wk(x)
        v = self.wv(x)

        # print("q", q.shape)
        # print("k", k.shape)
        # print("v", v.shape)

        # attn torch.Size([B, 8, 4096, 4096])
        attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, self.num_patches, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x