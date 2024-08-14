import math

import torch
import torch.nn as nn
from timm.layers import Mlp, DropPath

from .irpe import build_rpe, get_rpe_config


class CrossRPEAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., num_patches=None,
                 n_modalities=2):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.num_patches = num_patches
        self.n_modalities = n_modalities // 2
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.wq = nn.Linear(dim, dim, bias=qkv_bias)


        # self.q_learned = nn.Parameter(torch.zeros(1, 1, dim))
        # self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, dim))

        # image relative position encoding
        rpe_config = get_rpe_config(
            ratio=1.9,
            method="euc",
            mode='ctx',
            shared_head=True,
            skip=0,
            rpe_on='qkv',
        )
        self.rpe_q, self.rpe_k, self.rpe_v = build_rpe(rpe_config,
                                                       head_dim=head_dim,
                                                       num_heads=num_heads)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):

        B, N, C = x.shape
        # num_head == 8
        # q torch.Size([B, 8, 4096, 64])
        # BNC -> BNH(C/H) -> BHN(C/H)
        q = self.wq(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # BNC -> BNH(C/H) -> BHN(C/H)
        k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # BNC -> BNH(C/H) -> BHN(C/H)
        v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # attn torch.Size([B, 8, 4096, 4096])
        attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
        # print(attn.shape)
        # image relative position on keys
        if self.rpe_k is not None:
            attn += torch.cat(
                [self.rpe_k(q)[:, :, :, :1], self.rpe_k(q)[:, :, :, 1:].repeat(1, 1, 1, self.n_modalities)], dim=-1)

        # image relative position on queries
        if self.rpe_q is not None:
            attn += self.rpe_q(k * self.scale).transpose(2, 3)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # # image relative position on values
        # if self.rpe_v is not None:
        #     attn += self.rpe_v(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, self.num_patches, C)  # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class CrossRPEBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, num_patches=36, n_modalities=[], ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CrossRPEAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, n_modalities=n_modalities,
                                      qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, num_patches=num_patches)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        B, C, H, W = x.shape
        
        # flat spatial dimensions to patches
        x = x.view(B, C, -1).permute(0, 2, 1)
        # print("x", x.shape)
        x = self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.permute(0, 2, 1).view(B, C, H, W)
        # print("x", x.shape)

        return x