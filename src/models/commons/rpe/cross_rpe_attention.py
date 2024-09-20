import math

import numpy as np
import torch
import torch.nn as nn
from timm.models.layers import Mlp, DropPath
import torch.nn.functional as F

from .irpe import build_rpe, get_rpe_config


class CrossRPEAttention(nn.Module):
    def __init__(
            self, 
            dim, 
            num_heads=8, 
            qkv_bias=False, 
            qk_scale=None, 
            attn_drop=0., 
            proj_drop=0., 
            num_patches=None,
            n_modalities=2):
        super().__init__()
        print("dim", dim)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # find next square root integer to ensure skip=0 in rpe - don't forget to add padding to entry
        self.num_patches = math.ceil(np.sqrt(num_patches))**2 // 2
        self.num_patches_original = num_patches
        self.n_modalities = n_modalities
        # self.internal_dim = embedding_dim // downsample_rate

        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        # self.q_learned = nn.Parameter(torch.zeros(1, 1, dim))


        self.q_learned = nn.Parameter(torch.zeros(1, 1, dim))
        # self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, dim))

        # image relative position encoding
        rpe_config = get_rpe_config(
            ratio=1.9,
            method="euc",
            mode='ctx',
            shared_head=True,
            skip=0,
            rpe_on='k',# do we need more ? 
        )
        self.rpe_q, self.rpe_k, self.rpe_v = build_rpe(rpe_config,
                                                       head_dim=head_dim,
                                                       num_heads=num_heads)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):

        x = x.permute(0, 2, 1)
        pad = self.num_patches - x.shape[-1]
        # B , N, C
        x = F.pad(x, (0, int(pad))).permute(0, 2, 1)

        B, N, C = x.shape
        q_ = self.q_learned.expand(B, self.num_patches, -1).reshape(B, self.num_patches, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        print("q__", q_.shape)
        # print(self.wq(x).shape)
        q = self.wq(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # BNC -> BNH(C/H) -> BHN(C/H)
        k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # BNC -> BNH(C/H) -> BHN(C/H)
        v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        print("q", q.shape)
        print("k", k.shape)
        print("v", v.shape)

        # attn torch.Size([B, 8, 4096, 4096])
        attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
        print("attn", attn.shape)
        # image relative position on keys
        if self.rpe_k is not None:
            # qk_t0 = self.rpe_k(q[:, :, :4096, :])
            # qk_t1 = self.rpe_k(q[:, :, 4096:, :])

            # print("t0 qk", qk_t0.shape)
            print("rpe k", self.rpe_k(q)[:, :, :, :1].shape)
            print("rpe k extend", self.rpe_k(q).repeat(1, 1, 1, self.n_modalities).shape)
            print("full rpe k", self.rpe_k(q).shape)
            # change from omnisat - get back to original imp : https://github.com/microsoft/Cream/blob/main/iRPE/DETR-with-iRPE/models/rpe_attention/rpe_attention_function.py
            # attn += torch.cat(
            #     [self.rpe_k(q)[:, :, :, 1:].repeat(1, 1, 1, self.n_modalities)], dim=-1)

            attn += self.rpe_k(q, height=None, width=None)

        # image relative position on queries
        if self.rpe_q is not None:
            attn += self.rpe_q(k * self.scale).transpose(2, 3)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # # image relative position on values
        # if self.rpe_v is not None:
        #     attn += self.rpe_v(attn)
        print((attn @ v).transpose(1, 2).shape)
        print("n pathes", self.num_patches)
        x = (attn @ v).transpose(1, 2).reshape(B, self.num_patches, C)
        # print("attn @ v", x.shape)
        print("xxx", x.shape)
        print(self.proj.weight.shape)
        # x = x.reshape(B, self.num_patches, C)  # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
        # print("x in proj", x.shape)
        x = self.proj(x)
        x = self.proj_drop(x)
        # print("out attn func", x.shape)
        # remove padding
        print("xb", x.shape)
        x = x[:,  :self.num_patches_original, :]
        print("x cut", x.shape)
        return x


class CrossRPEBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, num_patches=36, n_modalities=[], ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CrossRPEAttention(dim, 
                                      num_heads=num_heads, 
                                      qkv_bias=qkv_bias, 
                                      n_modalities=n_modalities,
                                      qk_scale=qk_scale, 
                                      attn_drop=attn_drop, 
                                      proj_drop=drop, 
                                      num_patches=num_patches,
                                      )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        B, C, N, H, W = x.shape

        # flat spatial dimensions over modalities (2) => tokens dimensions (8192)
        x = x.view(B, -1, N, C).permute(0, 2, 1)

        #x = torch.rand((B, C, 91*91)).permute(0, 2, 1)

        print("x", x.shape)
        x = self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        print(x.shape)
        x = x.permute(0, 2, 1).reshape(B, -1, H, W).view(B*N, C, H, W)


        print("x out", x.shape)

        print("out")

        return x