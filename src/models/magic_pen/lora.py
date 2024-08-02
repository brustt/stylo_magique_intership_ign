# https://github.com/MathieuNlp/Sam_LoRA/blob/main/src/lora.py

from models.segment_anything.modeling.image_encoder_dev import ImageEncoderViT

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

class LoRA_qkv(nn.Module):
    """
    LoRA adaption for attention modules. Only for queries and values

    TODO: check if scaling values need to be add

    Arguments:
        qkv: Original block of attention
        linear_a_q: linear block for q
        linear_b_q: linear block for q
        linear_a_v: linear block for v
        linear_b_v: linear block for v

    Return:
        qkv(nn.Module): qkv block with all linear blocks added (equivalent to adding the matrix B*A)
    """

    def __init__(
            self,
            qkv,
            linear_a_q: nn.Module,
            linear_b_q: nn.Module,
            linear_a_v: nn.Module,
            linear_b_v: nn.Module,
    ):
        
        super(LoRA_qkv, self).__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.d_model = qkv.in_features
        self.w_identity = torch.eye(qkv.in_features)

    def forward(self, x: Tensor):
        qkv = self.qkv(x)
        q_ba = self.linear_b_q(self.linear_a_q(x))
        v_ba = self.linear_b_v(self.linear_a_v(x))
        qkv[:, :, :, :self.d_model] += q_ba #q part
        qkv[:, :, :, -self.d_model:] += v_ba #v part

        return qkv


class ImageEncoderViTLoRA(nn.Module):
    """
    Take a vit model and inject LoRa layer to attention matrix 

    TODO: add attention block selection

    Arguments:
        vit_model: vit
        rank: Rank of the matrix for LoRA    
    Return:
        None

    """

    def __init__(self, 
                 
                 vit_model: ImageEncoderViT, 
                 rank: int, 
                 ):
        super().__init__()
        self.rank = rank
        assert rank > 0
        # base_vit_dim = sam_model.image_encoder.patch_embed.proj.out_channels

     
        self.lora_layer = list(range(len(vit_model.blocks)))
        
        self.A_weights = []
        self.B_weights = []

        # apply on each layer
        for i, blk in enumerate(vit_model.blocks):

            w_qkv_linear = blk.attn.qkv
            self.d_model = w_qkv_linear.in_features

            w_a_linear_q = nn.Linear(self.d_model, self.rank, bias=False)
            w_b_linear_q = nn.Linear(self.rank, self.d_model, bias=False)
            w_a_linear_v = nn.Linear(self.d_model, self.rank, bias=False)
            w_b_linear_v = nn.Linear(self.rank, self.d_model, bias=False)
            

            self.A_weights.append(w_a_linear_q)
            self.B_weights.append(w_b_linear_q)
            self.A_weights.append(w_a_linear_v)
            self.B_weights.append(w_b_linear_v)

            blk.attn.qkv = LoRA_qkv(
                w_qkv_linear,
                w_a_linear_q,
                w_b_linear_q,
                w_a_linear_v,
                w_b_linear_v
            )

        self.reset_parameters()


    def reset_parameters(self):
        """
        Initialize the LoRA A and B matrices like in the paper
        """
        # Initalisation like in the paper
        for w_A in self.A_weights:
            nn.init.kaiming_uniform_(w_A.weight, a=np.sqrt(5))
        for w_B in self.B_weights:
            nn.init.zeros_(w_B.weight)


    def save_lora_parameters(self, filename: str):
        """Check if needed"""
        pass

    def load_lora_parameters(self, filename: str):
        """Check if needed"""
        pass 

