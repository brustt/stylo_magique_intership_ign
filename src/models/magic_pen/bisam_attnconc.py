
from copy import deepcopy
from enum import Enum
from commons.constants import DEVICE_MAP, IMG_SIZE
from models.commons.mask_items import ImgType
from models.commons.rpe.cross_rpe_attention import CrossRPEBlock
from models.magic_pen.bisam_abc import BiSamGeneric
from models.segment_anything.modeling.common import MLPBlock
import numpy as np
from omegaconf import DictConfig
import torch
from torch import nn
from torch.nn import functional as F
from timm.models.layers import Mlp, DropPath

from models.segment_anything.modeling.image_encoder_dev import (
    ImageEncoderViT,
)  # edited
from src.models.segment_anything.modeling.mask_decoder_dev import MaskDecoder  # edited
from src.models.segment_anything.modeling.prompt_encoder_dev import (
    PromptEncoder,
)  # edited

from typing import List, Dict, Any, Tuple, Type, Union
from models.segment_anything.modeling.transformer import Attention

import logging

# TO DO : define globally
logging.basicConfig(format="%(asctime)s - %(levelname)s ::  %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class BiSamAttn(BiSamGeneric):
    mask_threshold: float = 0.0

    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        params: Union[Dict, DictConfig],
        embedding_dim: int = 256,
        num_patches: int = 32*32
    ) -> None:
        """
        SAM predicts object masks from an batch of images and prompts

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__(image_encoder, prompt_encoder, mask_decoder, params)

        self.fusion_module = CrossMaskAttentionBlock(
            embedding_dim, # channels
            mlp_ratio=2., 
            qkv_bias=False, 
            drop=0., 
            attn_drop=0.,
            drop_path=0., 
            num_patches=num_patches, # spatial dim flatten
        )


    def forward(
        self,
        batched_input: Dict[str, torch.Tensor],
        multimask_output: bool,
        one_mask_for_all: bool = True,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        SAM implementation for bi-input and batch inference.
        Args:
            batched_input (List[Dict[str, Any]]): images batch with associated prompts (points)
            multimask_output (bool): multi_mask return
            one_mask_for_all (bool, optional): mask compute mode : per mask or for all mask

        Returns:
            List[Dict[str, torch.Tensor]]: dict return as prediction batch tensor
        """
        batch_size = batched_input[next(iter(batched_input))].shape[0]

        emb_A = self.image_encoder(self.preprocess(batched_input["img_A"]))
        emb_B = self.image_encoder(self.preprocess(batched_input["img_B"]))

        # concatenation tokens wise : B x 256 x 2 x 64 x 64
        self.image_embeddings = torch.stack([emb_A, emb_B], axis=2)

        # B x C x H x W
        fusion_embeddings = self.fusion_module(self.image_embeddings)
        
        print("fusion embeddings : ", fusion_embeddings.shape)

        if one_mask_for_all:
            # one inference for all points => unique mask(s)
            points = batched_input["point_coords"][:, None,...], batched_input["point_labels"][:, None,...]
        else:
            # one inference per point => mask(s) by point
            points = batched_input["point_coords"][:, :, None, :], batched_input["point_labels"][..., None]

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=points,
            boxes=None,
            masks=None
        )
        # print("sparse_embeddings :", sparse_embeddings.shape),
        # print("dense_embeddings :", dense_embeddings.shape),

        # low_res_mask : B x N x M x 256 x 256
        # N : number of prompt or 1 (one_mask_for_all)
        # M : number of mask per prompt (multimask output - 3 or 1)
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=fusion_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )            

        masks = self.upscale_masks(low_res_masks, IMG_SIZE)

        masks, iou_predictions = self.select_masks(
            masks, iou_predictions, multimask_output
        )        
        return masks, iou_predictions
    
    def select_masks(
        self, masks: torch.Tensor, iou_predictions: torch.Tensor, multimask_output: bool
    ) -> torch.Tensor:
        mask_slice = slice(1, None) if multimask_output else slice(0, 1)
        masks, iou_predictions = (
            masks[:, :, mask_slice, :, :],
            iou_predictions[:, :, mask_slice],
        )
        # remove dim of number of masks for multimask_output=False
        return masks.squeeze(2), iou_predictions

    def upscale_masks(
        self, masks: torch.Tensor, original_size: Tuple[int]
    ) -> torch.Tensor:

        init_shape = masks.shape
        masks = masks.flatten(0, 1)  # interpolate take 4D data not 5D
        masks = F.interpolate(
            masks, original_size, mode="bilinear", align_corners=False
        )
        masks = masks.reshape(init_shape[0], -1, *masks.shape[1:])
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std
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


class CrossMaskAttentionBlock(nn.Module):
    """
    Inspired from Omnisat and https://arxiv.org/pdf/2312.04869
    Use layers  from timm : droppath, MLP.
    """

    def __init__(
            self, 
            dim, 
            mlp_ratio=4., 
            qkv_bias=False, 
            drop=0., 
            attn_drop=0.,
            drop_path=0., 
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
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=drop)
        # get back spatial dim
        self.up_layer = nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2)

    def forward(self, x):
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