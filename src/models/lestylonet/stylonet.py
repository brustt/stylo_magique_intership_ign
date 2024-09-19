import logging
import re
from typing import Any, List, Tuple, Union
import torch
import torch.nn as nn
from torch.nn import functional as F
from omegaconf import DictConfig

from commons.constants import IMG_SIZE
from .factories import *


logger = logging.getLogger(__name__)


class LeStyloNet(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        
        # sam_components = create_sam_model(config)
        """ENCODER"""
        self.stem = create_stem_module(config)
        self.tuner_strategy = create_tuner_strategy(config)
        self.bitemporal_transformer_blocks = create_bitemporal_transformer_blocks(
            config,
            self.tuner_strategy 
        )
        self.encoder_neck = create_encoder_neck(config)
        """PROMPT ENCODER"""
        self.prompt_encoder = create_prompt_encoder(config)
        """PROMPT-TO-IMAGE"""
        self.prompt_to_image_fusion = create_prompt_to_image_module(config)
        """FUSION EMBEDDING"""
        self.bitemporal_embedding_fusion = create_bitemporal_embedding_fusion_module(
            config,
            create_fusion_strategy(config)
        )
        """PREDICTION HEADS"""
        self.iou_score_head = create_iou_score_head(config)
        self.mask_head = create_mask_head(config)
        
        # self.initialize_from_sam(sam_components)
        self.load_weights(config.get("sam_ckpt_path"), config.get("use_weights"))


    def forward(self, x, one_mask_for_all: bool = True, multimask_output: bool=True):
        (
            x1, 
            x2, 
            points_coords,
            points_labels 
        ) = x["img_A"], x["img_B"], x["point_coords"], x["point_labels"]
        
        """Encoder pass x1"""
        x1 = self.stem(self.preprocess(x1))
        # B x 64 x 64 x 768
        for blk in self.bitemporal_transformer_blocks:
            x1 = blk(x1)
        # B x 256 x 64 x 64
        x1 = self.encoder_neck(x1.permute(0, 3, 1, 2))
        """Encoder pass x2"""
        x2 = self.stem(self.preprocess(x2))
        # B x 64 x 64 x 768
        for blk in self.bitemporal_transformer_blocks:
            x2 = blk(x2)
        # B x 256 x 64 x 64
        x2 = self.encoder_neck(x2.permute(0, 3, 1, 2))
        """Prompt Encoder"""
        if one_mask_for_all:
            # one inference for all points => unique mask(s)
            points = points_coords[:, None,...], points_labels[:, None,...]
        else:
            # one inference per point => mask(s) by point
            points = points_coords[:, :, None, :], points_labels[..., None]

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=points,
            boxes=None,
            masks=None
        )
        """Fusion Module"""
        x = self.bitemporal_embedding_fusion(x1, x2)
        """Prompt-To-Image"""
        tokens, mask_embedding = self.prompt_to_image_fusion(
            image_embeddings=x,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
        )

        """Mask-Head"""
        low_res_masks = self.mask_head(tokens, mask_embedding)

        """IOU-Score-Head"""
        iou_pred = self.iou_score_head(tokens)

        masks = self.upscale_masks(low_res_masks, IMG_SIZE)

        masks, iou_predictions = self.select_masks(
            masks, iou_pred, multimask_output
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
    
    def initialize_from_sam(sam):
        pass
    
    def load_weights(self, checkpoint: str, use_weights: Union[Any, List]) -> None:
        if not checkpoint:
            logger.info("No SAM checkpoint provided. Skipping weight initialization.")
            return

        logger.info(f"Loading weights from SAM checkpoint: {checkpoint}")
        pretrained_weights = torch.load(checkpoint)

        if use_weights is None:
            # We use all weights
            self.load_state_dict(pretrained_weights, strict=False)
            logger.info("All weights loaded from SAM checkpoint")
        else:
            # We select weights to load 
            model_dict = self.state_dict()
            pretrained_weights = {k: v for k, v in pretrained_weights.items() if any([k.startswith(m) for m in use_weights])}
            model_dict.update(pretrained_weights)
            self.load_state_dict(model_dict, strict=False)
            logger.info(f"Weights loaded for: {use_weights}")

        # Let the tuner strategy handle any post-initialization steps
        self.tuner_strategy.post_init(self)

    def freeze_weights(self, ft_mode: str):
        def match_name(trainable_name: str, layer_name: str) -> bool:
            return bool(re.search(trainable_name, layer_name))

        # Let the tuner strategy handle the freezing of weights
        self.tuner_strategy.freeze_weights(self)

        # Freeze specific layers
        # TODO: update with new modular structure
        for name, param in self.named_parameters():
            if match_name("iou_prediction_head", name):
                param.requires_grad_(False)
            if match_name("prompt_encoder.mask_downscaling", name):
                param.requires_grad_(False)
        
        if hasattr(self.prompt_encoder, 'point_embeddings'):
            self.prompt_encoder.point_embeddings[2].requires_grad_(False)
            self.prompt_encoder.point_embeddings[3].requires_grad_(False)

        logger.info(f"Weights frozen according to {ft_mode} mode")