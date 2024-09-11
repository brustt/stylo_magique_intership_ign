import logging
import re
from typing import Any, List, Union
import torch
import torch.nn as nn
from omegaconf import DictConfig
from .modules import *
from .factories import *


logger = logging.getLogger(__name__)


class LeStyloNet(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        
        # sam_components = create_sam_model(config)
        
        self.stem = create_stem_module(config)
        self.tuner_strategy = create_tuner_strategy(config)
        # TODO: move to config
        if isinstance(self.tuner_strategy, LoraModule):
            self.tuner_strategy.modules_to_tune = ['qkv'] 

        self.bitemporal_transformer_blocks = create_bitemporal_transformer_blocks(
            config.transformer_blocks,
            self.tuner_strategy
        )
        self.encoder_neck = create_encoder_neck(config.encoder_neck)
        self.prompt_encoder = create_prompt_encoder(config.prompt_encoder)
        self.image_prompt_fusion = create_image_prompt_fusion_module(config.image_prompt_fusion)
        self.bitemporal_embedding_fusion = create_bitemporal_embedding_fusion_module(
            config.embedding_fusion,
            FusionStrategyModule.create(config.fusion_strategy)
        )
        self.iou_score_head = create_iou_score_head(config.iou_score_head)
        self.mask_head = create_mask_head(config.mask_head)
        
        # self.initialize_from_sam(sam_components)

    def forward(self, x):
        """Encoder pass"""
        # Implement the forward pass using all the modules
        x = self.stem(x)
        # B x 64 x 64 x 768
        for blk in self.blocks:
            x = blk(x)
        # print("x in neck", x.shape)
        x = self.neck(x.permute(0, 3, 1, 2))
        # B x 256 x 64 x 64
        """Prompt Encoder"""
        """Fusion Module"""
        """Prompt-To-Image"""
        """Mask-Head"""

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