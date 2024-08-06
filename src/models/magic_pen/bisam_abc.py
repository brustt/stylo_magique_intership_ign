from abc import ABC, abstractmethod
from copy import deepcopy
from enum import Enum
import re
from commons.constants import DEVICE_MAP, IMG_SIZE
from models.commons.mask_items import ImgType
from omegaconf import DictConfig
import torch
from torch import nn
from torch.nn import functional as F

from models.segment_anything.modeling.image_encoder_dev import (
    ImageEncoderViT,
)  # edited
from src.models.segment_anything.modeling.mask_decoder_dev import MaskDecoder  # edited
from src.models.segment_anything.modeling.prompt_encoder_dev import (
    PromptEncoder,
)  # edited

from typing import List, Dict, Any, Tuple, Union

import logging

# TO DO : define globally
logging.basicConfig(format="%(asctime)s - %(levelname)s ::  %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class BiSamGeneric(nn.Module, ABC):
    mask_threshold: float = 0.0

    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        params: Union[Dict, DictConfig],
    ) -> None:
        """
        SAM predicts object masks from an batch of images and prompts

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
        """
        super().__init__()

        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.image_embeddings = None

        self.register_buffer(
            "pixel_mean", torch.tensor([123.675, 116.28, 103.53]).view(-1, 1, 1), False
        )
        self.register_buffer(
            "pixel_std", torch.tensor([58.395, 57.12, 57.375]).view(-1, 1, 1), False)
        
         # let's prevent checkpoint forgetting
        if not params.get("sam_ckpt_path", None):
            raise ValueError("Please provide sam checkpoint")
        
        if not params.get("ft_mode", None):
            raise ValueError("Please provide ft mode")
        
        self.load_weights(params.get("sam_ckpt_path"), params.get("use_weights"))
        self.freeze_weights(params.get("ft_mode"))
    
    @abstractmethod
    def forward(self) -> List[Dict[str, torch.Tensor]]:
        raise NotImplementedError
        
            
    def load_weights(self, checkpoint: str, use_weights: Union[Any, List]) ->None:
        pretrained_weights = torch.load(checkpoint)
        if use_weights is None:
            # we use all weights
            self.load_state_dict(pretrained_weights, strict=True)
        else:
            # we select weights to load 
            model_dict = self.state_dict()
            pretrained_weights = {k: v for k, v in pretrained_weights.items() if any([k.startswith(m) for m in use_weights])}
            model_dict.update(pretrained_weights)
            self.load_state_dict(model_dict, strict=False)
            logger.info(f"Weights loaded for : {use_weights}")

    def freeze_weights(self, ft_mode: str):
        """
        # TODO: 
        #  - freeze layer on key layer selection / name
            - set init weights based on known distrib
        """
        def match_name(trainable_name: str, layer_name: str) -> bool:
            return bool(re.search(trainable_name, layer_name))
        
        if ft_mode == "probing":
            #self.image_encoder.requires_grad_(False)
            for l in self.image_encoder.parameters():
                l.requires_grad_(False)

        elif ft_mode == "adapter":
            #  ImageEncoderAdapterVit has adapter layer
            for name, l in self.image_encoder.named_parameters():
                if not match_name(ft_mode, name):
                    l.requires_grad_(False)

        elif ft_mode == "lora":
            for l in self.image_encoder.parameters():
                l.requires_grad_(False)

            self.image_encoder.init_lora_layers()


        # freeze layer not contributing to backpropagation
        for name, l in self.named_parameters():
            if match_name("iou_prediction_head", name):
                l.requires_grad_(False)
            if match_name("prompt_encoder.mask_downscaling", name):
                l.requires_grad_(False)
            self.prompt_encoder.point_embeddings[2].requires_grad_(False)
            self.prompt_encoder.point_embeddings[3].requires_grad_(False)