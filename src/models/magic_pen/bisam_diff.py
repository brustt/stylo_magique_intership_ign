from copy import deepcopy
from enum import Enum
from commons.constants import DEVICE_MAP, IMG_SIZE
from omegaconf import DictConfig
from src.models.commons.mask_items import ImgType
from src.models.magic_pen.bisam_abc import BiSamGeneric
import torch
from torch import nn
from torch.nn import functional as F

from src.models.segment_anything.modeling.image_encoder_dev import (
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

class BiSamDiff(BiSamGeneric):
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
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__(image_encoder, prompt_encoder, mask_decoder, params)

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

        self.image_embeddings_A = self.image_encoder(self.preprocess(batched_input["img_A"]))
        self.image_embeddings_B = self.image_encoder(self.preprocess(batched_input["img_B"]))

        # simple diff
        self.image_embeddings = self.image_embeddings_A - self.image_embeddings_B

        if one_mask_for_all:
            # one inference for all points => unique mask(s)
            points = batched_input["point_coords"][:, None,...], batched_input["point_labels"][:, None,...]
        else:
            # one inference per point => mask(s) by point
            points = batched_input["point_coords"][:, :, None, :], batched_input["point_labels"][..., None]


        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=points,
            boxes=None,
            masks=None,
        )
        # low_res_mask : B x N x M x 256 x 256
        # N : number of prompt or 1 (one_mask_for_all)
        # M : number of mask per prompt (multimask output - 3 or 1)
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=self.image_embeddings,
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
