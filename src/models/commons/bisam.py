from copy import deepcopy
from enum import Enum
from commons.constants import DEVICE_MAP
import torch
from torch import nn
from torch.nn import functional as F

from .mask_items import ImgType
from models.segment_anything.modeling.image_encoder_dev import (
    ImageEncoderViT,
)  # edited
from src.models.segment_anything.modeling.mask_decoder import MaskDecoder  # edited
from src.models.segment_anything.modeling.prompt_encoder import (
    PromptEncoder,
)  # edited

from typing import List, Dict, Any, Tuple

import logging

# TO DO : define globally
logging.basicConfig(format="%(asctime)s - %(levelname)s ::  %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SamModeInference(Enum):
    AUTO = "auto"
    INTERACTIVE = "interactive"


class BiSam2(nn.Module):
    mask_threshold: float = 0.0

    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        device: str=None,
        **kwargs
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
        super().__init__()

        self.device = DEVICE_MAP[device] if device else "cpu"
        self.image_encoder = image_encoder.to(self.device)
        self.prompt_encoder = prompt_encoder.to(self.device)
        self.mask_decoder = mask_decoder.to(self.device)
        self.image_embeddings = None
        self.pixel_mean = (
            torch.tensor([123.675, 116.28, 103.53]).view(-1, 1, 1).to(self.device)
        )
        self.pixel_std = (
            torch.tensor([58.395, 57.12, 57.375]).view(-1, 1, 1).to(self.device)
        )

        # self.register_buffer(
        #     "pixel_mean", torch.tensor([123.675, 116.28, 103.53]).view(-1, 1, 1), False
        # )
        # self.register_buffer("pixel_std", torch.tensor([58.395, 57.12, 57.375]).view(-1, 1, 1), False)

    @torch.no_grad()
    def forward(
        self,
        batched_input: Dict[str, torch.Tensor],
        multimask_output: bool,
        return_logits: bool = False,
        mode: SamModeInference = SamModeInference.AUTO,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        SAM implementation for bi-input and batch inference.
        TO DO :
        - add multi size prompts
        - force lightning associated device through python class wrapper

        Args:
            batched_input (List[Dict[str, Any]]): images batch with associated prompts (points)
            multimask_output (bool): multi_mask return
            return_logits (bool, optional): logits return. Defaults to False.

        Returns:
            List[Dict[str, torch.Tensor]]: dict return as prediction batch tensor
        """
        batch_size = batched_input[next(iter(batched_input))].shape[0]
        # print(f"Mode : {mode}"

        # print("device input B:", batched_input["img_B"].detach().clone().device)
        # print("device input pts coords :", batched_input["point_coords"].detach().clone().device)
        # print("device input pts labels:", batched_input["point_labels"].detach().clone().device)

        if mode.value == SamModeInference.AUTO.value:

            # input_images = torch.cat(
            #     [batched_input["img_A"], batched_input["img_B"]]
            # ).to(self.device)

            input_images = batched_input["img_B"].to(self.device)


        elif mode.value == SamModeInference.INTERACTIVE.value:
            # for training remove detach()
            input_images = batched_input["img_B"].to(self.device)

        else:
            raise ValueError(f"mode {mode} for SAM not recognized")
        
        input_images = self.preprocess(input_images)

        self.image_embeddings = self.image_encoder(input_images)

        outputs = []
        for i, curr_embedding in enumerate(self.image_embeddings):

            point_coords = batched_input["point_coords"][i]
            # remove padding points
            point_coords = point_coords[torch.sum(point_coords, dim=1) > 0]
            # remove padding points
            point_labels = batched_input["point_labels"][i]
            point_labels = point_labels[:point_coords.shape[0]]  

            # if we add batch dim, why it could not works with batch ?
            points = point_coords[None, :, :], point_labels[None, :]

            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=None,
                masks=None,
            )
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            print(low_res_masks.shape)
            
            # masks = self.upscale_masks(low_res_masks, original_size)

            # if not return_logits:
            #     masks = low_res_masks > self.mask_threshold

            outputs.append(
                {
                    "masks": low_res_masks,
                    "iou_predictions": iou_predictions,
                    #"low_res_logits": low_res_masks,
                }
            )
        
        masks = torch.stack([out["masks"] for out in outputs])
        iou_predictions = torch.stack([out["iou_predictions"] for out in outputs])

        return  {
                    "masks": masks,
                    "iou_predictions": iou_predictions,
                    #"low_res_logits": low_res_masks,
                }

    def upscale_masks(
        self, masks: torch.Tensor, original_size: Tuple[int]
    ) -> torch.Tensor:
        """Now not batched"""

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

    def get_image_embedding(self, img_type: ImgType = None) -> torch.Tensor:
        if self.image_embeddings is None:
            raise RuntimeError("Please compute batch images embedding first")
        # TODO: extraction by types is WRONG if only on type (query prompt) is provided
        if img_type == ImgType.A:
            return self.image_embeddings.view(
                self.image_embeddings.shape[0] // 2,
                -1,
                *self.image_embeddings.shape[-3:],
            )[:, 0, ...]
        elif img_type == ImgType.B:
            return self.image_embeddings.view(
                self.image_embeddings.shape[0] // 2,
                -1,
                *self.image_embeddings.shape[-3:],
            )[:, 1, ...]
        else:
            return self.image_embeddings
