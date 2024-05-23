import torch
from torch import nn
from torch.nn import functional as F

from segment_any_change.sa_dev_v0.modeling.image_encoder import (
    ImageEncoderViT,
)  # edited
from segment_any_change.sa_dev_v0.modeling.mask_decoder import MaskDecoder  # edited
from segment_any_change.sa_dev_v0.modeling.prompt_encoder import PromptEncoder  # edited

from typing import List, Dict, Any, Tuple

import logging

# TO DO : define globally
logging.basicConfig(format="%(asctime)s - %(levelname)s ::  %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class BiSam(nn.Module):
    mask_threshold: float = 0.0

    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
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
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.image_embeddings = None
        self.register_buffer(
            "pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False
        )
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

    @torch.no_grad()
    def forward(
        self,
        batched_input: Dict[str, torch.Tensor],
        multimask_output: bool,
        return_logits: bool = False,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        SAM implementation for bi-input and batch inference.
        TO DO : add multi size prompts

        Args:
            batched_input (List[Dict[str, Any]]): images batch with associated prompts (points)
            multimask_output (bool): multi_mask return
            return_logits (bool, optional): logits return. Defaults to False.

        Returns:
            List[Dict[str, torch.Tensor]]: dict return as prediction batch tensor
        """

        input_images = torch.cat([batched_input["img_A"], batched_input["img_B"]])
        original_size = input_images.shape[-2:]
        input_images = self.preprocess(input_images)
        self.image_embeddings = self.image_encoder(input_images)

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=(
                batched_input["point_coords"][:, :, None, :],
                batched_input["point_labels"][..., None],
            ),
            boxes=None,
            masks=None,
        )
        print(f"sparse_embeddings: {sparse_embeddings.shape}")
        print(f"dense_embeddings: {dense_embeddings.shape}")

        low_res_masks, iou_predictions = self.mask_decoder.predict_masks_batch(
            image_embeddings=self.image_embeddings,  # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, N, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, N, 256, 64, 64)
        )

        masks = self.upscale_masks(low_res_masks, original_size)

        mask_slice = slice(1, None) if multimask_output else slice(0, 1)
        masks = masks[:, :, mask_slice, :, :]
        iou_predictions = iou_predictions[:, :, mask_slice]

        if not return_logits:
            masks = masks > self.mask_threshold

        return {
            "masks": masks,
            "iou_predictions": iou_predictions,
            "low_res_logits": low_res_masks,
        }

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

    def get_image_embedding(self, idx=None) -> torch.Tensor:
        if self.image_embeddings is None:
            raise RuntimeError("Please compute batch images embedding first")
        if idx is None:
            return self.image_embeddings
        return self.image_embeddings[idx, ...]
