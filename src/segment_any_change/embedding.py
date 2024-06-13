from typing import Any, Tuple
import cv2
import numpy as np
import torch

from magic_pen.config import IMG_SIZE
from segment_any_change.masks.mask_items import ImgType
from segment_any_change.masks.mask_process import resize
from segment_any_change.sa_dev.predictor import SamPredictor


def resize2d(arr: np.ndarray, target_size: Tuple, method: int):
    return cv2.resize(src=arr, dsize=target_size, interpolation=method)


def compute_mask_embedding(mask: np.ndarray, img_embedding: np.ndarray) -> np.ndarray:
    """Compute mask embedding

    - resize mask to img dimension (256 x 64 x 64)
    - map img embedding to mask value
    - aggregate over not-nan values

    Notes : we could use directly low_res_mask (logits) from Sam Class => 256x256 logits mask

    Args:
        mask (np.ndarray): _description_
        img_embedding (np.ndarray): _description_

    Returns:
        np.ndarray: mask embedding dim 256
    """
    assert mask.shape[-2:] == img_embedding.shape[-2:]
    # mask = cv2.resize(src=mask, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
    mask = np.where(mask != 0.0, 1, np.nan)
    mask_embedding = np.nanmean(img_embedding * mask[None, ...], axis=(1, 2))
    mask_embedding = np.where(np.isnan(mask_embedding), 0, mask_embedding)
    return mask_embedding


def compute_mask_embedding_torch(
    masks: torch.Tensor, img_embedding: torch.Tensor
) -> torch.Tensor:
    """_summary_

    Args:
        mask (torch.Tensor): torch.uint8 - (B,N,Hm,Wm)
        img_embedding (torch.Tensor): (B,C,Hm,Wm)

    Returns:
         torch.Tensor:  B x N x C
    """

    # resize to mask dim 1024 x 1024
    img_embedding = resize(img_embedding, IMG_SIZE)
    # align dim B x C x N x Hm x Wm)
    masks = masks.unsqueeze(1).repeat(1, img_embedding.shape[1], 1, 1, 1)
    # align dim B x C x N x Hm x Wm)
    img_embedding = img_embedding.unsqueeze(2).repeat(1, 1, masks.shape[2], 1, 1)
    # mask no data
    img_embedding[(masks == 0)] = torch.nan
    # mask embedding from spatial dim not nan
    masks_embedding = torch.nanmean(img_embedding, dim=(3, 4))
    # B x N x C
    return masks_embedding.permute(0, 2, 1)


def get_img_embedding_normed(predictor: Any, img_type: ImgType) -> np.ndarray:
    """Invert affine transformation of the image encoder last LayerNorm Layer.
    Run for a batch


    Args:
        predictor (SamPredictor, BiSam): inference class for SAM

    Returns:
        np.ndarray: Scaled embedding
    """
    # workaround BiSam - not clean
    if type(predictor.model).__name__ == "BiSam":
        embedding = predictor.model.get_image_embedding(img_type)
    else:
        embedding = predictor.get_image_embedding()

    # get last layerNorm weights & biais to invert affine transformation
    w = predictor.model._modules["image_encoder"].neck[3].weight
    b = predictor.model._modules["image_encoder"].neck[3].bias
    embedding = (embedding.squeeze(0) - b[:, None, None]) / w[:, None, None]
    return embedding.detach()
