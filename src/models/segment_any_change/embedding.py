from typing import Any, Tuple
import cv2
from deprecated import deprecated
import numpy as np
import torch

from commons.constants import IMG_SIZE
from src.models.commons.mask_items import ImgType
from src.commons.utils import resize


@deprecated
def compute_mask_embedding_array(
    mask: np.ndarray, img_embedding: np.ndarray
) -> np.ndarray:
    """Compute mask embedding - `deprecated`

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


def compute_mask_embedding(masks: torch.Tensor, img_embedding: torch.Tensor):
    if masks.ndim > 3:
        return _compute_mask_embedding_batch_torch(masks, img_embedding)
    else:
        return _compute_mask_embedding_torch(masks, img_embedding)


def _compute_mask_embedding_torch(
    masks: torch.Tensor, img_embedding: torch.Tensor
) -> torch.Tensor:
    """_summary_

    masks : N x H x W
    img_embedding : C x He x We
    Args:
        masks (torch.Tensor): _description_
        img_embedding (torch.Tensor): _description_

    Returns:
        torch.Tensor: _description_
    """
    assert img_embedding.ndim == masks.ndim
    # resize to mask dim
    img_embedding = resize(img_embedding, masks.shape[-2:])
    # align dim C x N x Hm x Wm)
    # use view (expand) for memory efficiency
    masks = masks.unsqueeze(0).expand(img_embedding.shape[0], -1, -1, -1)
    # align dim C x N x Hm x Wm
    # need to copy() (repeat) for later assignation
    img_embedding = img_embedding.unsqueeze(1).repeat(1, masks.shape[1], 1, 1)
    # mask no data
    img_embedding[(masks == 0)] = torch.nan
    # mask embedding from spatial dim not nan
    masks_embedding = torch.nanmean(img_embedding, dim=(2, 3))
    # N x C
    return masks_embedding.permute(1, 0)


def _compute_mask_embedding_batch_torch(
    masks: torch.Tensor, img_embedding: torch.Tensor
) -> torch.Tensor:
    """_summary_

    Args:
        masks (torch.Tensor): torch.uint8 - (B,N,Hm,Wm)
        img_embedding (torch.Tensor): (B,C,Hm,Wm)

    Returns:
         torch.Tensor:  B x N x C
    """

    # resize to mask dim
    img_embedding = resize(img_embedding, masks.shape[-2:])
    # align dim B x C x N x Hm x Wm)
    # use view (expand) for memory efficiency
    masks = masks.unsqueeze(1).expand(-1, img_embedding.shape[1], -1, -1, -1)
    # align dim B x C x N x Hm x Wm)
    # need to copy() (repeat) for later assignation
    img_embedding = img_embedding.unsqueeze(2).repeat(1, 1, masks.shape[2], 1, 1)
    # mask no data
    img_embedding[(masks == 0)] = torch.nan
    # print("img", img_embedding.sum(dim=(2, 3)))
    # mask embedding from spatial dim not nan
    masks_embedding = torch.nanmean(img_embedding, dim=(3, 4))
    # print("masks_embedding", masks_embedding.sum(dim=(2)))

    # B x N x C
    return masks_embedding.permute(0, 2, 1)


def get_img_embedding_normed(model: Any, img_type: ImgType) -> torch.Tensor:
    """Invert affine transformation of the image encoder last LayerNorm Layer.
    Run for a batch


    Args:
        predictor (SamPredictor, BiSam): inference class for SAM

    Returns:
        np.ndarray: Scaled embedding
    """
    # workaround BiSam - not clean
    if type(model).__name__ == "BiSam":
        embedding = model.get_image_embedding(img_type)
    else:
        raise RuntimeError(f"Not implemented for {type(model).__name__}")
    # check for batch ==1
    # get last layerNorm weights & biais to invert affine transformation
    w = model._modules["image_encoder"].neck[3].weight
    b = model._modules["image_encoder"].neck[3].bias
    embedding = (embedding - b[:, None, None]) / w[:, None, None]
    return embedding.detach()
