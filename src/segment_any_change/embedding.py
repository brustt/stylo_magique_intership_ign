import cv2
import numpy as np

from segment_any_change.sa_dev.predictor import SamPredictor


def compute_mask_embedding(mask: np.ndarray, img_embedding: np.ndarray) -> np.ndarray:
    """Compute mask embedding

    - resize mask to img dimension (256 x 64 x 64)
    - map img embedding to mask value
    - aggregate over not-nan values

    Args:
        mask (np.ndarray): _description_
        img_embedding (np.ndarray): _description_

    Returns:
        np.ndarray: mask embedding dim 256
    """

    mask = cv2.resize(src=mask, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
    mask = np.where(mask != 0.0, 1, np.nan)
    mask_embedding = np.nanmean(img_embedding * mask[None, ...], axis=(1, 2))
    mask_embedding = np.where(np.isnan(mask_embedding), 0, mask_embedding)
    return mask_embedding


def get_img_embedding_normed(predictor: SamPredictor) -> np.ndarray:
    """Invert affine transformation of the image encoder last LayerNorm Layer.


    Args:
        predictor (SamPredictor): inference class for SAM

    Returns:
        np.ndarray: Scaled embedding
    """
    embedding = predictor.get_image_embedding()
    # get last layerNorm weights & biais to invert affine transformation
    w = predictor.model._modules["image_encoder"].neck[3].weight
    b = predictor.model._modules["image_encoder"].neck[3].bias
    embedding = (embedding.squeeze(0) - b[:, None, None]) / w[:, None, None]
    embedding = embedding.detach().cpu().numpy()

    return embedding
