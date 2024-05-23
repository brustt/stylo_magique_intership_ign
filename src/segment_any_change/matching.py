import itertools
import cv2
import numpy as np
from typing import Dict, List, Tuple
from deprecated import deprecated
from segment_any_change.embedding import compute_mask_embedding
from segment_any_change.mask_items import (
    ItemProposal,
    ListProposal,
    create_union_object,
)
from segment_any_change.utils import timeit, to_tensor
import torch


def neg_cosine_sim(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    return -(x1 @ x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


def temporal_matching(
    img_embedding_A: np.ndarray, img_embedding_B: np.ndarray, masks: List[Dict]
) -> Tuple[List[np.ndarray], List[np.ndarray], List[float]]:
    """Compute mask embedding and confidence score for both images for some masks (mt or mt+1)

    Args:
        img_embedding_A (np.ndarray): _description_
        img_embedding_B (np.ndarray): _description_
        masks (List[Dict]): _description_

    Returns:
        Tuple[List[np.ndarray], List[np.ndarray], List[float]]: bi-temporal mask embeddings and confidence score
    """
    x_t = [
        compute_mask_embedding(m["segmentation"].astype(np.uint8), img_embedding_A)
        for m in masks
    ]
    x_t1 = [
        compute_mask_embedding(m["segmentation"].astype(np.uint8), img_embedding_B)
        for m in masks
    ]
    chg_ci = [neg_cosine_sim(x, y) for x, y in zip(x_t, x_t1)]

    return x_t, x_t1, chg_ci


def cover_same_zone(mask_1, mask_2, th=0.6) -> bool:
    """Check if masks extents are compliant for union

    rule for union (IoU) : inter_area > (union_area * th)

    Args:
        mask_1 (_type_): whenever mt or mt+1
        mask_2 (_type_): whenever mt or mt+1
        th (float, optional): threshold of IoU. Defaults to 0.6.

    Returns:
        bool: _description_
    """
    inter_area = np.sum(np.logical_and(mask_1, mask_2))
    union_area = np.sum(np.logical_or(mask_1, mask_2))
    return inter_area > (union_area * th)


def semantic_change_mask(
    items: List[ItemProposal], agg_func: str = "sum"
) -> np.ndarray:

    agg_factory = {"sum": np.sum, "avg": np.mean}
    if agg_func not in agg_factory:
        raise ValueError("Please provide valid agg function")

    # N x H x W
    masks = items.masks
    ci = items.confidence_scores
    mask_ci = agg_factory[agg_func]((masks * ci[:, None, None]), axis=0)

    return mask_ci


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


@timeit
def proposal_matching(
    items_A: List[ItemProposal], items_B: List[ItemProposal], th_union: float = 0.6
) -> ListProposal:
    """Iterative masks fusion based on IoU treshold.

    Not optimal : lack some fusions

    Args:
        items_A (List[ItemProposal]): items computed from img A masks
        items_B (List[ItemProposal]): items computed from img B masks

    Returns:
        ListProposal: Change proposals
    """

    filter_items = ListProposal()
    f_items = items_A + items_B

    for insert_item in f_items:
        inserted = False
        for ref_item in filter_items:
            if cover_same_zone(insert_item.mask, ref_item.mask, th=th_union):
                filter_items.add_item(create_union_object(insert_item, ref_item))
                filter_items.rm_item(ref_item.id)
                inserted = True
        if not inserted:
            filter_items.add_item(insert_item)

    return filter_items
