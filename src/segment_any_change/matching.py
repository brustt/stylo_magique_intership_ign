import numpy as np
from typing import Any, Dict, List, Tuple, Union
from deprecated import deprecated
from tqdm import tqdm
from magic_pen.config import IMG_SIZE
from segment_any_change.embedding import (
    compute_mask_embedding,
    get_img_embedding_normed,
)
from segment_any_change.masks.mask_generator import SegAnyMaskGenerator
from segment_any_change.masks.mask_items import (
    ImgType,
    change_thresholding,
    ItemProposal,
    ListProposal,
    create_union_object,
)
from torch.nn.utils.rnn import pad_sequence

from segment_any_change.masks.mask_process import binarize_mask
from segment_any_change.sa_dev.utils.amg import MaskData
from segment_any_change.utils import (
    SegAnyChangeVersion,
    resize,
    timeit,
    to_degre_torch,
    to_numpy,
)
import torch
from torchvision.ops.boxes import batched_nms

import logging


# TO DO : define globally
logging.basicConfig(format="%(asctime)s - %(levelname)s ::  %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def stack_tensor_from_sam_output(img_anns: List[Dict], key: str="masks") -> torch.Tensor:
    # (Bx2) x max(NA,NB) x (H x W)
    if key:
        return pad_sequence([anns[key] for anns in img_anns], batch_first=True)
    else:
        return pad_sequence(img_anns, batch_first=True)



class BitemporalMatching:
    def __init__(
        self,
        model,
        th_change_proposals: Union[float, str],
        col_nms_threshold: float,
        version: SegAnyChangeVersion,
        **sam_kwargs,
    ) -> None:
        self.mask_generator = SegAnyMaskGenerator(model, **sam_kwargs)
        # useful for future experimentation
        self.seganyversion = version.value
        self.col_nms_threshold = col_nms_threshold
        self.filter_method = th_change_proposals

    def __call__(self, batch: Dict[str, torch.Tensor], **params) -> Any:
        logger.info(f"=== {self.seganyversion} ====")
        items_batch = self.run(batch, self.filter_method, **params)
        return items_batch

    @timeit
    def run(self, batch: Dict[str, torch.Tensor], filter_method: str, **params) -> Any:
        """
        Run Bitemporal matching - vectorized manner

        # TODO : check new return on sample

        """
        batch_filtered = []

        
        img_anns = self.mask_generator.generate(batch)
        batch_size = self.mask_generator.batch_size


        imgs_embedding_A = get_img_embedding_normed(self.mask_generator, ImgType.A)
        imgs_embedding_B = get_img_embedding_normed(self.mask_generator, ImgType.B)

        for i in range(batch_size):

            img_anns_A = img_anns[i]
            img_anns_B = img_anns[i + batch_size]
            img_anns_curr = [img_anns_A, img_anns_B]

            img_emb_A = imgs_embedding_A[i]
            img_emb_B = imgs_embedding_B[i]

            # masks = stack_tensor_from_sam_output(img_anns, key="masks")
            # masks_A, masks_B = masks[:batch_size], masks[batch_size:]
            
            # NA x H x W           
            masks_A =  img_anns_A["masks"]
            # NB x H x W
            masks_B =  img_anns_B["masks"]

            # TODO : clean outputs temporal_matching if not needed
            # t -> t+1
            # ci : N
            x_t_mA, _, ci = temporal_matching_torch(
                img_emb_A, img_emb_B, masks_A
            )
            # t+1 -> t
            # ci1 : N
            _, x_t1_mB, ci1 = temporal_matching_torch(
                img_emb_A, img_emb_B, masks_B
            )
            # 2, max(NA,NB) x H x W
            masks = pad_sequence([masks_A, masks_B], batch_first=True)
            # 2, max(NA,NB)
            confidence_scores = pad_sequence([ci, ci1], batch_first=True)

            # 2 x max(NA, NB) x 4
            bboxes = stack_tensor_from_sam_output(img_anns_curr, key="bbox")
            # 2 x max(NA, NB)
            iou_preds = stack_tensor_from_sam_output(img_anns_curr, key="predicted_iou")
            # 2 x max(NA, NB) x H x W
            masks_logits = stack_tensor_from_sam_output(img_anns_curr, key="masks_logits")


            print("NMS masks fusion")
            print("masks", masks.shape)
            print("masks i A", masks_A.shape)
            print("masks i B", masks_B.shape)
            print("ci", confidence_scores.shape)
            print("bboxes", bboxes.shape)
            print("ious", iou_preds.shape)
            print("masks_logits", masks_logits.shape)

            # use data structure define in sa_dev
            # check for OutofBoundError
            data = MaskData(
                masks=masks.flatten(0, 1),
                masks_logits=masks_logits.flatten(0, 1),
                bboxes=bboxes.flatten(0, 1),
                ci=confidence_scores.flatten(0, 1),
                iou_preds=iou_preds.flatten(0, 1),
            )
            # simple fusion based on NMS
            data = proposal_matching_nms(
                data=data,
                nms_threshold=self.mask_generator.box_nms_thresh,
                col_threshold=self.col_nms_threshold,
            )

            # apply change threshold
            data["chgt_angle"] = to_degre_torch(data["ci"])
            data, th = change_thresholding(data, method=self.filter_method)

            # we need to get back batch information for each prediction
            # data = reconstruct_batch(data, masks.shape[0])

            batch_filtered.append(data)
        
        # B x N x H x W - N : filtered masks from A & B
        masks = pad_sequence([elem["masks"] for elem in batch_filtered], batch_first=True)
        masks_logits = pad_sequence([elem["masks_logits"] for elem in batch_filtered], batch_first=True)
        iou_preds = pad_sequence([elem["iou_preds"] for elem in batch_filtered], batch_first=True)
        ci = pad_sequence([elem["ci"] for elem in batch_filtered], batch_first=True)
        
        # check if we catch some masks
        if masks_logits.shape[1]:
            masks_logits = resize(masks_logits, IMG_SIZE)
        # else:
        #     masks_logits =  torch.zeros((masks_logits.shape[0], 0, IMG_SIZE, IMG_SIZE))
        masks_bin = binarize_mask(masks_logits, self.mask_generator.mask_threshold)

        return dict(
            masks=masks_bin, # B x max(NA, NB) x H x W
            img_anns=img_anns,
            iou_preds=iou_preds, # B x max(NA, NB) 
            confidence_scores=ci, # B x max(NA, NB) 
        )


def neg_cosine_sim(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    # print(np.linalg.norm(x1))
    # print(np.linalg.norm(x2))
    # print("---")
    return -(x1 @ x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


def neg_cosine_sim_torch(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Compute negative cosime similarities on mask embedding from a image paire

    Args:
        x1 (torch.Tensor): embedding 1 : N x C
        x2 (torch.Tensor): embedding 2 : N x C

    Returns:
        torch.Tensor: negative similarities N
    """
    # only interested by element wise dot product
    dot_prod = torch.diagonal((x1 @ x2.permute(1, 0)), dim1=0, dim2=1)
    # vectors norms
    dm = torch.linalg.norm(x1, dim=1) * torch.linalg.norm(x2, dim=1)
    return -dot_prod / dm


@timeit
def proposal_matching_nms(
    data: MaskData, nms_threshold: float, col_threshold: str = "ci"
) -> MaskData:

    keep_by_nms = batched_nms(
        data["bboxes"].float(),
        data[col_threshold],
        torch.zeros_like(data["bboxes"][:, 0]),  # categories
        iou_threshold=nms_threshold,  # default SAM : 0.7 for iou - for ci need to search
    )
    data.filter(keep_by_nms)

    return data


@timeit
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


@timeit
def temporal_matching_torch(
    img_embedding_A: torch.Tensor, img_embedding_B: torch.Tensor, masks: torch.Tensor
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """Compute mask embedding and confidence score for both images for some masks (mt or mt+1)

    Args:
        img_embedding_A torch.Tensor: C x He x We
        img_embedding_B torch.Tensor: C x He x We
        masks torch.Tensor: C x H x W

    Returns:
        Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]: bi-temporal mask embeddings and confidence score
    """
    x_t = compute_mask_embedding(masks, img_embedding_A)
    x_t1 = compute_mask_embedding(masks, img_embedding_B)
    chg_ci = neg_cosine_sim_torch(x_t, x_t1)

    return x_t, x_t1, chg_ci


@deprecated
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


@deprecated
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


@timeit
def proposal_matching(
    items_A: List[ItemProposal],
    items_B: List[ItemProposal],
    th_union: float = 0.6,
    skip_fusion=False,
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

    if skip_fusion:
        for insert_item in f_items:
            filter_items.add_item(insert_item)
        return filter_items

    else:
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
