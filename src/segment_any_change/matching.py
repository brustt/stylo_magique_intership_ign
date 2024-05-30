import numpy as np
from typing import Any, Dict, List, Tuple
from deprecated import deprecated
from segment_any_change.embedding import compute_mask_embedding, get_img_embedding_normed
from magic_pen.config import DEVICE
from magic_pen.data.loader import BiTemporalDataset
from magic_pen.data.process import DefaultTransform
from segment_any_change.masks.mask_generator import SegAnyMaskGenerator
from segment_any_change.masks.mask_items import (
    FilteringType,
    ImgType,
    create_change_proposal_items,
    ItemProposal,
    ListProposal,
    create_union_object,
)
from torch.nn.utils.rnn import pad_sequence

from segment_any_change.model import BiSam
from segment_any_change.sa_dev.utils.amg import batched_mask_to_box
from segment_any_change.utils import (
    SegAnyChangeVersion,
    flush_memory,
    load_levircd_sample,
    load_sam,
    timeit,
)
import torch
from torch.utils.data import DataLoader
import logging

# TO DO : define globally
logging.basicConfig(format="%(asctime)s - %(levelname)s ::  %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class BitemporalMatching:
    def __init__(self, model, th_change_proposals, **sam_kwargs) -> None:
        self.mask_generator = SegAnyMaskGenerator(model, **sam_kwargs)
        self.seganyversion = SegAnyChangeVersion.RAW
        self.filter_method = th_change_proposals
        self.items_A = None
        self.items_B = None
    
    def __call__(self, 
                 batch: Dict[str, torch.Tensor], 
                 **params) -> Any:
        
        preds = []
        device = params.get("device", None) if params.get("device", None) else DEVICE
        items_batch = self.run(batch, self.filter_method, **params)
        return items_batch

    @timeit
    def run(self, batch: Dict[str, torch.Tensor], filter_method: str, **params) -> Any:
        """
        Run Bitemporal matching
        Keep implementation in numpy for simplicity
        """
        img_anns = self.mask_generator.generate(batch)
        print(f"return {len(img_anns)}")

        items_batch = []

        masks_A = self.extract_temporal_img(img_anns, ImgType.A)
        masks_B = self.extract_temporal_img(img_anns, ImgType.B)
        img_embedding_A = get_img_embedding_normed(self.mask_generator, ImgType.A)
        img_embedding_B = get_img_embedding_normed(self.mask_generator, ImgType.B)

        assert len(masks_A) == len(masks_B) == len(img_embedding_A) == len(img_embedding_B)

        for img_embA, mA, img_embB, mB in zip(img_embedding_A, masks_A, img_embedding_B, masks_B):
            # t -> t+1
            x_t_mA, _, ci = temporal_matching(img_embA, img_embB, mA)
            # t+1 -> t
            _, x_t1_mB, ci1 = temporal_matching(
                img_embA, img_embB, mB
            )

            # TO DO : review nan values : object loss after resize
            logger.info(f"nan values ci {np.sum(np.isnan(ci))}")
            logger.info(f"nan values ci1 {np.sum(np.isnan(ci1))}")


            self.items_A = create_change_proposal_items(
                masks=mA, ci=ci, type_img=ImgType.A, embeddings=x_t_mA
            )
            self.items_B = create_change_proposal_items(
                masks=mB, ci=ci1, type_img=ImgType.B, embeddings=x_t1_mB
            )

            match self.seganyversion:
                case SegAnyChangeVersion.RAW:
                    items_change = proposal_matching(self.items_A, self.items_B)
                    th = items_change.apply_change_filtering(
                        filter_method, FilteringType.Sup
                    )
                    items_batch.append(items_change)
                case _:
                    raise RuntimeError("SegAnyChange version unkwown")
                
        masks = pad_sequence([item_list.masks for item_list in items_batch]).permute(1, 0, 2, 3)
        iou_preds = pad_sequence([item_list.iou_preds for item_list in items_batch]).permute(1, 0)

        return dict(masks=masks, iou_preds=iou_preds)


    def extract_temporal_img(self, items: List[Dict], name: ImgType) -> List[Dict]:
        """Retrieve image temporality - keep old data format"""
        match name:
            case ImgType.A:
                return (
                    [[
                        {
                            "segmentation" : m, 
                            "iou_pred" : s
                        } for m,s in zip(d["masks"], d["predicted_iou"])
                        ] for d in items if d["img_type"] == ImgType.A])
            case ImgType.B:
                return (
                    [[
                        {
                            "segmentation" : m, 
                            "iou_pred" : s
                        } for m,s in zip(d["masks"], d["predicted_iou"])
                        ] for d in items if d["img_type"] == ImgType.B])
            case _:
                raise ValueError("Uncorrect image type")


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



if __name__ == "__main__":

    flush_memory()

    pair_img = load_levircd_sample(1, seed=42)
    path_label, path_A, path_B = pair_img.iloc[0]
    # default parameters for auto-generation
    sam_params = {
        "points_per_side": 10,  # lower for speed
        "points_per_batch": 64,  # not used
        "pred_iou_thresh": 0.88,
        "stability_score_thresh": 0.95,
        "stability_score_offset": 1.0,
        "box_nms_thresh": 0.7,
        "min_mask_region_area": 0,
    }
    logger.info("==== start ====")
    filter_change_proposals = "otsu"
    filter_query_sim = 70
    batch_size = 1
    model_type = "vit_b"

    ds = BiTemporalDataset(name="levir-cd", dtype="train", transform=DefaultTransform())
    dataloader = DataLoader(ds, batch_size=batch_size)
    masks_loop = []

    model = load_sam(
        model_type=model_type, model_cls=BiSam, version="dev", device=DEVICE
    )

    matcher = BitemporalMatching(model, **sam_params)

    logger.info("--- Bitemporal matching ---")

    for i_batch, batch in enumerate(dataloader):
        if i_batch == 1:
            break

        logger.info(f"Run batch {i_batch}")

        items_change = matcher.run(
            batch=batch,
            filter_method=filter_change_proposals,
        )
        print(f"Done : {len(items_change)}")