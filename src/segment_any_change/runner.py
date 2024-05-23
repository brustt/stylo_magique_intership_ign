import numpy as np
from magic_pen.config import DEVICE
from magic_pen.data.loader import BiTemporalDataset
from magic_pen.data.process import DefaultTransform
from segment_any_change.embedding import get_img_embedding_normed
from segment_any_change.mask_generator import SegAnyMaskGenerator
from segment_any_change.mask_items import (
    FilteringType,
    ImgType,
    create_change_proposal_items,
)
from segment_any_change.matching import proposal_matching, temporal_matching
from segment_any_change.model import BiSam
from segment_any_change.utils import (
    SegAnyChangeVersion,
    batch_to_list,
    flush_memory,
    load_img,
    load_levircd_sample,
    load_sam,
    to_degre,
    timeit,
)
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Any, Tuple
import logging

# TO DO : define globally
logging.basicConfig(format="%(asctime)s - %(levelname)s ::  %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class BitemporalMatching:
    def __init__(self, model, **sam_kwargs) -> None:
        self.mask_generator = SegAnyMaskGenerator(model, **sam_kwargs)
        self.seganyversion = SegAnyChangeVersion.RAW
        self.items_A = None
        self.items_B = None

    @timeit
    def run(self, batch: Dict[str, torch.Tensor], filter_method: str, **params) -> Any:
        """
        Run Bitemporal matching
        Keep implementation in numpy for simplicity
        """
        img_anns = self.mask_generator.generate(batch)
        for item in img_anns:
            masks_A = self.extract_temporal_img(item["masks"], ImgType.A)
            masks_B = self.extract_temporal_img(item["masks"], ImgType.B)
            img_embedding_A = get_img_embedding_normed(self.mask_generator, ImgType.A)
            # print(f"N masks A : {len(masks_A)}")
            img_embedding_B = get_img_embedding_normed(self.mask_generator, ImgType.B)
            # t -> t+1
            x_t_mA, _, ci = temporal_matching(img_embedding_A, img_embedding_B, masks_A)
            # t+1 -> t
            _, x_t1_mB, ci1 = temporal_matching(
                img_embedding_A, img_embedding_B, masks_B
            )

            # TO DO : review nan values : object loss after resize
            logger.info(f"nan values ci {np.sum(np.isnan(ci))}")
            logger.info(f"nan values ci1 {np.sum(np.isnan(ci1))}")

            self.items_A = create_change_proposal_items(
                masks=masks_A, ci=ci, type_img=ImgType.A, embeddings=x_t_mA
            )
            self.items_B = create_change_proposal_items(
                masks=masks_B, ci=ci1, type_img=ImgType.B, embeddings=x_t1_mB
            )

            match self.seganyversion:
                case SegAnyChangeVersion.RAW:
                    items_change = proposal_matching(self.items_A, self.items_B)
                    th = items_change.apply_change_filtering(
                        filter_method, FilteringType.Sup
                    )
                case _:
                    raise RuntimeError("SegAnyChange version unkwown")

            return (
                items_change,
                th,
            )

    def extract_temporal_img(self, masks: np.ndarray, name: ImgType):
        """Compliant format with old code"""
        match name:
            case ImgType.A:
                return [{"segmentation": masks[_]} for _ in range(0, len(masks), 2)]
            case ImgType.B:
                return [{"segmentation": masks[_]} for _ in range(1, len(masks), 2)]
            case _:
                raise ValueError("Uncorrect image type")


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
