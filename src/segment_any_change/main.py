from typing import Any, List
import numpy as np
from segment_any_change.embedding import get_img_embedding_normed
from segment_any_change.mask_items import ImgType, create_change_proposal_items
from segment_any_change.matching import proposal_matching, temporal_matching
from segment_any_change.utils import load_img, load_levircd_sample, load_sam

from segment_any_change.mask_generator import SegAnyMaskGenerator
import logging

# TO DO : define globally
logging.basicConfig(format="%(asctime)s - %(levelname)s ::  %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SegAnyChange:
    def __init__(self, model_type: str, **sam_kwargs) -> None:
        model = load_sam(model_type)
        self.mask_generator = SegAnyMaskGenerator(model, **sam_kwargs)
        self.items_A = None
        self.items_B = None

    def bitemporal_matching(
        self, img_A: np.ndarray, img_B: np.ndarray, filter_method: str, **params
    ) -> Any:

        # Image A : embedding + masks
        masks_A = self.mask_generator.generate(img_A)
        print(f"N masks A : {len(masks_A)}")
        img_embedding_A = get_img_embedding_normed(self.mask_generator.predictor)

        # Image B : embedding + masks
        masks_B = self.mask_generator.generate(img_B)
        print(f"N masks B : {len(masks_B)}")
        img_embedding_B = get_img_embedding_normed(self.mask_generator.predictor)

        # t -> t+1
        x_t_mA, x_t1_mA, ci = temporal_matching(
            img_embedding_A, img_embedding_B, masks_A
        )
        # t+1 -> t
        x_t_mB, x_t1_mB, ci1 = temporal_matching(
            img_embedding_A, img_embedding_B, masks_B
        )

        # TO DO : review nan values : object loss after resize
        print(f"nan values ci {np.sum(np.isnan(ci))}")
        print(f"nan values ci1 {np.sum(np.isnan(ci1))}")

        self.items_A = create_change_proposal_items(masks_A, ci, ImgType.A)
        self.items_B = create_change_proposal_items(masks_B, ci1, ImgType.B)

        # filter on sim/chgt_angle before union ?
        logger.info("Proposal Matching ...")
        items_change = proposal_matching(self.items_A, self.items_B)
        items_change = items_change.apply_change_filtering(filter_method)

        # tmp_return = {
        #     "A": {"global_mask":masks_A, "img_embedding":img_embedding_A},
        #     "B": {"global_mask":masks_B, "img_embedding":img_embedding_B},
        #     "mask_embedding":{"t->t+1":(x_t_mA, x_t1_mA), "t+1->t":(x_t_mB, x_t1_mB)},
        #     "conf":{"t->t+1":ci, "t+1->t":ci1}

        # }
        # return tmp_return

        return items_change

    def get_mask_proposal(self, temp_type: ImgType, idx=None) -> List[np.ndarray]:

        dict_type = {
            ImgType.A: self.items_A,
            ImgType.B: self.items_B,
        }
        if temp_type not in dict_type:
            raise KeyError("please provide valid data type : A, B")
        if idx is None:
            return [i.mask.astype(np.uint8) for i in dict_type[temp_type]]


if __name__ == "__main__":
    pair_img = load_levircd_sample(1, seed=42)
    path_label, path_A, path_B = pair_img.iloc[0]
    # default parameters for auto-generation
    sam_params = {
        "points_per_side": 32,
        "points_per_batch": 64,
        "pred_iou_thresh": 0.88,
        "stability_score_thresh": 0.95,
        "stability_score_offset": 1.0,
        "box_nms_thresh": 0.7,
        "crop_n_layers": 0,
        "crop_nms_thresh": 0.7,
        "crop_overlap_ratio": 512 / 1500,
        "crop_n_points_downscale_factor": 1,
        "point_grids": None,
        "min_mask_region_area": 0,
        "output_mode": "binary_mask",
    }
    logger.info("==== start ====")
    segany = SegAnyChange(model_type="vit_b", **sam_params)
    res = segany.bitemporal_matching(
        img_A=load_img(path_A), img_B=load_img(path_B), filter_method="otsu"
    )
    print(f"Done : {len(res)}")
