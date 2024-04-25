import itertools
from typing import Any, Dict, List, Tuple
import pandas as pd
import numpy as np 
import cv2
from segment_any_change.utils import load_img, load_levircd_sample, load_sam

from segment_any_change.mask_generator import SegAnyMaskGenerator 
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s ::  %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def get_img_embedding_normed(predictor):
    embedding = predictor.get_image_embedding()
    # get last layerNorm weights & biais to invert affine transformation
    w = predictor.model._modules["image_encoder"].neck[3].weight
    b = predictor.model._modules["image_encoder"].neck[3].bias
    print(embedding.shape)
    embedding = (embedding.squeeze(0) - b[:, None, None]) / w[:, None, None]
    embedding = embedding.detach().cpu().numpy()

    return embedding

def generate_masks(mask_generator, image):
    masks = mask_generator.generate(image)
    #masks = [m["segmentation"].astype(np.uint8) for m in masks]
    return masks

class SegAnyChange:
    def __init__(self, img_A: np.ndarray, img_B: np.ndarray, model_type: str, **sam_kwargs) -> None:
        self.img_A = img_A
        self.img_B = img_B
        model = load_sam(model_type)
        self.mask_generator = SegAnyMaskGenerator(model, **sam_kwargs)

    def bitemporal_matching(self, **params) -> Any:

        # Image A : embedding + masks
        self.masks_A = generate_masks(self.mask_generator, self.img_A)
        print(f"N masks A : {len(self.masks_A)}")
        img_embedding_A = get_img_embedding_normed(self.mask_generator.predictor)
        self.mask_generator.predictor.reset_image()
        
        # Image B : embedding + masks
        self.masks_B = generate_masks(self.mask_generator, self.img_B)
        print(f"N masks B : {len(self.masks_A)}")
        img_embedding_B = get_img_embedding_normed(self.mask_generator.predictor)
        self.mask_generator.predictor.reset_image()
        
        # t -> t+1
        x_t_mA, x_t1_mA, chg_ci = self.temporal_matching(img_embedding_A, img_embedding_B, self.get_mask_proposal("A"))

        # t+1 -> t
        x_t_mB, x_t1_mB, chg_ci1 = self.temporal_matching(img_embedding_A, img_embedding_B, self.get_mask_proposal("B"))
        # proposal_matching
        # what to do with matching and confidence score ci and ci1 ??

        tmp_return = {
            "A": {"global_mask":self.masks_A, "img_embedding":img_embedding_A},
            "B": {"global_mask":self.masks_B, "img_embedding":img_embedding_B},
            "mask_embedding":{"t->t+1":(x_t_mA, x_t1_mA), "t+1->t":(x_t_mB, x_t1_mB)},
            "conf":{"t->t+1":chg_ci, "t+1->t":chg_ci1}

        }
        return tmp_return
    
    def temporal_matching(self, img_embedding_A: np.ndarray,  img_embedding_B: np.ndarray, mask: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray], List[float]]:
        x_t = [self.compute_mask_embedding(m, img_embedding_A) for m in mask]
        x_t1 = [self.compute_mask_embedding(m, img_embedding_B) for m in mask]
        chg_ci= [self.neg_cosine_sim(x, y) for x, y in zip(x_t, x_t1)]

        return x_t, x_t1, chg_ci

    
    def compute_mask_embedding(self, mask: np.ndarray, img_embedding: np.ndarray) -> np.ndarray:
        # 256 x 64 x 64

        mask = cv2.resize(src=mask, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
        mask = np.where(mask !=0., 1, np.nan)
        mask_embedding = np.nanmean(img_embedding*mask[None, ...], axis=(1, 2))
        mask_embedding = np.where(np.isnan(mask_embedding),0 , mask_embedding)
        return mask_embedding
    
    def neg_cosine_sim(self, x1, x2): 
        # need to be equals norm(x1) == norm(x2) == sqrt(256)
        return - (x1 @ x2) / (np.linalg.norm(x1)*np.linalg.norm(x2))
    

    def proposal_matching(self, mask_A: np.ndarray, mask_B: np.ndarray) -> Dict[int, int]:
        """Compute overlap mapping between two sets of binary masks

        - Add threshold for association

        Args:
            mask_A (np.ndarray): mask from It
            mask_B (np.ndarray): mask from It+1

        Returns:
            Dict[int, int]: Dict of mask indices which intersects (mask_A_idx : [mask_B_idx1...mask_B_idx2])
        """
        inter_func = lambda x,y: np.count_nonzero(np.logical_and(x, y)) > 1

        overlap = {}
        prod_mt = list(zip(np.arange(len(mask_A)), mask_A))
        prod_mt1 = list(zip(np.arange(len(mask_B)), mask_B))

        for (mt_, mt1_) in itertools.product(prod_mt, prod_mt1):
            if inter_func(mt_[1], mt1_[1]):
                if mt_[0] in overlap:
                    overlap[mt_[0] ].append(mt1_[0])
                else:
                    overlap[mt_[0] ] = [mt1_[0]]
        
        return overlap
    
    def get_mask_proposal(self, temp_type: str , idx=None):
        
        dict_type = {
            "A": self.masks_A,
            "B": self.masks_B,
        }
        if temp_type not in dict_type:
            raise KeyError("please provide valid data type : A, B")
        if idx is None:
            return [m["segmentation"].astype(np.uint8) for m in dict_type[temp_type]]
            




if __name__ == "__main__":
    pair_img = load_levircd_sample(1, seed=42)
    path_label,path_A, path_B = pair_img.iloc[0]
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
        "crop_n_points_downscale_factor":1,
        "point_grids": None,
        "min_mask_region_area": 0,
        "output_mode": "binary_mask",
    }
    logger.info("==== start ====")
    segany = SegAnyChange(
        img_A=load_img(path_A),
        img_B=load_img(path_B),
        model_type="vit_b",
        **sam_params
    )
    #res = segany.bitemporal_matching()