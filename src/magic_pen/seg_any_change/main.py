"""
- create data loader
- get images embedding from sam
- get masks from grid points


- remove affine transformation (learnable parameters weights and biais from last LayerNorm2d layer in segment anything)
- re-install segment_anything with pip install -e .

"""
import itertools
from typing import Any, Dict
import pandas as pd
import numpy as np 
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
from segment_anything.modeling import Sam
import cv2
from magic_pen.config import DEVICE, sam_model_large, sam_dict_checkpoint
from magic_pen.seg_any_change.utils import load_levircd_sample

from magic_pen.seg_any_change.mask_generator import SegAnyMaskGenerator 

def load_img(path: str):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def load_sam(model_type: str):
    sam = sam_model_registry[model_type](checkpoint=sam_dict_checkpoint[model_type])
    _ = sam.to(device=DEVICE)
    return sam


def embedding_from_predictor(predictor):
    embedding = predictor.get_image_embedding()
    embedding = embedding.detach().cpu().numpy().squeeze()
    return embedding

def generate_masks(mask_generator, image):
    masks = mask_generator.generate(image)
    masks = [m["segmentation"].astype(np.uint8) for m in masks]
    return masks

class SegAnyChange:
    def __init__(self, img_A: np.ndarray, img_B: np.ndarray, model_type: str, **sam_kwargs) -> None:
        self.img_A = img_A
        self.img_B = img_B
        sam = load_sam(model_type)
        self.mask_generator = SegAnyMaskGenerator(sam, **sam_kwargs)

    def bitemporal_matching(self, **params) -> Any:

        masks_A = generate_masks(self.mask_generator, self.img_A)
        print(f"N masks A : {len(masks_A)}")
        img_embedding_A = embedding_from_predictor(self.mask_generator.predictor)
        self.mask_generator.predictor.reset_image()
        
        masks_B = generate_masks(self.mask_generator, self.img_B)
        print(f"N masks B : {len(masks_A)}")

        img_embedding_B = embedding_from_predictor(self.mask_generator.predictor)
        self.mask_generator.predictor.reset_image()
        
        # t -> t+1
        x_t_mA = [self.compute_mask_embedding(m, img_embedding_A) for m in masks_A]
        x_t1_mA = [self.compute_mask_embedding(m, img_embedding_B) for m in masks_A]
        chg_ci= [self.neg_cosine_sim(x, y) for x, y in zip(x_t_mA, x_t1_mA)]

        # t+1 -> t
        x_t_mB = [self.compute_mask_embedding(m, img_embedding_A) for m in masks_B]
        x_t1_mB = [self.compute_mask_embedding(m, img_embedding_B) for m in masks_B]
        chg_ci1 = [self.neg_cosine_sim(x, y) for x, y in zip(x_t_mB, x_t1_mB)]

        # proposal_matching
        # what to do with matching and confidence score ci and ci1 ??

        tmp_return = {
            "A": {"global_mask":masks_A, "img_embedding":img_embedding_A},
            "B": {"global_mask":masks_B, "img_embedding":img_embedding_B},
            "mask_embedding":{"t->t+1":(x_t_mA, x_t1_mA), "t+1->t":(x_t_mB, x_t1_mB)},
            "conf":{"t->t+1":chg_ci, "t+1->t":chg_ci1}

        }
        return tmp_return
            
    
    def compute_mask_embedding(self, mask: np.ndarray, img_embedding: np.ndarray) -> np.ndarray:
        # 256 x 64 x 64
        mask = mask.reshape(img_embedding.shape)
        mask = np.where(mask!=0,1,np.nan)
        mask_embedding = np.nanmean(img_embedding*mask, axis=(1, 2))
        mask_embedding = np.where(np.isnan(mask_embedding),0,1)
        return mask_embedding
    
    def neg_cosine_sim(self, x1, x2): 
        assert np.linalg.norm(x1) == np.linalg.norm(x2)
        return - (x1 @ x2) / np.linalg.norm(x1)**2
    

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




if __name__ == "__main__":
    pair_img = load_levircd_sample(1, seed=42)
    path_label,path_A, path_B = pair_img.iloc[0]
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
    segany = SegAnyChange(
        img_A=load_img(path_A),
        img_B=load_img(path_B),
        model_type="vit_b",
        **sam_params
    )

    res = segany.bitemporal_matching()