"""
- create data loader
- get images embedding from sam
- get masks from grid points


- remove affine transformation (learnable parameters weights and biais from last LayerNorm2d layer in segment anything)
- re-install segment_anything with pip install -e .

"""
from typing import Any
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
        img_embedding_A = embedding_from_predictor(self.mask_generator.predictor)
        self.mask_generator.predictor.reset_image()
        
        masks_B = generate_masks(self.mask_generator, self.img_B)
        img_embedding_B = embedding_from_predictor(self.mask_generator.predictor)
        self.mask_generator.predictor.reset_image()
        
        # t -> t+1
        x_ti = [self.compute_mask_embedding(m, img_embedding_A) for m in masks_A]
        x_t1i1 = [self.compute_mask_embedding(m, img_embedding_B) for m in masks_A]
        chg_ci= [self.neg_cosine_sim(x, y) for x, y in zip(x_ti, x_t1i1)]

        # t+1 -> t
        x_ti = [self.compute_mask_embedding(m, img_embedding_A) for m in masks_B]
        x_t1i1 = [self.compute_mask_embedding(m, img_embedding_B) for m in masks_B]
        chg_ci1 = [self.neg_cosine_sim(x, y) for x, y in zip(x_ti, x_t1i1)]

        tmp_return = {
            "A": {"mask":masks_A,"img_embedding":img_embedding_A},
            "B": {"mask":masks_B,"img_embedding":img_embedding_B},
            "conf":[chg_ci, chg_ci1]

        }
        return tmp_return
            
    
    def compute_mask_embedding(self, mask: np.ndarray, img_embedding: np.ndarray) -> np.ndarray:
        # remove zero values : np.nanmean(np.where(matrix!=0,matrix,np.nan),(1, 2))
        return np.mean(img_embedding*mask.reshape(img_embedding.shape), axis=(1, 2))
    
    def neg_cosine_sim(self, x1, x2): 
        assert np.linalg.norm(x1) == np.linalg.norm(x2)
        return - (x1 @ x2) / np.linalg.norm(x1)**2




if __name__ == "__main__":
    pair_img = load_levircd_sample(1, seed=42)
    path_label,path_A, path_B = pair_img.iloc[0]
    segany = SegAnyChange(
        img_A=load_img(path_A),
        img_B=load_img(path_B),
        model_type="vit_b"
    )

    res = segany.bitemporal_matching()