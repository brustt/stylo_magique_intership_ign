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
from segment_anything import SamPredictor, sam_model_registry
from segment_anything.modeling import Sam
import cv2
from magic_pen.config import DEVICE, sam_model_large
from magic_pen.seg_any_change.utils import load_levircd_sample

def load_img(path: str):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def init_encoder(model_type: str):
    sam = sam_model_registry[model_type](checkpoint=sam_model_large)
    _ = sam.to(device=DEVICE)
    encoder = SamPredictor(sam_model=sam)
    return encoder


def get_embedding(img: np.ndarray, encoder: Any):
    encoder.set_image(img, image_format="RGB")
    embedding = encoder.get_image_embedding()
    return embedding

def pipeline_encoding(path_A: str, path_B: str, model_type="vit_h"):
    
    img_A = load_img(path_A)
    img_B = load_img(path_B)

    encoder = init_encoder(model_type)

    embedding_A = get_embedding(img_A, encoder)
    embedding_B = get_embedding(img_B, encoder)

    print(embedding_A.shape)
    print(embedding_B.shape)

    # check zero mean and unit variance




if __name__ == "__main__":
    df = load_levircd_sample(size=1)
    path_A, path_B, label = df["A"].item(), df["B"].item(), df["label"].item()
    embs = pipeline_encoding(path_A, path_B)