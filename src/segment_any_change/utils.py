from typing import Any, Union
import pandas as pd
import os
from magic_pen.config import levirCD_path, SEED
from pathlib import Path
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import cv2
from magic_pen.io import make_path
from segment_any_change.sa_dev import sam_model_registry
from magic_pen.config import DEVICE, sam_model_large, sam_dict_checkpoint

def load_img_cv2(path: str):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def load_sam(model_type: str):
    sam = sam_model_registry[model_type](checkpoint=sam_dict_checkpoint[model_type])
    _ = sam.to(device=DEVICE)
    return sam


def load_img(img_path):
    img = io.imread(img_path)
    return img

def show_img(img, show_axis=False):
    io.imshow(img)
    if not show_axis:
        plt.axis("off")

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

def load_levircd_sample(size: Union[int, float, Any]=None, data_type="train", seed=SEED) -> pd.DataFrame:
    """Sample levir-cd pair images with change label (paths)

    Args:
        size (Union[int, float, Any], optional): number of sample | frac of data_type | entire set (None). Defaults to None.
        data_type (str, optional): dataset type : train, test, val. Defaults to "train".
        seed (int, optional): seed. Defaults to SEED.

    Raises:
        ValueError: invalid data_type

    Returns:
        pd.DataFrame: data files paths (bitemporal images path and label)
    """
    
    path_dict = {
        "train": make_path("train", levirCD_path),
        "test": make_path("test", levirCD_path),
        "val": make_path("val", levirCD_path),
    }
    path = path_dict.get(data_type, None)

    if not path:
        raise ValueError("Please provide valid data_type : train, test, val")

    labels = sorted([make_path(_, path, "label") for _ in os.listdir(Path(path, "label")) if Path(_).suffix == ".png"])
    files_A = sorted([make_path(_, path, "A") for _ in os.listdir(Path(path, "A")) if Path(_).suffix == ".png"])
    files_B = sorted([make_path(_, path, "B") for _ in os.listdir(Path(path, "B")) if Path(_).suffix == ".png"])

    df = pd.DataFrame({"label": labels, "A": files_A, "B": files_B})

    if isinstance(size, float):
        return df.sample(frac=size, random_state=seed)
    elif isinstance(size, int):
        return df.sample(n=size, random_state=seed)
    else:
        return df

if __name__ == "__main__":
    df = load_levircd_sample(size=10, data_type="train")
    print(df.shape)
    print(df)