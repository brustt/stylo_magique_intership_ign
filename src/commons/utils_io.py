import os
import pickle
from typing import Dict, Any, Union
from commons.config import *
from src.models.segment_anything.build_sam_dev import sam_model_registry
from src.models.segment_anything.build_sam import (
    sam_model_registry as sam_model_registry_v0,
)
from commons.config import DEVICE, SAM_DICT_CHECKPOINT
import skimage.io as io


def make_path(file_name, *path):
    return os.path.join(*path, file_name)


def check_dir(*path):
    os.makedirs(os.path.join(*path), exist_ok=True)
    return os.path.join(*path)


def save_pickle(data, output_path):
    with open(output_path, "wb") as f:
        pickle.dump(data, f)


def load_pickle(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data

def load_sam(
    model_type: str, model_cls: Any = None, version: str = "dev", device: str = DEVICE
):

    sam = None

    match version:
        case "dev":
            sam = sam_model_registry[model_type](
                checkpoint=SAM_DICT_CHECKPOINT[model_type], model=model_cls
            ).to(device=device)
        case "raw":
            sam = sam_model_registry_v0[model_type](
                checkpoint=SAM_DICT_CHECKPOINT[model_type]
            ).to(device=device)
        case _:
            raise ValueError(
                "Please provide valid sam verison implementation : dev, raw"
            )
    return sam


def load_img(img_path):
    img = io.imread(img_path)
    return img
