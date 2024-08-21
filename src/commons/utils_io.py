import os
import pickle
from typing import Dict, Any, List, Union
import pathlib

import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig
import torch
from src.models.segment_anything.build_sam_dev import sam_model_registry
from src.models.segment_anything.build_sam import (
    sam_model_registry as sam_model_registry_v0,
)
from src.models.segment_anything.build_sam_v2 import sam_model_registry as sam_model_registry_v2

from commons.constants import DEVICE, SAM_DICT_CHECKPOINT
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
    model_type: str, 
    model_cls: Any = None, 
    version: str = "dev", 
    device: str = DEVICE, 
    is_strict: bool=True,
    init_ckpt: bool=True,
    embed_dim: int=256,
):

    sam = None
    ckpt = None
    if init_ckpt:
        ckpt = SAM_DICT_CHECKPOINT[model_type]

    match version:
        case "rawb":
            sam = sam_model_registry_v2[model_type](
                checkpoint=ckpt, model=model_cls
            ).to(device=device)
        case "dev":
            sam = sam_model_registry[model_type](
                checkpoint=ckpt, model=model_cls, is_strict=is_strict,  embed_dim=embed_dim
            ).to(device=device)
        case "raw":
            sam = sam_model_registry_v0[model_type](
                checkpoint=ckpt, is_strict=is_strict
            ).to(device=device)
        case _:
            raise ValueError(
                "Please provide valid sam verison implementation : dev, raw"
            )
    return sam


def load_img(img_path):
    img = io.imread(img_path)
    return img



def load_config(list_args: List[str]) -> DictConfig:
    GlobalHydra.instance().clear()
    # Initialize the Hydra configuration
    hydra.initialize(config_path="../../configs", version_base=None)
    # Compose the configuration with the desired environment override
    cfg = hydra.compose(config_name="train", overrides=list_args)
    
    return cfg

def load_ckpt_sam(sam, checkpoint=None):
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        sam.load_state_dict(state_dict)
    return sam

def create_folder(path: str,
                  parents: bool = True,
                  exist_ok: bool = True):
    """create folder with the whole hierarchy if required

    Parameters
    ----------
    path complete path of the folder

    Returns
    -------

    """
    pathlib.Path(path).mkdir(parents=parents,
                             exist_ok=exist_ok)
