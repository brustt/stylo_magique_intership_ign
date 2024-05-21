from dataclasses import dataclass
from enum import Enum
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from typing import List, Tuple, Dict, Any
from magic_pen.config import *
from PIL import Image

# Ignore warnings
import warnings

from magic_pen.data.process import generate_grid_prompt
from magic_pen.io import load_levircd_sample
from segment_any_change.utils import load_img

warnings.filterwarnings("ignore")

def get_ds_path(ds_name: str) -> str:
    data_sources = {
        "levir-cd":levirCD_path,
    }
    if not (ds_name in data_sources):
        raise ValueError("Please provide valid dataset name")

    return data_sources[ds_name]

def load_ds(ds_name: str, **kwargs) -> pd.DataFrame:
    data_sources_loader = {
        "levir-cd": load_levircd_sample,
    }
    if not (ds_name in data_sources_loader):
        raise ValueError("Please provide valid dataset name")

    return data_sources_loader[ds_name](**kwargs)
        
@dataclass
class MetaItem:
    A_path: str
    B_path: str
    label_path: str
        

class BiTemporalDataset(Dataset):
    def __init__(self,
                 name: str=None,
                 items: List[MetaItem]=None,
                 dtype: str = "train", 
                 transform: Any=None,
                 seed: int=SEED) -> None:
        
        if not any([name, items]):
            raise("Please provide at least items or dataset name")
        
        self.items = load_ds(ds_name=name, data_type=dtype) if items is None else items
        self.transform = transform
        self.seed = seed
    
    def __len__(self) -> int:
        return self.items.shape[0]

    def __getitem__(self, index) -> Any:
        label_path, A_path, B_path = self.items.iloc[index].values
        img_A = load_img(A_path)
        img_B = load_img(B_path)
        label = load_img(label_path)

        sample = {
            "img_A":img_A,
            "img_B":img_B,
            "label":label,
            "index": index
        }

        # add generation prompts based on each label zone

        if self.transform:
            sample = self.transform(sample)

        return sample


class PromptDataset(Dataset):
    def __init__(self, 
                 path: str=None, 
                 prompts :Any=None,
                 length: int = None,
                 n_points: int=32) -> None:
        
        if not any([length, prompts]):
            raise RuntimeError("Length or prompts should be specified")
        
        self.length = length
        self.prompts = self.generate_grid(n_points) if prompts is None else prompts
    
    def generate_grid(self, n_points: int):
        return np.tile(generate_grid_prompt(n_points), (self.length, 1, 1))
    
    def __len__(self) -> int:
        return self.prompts.shape[0]
    
    def __getitem__(self, index) -> Any:
        return {"point_coord": self.prompts[index]}