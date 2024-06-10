from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd
from src.commons.constants import SECOND_NO_CHANGE_RGB, SECOND_RGB_TO_CAT, NamedDataset
from magic_pen.config import SECOND_PATH, levirCD_path, SEED
from PIL import Image
from torch.utils.data import Dataset

# Ignore warnings
import warnings

from magic_pen.data.process import generate_grid_prompt
from magic_pen.utils_io import load_levircd_sample, load_second_sample
from segment_any_change.utils import load_img
import rasterio as rio
from rasterio.plot import reshape_as_image

warnings.filterwarnings("ignore")


def get_ds_path(ds_name: str) -> str:
    data_sources = {
        NamedDataset.LEVIR_CD.value: levirCD_path,
        NamedDataset.SECOND.value: SECOND_PATH,
    }
    if not (ds_name in data_sources):
        raise ValueError("Please provide valid dataset name")

    return data_sources[ds_name]


def load_ds(ds_name: str, **kwargs) -> pd.DataFrame:
    data_sources_loader = {
        NamedDataset.LEVIR_CD.value: load_levircd_sample,
        NamedDataset.SECOND.value: load_second_sample,
    }
    if not (ds_name in data_sources_loader):
        raise ValueError("Please provide valid dataset name")

    return data_sources_loader[ds_name](**kwargs)


class BiTemporalDataset(Dataset):
    def __init__(
        self,
        name: str = None,
        dtype: str = "train",
        transform: Any = None,
        seed: int = SEED,
    ) -> None:

        if name is None:
            raise ("Please provide at least items or dataset name")
        self.items = load_ds(ds_name=name, data_type=dtype)

        self.transform = transform
        self.seed = seed
        self.name = name

    def __len__(self) -> int:
        return self.items.shape[0]

    def __getitem__(self, index) -> Any:
        row = self.items.iloc[index]

        if self.name == NamedDataset.LEVIR_CD.value:
            img_A = load_img(row["A"])
            img_B = load_img(row["B"])
            label = load_img(row["label"])

        elif self.name == NamedDataset.SECOND.value:
            img_A = load_img(row["A"])
            img_B = load_img(row["B"])
            label_A = load_img(row["label_A"])
            # label_B = load_img(row["label_B"])
            label = (
                np.any(label_A != SECOND_NO_CHANGE_RGB, axis=-1).astype(np.uint8) * 255
            )

        sample = {"img_A": img_A, "img_B": img_B, "label": label, "index": index}

        # add generation prompts based on each label zone

        if self.transform:
            sample = self.transform(sample)

        return sample


class PromptDataset(Dataset):
    def __init__(
        self,
        path: str = None,
        prompts: Any = None,
        length: int = None,
        n_points: int = 32,
    ) -> None:

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
