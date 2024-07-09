from dataclasses import asdict, dataclass
from typing import Any, Union
import numpy as np
import pandas as pd
from commons.constants import SECOND_NO_CHANGE_RGB, SECOND_RGB_TO_CAT, NamedDataset
from commons.config import SECOND_PATH, LEVIRCD_PATH, SEED
from PIL import Image
from torch.utils.data import Dataset
from omegaconf import OmegaConf, DictConfig

# Ignore warnings
import warnings

from src.data.process import generate_grid_prompt, generate_prompt
from commons.utils_io import load_levircd_sample, load_second_sample
from src.models.segment_any_change.config_run import ExperimentParams
from src.commons.utils import load_img

warnings.filterwarnings("ignore")


def get_ds_path(ds_name: str) -> str:
    data_sources = {
        NamedDataset.LEVIR_CD.value: LEVIRCD_PATH,
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
        params: Union[DictConfig, ExperimentParams] = None,
    ) -> None:

        if name is None:
            raise RuntimeError("Please provide at least items or dataset name")

        if not any([params.prompt_type, params.n_prompt]):
            raise RuntimeError(
                "Please provide prompt generation parameter : prompt_type and points_per_side"
            )

        self.items = load_ds(ds_name=name, data_type=dtype)

        self.transform = transform
        self.seed = seed
        self.name = name

        # warning override params to default if not exists
        self.params = (
            params if isinstance(params, DictConfig) else OmegaConf.structured(params)
        )

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

        sample = {
            "img_A": img_A,
            "img_B": img_B,
            "label": label,
        }

        if self.transform:
            sample = self.transform(sample)

        prompt_coords, prompt_labels = generate_prompt(
            sample["label"], self.params.prompt_type, self.params.n_prompt, self.params
        )
        # note : point coords are computed on transformed img (may be resized)

        sample = sample | dict(
            index=index, point_coords=prompt_coords, point_labels=prompt_labels
        )

        return sample
