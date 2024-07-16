import os
from pathlib import Path
from typing import Any, Union
import numpy as np
import pandas as pd
from commons.constants import SECOND_NO_CHANGE_RGB, NamedDataset
from commons.constants import SECOND_PATH, LEVIRCD_PATH, SEED
import torch
from torch.utils.data import Dataset
from omegaconf import OmegaConf, DictConfig

# Ignore warnings
import warnings

from src.data.process import generate_prompt
from commons.utils_io import make_path
from src.models.segment_any_change.config_run import ExperimentParams
from src.commons.utils import extract_number, load_img

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

            label_path = row["label"]
            print(row["label"])

        elif self.name == NamedDataset.SECOND.value:
            img_A = load_img(row["A"])
            img_B = load_img(row["B"])
            label_A = load_img(row["label_A"])
            # label_B = load_img(row["label_B"])
            label = (
                np.any(label_A != SECOND_NO_CHANGE_RGB, axis=-1).astype(np.uint8) * 255
            )
            print(row["label_A"])
            label_path = row["label_A"]

        sample = {
            "img_A": img_A,
            "img_B": img_B,
            "label": label,
            "label_path": label_path,
        }

        if self.transform:
            sample = self.transform(sample)

        prompt_coords, prompt_labels, new_label = generate_prompt(
            sample["label"], self.params.prompt_type, self.params.n_prompt, self.params
        )
        #print("LABEL", torch.unique(new_label.flatten()))
        # check if we need to process labels
        # note : point coords are computed on transformed img (may be resized)
        sample = sample | dict(
            label=new_label, index=index, point_coords=prompt_coords, point_labels=prompt_labels
        )
        return sample


def load_second_sample(
    size: Union[int, float, Any] = None, data_type="train", seed=SEED
) -> pd.DataFrame:
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
        "train": make_path("SECOND_train_set", SECOND_PATH),
        "test": make_path("SECOND_total_test/test", SECOND_PATH),
    }
    path = path_dict.get(data_type, None)

    if not path:
        raise ValueError("Please provide valid data_type : train, test, val")

    labels_A = sorted(
        [
            make_path(_, path, "label1")
            for _ in os.listdir(Path(path, "label1"))
            if Path(_).suffix == ".png"
        ]
    )

    labels_B = sorted(
        [
            make_path(_, path, "label2")
            for _ in os.listdir(Path(path, "label2"))
            if Path(_).suffix == ".png"
        ]
    )

    files_A = sorted(
        [
            make_path(_, path, "im1")
            for _ in os.listdir(Path(path, "im1"))
            if Path(_).suffix == ".png"
        ]
    )
    files_B = sorted(
        [
            make_path(_, path, "im2")
            for _ in os.listdir(Path(path, "im2"))
            if Path(_).suffix == ".png"
        ]
    )

    df = pd.DataFrame(
        {"label_A": labels_A, "label_B": labels_B, "A": files_A, "B": files_B}
    )

    if isinstance(size, float):
        return df.sample(frac=size, random_state=seed)
    elif isinstance(size, int):
        return df.sample(n=size, random_state=seed)
    else:
        return df


def load_levircd_sample(
    size: Union[int, float, Any] = None, data_type="train", seed=SEED
) -> pd.DataFrame:
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
        "train": make_path("train", LEVIRCD_PATH),
        "test": make_path("test", LEVIRCD_PATH),
        "val": make_path("val", LEVIRCD_PATH),
    }
    path = path_dict.get(data_type, None)

    if not path:
        raise ValueError("Please provide valid data_type : train, test, val")

    labels = sorted(
        [
            make_path(_, path, "label")
            for _ in os.listdir(Path(path, "label"))
            if Path(_).suffix == ".png"
        ],
        key=extract_number,
    )
    files_A = sorted(
        [
            make_path(_, path, "A")
            for _ in os.listdir(Path(path, "A"))
            if Path(_).suffix == ".png"
        ],
        key=extract_number,
    )

    files_B = sorted(
        [
            make_path(_, path, "B")
            for _ in os.listdir(Path(path, "B"))
            if Path(_).suffix == ".png"
        ],
        key=extract_number,
    )

    df = pd.DataFrame({"label": labels, "A": files_A, "B": files_B})

    if isinstance(size, float):
        return df.sample(frac=size, random_state=seed)
    elif isinstance(size, int):
        return df.sample(n=size, random_state=seed)
    else:
        return df
