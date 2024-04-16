from typing import Any, Union
import pandas as pd
import os
from magic_pen.config import levirCD_path, SEED
from pathlib import Path
import numpy as np

from magic_pen.io import make_path

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