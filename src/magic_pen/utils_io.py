import os
import pathlib
import pickle
import pandas as pd
from typing import Dict, Any, Union
from magic_pen.config import *


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
        "train": make_path("train", levirCD_path),
        "test": make_path("test", levirCD_path),
        "val": make_path("val", levirCD_path),
    }
    path = path_dict.get(data_type, None)

    if not path:
        raise ValueError("Please provide valid data_type : train, test, val")

    labels = sorted(
        [
            make_path(_, path, "label")
            for _ in os.listdir(Path(path, "label"))
            if Path(_).suffix == ".png"
        ]
    )
    files_A = sorted(
        [
            make_path(_, path, "A")
            for _ in os.listdir(Path(path, "A"))
            if Path(_).suffix == ".png"
        ]
    )
    files_B = sorted(
        [
            make_path(_, path, "B")
            for _ in os.listdir(Path(path, "B"))
            if Path(_).suffix == ".png"
        ]
    )

    df = pd.DataFrame({"label": labels, "A": files_A, "B": files_B})

    if isinstance(size, float):
        return df.sample(frac=size, random_state=seed)
    elif isinstance(size, int):
        return df.sample(n=size, random_state=seed)
    else:
        return df
