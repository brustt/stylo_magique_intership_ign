from functools import wraps
from typing import Tuple, Union
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import cv2
from magic_pen.io import load_levircd_sample
from segment_any_change.sa_dev import sam_model_registry
from magic_pen.config import DEVICE, sam_dict_checkpoint
import time
from collections.abc import Iterable
import logging
import torch

# TO DO : define globally
logging.basicConfig(format="%(asctime)s - %(levelname)s ::  %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def flush_memory():
    import gc

    gc.collect()
    torch.cuda.empty_cache()


def to_tensor(arr: np.ndarray, transpose:bool=False, dtype=torch.float, device=DEVICE) -> torch.Tensor:
    if transpose:
        arr = arr.transpose((2, 0, 1))
    return torch.as_tensor(
                arr, dtype=dtype, device=device
            )


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
    sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones(
        (
            sorted_anns[0]["segmentation"].shape[0],
            sorted_anns[0]["segmentation"].shape[1],
            4,
        )
    )
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann["segmentation"]
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


def rad_to_degre(x):
    return x * 180 / np.pi


def to_degre(x):
    return rad_to_degre(np.arccos(-x))


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        logger.info(f"Function {func.__name__} Took {total_time:.4f} seconds")
        return result
    return timeit_wrapper


def flatten(xs):
    for x in xs:
        if isinstance(x, Iterable) and not isinstance(x, (dict)):
            yield from flatten(x)
        else:
            yield x


if __name__ == "__main__":
    df = load_levircd_sample(size=10, data_type="train")
    print(df.shape)
    print(df)
