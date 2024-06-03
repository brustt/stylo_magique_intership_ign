from enum import Enum
from functools import wraps
import re
from typing import Any, Dict, List, Tuple, Union
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import cv2
from magic_pen.utils_io import load_levircd_sample
from segment_any_change.sa_dev import sam_model_registry
from segment_any_change.sa_dev_v0 import sam_model_registry as sam_model_registry_v0

from magic_pen.config import DEVICE, sam_dict_checkpoint
import time
from collections.abc import Iterable
import logging
import torch
from skimage.exposure import match_histograms

# TO DO : define globally
logging.basicConfig(format="%(asctime)s - %(levelname)s ::  %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

NPDTYPE_TO_OPENCV_DTYPE = {
    np.uint8: cv2.CV_8U,
    np.uint16: cv2.CV_16U,
    np.int32: cv2.CV_32S,
    np.float32: cv2.CV_32F,
    np.float64: cv2.CV_64F,
    np.dtype("uint8"): cv2.CV_8U,
    np.dtype("uint16"): cv2.CV_16U,
    np.dtype("int32"): cv2.CV_32S,
    np.dtype("float32"): cv2.CV_32F,
    np.dtype("float64"): cv2.CV_64F,
}


class SegAnyChangeVersion(Enum):
    RAW = "v0"
    MAP = "v1"


def flush_memory():
    import gc

    gc.collect()
    torch.cuda.empty_cache()


def to_tensor(
    arr: np.ndarray, transpose: bool = True, dtype=torch.float, device=DEVICE
) -> torch.Tensor:
    if transpose:
        arr = arr.transpose(2, 0, 1)
    return torch.as_tensor(arr, dtype=dtype, device=device)


def to_numpy(tensor: torch.Tensor, transpose: bool = True, dtype=None) -> np.ndarray:
    if transpose:
        tensor = tensor.permute(1, 2, 0)
    dtype = dtype if dtype else torch.float
    return tensor.to(dtype).detach().cpu().numpy()


def load_img_cv2(path: str):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def show_pair_img(img_A: Union[str, np.ndarray], img_B: Union[str, np.ndarray]):
    if isinstance(img_A, str):
        img_A = load_img(img_A)
    if isinstance(img_B, str):
        img_B = load_img(img_B)
    pair = np.hstack((img_A, img_B))
    show_img(pair)


def batch_to_list(batch: Dict[str, Any]) -> List[Dict[str, Any]]:
    batch_size = next(iter(batch.values())).size(0)
    batch_list = []
    for i in range(batch_size):
        elem = {k: v[i].unsqueeze(0) for k, v in batch.items()}
        batch_list.append(elem)
    return batch_list


def load_sam(
    model_type: str, model_cls: Any=None, version: str = "dev", device: str = DEVICE
):

    sam = None

    match version:
        case "dev":
            sam = sam_model_registry[model_type](
                checkpoint=sam_dict_checkpoint[model_type], model=model_cls
            ).to(device=device)
        case "raw":
            sam = sam_model_registry_v0[model_type](
                checkpoint=sam_dict_checkpoint[model_type]
            ).to(device=device)
        case _:
            raise ValueError(
                "Please provide valid sam verison implementation : dev, raw"
            )
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


def show_pair_img(img_A: Union[str, np.ndarray], img_B: Union[str, np.ndarray]):
    if isinstance(img_A, str):
        img_A = load_img(img_A)
    if isinstance(img_B, str):
        img_B = load_img(img_B)
    pair = np.hstack((img_A, img_B))
    show_img(pair)


def show_masks(masks, plt, alpha=0.7):
    if len(masks) == 0:
        return
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((masks.shape[1], masks.shape[2], 4))
    img[:, :, 3] = 0
    for ann in masks:
        color_mask = np.concatenate([np.random.random(3), [alpha]])
        img[ann] = color_mask
    ax.imshow(img)
    return img


def show_points(coords, labels, ax, marker_size=25):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color="green",
        marker="o",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )
    ax.scatter(
        neg_points[:, 0],
        neg_points[:, 1],
        color="red",
        marker="o",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )


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

def substring_present(substring, string):
    pattern = re.compile(re.escape(substring))
    match = pattern.search(string)
    return match is not None

def apply_histogram(
    img: np.ndarray, reference_image: np.ndarray, blend_ratio: float = 0.5
) -> np.ndarray:
    """Apply histogram matching to an image.
    Parameters
    ----------
    img : np.ndarray
        Image to apply histogram matching on.
    reference_image : np.ndarray
        Image to use as reference for histogram matching.
    blend_ratio : float, optional
        Ratio of blending between the original image and the histogram matched image, by default 0.5
    Returns
    -------
    np.ndarray
        The image with histogram matching applied.
    """
    if img.dtype != reference_image.dtype:
        raise RuntimeError(
            f"Dtype of image and reference image must be the same. Got {img.dtype} and {reference_image.dtype}"
        )
    reference_image = cv2.resize(reference_image, dsize=(img.shape[1], img.shape[0]))
    channel_axis = img.ndim - 1
    matched = match_histograms(
        np.squeeze(img), np.squeeze(reference_image), channel_axis=channel_axis
    )
    img = cv2.addWeighted(
        matched,
        blend_ratio,
        img,
        1 - blend_ratio,
        0,
        dtype=NPDTYPE_TO_OPENCV_DTYPE[img.dtype],
    )
    return img


def create_overlay_outcome_cls(tp: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor) -> np.ndarray:
    # cannot assign list to tensor => convert to np
    overlay = np.zeros((*tp.shape, 3), dtype=np.uint8)
    
    tp = to_numpy(tp, transpose=False)
    fp = to_numpy(fp, transpose=False)
    fn = to_numpy(fn, transpose=False)

    # tps in green - we keep masks summed
    overlay[tp >= 1] = [0, 255, 0]
    
    # fn in red
    overlay[fn >= 1] = [255, 0, 0]
    
    # fp orange
    overlay[fp >= 1] = [255, 165, 0]

    return to_tensor(overlay, transpose=True)

if __name__ == "__main__":
    df = load_levircd_sample(size=10, data_type="train")
    print(df.shape)
    print(df)
