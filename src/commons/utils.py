from enum import Enum
from functools import wraps
import re
from typing import Any, Dict, List, Tuple, Union
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import cv2
from commons.utils_io import load_img

from commons.config import DEVICE
import time
from collections.abc import Iterable
import logging
import torch
from skimage.exposure import match_histograms
import seaborn as sns  # type: ignore
import torch.nn.functional as F
from torchvision.utils import make_grid

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
    AUTHOR = "v2"


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


def resize(
    tensor: Union[torch.Tensor, np.ndarray], target_size: Tuple[int, int]
) -> torch.Tensor:

    if not isinstance(tensor, torch.Tensor):
        tensor = torch.as_tensor(tensor)

    if tensor.ndim < 4:
        tensor = tensor.unsqueeze(0)

    if tensor.dtype != torch.float32:
        tensor = tensor.to(torch.float32)

    tensor = F.interpolate(tensor, target_size, mode="bicubic", align_corners=False)

    return tensor.squeeze(0)


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


def show_prediction_sample(output: Dict, idx: int = None):
    """Show sample of given batch in a row plot :
    - img_A
    - img_B
    - label
    - prediction (masks aggregated)
    ...
    Args:
        output (Dict): output model with batch
        idx (int, optional): sample index in the batch. Defaults to None.
    """
    masks = output["pred"]["masks"].cpu().squeeze(0)
    img_A = output["batch"]["img_A"].cpu().squeeze(0)
    img_B = output["batch"]["img_B"].cpu().squeeze(0)
    label = output["batch"]["label"].cpu().squeeze(0)
    # raw_masks = output["pred"]["raw_masks"].cpu().squeeze(0)
    masks_change = output["pred"]["all_changes"].cpu().squeeze(0)

    prompts = output["batch"]["point_coords"].cpu().squeeze(0)

    if idx is not None:
        masks = masks[idx].squeeze(0)
        img_A = img_A[idx].squeeze(0)
        img_B = img_B[idx].squeeze(0)
        label = label[idx].squeeze(0)
        prompts = prompts[idx].squeeze(0)
        # raw_masks = raw_masks[idx].squeeze(0)
        masks_change = masks_change[idx].squeeze(0)

    if masks.ndim == 3:
        masks = torch.sum(masks, dim=0)
        # raw_masks = torch.sum(raw_masks, dim=0)
        masks_change = torch.sum(masks_change, dim=0)

    imgs = [img_A, img_B, label, masks_change, masks]
    names = ["img_A", "img_B", "label", "all_changes", "sim_changes"]

    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False, figsize=(10, 10))
    for i, sample in enumerate(zip(imgs, names)):
        img, name = sample
        if name.startswith("im"):
            img = to_numpy(img, transpose=True) / 255
            axs[0, i].imshow(img)

        else:
            img = to_numpy(img, transpose=False)
            axs[0, i].imshow(img, cmap="grey")

        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        axs[0, i].set_title(name)
        if name == "img_B":
            if prompts.shape[0] < 100:  # prevent showing grid
                colors = [
                    np.random.choice(range(256), size=3) / 255
                    for _ in range(len(prompts))
                ]
                for pt, c in zip(prompts, colors):
                    axs[0, i].scatter(*pt, color=c, marker="*", s=50)


def rad_to_degre(x):
    return x * 180 / np.pi


def to_degre(x):
    return rad_to_degre(np.arccos(-x))


def to_degre_torch(x):
    return rad_to_degre(torch.arccos(-x))


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


def rm_substring(string, substring):
    return string.replace(substring, "")


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


def shift_range_values(arr, new_bounds=[0, 1]):
    old_range = torch.max(arr) - torch.min(arr)
    new_range = new_bounds[1] - new_bounds[0]
    shit_arr = (((arr - torch.min(arr)) * new_range) / old_range) + new_bounds[0]
    return shit_arr


def create_overlay_outcome_cls(
    tp: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor
) -> np.ndarray:
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


def get_units_cnt_px(tp, fp, tn, fn):
    """Pixel level counts"""

    cnt_tp = torch.count_nonzero(tp)
    cnt_fp = torch.count_nonzero(fp)
    cnt_fn = torch.count_nonzero(fn)
    cnt_tn = torch.count_nonzero(tn)

    return cnt_tp, cnt_fp, cnt_fn, cnt_tn


def get_units_cnt_obj(tp, fp, tn, fn):
    return tp, fp, tn, fn


def plot_confusion_matrix(confusion_matrix, fig_return: bool = True):
    """Tested for binary classification"""
    # create the confusion matrix as a numpy array
    # confusion_matrix = confusion_matrix / np.sum(confusion_matrix)
    # create a heatmap of the confusion matrix using seaborn
    # print(confusion_matrix)
    ax = sns.heatmap(
        confusion_matrix,
        annot=True,
        cmap="YlGnBu",
        fmt=".2f",
        xticklabels=["no change", "change"],
        yticklabels=["no change", "change"],
        cbar_kws={"shrink": 0.5},
        vmin=0,
        vmax=1,
    )

    # add labels and title to the plot
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    if not fig_return:
        # show the plot
        plt.show()
    else:
        fig = ax.get_figure()
        plt.close()
        return fig


def create_grid_batch(preds, batch, tp, fp, fn) -> np.ndarray:
    """create image grid from sample (imgA, imgB), label and masks predictions"""
    sample = []
    images_A = batch["img_A"].cpu()
    images_B = batch["img_B"].cpu()
    labels = batch["label"].cpu()
    img_outcome_cls = torch.zeros(images_A.shape[-2:])

    # to batchify ?
    for i in range(images_A.size(0)):

        img_A = images_A[i]
        img_B = images_B[i]
        # Align to 3 channels
        label = labels[i].unsqueeze(0).repeat(3, 1, 1)
        img_outcome_cls = create_overlay_outcome_cls(tp[i], fp[i], fn[i])

        # Stack individual masks and align to 3 channels
        pred = (
            torch.sum(preds[i, ...], axis=0)
            .unsqueeze(0)
            .repeat(3, 1, 1)
            .to(torch.uint8)
        )
        pred = shift_range_values(pred, new_bounds=[0, 255]).to(torch.uint8)
        row = torch.stack((img_A, img_B, label, pred, img_outcome_cls), dim=0)
        # combined stack as row
        row = make_grid(row, nrow=row.shape[0], padding=20, pad_value=1, normalize=True)
        sample.append(row)

    grid = make_grid(sample, nrow=1, padding=20, pad_value=1, scale_each=True)

    return grid


def extract_preds_cls(
    outputs: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    tp = outputs["pred_UnitsMetricCounts"]["tp_indices"]
    fp = outputs["pred_UnitsMetricCounts"]["fp_indices"]
    fn = outputs["pred_UnitsMetricCounts"]["fn_indices"]
    tn = outputs["pred_UnitsMetricCounts"]["tn_indices"]

    return tp, fp, fn, tn

def extract_number(file_path):
    match = re.search(r'_(\d+).png', file_path)
    if match:
        return int(match.group(1))
    return np.inf  

