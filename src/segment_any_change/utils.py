from functools import wraps
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import cv2
from magic_pen.io import load_levircd_sample
from segment_any_change.sa_dev import sam_model_registry
from magic_pen.config import DEVICE, sam_dict_checkpoint
import time


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
        print(f"Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds")
        return result

    return timeit_wrapper


if __name__ == "__main__":
    df = load_levircd_sample(size=10, data_type="train")
    print(df.shape)
    print(df)
