from typing import Any, Dict, Optional, Tuple, Union
from deprecated import deprecated
from omegaconf import DictConfig
import torch
import numpy as np
import torch.nn.functional as F

from commons.config import DEVICE, IMG_SIZE
from src.models.commons.mask_process import extract_object_from_batch
from src.models.segment_anything.utils.transforms import ResizeLongestSide

from src.models.segment_anything.utils.amg import build_point_grid
from src.commons.utils import apply_histogram

PX_MEAN = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1).to(DEVICE)
PX_STD = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1).to(DEVICE)


def generate_grid_prompt(n_points, img_size: int = IMG_SIZE) -> np.ndarray:
    return build_point_grid(n_points) * img_size


# TODO : refacto generate_prompt() inputs cleaner
def generate_prompt(
    mask, dtype: str, n_point: int, kwargs: Union[Dict, DictConfig]
) -> torch.Tensor:
    """Generate n_point prompts for a mask : grid or sample mode (dtype)"""
    img_size = mask.shape[-1]
    match dtype:
        case "grid":
            point_per_side = int(np.sqrt(n_point))
            prompt = torch.as_tensor(
                generate_grid_prompt(point_per_side, img_size=img_size)
            )
            labels = torch.ones(len(prompt))
        case "sample":
            loc = kwargs.get("loc", "center")
            prompt, labels = PointSampler().sample(mask, n_point, loc=loc)
        case _:
            raise ValueError("Please provide valid prompt builder name")

    return prompt.to(torch.float32), labels.to(torch.int8)


class PointSampler:
    """Prompt sampler - restricted to points"""

    def sample(self, mask: torch.Tensor, n_point: int, loc: str):

        # empty return
        sample_coords = torch.zeros((n_point, 2), dtype=torch.float32)

        _register_sample_method = {
            "random": self.draw_random_point,
            "center": self.draw_center_point,
        }
        if loc not in list(_register_sample_method):
            raise ValueError(
                f"loc method not valid. Valid values for loc : {list(_register_sample_method)}"
            )

        if mask.ndim < 3:
            mask = mask.unsqueeze(0)

        # extract shapes from mask
        shapes = extract_object_from_batch(mask).squeeze(0)

        print("SHAPE", shapes.shape)
        # check if there is some shapes
        if shapes.shape[0] > 1 or torch.sum(shapes):
            # we sample with replacement to keeping same tensor dimensions over batch if not enough shapes
            id_draw = torch.multinomial(
                torch.arange(shapes.shape[0], dtype=torch.float),
                n_point,
                replacement=False if shapes.shape[0] >= n_point else True
            )
            # get the coord of the pixels shapes (M x 3) - M number of not zeros pixels
            coords_candidates = torch.nonzero(shapes[id_draw]).to(torch.float)

            # iterate over the shapes
            sample_coords = torch.stack(
                [
                    # sample on masked data based on shape index - ignore index dim => (N, 2)
                    _register_sample_method[loc](
                        coords_candidates[coords_candidates[:, 0] == s][:, 1:]
                    )
                    for s in torch.unique(coords_candidates[:, 0])
                ]
            )


        # simulate point type (foreground / background)
        labels_points = torch.ones(len(sample_coords))

        return sample_coords, labels_points

    def draw_random_point(self, shape):
        """draw one random point from shape"""
        idx = torch.multinomial(
            torch.arange(shape.shape[0], dtype=torch.float), num_samples=1
        ).squeeze(0)
        # invert pixels coords to x, y
        return torch.flip(shape[idx], dims=(0,))

    def draw_center_point(self, shape):
        # TODO: modify with weighted avg approximation
        return torch.flip(torch.mean(shape, dim=0).to(int), dims=(0,))


class DefaultTransform:
    """Scale Img to square IMG_SIZE preserving original ratio and pad"""

    def __init__(self, half_precision: bool = False) -> None:
        self.transform = ResizeLongestSide(IMG_SIZE[0])
        self.precision = half_precision

    def __call__(self, sample: Dict) -> Dict:

        img_A = self.transform.apply_image(sample["img_A"])
        img_B = self.transform.apply_image(sample["img_B"])

        img_A = apply_histogram(img_A, img_B, blend_ratio=0.5)

        img_A = self.process(img_A)
        img_B = self.process(img_B)
        label = self.process(sample["label"])

        new_sample = sample | dict(img_A=img_A, img_B=img_B, label=label)
        return new_sample

    def process(self, input: np.ndarray) -> torch.Tensor:

        input_tensor = self.to_tensor(input)
        input_tensor = self.pad_tensor(input_tensor)

        if self.precision:
            input_tensor = input_tensor.half()
        return input_tensor

    def to_tensor(self, img: np.ndarray) -> torch.Tensor:
        input_image_torch = torch.as_tensor(img, device=DEVICE, dtype=torch.float)
        if input_image_torch.ndim > 2:
            input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()
        return input_image_torch

    def pad_tensor(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        # IMG_SIZE is the encoder image img_size
        padh = IMG_SIZE[0] - h
        padw = IMG_SIZE[0] - w
        x = F.pad(x, (0, padw, 0, padh))
        return x
