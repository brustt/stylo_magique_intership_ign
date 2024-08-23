from typing import Any, Dict, List, Optional, Tuple, Union
from deprecated import deprecated
from omegaconf import DictConfig
import torch
import numpy as np
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils import data

from commons.constants import DEVICE, IMG_SIZE
from src.models.commons.mask_process import binarize_mask, extract_object_from_batch
from src.models.segment_anything.utils.transforms import ResizeLongestSide

from src.models.segment_anything.utils.amg import build_point_grid
from src.commons.utils import apply_histogram

def generate_grid_prompt(n_points, img_size: int = IMG_SIZE) -> np.ndarray:
    return build_point_grid(n_points) * img_size

@deprecated
def collate_align_prompt(input: List[Any]):
    """Stack tensors with different size (prompts) before create batch - used in dataloader"""

    prompt_pts = [d["point_coords"] for d in input]
    prompt_labels = [d["point_labels"] for d in input]

    # we set torch.inf as value to ignore prompt
    batch_pts = pad_sequence(prompt_pts, batch_first=True, padding_value=torch.inf)
    # negative prompt : 0 is ignore - (same as positive prompt btw)
    batch_labels = pad_sequence(prompt_labels, batch_first=True, padding_value=0)

    for i in range(len(input)):
        input[i]["point_coords"] = batch_pts[i]
        input[i]["point_labels"] = batch_labels[i]

    return data._utils.collate.default_collate(input)

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
            loc = kwargs.get('loc', "center")
            n_shape = kwargs.get('n_shape', None)
            prompt, labels = PointSampler().sample(mask=mask,  n_point_per_shape=n_point, loc=loc, n_shape=n_shape)
        case _:
            raise ValueError("Please provide valid prompt builder name")

    return prompt.to(torch.float32), labels.to(torch.int8)

class PointSampler:
    """Prompt sampler - restricted to points
    
    Each generated points is under coordinates format (X,Y) in pixels.
    """
    MIN_AREA = 25
    
    def __init__(self):
        self._register_sample_method = {
            "random": self.draw_random_point,
            "center": self.draw_center_point,
        }

    def sample_candidates_shapes(self, shapes: torch.Tensor, n_shape: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # assign equals weights
        probs = torch.ones(shapes.shape[0]) / shapes.shape[0]
        # Sample the shapes
        id_candidates_shapes = torch.multinomial(
            probs,
            n_shape,
            # we sample with replacement to keeping same tensor dimensions over batch if not enough shapes
            replacement=False if (shapes.shape[0] >= n_shape) else True,
        )
        # get the coord of the pixels shapes (M x 3) - M number of not zeros pixels
        coords_candidates = torch.nonzero(shapes[id_candidates_shapes]).to(torch.float)
        return coords_candidates, id_candidates_shapes
        
    def sample(self, mask: torch.Tensor,  n_point_per_shape: int, loc: str, n_shape: int):
        """
        Sample m points over n random shape
        Return new label if a subset of shapes (n_shape) is selected
        """

        if loc not in list(self._register_sample_method):
            raise ValueError(
                f"loc method not valid. Valid values for loc : {list(self._register_sample_method)}"
            )
        if not n_shape:
            raise ValueError("please provide n_shape to sample. One point per shape")
            
        if mask.ndim < 3:
            mask = mask.unsqueeze(0)
        # track id shapes if we a subset of shapes
        id_selected_shapes = None
        
        # extract shapes from mask - squeeze batch dimension
        shapes = extract_object_from_batch(mask).squeeze(0)
        print("shape", shapes.shape)
        # filter on areas
        areas = torch.sum(shapes, dim=(1, 2))
        indices = torch.where(areas > self.MIN_AREA)[0]
        shapes = shapes[indices,:,:]

        # check if there is some shapes extracted - check sum for no-shapes return
        # check > 1 first for speed in case of shapes - return no shapes :  (1 x) 1 x H x W
        if shapes.shape[0] > 1 or torch.sum(shapes):
            # extract all shapes (max of batch) if there are not enough shapes
            n_shape = min(n_shape,  shapes.shape[0])
            coords_candidates, id_selected_shapes = self.sample_candidates_shapes(shapes, n_shape)
            # first column of coords_candidates == index of shape
            # iterate over the shapes
            sample_coords = torch.cat(
                [
                    # mask coordinates based on shape index
                    # select only coordinates dims for _register_sample_method : [:, 1:] => (N, 2)
                    self._register_sample_method[loc](
                        coords_candidates[coords_candidates[:, 0] == s][:, 1:], n_point_per_shape
                    )
                    for s in torch.unique(coords_candidates[:, 0])
                ]
            )
            # simulate point type (foreground / background) - foreground default
            labels_points = torch.ones(len(sample_coords))
        else:
            # empty return = sample random points
            #sample_coords = torch.zeros((n_shape*n_point_per_shape, 2), dtype=torch.float32) - 1000
            sample_coords = torch.as_tensor(np.random.randint(0, mask.shape[-1], size=(n_shape, 2)))
            # label - 1
            labels_points = torch.zeros(len(sample_coords)) - 1

        return sample_coords, labels_points

    def draw_random_point(self, shape, n_point):
        """draw one random point from shape"""
        idx = torch.multinomial(
            torch.ones(shape.shape[0], dtype=torch.float), num_samples=n_point
        )
        # invert pixels coords to (x, y) format
        return torch.flip(shape[idx], dims=(1,))

    def draw_center_point(self, shape, n_point):
        """
        Sample approximation center. Proxy for hard concave object where "natural center" (simple average) doesn't belong to the shape.
        shape : (M, 2) : (shape's pixels, px coordinates dim)
        """
        # proxy for hard concave object
        visible_center = torch.mean(shape, dim=0).to(int)
        # euclidean distance
        dist_center = torch.cdist(visible_center.unsqueeze(0).to(torch.float), shape, p=2.).view(-1)
        idx = torch.nonzero(dist_center).view(-1)
        dist_center, shape = dist_center[idx], shape[idx,...]
        # sample point from inverse distance weighting => in favor of "closest center" point - take first 50 pts arbitrary
        # values, indices  = torch.topk(dist_center, min(50, dist_center.shape[0]), largest=False, sorted=True)
        # values, indices = values.view(-1), indices.view(-1)
        idx = torch.multinomial(
            1/dist_center, num_samples=n_point
        )
        # flip to convert to (x, y) format
        return torch.flip(shape[idx], dims=(1,))


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
        label = binarize_mask(self.process(sample["label"]), th=0.)

        new_sample = sample | dict(img_A=img_A, img_B=img_B, label=label)
        return new_sample

    def process(self, input: np.ndarray) -> torch.Tensor:

        input_tensor = self.to_tensor(input)
        input_tensor = self.pad_tensor(input_tensor)

        if self.precision:
            input_tensor = input_tensor.half()
        return input_tensor

    def to_tensor(self, img: np.ndarray) -> torch.Tensor:

        input_image_torch = torch.tensor(img, dtype=torch.float32)

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
