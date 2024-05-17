from typing import Any, Dict, Optional, Tuple
from deprecated import deprecated
import torch
import numpy as np
import torch.nn.functional as F

from magic_pen.config import DEVICE, IMG_SIZE
from segment_any_change.sa_dev.utils.transforms import ResizeLongestSide

from segment_any_change.sa_dev.utils.amg import build_point_grid

PX_MEAN = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1).to(DEVICE)
PX_STD = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1).to(DEVICE)



def generate_grid_prompt(n_points) -> np.ndarray:
    return build_point_grid(n_points)

def prepare_prompts(point_coords: Optional[np.ndarray] = None,
                    point_labels: Optional[np.ndarray] = None,
                    box: Optional[np.ndarray] = None,
                    mask_input: Optional[np.ndarray] = None,
                    original_size: Optional[Tuple[int, int]] = IMG_SIZE) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    
    # check resizeLongestSide can take batch prompts and images
    transform = ResizeLongestSide(IMG_SIZE[0])
    
    coords_torch, labels_torch, box_torch, mask_input_torch = None, None, None, None 
    if point_coords is not None:
        assert (
            point_labels is not None
        ), "point_labels must be supplied if point_coords is supplied."
        point_coords = transform.apply_coords(point_coords, original_size)

    coords_torch = torch.as_tensor(
    point_coords, dtype=torch.float, device=DEVICE
    )
    labels_torch = torch.as_tensor(
        point_labels, dtype=torch.int, device=DEVICE
    )            
    
    coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]

    if box is not None:
        box = transform.apply_boxes(box, original_size)
        box_torch = torch.as_tensor(box, dtype=torch.float, device=DEVICE)
        box_torch = box_torch[None, :]
    if mask_input is not None:
        mask_input_torch = torch.as_tensor(
            mask_input, dtype=torch.float, device=DEVICE
        )
        mask_input_torch = mask_input_torch[None, :, :, :]

    return (coords_torch,
            labels_torch,
            box_torch,
            mask_input_torch)


class DefaultTransform:
    """Scale Img to square IMG_SIZE preserving original ratio and pad"""
    def __init__(self) -> None:
        self.transform = ResizeLongestSide(IMG_SIZE[0])

    def __call__(self, sample: Dict) -> Dict:

        img_A, img_B, label = sample.values()

        # ResizeLongestSide can be apply on a batch
        img_A = self.process(img_A)
        img_B = self.process(img_B)
        label = self.process(label)

        return {
            "img_A":img_A,
            "img_B":img_B,
            "label":label
        }

    def process(self, input: np.ndarray) -> torch.Tensor:

        input = self.transform.apply_image(input)
        input_tensor = self.to_tensor(input)
        input_tensor = self.pad_tensor(
            input_tensor
        )
        return input_tensor
    

    def to_tensor(self, img: np.ndarray) -> torch.Tensor:
        input_image_torch = torch.as_tensor(img, device=DEVICE)
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


@deprecated
def prepare_image(input_image: np.ndarray):

    if input_image.shape[-1] != 3:
        raise RuntimeError("Please provide 3 channels image")
    
    transform = ResizeLongestSide(IMG_SIZE[0])
    input_image = transform.apply_image(input_image)

    input_image_torch = torch.as_tensor(input_image, device=DEVICE)
    input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[
        None, :, :, :
    ]
    input_image_torch = preprocess_image(
        input_image_torch
    )
    return input_image_torch



def preprocess_image(x: torch.Tensor) -> torch.Tensor:
    """Normalize pixel values and pad to a square input."""
    x = normalize_img(x)
    x = pad_tensor(x)
    return x

def normalize_img(x:torch.Tensor,pixel_mean: torch.Tensor=PX_MEAN, pixel_std: torch.Tensor=PX_STD) -> torch.Tensor:
    x = (x - pixel_mean) / pixel_std
    return x

def pad_tensor(x: torch.Tensor) -> torch.Tensor:
    h, w = x.shape[-2:]
    # IMG_SIZE is the encoder image img_size
    padh = IMG_SIZE[0] - h
    padw = IMG_SIZE[0] - w
    x = F.pad(x, (0, padw, 0, padh))
    return x