from typing import Optional, Tuple
import torch
import numpy as np
import torch.nn.functional as F

from magic_pen.config import DEVICE, IMG_SIZE
from segment_any_change.sa_dev.utils.transforms import ResizeLongestSide

"""
inference for batch images
"""







def predict_batch(model, input_points, input_labels):

    pass