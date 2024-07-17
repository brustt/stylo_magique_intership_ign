# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .build_sam_dev import (
    build_sam,
    build_sam_vit_h,
    build_sam_vit_l,
    build_sam_vit_b,
    sam_model_registry,
)
from .build_sam import (
    build_sam as build_sam_ori,
    build_sam_vit_h as build_sam_vit_h_ori,
    build_sam_vit_l as build_sam_vit_l_ori,
    build_sam_vit_b as build_sam_vit_b_ori,
    sam_model_registry as sam_model_registry_ori,
)
from .predictor import SamPredictor
from .automatic_mask_generator import SamAutomaticMaskGenerator
