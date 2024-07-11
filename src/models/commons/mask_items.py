from copy import deepcopy
from enum import Enum
import itertools
from dataclasses import dataclass, field
from typing import Any, Dict, ItemsView, List, Optional, Tuple, Union
from deprecated import deprecated
import numpy as np
from skimage.filters import threshold_otsu
import torch
from src.models.segment_anything.utils.amg import MaskData
from src.commons.utils import to_numpy


class ImgType(Enum):
    A = 0
    B = 1


class FilteringType(Enum):
    Inf = "inf"
    Sup = "sup"


class MaskData:
    """
    A structure for storing masks and their related data in batched format.
    Implements basic filtering and concatenation.
    """

    def __init__(self, **kwargs) -> None:
        for v in kwargs.values():
            assert isinstance(
                v, (list, np.ndarray, torch.Tensor)
            ), "MaskData only supports list, numpy arrays, and torch tensors."
        self._stats = dict(**kwargs)

    def __setitem__(self, key: str, item: Any) -> None:
        assert isinstance(
            item, (list, np.ndarray, torch.Tensor)
        ), "MaskData only supports list, numpy arrays, and torch tensors."
        self._stats[key] = item

    def __delitem__(self, key: str) -> None:
        del self._stats[key]

    def __getitem__(self, key: str) -> Any:
        return self._stats[key]

    def items(self) -> ItemsView[str, Any]:
        return self._stats.items()

    def filter(self, keep: torch.Tensor) -> None:
        for k, v in self._stats.items():
            if v is None:
                self._stats[k] = None
            elif isinstance(v, torch.Tensor):
                self._stats[k] = v[torch.as_tensor(keep, device=v.device)]
            elif isinstance(v, np.ndarray):
                self._stats[k] = v[keep.detach().cpu().numpy()]
            elif isinstance(v, list) and keep.dtype == torch.bool:
                self._stats[k] = [a for i, a in enumerate(v) if keep[i]]
            elif isinstance(v, list):
                self._stats[k] = [v[i] for i in keep]
            else:
                raise TypeError(f"MaskData key {k} has an unsupported type {type(v)}.")

    def cat(self, new_stats: "MaskData") -> None:
        for k, v in new_stats.items():
            if k not in self._stats or self._stats[k] is None:
                self._stats[k] = deepcopy(v)
            elif isinstance(v, torch.Tensor):
                self._stats[k] = torch.cat([self._stats[k], v], dim=0)
            elif isinstance(v, np.ndarray):
                self._stats[k] = np.concatenate([self._stats[k], v], axis=0)
            elif isinstance(v, list):
                self._stats[k] = self._stats[k] + deepcopy(v)
            else:
                raise TypeError(f"MaskData key {k} has an unsupported type {type(v)}.")

    def to_numpy(self) -> None:
        for k, v in self._stats.items():
            if isinstance(v, torch.Tensor):
                self._stats[k] = v.detach().cpu().numpy()


def thresholding(
    data: MaskData, attr: str, method: Union[str, float], filtering_type: FilteringType
) -> Any:
    """Apply Thresholding based on change angle"""

    # print(f"thresholding : {method} for {attr}")

    method_factory = {
        "otsu": apply_otsu,
        "th": apply_th,
    }
    params = {}
    data_ = deepcopy(data)

    if (method not in method_factory and isinstance(method, str)) or method is None:
        raise ValueError(
            f"Please provide valid filtering method : {list(method_factory)}"
        )
    if isinstance(method, (float, int)):
        params["th"] = method
        method = "th"

    data_, th = method_factory[method](data_, attr, filtering_type, **params)
    return data_, th


def apply_otsu(
    data: MaskData,
    attr: str,
    filtering_type: FilteringType,
) -> Tuple[MaskData, float]:
    # set otsu threshold on batch
    arr = to_numpy(data[attr], transpose=False)
    th = threshold_otsu(arr[~np.isnan(arr)])
    # print(th)
    return apply_th(data, attr, filtering_type, th)


def apply_th(
    data: MaskData,
    attr: str,
    filtering_type: FilteringType,
    th: float,
) -> Tuple[MaskData, float]:
    if filtering_type == FilteringType.Inf:
        keep_indices = torch.where(data[attr] < th)[0]
    elif filtering_type == FilteringType.Sup:
        keep_indices = torch.where(data[attr] > th)[0]
    else:
        raise AttributeError("Please provide valid filtering type")
    data.filter(keep_indices)
    return data, th
