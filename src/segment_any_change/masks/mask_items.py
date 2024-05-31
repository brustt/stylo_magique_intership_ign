from enum import Enum
import itertools
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from skimage.filters import threshold_otsu
import torch
from magic_pen.config import DEVICE
from segment_any_change.utils import flatten, to_degre


class ImgType(Enum):
    A = 0
    B = 1


class FilteringType(Enum):
    Inf = "inf"
    Sup = "sup"


@dataclass
class ItemProposal:
    """
    Base class for change proposal item
    """

    mask: np.ndarray
    embedding: np.ndarray
    confidence_score: float
    id: int = field(default_factory=itertools.count().__next__, init=False)
    meta: List[Dict]
    iou_pred: Union[np.ndarray, float]
    chgt_angle: float = None
    from_img: List[ImgType] = None

    def __post_init__(self):
        if self.chgt_angle is None:
            self.chgt_angle = to_degre(self.confidence_score)

    # not sure it's the best way with dataclass
    def setter(self, varname, value):
        return setattr(self, varname, value)


@dataclass
class ListProposal:
    """
    List wrapper for change proposal

    Raises:
        ValueError: not a valid filtering method

    Returns:
        _type_: List[ItemProposal] # to change to self
    """

    items: Optional[List[ItemProposal]] = None
    mask_ci: np.ndarray = None

    def __post_init__(self) -> None:
        if self.items is None:
            self.items = []

    def add_item(self, item) -> None:
        if item.id not in [_.id for _ in self.items]:
            self.items.append(item)

    @property
    def masks(self) -> torch.Tensor:
        return torch.as_tensor(
            np.stack([m.mask for m in self.items]), dtype=torch.int8, device=DEVICE
        )

    @property
    def iou_preds(self) -> torch.Tensor:
        return torch.as_tensor(
            np.stack([m.iou_pred for m in self.items]), dtype=torch.float, device=DEVICE
        )

    @property
    def confidence_scores(self) -> np.ndarray:
        return np.stack([m.confidence_score for m in self.items])

    def set_mask_ci(self, mask: np.ndarray) -> None:
        self.mask_ci = mask

    def set_items(self, items: List[ItemProposal]) -> None:
        self.items = items

    def rm_item(self, id: int) -> None:
        self.items = [_ for _ in self.items if _.id != id]

    def __getitem__(self, idx) -> ItemProposal:
        return self.items[idx]

    def __len__(self) -> int:
        return len(self.items)

    def apply_change_filtering(
        self, method: str, mode: FilteringType, **kwargs
    ) -> float:
        method_factory = {
            "otsu": apply_otsu_items,
            "th": apply_th_items,
        }
        if method is None:
            return None
        if method not in method_factory and isinstance(method, str):
            raise ValueError(
                f"Please provide valid filtering method : {list(method_factory)}"
            )
        if isinstance(method, (float, int)):
            kwargs["th"] = method
            method = "th"

        self.items, th = method_factory[method](self.items, mode, **kwargs)
        return th

    def update_field(self, field, values: List[Any]) -> None:
        if len(values) != len(self.items):
            raise ValueError("Values length and ListProposal length should be the same")
        for i, v in enumerate(values):
            self.items[i].setter(field, v)


def apply_otsu_items(
    items: List[ItemProposal], mode: FilteringType
) -> Tuple[List[ItemProposal], float]:
    th = threshold_otsu(np.array([_.chgt_angle for _ in items]))
    return apply_th_items(items, mode, th)


def apply_th_items(
    items: List[ItemProposal], mode: FilteringType, th: float
) -> Tuple[List[ItemProposal], float]:
    sup_filtering = lambda l, th: [item for item in l if item.chgt_angle >= th]
    inf_filtering = lambda l, th: [item for item in l if item.chgt_angle <= th]

    mode_dict = {FilteringType.Inf: inf_filtering, FilteringType.Sup: sup_filtering}
    return (mode_dict[mode](items, th), th)


def create_union_object(item_A: ItemProposal, item_B: ItemProposal) -> ItemProposal:
    """Create union of two object

    Fusion of extent
    - Mean of confidence score / angle
    - concat other attributes

    Args:
        item_A (ItemProposal): item 1
        item_B (ItemProposal): item 2

    Returns:
        ItemProposal: new item created
    """
    # filter on sim before merge ?
    return ItemProposal(
        mask=np.logical_or(item_A.mask, item_B.mask).astype(np.uint8),
        confidence_score=np.mean([item_A.confidence_score, item_B.confidence_score]),
        meta=([item_A.meta] + [item_B.meta]),
        chgt_angle=np.mean([item_A.chgt_angle, item_B.chgt_angle]),
        from_img=[item_A.from_img, item_B.from_img],
        embedding=np.mean([item_A.embedding, item_B.embedding], axis=0),
        iou_pred=np.mean([item_A.iou_pred, item_B.iou_pred], axis=0),
    )


def create_change_proposal_items(
    masks: List[Dict], ci: List[float], type_img: ImgType, embeddings: np.ndarray
) -> List[ItemProposal]:
    return [
        ItemProposal(
            mask=m["segmentation"],
            confidence_score=c,
            meta={k: v for k, v in m.items() if k != "segmentation"},
            from_img=type_img,
            embedding=emb,
            iou_pred=m["iou_pred"],
        )
        for m, c, emb in zip(masks, ci, embeddings)
        if not np.isnan(c)
    ]


def thresholding_factory(
    arr: np.ndarray, method, mode: FilteringType, **kwargs
) -> Tuple[np.ndarray, float]:
    method_factory = {
        "otsu": apply_otsu,
        "th": apply_th,
    }
    if method is None:
        return None
    if method not in method_factory and isinstance(method, str):
        raise ValueError(
            f"Please provide valid filtering method : {list(method_factory)}"
        )
    if isinstance(method, (float, int)):
        kwargs["th"] = method
        method = "th"

    arr, th = method_factory[method](arr, mode, **kwargs)
    return arr, th


def apply_otsu(arr: np.ndarray, mode: FilteringType) -> Tuple[np.ndarray, float]:
    th = threshold_otsu(arr)
    return apply_th(arr, mode, th)


def apply_th(
    arr: np.ndarray, mode: FilteringType, th: Union[int, float]
) -> Tuple[np.ndarray, float]:
    mode_dict = {
        FilteringType.Inf: lambda l, th: (arr <= th).astype(np.uint8),
        FilteringType.Sup: lambda arr, th: (arr >= th).astype(np.uint8),
    }
    return (mode_dict[mode](arr, th), th)
