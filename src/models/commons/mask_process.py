import enum
from typing import Dict, List, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
from commons.config import DEVICE
from src.models.segment_anything.utils.amg import (
    MaskData,
    remove_small_regions,
    batched_mask_to_box,
    calculate_stability_score,
    mask_to_rle_pytorch,
    rle_to_mask,
)
from torchvision.ops.boxes import batched_nms
import kornia as K
from torch.nn.utils.rnn import pad_sequence
import logging

from src.commons.utils import to_numpy

# TO DO : define globally
logging.basicConfig(format="%(asctime)s - %(levelname)s ::  %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def nms_wrapper(data, box_nms_thresh):
    keep_by_nms = batched_nms(
        data["boxes"].float(),
        data["iou_preds"],
        torch.zeros_like(data["boxes"][:, 0]),  # categories
        iou_threshold=box_nms_thresh,
    )
    return keep_by_nms


def postprocess_masks(masks, iou_preds, points):

    box_nms_thresh = 0.7
    min_area = 0.0

    data = MaskData(
        masks=masks.flatten(0, 1),
        iou_preds=iou_preds.flatten(0, 1),
        points=torch.as_tensor(points.repeat(masks.shape[1], axis=0)),
    )
    print(data["masks"].shape[0])

    data = filters_masks(data)
    data["boxes"] = batched_mask_to_box(data["masks_binary"])

    keep_by_nms = nms_wrapper(data, box_nms_thresh)
    data.filter(keep_by_nms)
    print(data["masks"].shape[0])

    if min_area > 0.0:
        data["rles"] = mask_to_rle_pytorch(data["masks_binary"])
        data = postprocess_small_regions(data, min_area, box_nms_thresh)

    return data["masks"], data["masks_binary"], data["iou_preds"]


def postprocess_small_regions(
    mask_data: MaskData, min_area: int, nms_thresh: float
) -> MaskData:
    """
    Removes small disconnected regions and holes in masks, then reruns
    box NMS to remove any new duplicates.

    Edits mask_data in place.

    Requires open-cv as a dependency.
    """
    if len(mask_data["rles"]) == 0:
        return mask_data

    # Filter small disconnected regions and holes
    new_masks = []
    scores = []
    for rle in mask_data["rles"]:
        mask = rle_to_mask(rle)

        mask, changed = remove_small_regions(mask, min_area, mode="holes")
        unchanged = not changed
        mask, changed = remove_small_regions(mask, min_area, mode="islands")
        unchanged = unchanged and not changed

        new_masks.append(torch.as_tensor(mask).unsqueeze(0))
        # Give score=0 to changed masks and score=1 to unchanged masks
        # so NMS will prefer ones that didn't need postprocessing
        scores.append(float(unchanged))

    # Recalculate boxes and remove any new duplicates
    masks = torch.cat(new_masks, dim=0)
    boxes = batched_mask_to_box(masks)
    keep_by_nms = batched_nms(
        boxes.float(),
        torch.as_tensor(scores),
        torch.zeros_like(boxes[:, 0]),  # categories
        iou_threshold=nms_thresh,
    )
    mask_data.filter(keep_by_nms)
    return mask_data


def filters_masks(
    data,
    mask_threshold=0.0,
    pred_iou_thresh: float = 0.88,  # could be lower
    stability_score_thresh: float = 0.95,  # could be lower
    stability_score_offset: float = 1.0,
    return_logits: bool = True,
) -> MaskData:

    # Filter by predicted IoU
    if pred_iou_thresh > 0.0:
        keep_mask = data["iou_preds"] > pred_iou_thresh
        data.filter(keep_mask)

    print(f'filter iou_th : {data["masks"].shape[0]}')

    # Calculate stability score
    data["stability_score"] = calculate_stability_score(
        data["masks"],
        mask_threshold,
        stability_score_offset,
    )
    if stability_score_thresh > 0.0:
        keep_mask = data["stability_score"] >= stability_score_thresh
        data.filter(keep_mask)

    print(f' filter stability_score : {data["masks"].shape[0]}')

    if not return_logits:
        # Threshold masks and calculate boxes
        data["masks_binary"] = data["masks"] > mask_threshold
        print(f' filter mask_threshold : {data["masks"].shape[0]}')

    return data


def extract_object_from_batch(masks: torch.Tensor) -> torch.Tensor:
    """Extract individual mask objects from a batch of masks B x H x W

    Args:
        masks (torch.Tensor): (B x H x W) batch masks

    Returns:
        torch.Tensor: (B x N x H x W) with N total objects of batch
    """

    def remap_values(remapping, x):
        index = torch.bucketize(x.ravel(), remapping[0])
        return remapping[1][index].reshape(x.shape)

    # B x 1 x H x W -> B x H x W. Normalize to [0, 1]
    labels_masks = K.contrib.connected_components(
        masks.unsqueeze(1).to(torch.float) / torch.max(masks), num_iterations=500
    ).squeeze(1)
    uniques = torch.unique(labels_masks)
    # Labels from K.connected_component are not 0-index based and sequential integers
    # Need a mapping for one-hot
    remapping = uniques, torch.arange(end=len(uniques))
    remapped_batch = remap_values(remapping, labels_masks)
    # B x H x W => B x N x H x W
    masks = F.one_hot(remapped_batch.long(), len(uniques)).permute(0, 3, 1, 2)
    # remove background mask, i.e no-change
    masks = masks[
        :,
        1:,
        :,
    ]

    if masks.shape[1] == 0:
        # no object found in batch : masks => B x 0 x H x W
        # return B x 1 x H x W zeros masks
        return torch.zeros(masks.shape[0], 1, *masks.shape[-2:])

    return masks


def extract_individual_object_from_mask(masks: torch.Tensor) -> torch.Tensor:
    assert masks.ndim == 3  # assert batch
    # extract normalization somewhere else
    norm_factor = torch.max(masks)
    labels_masks = K.contrib.connected_components(
        masks.unsqueeze(1).to(torch.float) / norm_factor, num_iterations=500
    )
    # B x 1 x H x W => B x H x W
    labels_masks = labels_masks.view(masks.shape[0], *labels_masks.shape[-2:])
    batch_masks = []
    # loop over img batch
    for m in labels_masks:
        batch_masks.append(_extract_obj(m))
    # N x B x H x W => B x N x H x W
    batch_masks = pad_sequence(batch_masks).permute(1, 0, 2, 3)
    return batch_masks


def _extract_obj(tensor: torch.Tensor) -> torch.Tensor:
    """
    Create individual binary mask from one array with annotated shapes
    TODO: vectorized
    """
    all_masks = []
    # check unique give sorted values
    id_shapes = torch.unique(tensor)[1:]
    for shape in id_shapes:
        all_masks.append(torch.where(tensor == shape, 1, 0))
    if all_masks:
        return torch.stack(
            all_masks
        )  # RuntimeError: stack expects a non-empty TensorList
    else:
        # we don't have object extracted => create empty masks for every
        return torch.zeros((1, *tensor.shape))


def select_type_scores_decision_mAP(type_decision_mAP) -> str:
    match type_decision_mAP:
        case "ci":
            return "confidence_scores"
        case "iou":
            return "iou_preds"
        case _:
            raise ValueError("Type decision not valid")


def _mask_processing(
    data_type: str,
    preds: Dict[str, torch.Tensor] = None,
    labels: torch.Tensor = None,
    type_decision_mAP: str = "ci",
) -> List[Dict]:
    """Format input data for Torchmetrics mAP - iou_segm mode
    - compute indivudals masks
    - associate scores to compute classification threshold

    Args:
        data_type (str): Union["pred", "label"]
        preds (Dict[str, torch.Tensor], optional): masks predictions and scores. Defaults to None.
        labels (torch.Tensor, optional): binary masks labels. Defaults to None.
        type_decision_mAP (str, optional): type of score to choose for decision. Defaults to "ci".

    Returns:
        List[Dict]: pred or labels formatted with associated individuals masks
    """

    def _mask_processing_preds(
        preds: Dict[str, torch.Tensor], type_decision_mAP: str = "ci"
    ) -> List[Dict]:
        # TO DO : check if padding doesn't inlfuence bbox generation
        assert isinstance(preds, dict), "Invalid data type"

        key = select_type_scores_decision_mAP(type_decision_mAP)

        preds = [
            {
                "masks": mask,
                "labels": torch.zeros(len(mask), dtype=torch.uint8),
                "scores": s,
            }
            for mask, s in zip(preds["masks"].to(torch.bool), preds[key])
        ]
        return preds

    def _mask_processing_labels(labels: torch.Tensor) -> List[Dict]:
        # TO DO : check if padding doesn't inlfuence bbox generation
        assert isinstance(labels, torch.Tensor), "Invalid data type"

        labels = extract_object_from_batch(labels)

        gt = [
            {
                "masks": mask,
                "labels": torch.zeros(len(mask), dtype=torch.uint8),
            }
            for mask in labels.to(torch.uint8)
        ]
        return gt

    if not any([preds is not None, labels is not None]):
        raise RuntimeError("Please provide at least one of the labels or preds")

    match data_type:
        case "pred":
            return _mask_processing_preds(
                preds=preds, type_decision_mAP=type_decision_mAP
            )
        case "label":
            return _mask_processing_labels(labels=labels)


def _bbox_processing(
    data_type: str,
    preds: Dict[str, torch.Tensor] = None,
    labels: torch.Tensor = None,
    type_decision_mAP: str = "ci",
) -> List[Dict]:
    """Format input data for Torchmetrics mAP - bbox mode
    - compute indivudals bbox from masks
    - associate scores to compute classification threshold

    Args:
        data_type (str): Union["pred", "label"]
        preds (Dict[str, torch.Tensor], optional): masks predictions and scores. Defaults to None.
        labels (torch.Tensor, optional): binary masks labels. Defaults to None.
        type_decision_mAP (str, optional): type of score to choose for decision. Defaults to "ci".

    Returns:
        List[Dict]: pred or labels formatted with associated individuals bbox
    """

    def _bbox_processing_preds(
        preds: Dict[str, torch.Tensor], type_decision_mAP: str = "ci"
    ) -> List[Dict]:

        assert isinstance(preds, dict), "Invalid data type"

        key = select_type_scores_decision_mAP(type_decision_mAP)

        masks_boxes = batched_mask_to_box(preds["masks"].to(torch.bool))
        # TO DO : check if padding doesn't inlfuence bbox generation
        preds = [
            {
                "boxes": im_bbox,
                "labels": torch.zeros(len(im_bbox), dtype=torch.int8),
                "scores": iou,
            }
            for im_bbox, iou in zip(masks_boxes, preds[key])
        ]
        return preds

    def _bbox_processing_labels(labels: torch.Tensor) -> List[Dict]:

        assert isinstance(labels, torch.Tensor), "Invalid data type"

        labels = extract_object_from_batch(labels)
        labels_boxes = batched_mask_to_box(labels.to(torch.bool))
        gt = [
            {"boxes": il_bbox, "labels": torch.zeros(len(il_bbox), dtype=torch.int8)}
            for il_bbox in labels_boxes
        ]
        return gt

    if not any([preds is not None, labels is not None]):
        raise RuntimeError("Please provide at least one of the labels or preds")

    match data_type:
        case "pred":
            return _bbox_processing_preds(
                preds=preds, type_decision_mAP=type_decision_mAP
            )
        case "label":
            return _bbox_processing_labels(labels=labels)


def binarize_mask(masks: torch.Tensor, th: float) -> torch.Tensor:
    return masks > th
