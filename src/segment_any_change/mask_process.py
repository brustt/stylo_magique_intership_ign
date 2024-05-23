import torch
from segment_any_change.sa_dev.utils.amg import (
    MaskData,
    remove_small_regions,
    batched_mask_to_box,
    calculate_stability_score,
    mask_to_rle_pytorch,
    rle_to_mask,
)
from torchvision.ops.boxes import batched_nms


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
    data["boxes"] = batched_mask_to_box(data["masks"])

    keep_by_nms = nms_wrapper(data, box_nms_thresh)
    data.filter(keep_by_nms)
    print(data["masks"].shape[0])

    if min_area > 0.0:
        data["rles"] = mask_to_rle_pytorch(data["masks"])
        data = postprocess_small_regions(data, min_area, box_nms_thresh)

    return data["masks"], data["iou_preds"]


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


def filters_masks(data):

    mask_threshold = 0.0
    pred_iou_thresh: float = 0.88  # could be lower
    stability_score_thresh: float = 0.95  # could be lower
    stability_score_offset: float = 1.0

    # Filter by predicted IoU
    if pred_iou_thresh > 0.0:
        keep_mask = data["iou_preds"] > pred_iou_thresh
        data.filter(keep_mask)
    print(f' filter iou_th : {data["masks"].shape[0]}')

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

    # Threshold masks and calculate boxes
    data["masks"] = data["masks"] > mask_threshold
    print(f' filter mask_threshold : {data["masks"].shape[0]}')

    return data
