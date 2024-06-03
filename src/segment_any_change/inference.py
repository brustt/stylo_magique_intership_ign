from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Union

import torch
from magic_pen.data.loader import BiTemporalDataset, DataSample
from magic_pen.data.process import DefaultTransform
from magic_pen.config import DEVICE, logs_dir
from magic_pen.dummy import DummyModel
from segment_any_change.matching import BitemporalMatching
from segment_any_change.model import BiSam
from segment_any_change.utils import load_sam
from magic_pen.utils_io import load_levircd_sample


@dataclass
class ExperimentParams:
    # global
    model_type: str
    batch_size: int
    output_dir: Union[str, Path]
    logs_dir: Union[str, Path]
    # seg any change
    th_change_proposals: str
    # sam mask generation
    points_per_side: int
    points_per_batch: int
    pred_iou_thresh: float
    stability_score_thresh: float
    stability_score_offset: float
    box_nms_thresh: float
    min_mask_region_area: float


def choose_model(is_debug, params):

    if is_debug:
        return DummyModel(3, 1).to(DEVICE)
    else:
        sam = load_sam(
            model_type=params.model_type, model_cls=BiSam, version="dev", device=DEVICE
        )
        return BitemporalMatching(
            model=sam,
            th_change_proposals=params.th_change_proposals,
            points_per_side=params.points_per_side,
            points_per_batch=params.points_per_batch,
            pred_iou_thresh=params.pred_iou_thresh,
            stability_score_thresh=params.stability_score_thresh,
            stability_score_offset=params.stability_score_offset,
            box_nms_thresh=params.box_nms_thresh,
            min_mask_region_area=params.min_mask_region_area,
        )


def load_default_sam_params():
    # experiment parameters
    exp_params = {
        "batch_size": 4,
        "model_type": "vit_h",
    }

    seganychange_params = {
        "th_change_proposals": 0.0,
    }

    # sam mask generation
    sam_params = {
        "points_per_side": 32,  # lower for speed
        "points_per_batch": 64,  # not used
        "pred_iou_thresh": 0.88,  # configure lower for exhaustivity
        "stability_score_thresh": 0.95,  # configure lower for exhaustivity
        "stability_score_offset": 1.0,
        "box_nms_thresh": 0.7,
        "min_mask_region_area": 0,
    }

    dir_params = {
        "output_dir": f"lightning_logs/predictions-{exp_params['model_type']}",
        "logs_dir": logs_dir,
    }

    return ExperimentParams(
        **(exp_params | seganychange_params | sam_params | dir_params)
    )


def load_partial_ds(ds_name: str, dtype, indices: Sequence[int] = None):
    ds = BiTemporalDataset(name=ds_name, dtype=dtype, transform=DefaultTransform())
    if indices:
        ds = torch.utils.data.Subset(ds, indices)
    print(f"DATASET SUBSET : {len(ds)}")

    return ds


def infer_on_sample(
    A_path: str, B_path: str, label_path: str, model: Any = None
) -> Dict[str, Any]:

    if model is None:
        model = choose_model(is_debug=False, params=load_default_sam_params())

    item = DataSample(A_path=A_path, B_path=B_path, label_path=label_path)

    ds = BiTemporalDataset(items=item, transform=DefaultTransform())

    dloader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)

    outputs = []
    for i, batch in enumerate(dloader):
        with torch.no_grad():
            outputs.append({"pred": model(batch), "batch": batch})
    return outputs[0]


def partial_inference(
    model: Any = None,
    ds_name: str = "levir-cd",
    dtype: str = "test",
    batch_size: int = None,
    indices: Sequence[int] = None,
) -> List[Dict]:

    if model is None:
        model = choose_model(is_debug=False, params=load_default_sam_params())
    if batch_size is None:
        batch_size = 2

    ds = load_partial_ds(ds_name, dtype, indices)

    dloader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)
    outputs = []
    for i, batch in enumerate(dloader):
        with torch.no_grad():
            outputs.append({"prediction": model(batch), "batch": batch})
    return outputs


if __name__ == "__main__":

    pair_img = load_levircd_sample(1, seed=42)
    path_label, path_A, path_B = pair_img.iloc[0]

    output = infer_on_sample(
        A_path=path_A, B_path=path_B, label_path=path_label, model=None
    )

    # from segment_any_change.masks.mask_generator import SegAnyMaskGenerator
    # params = load_default_sam_params()

    # sam = load_sam(
    #             model_type=params.model_type,
    #             model_cls=BiSam,
    #             version="dev",
    #             device=DEVICE
    #             )

    # item = DataSample(A_path=path_A,
    #                     B_path=path_B,
    #                     label_path=path_label)

    # ds = BiTemporalDataset(items=item, transform=DefaultTransform())
    # dloader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)
    # batch = next(iter(dloader))

    # mask_generator = SegAnyMaskGenerator(model=sam,
    #                                 points_per_side=params.points_per_side,
    #                                 points_per_batch=params.points_per_batch,
    #                                 pred_iou_thresh=params.pred_iou_thresh,
    #                                 stability_score_thresh=params.stability_score_thresh,
    #                                 stability_score_offset=params.stability_score_offset,
    #                                 box_nms_thresh=params.box_nms_thresh,
    #                                 min_mask_region_area=params.min_mask_region_area)

    # img_anns = mask_generator.generate(batch)

    # import pickle

    # with open(f"tmp/generator_return_{params.model_type}.pkl", "wb") as fp:
    #     pickle.dump(img_anns, fp)
    #     print("masks dumped ! ")
