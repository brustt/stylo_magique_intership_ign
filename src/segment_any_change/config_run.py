"""
tmp file for exploration and experiment params runs
"""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Union

from magic_pen.config import DEVICE, logs_dir
from magic_pen.dummy import DummyModel
from segment_any_change.eval import UnitsMetricCounts
from magic_pen.utils_io import check_dir
from segment_any_change.matching import BitemporalMatching
from segment_any_change.utils import SegAnyChangeVersion, load_sam
from torchmetrics.detection import MeanAveragePrecision
from torchmetrics.classification import (
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
)
from segment_any_change.model import BiSam


from magic_pen.utils_io import check_dir


@dataclass
class ExperimentParams:
    # global
    model_type: str
    batch_size: int
    output_dir: Union[str, Path]
    logs_dir: Union[str, Path]
    ds_name: str  # check for existance - Enum type
    # seg any change
    th_change_proposals: str
    seganychange_version: SegAnyChangeVersion
    col_nms_threshold: str
    # sam mask generation
    points_per_side: int
    points_per_batch: int
    pred_iou_thresh: float
    stability_score_thresh: float
    stability_score_offset: float
    box_nms_thresh: float
    min_mask_region_area: float
    engine_metric: Dict
    # exp
    exp_id: str
    exp_name: str
    # run
    num_worker: int
    n_job_by_node: int
    dev: bool  # True : infer with smaller model and less points from grid


def choose_model(is_debug, params):

    if is_debug:
        return DummyModel(3, 1).to(DEVICE)
    else:
        sam = load_sam(
            model_type=params.model_type, model_cls=BiSam, version="dev", device=DEVICE
        )
        # set to float16 - for cuda runtime
        return BitemporalMatching(
            model=sam,
            version=params.seganychange_version,
            th_change_proposals=params.th_change_proposals,
            points_per_side=params.points_per_side,
            points_per_batch=params.points_per_batch,
            pred_iou_thresh=params.pred_iou_thresh,
            stability_score_thresh=params.stability_score_thresh,
            stability_score_offset=params.stability_score_offset,
            box_nms_thresh=params.box_nms_thresh,
            min_mask_region_area=params.min_mask_region_area,
            col_nms_threshold=params.col_nms_threshold,
        )


def load_debug_cli_params():
    args = argparse.Namespace()
    args.ds_name = "levir-cd"
    args.dev = False
    args.n_job_by_node = 1
    args.batch_size = 2
    params = vars(args)
    return params


def load_default_metrics(**kwargs):
    return [
        BinaryF1Score(),
        BinaryPrecision(),
        BinaryRecall(),
        UnitsMetricCounts(),
        MeanAveragePrecision(iou_type=kwargs.get("iou_type_mAP", "segm")),
    ]


def load_exp_params(**params):
    if params.get("num_worker", None) is None:
        params["num_worker"] = params["batch_size"]

    if params["dev"]:
        return load_fast_exp_params(**params)
    else:
        return load_default_exp_params(**params)


def load_fast_exp_params(**params):
    # fast inference
    new_params = {
        "model_type": "vit_b",
        "points_per_side": 5,  # lower for speed
    }

    params = load_default_exp_params(**params)

    return ExperimentParams(
        **(
            asdict(params)
            | new_params  # merge other parameters - overwrite existing ones
        )
    )


def load_default_exp_params(**params):
    project = "seganychange"
    # experiment parameters
    exp_params = {
        "batch_size": params["batch_size"],
        "model_type": "vit_h",
        "ds_name": params["ds_name"],
    }
    exp_params["exp_id"] = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_params["exp_name"] = "seganychange_repr"
    # '_'.join([datetime.now().strftime('%Y%m%d'), exp_params["ds_name"], exp_params["model_type"]])

    seganychange_params = {
        "th_change_proposals": 0.0,
        "col_nms_threshold": "ci",  # ci | iou_preds
        "seganychange_version": SegAnyChangeVersion.AUTHOR,
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
        "output_dir": check_dir(
            logs_dir,
            project,
            exp_params["ds_name"],
            f"predictions-{exp_params['model_type']}",
        ),
        "logs_dir": check_dir(
            logs_dir, project, exp_params["ds_name"], exp_params["exp_name"]
        ),
    }

    engine_metric_params = {
        "engine_metric": {"iou_type_mAP": "segm", "type_decision_mAP": "ci"}
    }

    return ExperimentParams(
        **(
            exp_params
            | seganychange_params
            | sam_params
            | dir_params
            | engine_metric_params
            | params  # merge other parameters - overwrite existing one
        )
    )
