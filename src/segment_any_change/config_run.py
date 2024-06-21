"""
tmp file for exploration and experiment params runs
"""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import re
from typing import Any, Dict, Optional, Union

from commons.constants import NamedModels
from magic_pen.config import DEVICE, logs_dir
from magic_pen.dummy import DummyModel
from segment_any_change.eval import UnitsMetricCounts
from magic_pen.utils_io import check_dir
from segment_any_change.matching import BitemporalMatching
from segment_any_change.query_prompt import SegAnyPrompt
from segment_any_change.utils import SegAnyChangeVersion, load_sam
from torchmetrics.detection import MeanAveragePrecision
from torchmetrics.classification import (
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
    BinaryJaccardIndex,
    BinaryConfusionMatrix
)
from segment_any_change.model import BiSam
from magic_pen.utils_io import check_dir


# TODO : refacto with hydra & hydra-zen


@dataclass
class ExperimentParams:
    # global
    model_name: NamedModels
    model_type: str
    batch_size: int
    output_dir: Union[str, Path]
    logs_dir: Union[str, Path]
    ds_name: str  # check for existance - Enum type
    # seg any change
    th_change_proposals: str
    seganychange_version: SegAnyChangeVersion
    col_nms_threshold: str
    # prompt engine
    th_sim: Any
    n_points_grid: int
    # sam mask generation
    prompt_type: int
    n_prompt: int # number of prompt to sample
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
    # sam mask generation
    loc: Optional[str] = None # pormpt sampling : center or random



def choose_model(params: ExperimentParams):

    if params.model_name  == NamedModels.DUMMY.value:

        return DummyModel(3, 1).to(DEVICE)

    elif params.model_name  == NamedModels.SEGANYMATCHING.value:

        sam = load_sam(
            model_type=params.model_type, model_cls=BiSam, version="dev", device=DEVICE
        )
        # set to float16 - for cuda runtime
        return matching_engine == BitemporalMatching(
            model=sam,
            version=params.seganychange_version,
            **asdict(params)
        )
    
    elif params.model_name  == NamedModels.SEGANYPROMPT.value:

        sam = load_sam(
            model_type=params.model_type, model_cls=BiSam, version="dev", device=DEVICE
        )

        matching_engine = BitemporalMatching(
            model=sam,
            version=params.seganychange_version,
            **asdict(params)
        )

        return SegAnyPrompt(matching_engine, **asdict(params))
    else:
        raise RuntimeError(f"Model {params.model_name} not known")



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
        BinaryJaccardIndex(),
        UnitsMetricCounts(),
        MeanAveragePrecision(iou_type=kwargs.get("iou_type_mAP", "segm"), max_detection_thresholds=kwargs.get("max_detection_thresholds", None)),
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
        "n_prompt": 16*16,  # lower for speed
    }

    default_params = load_default_exp_params(**params)

    return ExperimentParams(
        **(
            asdict(default_params)
            | new_params  # merge other parameters - overwrite existing ones
            | params
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
    exp_params["exp_name"] = "seganychange_repr_change_th"
    # '_'.join([datetime.now().strftime('%Y%m%d'), exp_params["ds_name"], exp_params["model_type"]])

    seganychange_params = {
        "prompt_type":"grid",
        "n_points_grid": 1024,
        "loc":"center", # only for prompt_type == sample
        "th_change_proposals": 60,
        "col_nms_threshold": "ci",  # ci | iou_preds
        "seganychange_version": SegAnyChangeVersion.AUTHOR,
        "th_sim":"otsu"
    }

    # sam mask generation
    sam_params = {
        "n_prompt": 1024,  # lower for speed
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
        "engine_metric": {
            "iou_type_mAP": "segm", 
            "type_decision_mAP": "ci",
            "max_detection_thresholds": [10, 100, 1000]
            }
    }

    if params.get("th_change_proposals", None) and isinstance(params.get("th_change_proposals"), str):
        if not re.match('[a-z]', params.get("th_change_proposals")):
            print("hjk")
            params["th_change_proposals"] = float(params["th_change_proposals"])

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
