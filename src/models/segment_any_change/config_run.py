"""
tmp file for exploration and experiment params runs
"""

import argparse
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
import re
from typing import Any, Dict, Optional, Union

from commons.constants import NamedModels
from commons.config import DEVICE, LOGS_PATH
from commons.eval import UnitsMetricCounts
from commons.utils_io import check_dir, load_sam
from .matching import BitemporalMatching
from models.segment_any_change.query_prompt import SegAnyPrompt
from src.commons.utils import SegAnyChangeVersion
from torchmetrics.detection import MeanAveragePrecision
from torchmetrics.classification import (
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
    BinaryJaccardIndex,
    BinaryConfusionMatrix,
)
from models.commons.model import BiSam
from commons.utils_io import check_dir


# TODO : refacto with hydra & hydra-zen

# TODO : add default value here


@dataclass
class ExperimentParams:
    # global
    model_name: NamedModels = NamedModels.SEGANYMATCHING
    model_type: str = "vit_b"
    batch_size: int = 2
    output_dir: Union[str, Path] = "outdir"
    logs_dir: Union[str, Path] = "logdir"
    ds_name: str = "levircd"  # check for existance - Enum type
    # seg any change
    th_change_proposals: str = "otsu"
    seganychange_version: SegAnyChangeVersion = SegAnyChangeVersion.AUTHOR
    col_nms_threshold: str = "col"
    # prompt engine
    th_sim: Any = 60
    n_points_grid: int = 12
    # sam mask generation
    prompt_type: int = 12
    n_prompt: int = 12  # number of prompt to sample
    pred_iou_thresh: float = 12
    stability_score_thresh: float = 12
    stability_score_offset: float = 12
    box_nms_thresh: float = 12
    min_mask_region_area: float = 12
    engine_metric: Dict = field(default_factory=lambda: dict(key=12))
    # exp
    exp_id: str = "12"
    exp_name: str = "12"
    # run
    num_worker: int = 1
    n_job_by_node: int = 1
    dev: bool = True  # True : infer with smaller model and less points from grid
    # sam mask generation
    loc: Optional[str] = None  # pormpt sampling : center or random


def load_debug_cli_params():
    args = argparse.Namespace()
    args.ds_name = "levir-cd"
    args.dev = False
    args.n_job_by_node = 1
    args.batch_size = 2
    args.model_name = NamedModels.SEGANYMATCHING.value
    params = vars(args)
    return params


def load_default_metrics(**kwargs):
    return [
        BinaryF1Score(),
        BinaryPrecision(),
        BinaryRecall(),
        BinaryJaccardIndex(),
        UnitsMetricCounts(),
        MeanAveragePrecision(
            iou_type=kwargs.get("iou_type_mAP", "segm"),
            max_detection_thresholds=kwargs.get("max_detection_thresholds", None),
        ),
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
    project = "seganychange"
    new_params = {
        "model_type": "vit_b",
        "n_points_grid": 32,  # lower for speed
        "ds_name": params["ds_name"],
        "exp_name": "seganychange_repr_change_th",
    }

    dir_params = {
        "output_dir": check_dir(
            LOGS_PATH,
            project,
            new_params["ds_name"],
            f"predictions-{new_params['model_type']}",
        ),
        "logs_dir": check_dir(
            LOGS_PATH,
            project,
            new_params["ds_name"],
            new_params["exp_name"],
            new_params["model_type"],
        ),
    }

    default_params = load_default_exp_params(**params)

    return ExperimentParams(
        **(
            asdict(default_params)
            | new_params  # merge other parameters - overwrite existing ones
            | dir_params
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
        "prompt_type": "grid",
        "n_points_grid": 1024,
        "loc": "center",  # only for prompt_type == sample
        "th_change_proposals": 60,
        "col_nms_threshold": "ci",  # ci | iou_preds
        "seganychange_version": SegAnyChangeVersion.AUTHOR,
        "th_sim": 0.9,
    }

    # sam mask generation
    sam_params = {
        "n_prompt": 3,  # lower for speed
        "pred_iou_thresh": 0.88,  # configure lower for exhaustivity
        "stability_score_thresh": 0.95,  # configure lower for exhaustivity
        "stability_score_offset": 1.0,
        "box_nms_thresh": 0.7,
        "min_mask_region_area": 0,
    }

    dir_params = {
        "output_dir": check_dir(
            LOGS_PATH,
            project,
            exp_params["ds_name"],
            f"predictions-{exp_params['model_type']}",
        ),
        "logs_dir": check_dir(
            LOGS_PATH,
            project,
            exp_params["ds_name"],
            exp_params["exp_name"],
            exp_params["model_type"],
        ),
    }

    engine_metric_params = {
        "engine_metric": {
            "iou_type_mAP": "segm",
            "type_decision_mAP": "ci",
            "max_detection_thresholds": [10, 100, 1000],
        }
    }

    if params.get("th_change_proposals", None) and isinstance(
        params.get("th_change_proposals"), str
    ):
        if not re.match("[a-z]", params.get("th_change_proposals")):
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
