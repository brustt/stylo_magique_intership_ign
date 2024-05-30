from dataclasses import dataclass
import os
from typing import Any, List, Optional, Tuple, Union
import torch
from segment_any_change.eval import PX_METRICS, MetricEngine
from segment_any_change.matching import BitemporalMatching
from segment_any_change.tensorboard_callback import CustomWriter, TensorBoardCallbackLogger
from src.magic_pen.dummy import DummyModel
from magic_pen.config import DEVICE, SEED, project_path, logs_dir
import pytorch_lightning as pl
from pytorch_lightning.profilers import PyTorchProfiler
from magic_pen.data.datamodule import CDDataModule
from segment_any_change.model import BiSam
from segment_any_change.task import CDModule
from segment_any_change.utils import flush_memory, load_sam
import logging
from pytorch_lightning.loggers import TensorBoardLogger
from pathlib import Path

# TO DO : define globally
logging.basicConfig(format="%(asctime)s - %(levelname)s ::  %(message)s")


def choose_model(is_debug, params):

    if is_debug:
        return DummyModel(3, 1).to(DEVICE)
    else:
        sam = load_sam(
            model_type=params.model_type, 
            model_cls=BiSam,
            version="dev", 
            device=DEVICE
            )
        return BitemporalMatching(model=sam, 
                                th_change_proposals=params.th_change_proposals,
                                points_per_side=params.points_per_side,
                                points_per_batch=params.points_per_batch,
                                pred_iou_thresh=params.pred_iou_thresh,
                                stability_score_thresh=params.stability_score_thresh,
                                stability_score_offset=params.stability_score_offset,
                                box_nms_thresh=params.box_nms_thresh,
                                min_mask_region_area=params.min_mask_region_area)

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
 
        
def main(params: ExperimentParams, is_debug: bool=False):

    flush_memory()
    pl.seed_everything(seed=SEED)

    logger = TensorBoardLogger(save_dir=params.logs_dir, name="debug")

    profiler = PyTorchProfiler(
        on_trace_ready=torch.profiler.tensorboard_trace_handler(params.logs_dir),
        schedule=torch.profiler.schedule(skip_first=10, wait=1, warmup=1, active=20) # parameters meaning ?
    )

    model = choose_model(is_debug, params)

    pl_module = CDModule(model=model, metrics=PX_METRICS)
    
    dm = CDDataModule(name="levir-cd", batch_size=params.batch_size)

    callbacks = [
        TensorBoardCallbackLogger(), 
        CustomWriter(output_dir= params.output_dir, write_interval="epoch")
    ]
    trainer = pl.Trainer(
        logger=logger,
        accelerator=DEVICE,
        profiler=profiler,
        callbacks=callbacks
    )

    output = trainer.predict(pl_module, dm, return_predictions=True)
    return output

if __name__ == "__main__":

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # experiment parameters
    exp_params = {
        "batch_size": 4,
        "model_type": "vit_b",
    }

    seganychange_params = {
        "th_change_proposals": "otsu",
    }

    # sam mask generation
    sam_params = {
        "points_per_side": 10, #lower for speed
        "points_per_batch": 64, # not used
        "pred_iou_thresh": 0.88, # configure lower for exhaustivity
        "stability_score_thresh": 0.95, # configure lower for exhaustivity
        "stability_score_offset": 1.0,
        "box_nms_thresh": 0.7,
        "min_mask_region_area": 0,
    }

    dir_params = {
        "output_dir": f"lightning_logs/predictions-{exp_params['model_type']}",
        "logs_dir": logs_dir

    }

    params = ExperimentParams(**(exp_params | seganychange_params | sam_params | dir_params))

    main(params, is_debug=False)



