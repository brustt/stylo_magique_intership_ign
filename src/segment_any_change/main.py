from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union
import torch
from segment_any_change.eval import PX_METRICS

from segment_any_change.inference import ExperimentParams, choose_model
from segment_any_change.tensorboard_callback import (
    CustomWriter,
    TensorBoardCallbackLogger,
)

from magic_pen.config import DEVICE, SEED, logs_dir
import pytorch_lightning as pl
from pytorch_lightning.profilers import PyTorchProfiler
from magic_pen.data.datamodule import CDDataModule
from segment_any_change.task import CDModule
from segment_any_change.utils import flush_memory
import logging
from pytorch_lightning.loggers import TensorBoardLogger
from pathlib import Path

# TO DO : define globally
logging.basicConfig(format="%(asctime)s - %(levelname)s ::  %(message)s")


def main(params, is_debug: bool = False):

    flush_memory()
    pl.seed_everything(seed=SEED)

    logger = TensorBoardLogger(save_dir=params.logs_dir, name="debug")

    profiler = PyTorchProfiler(
        on_trace_ready=torch.profiler.tensorboard_trace_handler(params.logs_dir),
        schedule=torch.profiler.schedule(
            skip_first=10, wait=1, warmup=1, active=20
        ),  # parameters meaning ?
    )

    model = choose_model(is_debug, params)

    pl_module = CDModule(model=model, metrics=PX_METRICS)

    dm = CDDataModule(name="levir-cd", batch_size=params.batch_size)

    callbacks = [
        TensorBoardCallbackLogger(),
        CustomWriter(output_dir=params.output_dir, write_interval="epoch"),
    ]
    trainer = pl.Trainer(
        logger=logger, accelerator=DEVICE, profiler=profiler, callbacks=callbacks
    )

    output = trainer.predict(pl_module, dm, return_predictions=True)
    return output


if __name__ == "__main__":

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # experiment parameters
    exp_params = {
        "batch_size": 2,
        "model_type": "vit_b",
    }

    seganychange_params = {
        "th_change_proposals": "otsu",
    }

    # sam mask generation
    sam_params = {
        "points_per_side": 10,  # lower for speed
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

    params = ExperimentParams(
        **(exp_params | seganychange_params | sam_params | dir_params)
    )

    main(params, is_debug=False)
