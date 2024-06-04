from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union
import torch
from torchmetrics import Metric

from segment_any_change.inference import (
    ExperimentParams,
    choose_model,
    load_default_metrics,
    load_default_exp_params,
)
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


def main(
    params: ExperimentParams,
    metrics: List[Metric],
    is_debug: bool = False,
):

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

    pl_module = CDModule(model=model, metrics=metrics, **params.engine_metric)

    dm = CDDataModule(name=params.ds_name, batch_size=params.batch_size)

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

    params = load_default_exp_params()
    metrics = load_default_metrics(**params.engine_metric)
    main(params, metrics=metrics, is_debug=False)
