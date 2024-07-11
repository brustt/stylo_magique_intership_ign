from typing import Any, List, Optional, Tuple, Union
from src.models.segment_any_change.config_run import (
    ExperimentParams,
    load_debug_cli_params,
    load_exp_params,
    load_default_metrics,
)
from src.models.commons.choose_model import choose_model
import torch
from torchmetrics import Metric
from commons.tensorboard_callback import (
    PredictionWriter,
    TensorBoardCallbackLogger,
)

from commons.config import DEVICE, PROJECT_PATH, SEED
import pytorch_lightning as pl
from pytorch_lightning.profilers import PyTorchProfiler
from src.data.datamodule import CDDataModule
from models.segment_any_change.task import CDModule
from src.commons.utils import flush_memory
import logging
from pytorch_lightning.loggers import TensorBoardLogger
from pathlib import Path
import argparse
import sys
from dataclasses import asdict
from pprint import pprint

# TO DO : define globally
logging.basicConfig(format="%(asctime)s - %(levelname)s ::  %(message)s")


def main(
    params: ExperimentParams,
    metrics: List[Metric],
    is_debug: bool = False,
):

    flush_memory()
    pl.seed_everything(seed=SEED)

    logger = TensorBoardLogger(
        save_dir=params.logs_dir, name=None, version=params.exp_id
    )

    profiler = PyTorchProfiler(
        on_trace_ready=torch.profiler.tensorboard_trace_handler(params.logs_dir),
        schedule=torch.profiler.schedule(
            skip_first=10, wait=1, warmup=1, active=20
        ),  # parameters meaning ?
        sort_by_key="cpu_memory_usage",
    )

    model = choose_model(params)

    pl_module = CDModule(model=model, metrics=metrics, params=params)

    dm = CDDataModule(name=params.ds_name, params=params)

    callbacks = [
        TensorBoardCallbackLogger(params),
        # PredictionWriter(output_dir=params.output_dir, write_interval=1)
    ]
    trainer = pl.Trainer(
        logger=logger, accelerator=DEVICE, profiler=profiler, callbacks=callbacks
    )

    trainer.test(model=pl_module, datamodule=dm)


if __name__ == "__main__":

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # track run mode (debug)
    gettrace = getattr(sys, "gettrace", None)

    if not gettrace():
        # CLI
        parser = argparse.ArgumentParser(description="A simple CLI parser")
        parser.add_argument("--model_name", type=str, help="The name of the dataset")
        parser.add_argument("--ds_name", type=str, help="The name of the dataset")
        parser.add_argument(
            "--n_job_by_node", type=int, help="Number of job by node", default=1
        )
        parser.add_argument("--batch_size", type=int, help="Batch size", default=2)
        parser.add_argument(
            "--th_change_proposals", type=str, help="Change Threshold", default=None
        )

        parser.add_argument(
            "--dev",
            help="Fast inference - light model and prompts",
            default=False,
            action=argparse.BooleanOptionalAction,
        )

        params = vars(parser.parse_args())
    else:
        # debug mode
        logger.info("DEBUG MODE")
        params = load_debug_cli_params()

    params = load_exp_params(**params)
    metrics = load_default_metrics(**params.engine_metric)
    pprint(asdict(params), sort_dicts=False)
    main(params, metrics=metrics, is_debug=False)
