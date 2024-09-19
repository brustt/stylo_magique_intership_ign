import os
from typing import Any, Dict, List, Tuple

from commons.constants import SEED
from commons.instantiators import instantiate_callbacks
from commons.utils import flush_memory
import hydra
from pytorch_lightning import Callback
from lightning.pytorch import (
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
import torch

from src.commons import (
    RankedLogger,
    extras,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)

@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    metric_dict, object_dict = None, None
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating module <{cfg.model}>")
    module: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating loggers...")
    #logger: List[Logger] = instantiate_loggers(cfg.get("logger"))   
    
    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for evaluation.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    flush_memory()
    seed_everything(seed=cfg.seed)
    # apply extra utilities
    extras(cfg)
    print(cfg.model)
    train(cfg)


if __name__ == "__main__":
    main()
