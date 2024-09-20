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
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating module <{cfg.model.instance._target_}>")
    module: LightningModule = hydra.utils.instantiate(cfg.model.instance)

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.callbacks)

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, logger=logger, callbacks=callbacks
    )

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": module,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    # not used
    if cfg.get("compile"):
        log.info("Compiling model!")
        module = torch.compile(module)

    log.info("Starting testing!")
    # ckpt is provide in BitemporalMatching for SegmentAnyChange models otherwise it sould be mention here
    #trainer.test(model=module, datamodule=datamodule)
    if cfg.get("train"):
            log.info("Starting training!")
            # no ckpt for first training sessions
            trainer.fit(model=module, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path", None))

    train_metrics = trainer.callback_metrics


        
    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=module, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

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
