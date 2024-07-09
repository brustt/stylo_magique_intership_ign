import os
from typing import Any, Dict, List, Tuple

import hydra
import rootutils
from lightning.pytorch import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from src.commons import (
    RankedLogger,
    extras,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Tuple[dict, dict] with metrics and dict with all instantiated objects.
    """
    # assert cfg.ckpt_path
    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    print(cfg.keys())

    # log.info(f"Instantiating sam <{cfg.model.network.model._target_}>")
    # sam: Any = hydra.utils.instantiate(cfg.model.network)

    log.info(cfg)

    log.info(f"Instantiating module <{cfg.model.instance._target_}>")
    module: LightningModule = hydra.utils.instantiate(cfg.model.instance)
    # instantiate ok : pl module
    # print(type(module))
    # # instantiate ok : cls bitemporal matching
    # print(type(module.model))
    # print(type(module.model.mask_generator))

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

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

    log.info("Starting testing!")
    trainer.test(model=module, datamodule=datamodule)
    # trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)

    # # for predictions use trainer.predict(...)
    # # predictions = trainer.predict(model=model, dataloaders=dataloaders, ckpt_path=cfg.ckpt_path)

    # metric_dict = trainer.callback_metrics

    metric_dict, object_dict = None, None

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval_test.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for evaluation.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    # extras(cfg)
    evaluate(cfg)


if __name__ == "__main__":
    print(os.environ["PROJECT_PATH"])
    main()
