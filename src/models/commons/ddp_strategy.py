from datetime import timedelta
from typing import List, Literal, Any, Callable
from dataclasses import dataclass

from omegaconf import DictConfig
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.strategies.ddp import (default_pg_timeout, Precision,
                                              ClusterEnvironment, CheckpointIO)
from torch.distributed.algorithms.ddp_comm_hooks import (
    default_hooks as default,
    powerSGD_hook as powerSGD,
)
from torch.distributed.algorithms.ddp_comm_hooks import post_localSGD_hook as post_localSGD
COMM_HOOKS = {"all_reduced": default.allreduce_hook,
              "bf16_compress_hook": default.bf16_compress_hook,
              "fp16_compress_hook": default.fp16_compress_hook,
              "power_sgd_hook": powerSGD.powerSGD_hook}
COMM_WRAPPERS = {"bf16_compress_wrapper": default.bf16_compress_wrapper,
                 "fp16_compress_wrapper": default.fp16_compress_wrapper,}


def build_ddp_strategy(cfg: DictConfig) -> DDPStrategy:

    ddp_comm_state = None
    ddp_comm_hook = COMM_HOOKS[str(cfg.get("extras.ddp_comm_hook"))] \
        if cfg.get("extras.ddp_comm_hook") else None
    ddp_comm_wrapper = COMM_WRAPPERS[str(cfg.get("extras.ddp_comm_wrapper"))] \
        if cfg.get("extras.ddp_comm_wrapper") else None
    model_averaging_period = None

    if cfg.get("trainer.strategy"):
        del cfg["trainer.strategy"]
    if cfg.get("extras.ddp_comm_state"):
        if cfg.get("ddp_comm_state") == "power_sgd":
            ddp_comm_state = powerSGD.PowerSGDState(process_group=None,
                                                    matrix_approximation_rank=1,
                                                    start_powerSGD_iter=5000, ) \
                if cfg.get("extras.use_power_sgd") else None

        elif cfg.get("ddp_comm_state") == "local_sgd":
            ddp_comm_state = post_localSGD.PostLocalSGDState(
                process_group=None,
                subgroup=None,
                start_localSGD_iter=8,
            ),
            ddp_comm_hook = post_localSGD.post_localSGD_hook,
            model_averaging_period = 4,
        else:
            pass
    return DDPStrategy(ddp_comm_state=ddp_comm_state,
                       ddp_comm_hook=ddp_comm_hook,
                       ddp_comm_wrapper=ddp_comm_wrapper,
                       model_averaging_period=model_averaging_period,
                       timeout=default_pg_timeout,
                       start_method="popen",)
