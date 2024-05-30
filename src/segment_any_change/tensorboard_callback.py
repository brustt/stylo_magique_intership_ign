from typing import Any
import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import BasePredictionWriter
import os
import logging


# TO DO : define globally
logging.basicConfig(format="%(asctime)s - %(levelname)s ::  %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class TensorBoardCallbackLogger(Callback):
    def __init__(self, log_n_steps=5):
        super().__init__()
        self.log_n_steps = log_n_steps

    def add_metric(self, key, value, pl_module, trainer):
        pl_module.logger.experiment.add_scalar(key, value, global_step=trainer.global_step)


    def on_predict_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        if batch_idx % self.log_n_steps == 0:
            for key, metric in outputs.items():
                self.add_metric(key, metric, pl_module, trainer)



class CustomWriter(BasePredictionWriter):
    """see imp for distributed computing"""

    def __init__(self, output_dir, write_interval):
        super().__init__(write_interval)
        self.output_dir = output_dir

    # def write_on_batch_end(
    #     self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx
    # ):
    #     torch.save(prediction, os.path.join(self.output_dir, dataloader_idx, f"{batch_idx}.pt"))

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        torch.save(predictions, os.path.join(self.output_dir, "predictions.pt"))
        logger.info(os.path.join(self.output_dir, "predictions.pt"))