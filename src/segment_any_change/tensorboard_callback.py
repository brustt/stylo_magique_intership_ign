from typing import Any
import numpy as np
import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import BasePredictionWriter
import torchvision
import os
import logging

from segment_any_change.utils import create_overlay_outcome_cls, substring_present, to_numpy


# TO DO : define globally
logging.basicConfig(format="%(asctime)s - %(levelname)s ::  %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TensorBoardCallbackLogger(Callback):
    """
    doc summaryWriter Tensorboard : https://tensorboardx.readthedocs.io/en/stable/tensorboard.html#tensorboardX.SummaryWriter
    """
    _units_key_name = "UnitsMetricCounts"

    def __init__(self, log_n_steps=5):
        super().__init__()
        self.log_n_steps = log_n_steps

    def add_metric(self, key, value, pl_module, trainer):
        pl_module.logger.experiment.add_scalar(
            key, value, global_step=pl_module.global_step
        )

    def create_grid_batch(self, outputs, batch) -> np.ndarray:
        """create image grid from sample (imgA, imgB), label and masks predictions"""
        sample = []

        images_A = batch["img_A"].cpu()
        images_B = batch["img_B"].cpu()
        labels = batch["label"].cpu()
        preds = outputs["pred"]["masks"].cpu()
        
        def shift_range_values(arr, new_bounds=[0, 1]): 
            old_range = (torch.max(arr) - torch.min(arr))  
            new_range = (new_bounds[1] - new_bounds[0])  
            shit_arr = (((arr - torch.min(arr)) * new_range) / old_range) + new_bounds[0]
            return shit_arr
    

        for i in range(images_A.size(0)):

            img_A = images_A[i]
            img_B = images_B[i]
            # Align to 3 channels
            label = labels[i].unsqueeze(0).repeat(3, 1, 1)
            img_outcome_cls = torch.zeros_like(img_A)
            
            # pass key pred_UnitsMetricCounts to params
            if any([substring_present(self._units_key_name, key) for key in list(outputs["metrics"])]):
                tp = outputs["metrics"]["pred_UnitsMetricCounts"]["tp_indices"].squeeze(0)[i,...]
                fp = outputs["metrics"]["pred_UnitsMetricCounts"]["fp_indices"].squeeze(0)[i,...]
                fn = outputs["metrics"]["pred_UnitsMetricCounts"]["fn_indices"].squeeze(0)[i,...]
                
                img_outcome_cls = create_overlay_outcome_cls(tp, fp, fn)

            # Stack individual masks and align to 3 channels
            print(f"pred : {preds[i, ...].shape}")
            print(f"pred type : {preds[i, ...].dtype}")
            print(f"label type : {preds[i, ...].dtype}")
            pred = torch.sum(preds[i, ...], axis=0).unsqueeze(0).repeat(3, 1, 1).to(torch.uint8)
            print(f"pred nonzeros: {torch.nonzero(pred).shape}")
            print(f"pred type: {pred.shape}")
            print(f"pred non zeros cnt: {torch.sum(pred)}")
            print(f"pred non zeros unique: {torch.unique(pred)}")

            pred = shift_range_values(pred, new_bounds=[0, 255]).to(torch.uint8)


            print("-----")
            row = torch.stack((img_A, img_B, label, pred, img_outcome_cls), dim=0)
            
            # combined stack as row
            row = torchvision.utils.make_grid(
                row, nrow=row.shape[0], padding=20, pad_value=1, normalize=True
            )
            sample.append(row)

        grid = torchvision.utils.make_grid(
            sample, nrow=1, padding=20, pad_value=1, scale_each=True
        )

        return grid

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if batch_idx % self.log_n_steps == 0:
            for key, metric in outputs["metrics"].items():
                if not substring_present(self._units_key_name, key):
                    self.add_metric(key, metric, pl_module, trainer)

        if batch_idx == 0:
            img_sample = self.create_grid_batch(outputs, batch)
            pl_module.logger.experiment.add_image(
                "first_batch", img_sample, pl_module.global_step, dataformats="CHW"
            )
            # pl_module.logger.experiment.add_text(
            #     "first_batch", "imgA, imgB, label, prediction"
            # )

      
    def on_predict_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        pass

class CustomWriter(BasePredictionWriter):
    """see imp for distributed computing"""

    def __init__(self, output_dir, write_interval):
        super().__init__(write_interval)
        self.output_dir = output_dir

    def write_on_batch_end(
        self,
        trainer,
        pl_module,
        prediction,
        batch_indices,
        batch,
        batch_idx,
        dataloader_idx,
    ):
        torch.save(prediction, os.path.join(self.output_dir, "predictions.pt"))
        logger.info(os.path.join(self.output_dir, "predictions.pt"))

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        to_write = ["predictions", "batch_idx"]
        #output = [{k: v for k, v in p.items()} for p in predictions]
        #torch.save(output, os.path.join(self.output_dir, "predictions.pt"))
        #logger.info(os.path.join(self.output_dir, "predictions.pt"))
