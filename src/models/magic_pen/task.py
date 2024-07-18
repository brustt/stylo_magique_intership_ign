import os
from typing import Any, Dict, List, Union
from commons.utils import show_prediction_sample, to_numpy
from commons.utils_io import save_pickle
import lightning.pytorch as pl
from omegaconf import DictConfig
import pandas as pd
import torch
from torchmetrics import MetricCollection
from .config_run import ExperimentParams
import torch.nn as nn
from torchmetrics.classification import BinaryF1Score, BinaryJaccardIndex, BinaryRecall, BinaryPrecision
"""
implement simpler metrics logger
"""

class MagicPenModule(pl.LightningModule):
    multimask_output = False

    def __init__(self, network, params):

        super().__init__()
        self.params = params
        self.model = network
        self.load_weights(params.get("sam_ckpt_path"))
        self.freeze_weigts()
        self.loss = nn.BCEWithLogitsLoss()
        self.train_metrics = MetricCollection( [
                BinaryJaccardIndex(),
            ], prefix="train")
        
        self.val_metrics =  MetricCollection( [
                BinaryJaccardIndex(),
            ], prefix="val")
        
        self.test_metrics =  MetricCollection( [
                BinaryJaccardIndex(),
            ], prefix="test")
    
    def load_weights(self, checkpoint):
        self.model.load_state_dict(torch.load(checkpoint))

    def freeze_weigts(self):
        """
        # TODO: freeze layer on key layer selection / name
        """
        self.model.image_encoder.requires_grad_(False)
        self.model.prompt_encoder.requires_grad_(False)

    def forward(self, x):
        # try with multimask_output == True and select best one
        preds, ious =  self.model(x, multimask_output=self.multimask_output)
        # remove nan for no-prompt values in the prompt dimension
        preds = torch.nan_to_num(preds, nan=0.)
        return preds, ious
    
    def _step(self, batch):
        preds, ious  = self.forward(batch)
        # sum logits over all masks
        preds = torch.sum(preds, dim=1)
        loss = self.loss(preds, batch["label"])
        return preds, loss

    def training_step(self, batch, batch_idx):
        preds, loss = self._step(batch)
        return {"pred": preds, "loss": loss}

    def validation_step(self, batch, batch_idx):
        preds, loss = self._step(batch)
        return {"pred": preds, "loss": loss}
    
    def test_step(self, batch, batch_idx):
        preds, loss = self._step(batch)
        return {"pred": preds, "loss": loss}

    def on_train_batch_end(self, outputs, batch, batch_idx):
        loss, preds = outputs["loss"], outputs["pred"]
        self.train_metrics.update(preds, batch["label"])
        self.log(
            f"train/loss",
            loss,
            sync_dist=True,
            on_step=True,
            on_epoch=True,
        )
        if batch_idx % 5 == 0:
            bs = preds.shape[0]
            for b_i in range(bs):
                fig = show_prediction_sample((outputs|dict(batch=batch)), idx=b_i)
                self.logger.experiment.add_figure(
                        f"sample_{self.current_epoch}_{batch_idx}",
                        fig,
                    )
        if self.current_epoch % 5 == 0:
            sm= nn.Sigmoid()
            self.logger.experiment.add_histogram(
                    f"hist_preds_{self.current_epoch}",
                    sm(preds[0].flatten()),
                )

    def on_validation_batch_end(self, outputs, batch, batch_idx):
        loss, preds = outputs["loss"], outputs["pred"]
        self.val_metrics.update(preds, batch["label"])
        self.log(
            f"val/loss",
            loss,
            sync_dist=True,
            on_step=True,
            on_epoch=True,
        )

    def on_test_batch_end(self, outputs, batch, batch_idx):
        loss, preds = outputs["loss"], outputs["pred"]
        self.test_metrics.update(preds, batch["label"])
        self.log(
            f"test/loss",
            loss,
            sync_dist=True,
            on_step=True,
            on_epoch=True,
        )

    def on_validation_epoch_end(self):
        metrics = self.val_metrics.compute()
        for metric_name, metric_value in metrics.items():
            self.log(
                f"val/{metric_name}",
                metric_value,
                sync_dist=True,
                on_step=False,
                on_epoch=True,
            )

    def on_train_epoch_end(self):
        metrics = self.train_metrics.compute()
        for metric_name, metric_value in metrics.items():
            self.log(
                f"train/{metric_name}",
                metric_value,
                sync_dist=True,
                on_step=False,
                on_epoch=True,
            )
        
        

    def on_test_epoch_end(self):
        metrics = self.test_metrics.compute()
        for metric_name, metric_value in metrics.items():
            self.log(
                f"test/{metric_name}",
                metric_value,
                sync_dist=True,
                on_step=False,
                on_epoch=True,
            )

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters())
        
        return {"optimizer": optimizer}
