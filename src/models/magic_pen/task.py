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
import re

import logging

# TO DO : define globally
logging.basicConfig(format="%(asctime)s - %(levelname)s ::  %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


_register_layer_not_used = [
    "model.prompt_encoder.point_embeddings.2.weight",
    "model.prompt_encoder.point_embeddings.3.weight",
    "model.mask_decoder.iou_prediction_head.layers.0.weight",
    "model.mask_decoder.iou_prediction_head.layers.0.bias", 
    "model.mask_decoder.iou_prediction_head.layers.1.weight",
    "model.mask_decoder.iou_prediction_head.layers.1.bias", 
    "model.mask_decoder.iou_prediction_head.layers.2.weight", 
    "model.mask_decoder.iou_prediction_head.layers.2.bias",
]
    


class MagicPenModule(pl.LightningModule):
    multimask_output = False

    def __init__(self, 
                 network, 
                 optimizer, 
                 scheduler, 
                 loss, 
                 task_name: str,
                 compile: bool=True):

        super().__init__()

        self.model = network
        self.task_name = task_name
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss = loss
        self.compile = compile
        self.sgmd = nn.Sigmoid()

        # save all parameters with Lightning for checkpoints : access with Module.hparams
        # self.save_hyperparameters() # throw error
        
        self.train_metrics = MetricCollection( [
                BinaryJaccardIndex(),
            ], prefix="train")
        
        self.val_metrics =  MetricCollection( [
                BinaryJaccardIndex(),
            ], prefix="val")
        
        self.test_metrics =  MetricCollection( [
                BinaryJaccardIndex(),
            ], prefix="test")
        
        self.train_loss = []
        self.val_loss = []
        self.train_epoch_mean = None
        self.val_epoch_mean = None

    def on_train_start(self):
        if self.compile:
            self.model = torch.compile(self.model)
    
    def on_before_backward(self, loss):
        # for name, param in self.named_parameters():
            # print(f'on before backward, param name: {name}, grad status: {param.grad.shape if param.grad is not None else None}, grad require: {param.requires_grad}')
        pass

    def on_after_backward(self):
        # for name, param in self.named_parameters():
            # print(f'on after backward, param name: {name}, grad status: {param.grad.shape if param.grad is not None else None}, grad require: {param.requires_grad}')
        pass

    def forward(self, x):
        preds, ious =  self.model(x, multimask_output=self.multimask_output)
        # current out : B x 1 x 1024 x 1024
        return preds, ious
    
    def _step(self, batch):
        preds, ious  = self.forward(batch)
        preds = preds.squeeze()
        # align dim - case batch 1
        preds = preds.expand(*batch["label"].shape)
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
        self.train_loss.append(loss)
        self.train_metrics.update(self.sgmd(preds), batch["label"])

        if self.current_epoch % 10 == 0:
            if batch_idx % 50 == 0:
                bs = preds.shape[0]
                for b_i in range(bs):
                    fig = show_prediction_sample((outputs|dict(batch=batch)), idx=b_i)
                    self.logger.experiment.add_figure(
                            f"sample_{self.current_epoch}_{batch_idx}_{b_i}",
                            fig,
                        )
        # if self.current_epoch % 10 == 0:
        #     if batch_idx == 0:
        #         sm= nn.Sigmoid()
        #         self.logger.experiment.add_histogram(
        #                 f"hist_preds_{self.current_epoch}",
        #                 sm(preds[0].flatten()),
        #             )

    def on_validation_batch_end(self, outputs, batch, batch_idx):
        loss, preds = outputs["loss"], outputs["pred"]
        self.val_metrics.update(self.sgmd(preds), batch["label"])
        self.val_loss.append(loss)

    def on_test_batch_end(self, outputs, batch, batch_idx):
        loss, preds = outputs["loss"], outputs["pred"]
        self.test_metrics.update(self.sgmd(preds), batch["label"])


    def on_validation_epoch_end(self):
        metrics = self.val_metrics.compute()
        for metric_name, metric_value in metrics.items():
            self.log(
                f"val/{metric_name}",
                metric_value,
                sync_dist=True,
                on_step=False,
                on_epoch=True,
                prog_bar=True
            )
        
        self.val_epoch_mean = torch.stack(self.val_loss).mean()
        self.log(
            "val/loss", 
            self.val_epoch_mean, 
            on_step=False, 
            on_epoch=True, 
            sync_dist=True, 
            logger=True
        )
        # free up the memory
        self.val_loss.clear()
        

    def on_train_epoch_end(self):
        metrics = self.train_metrics.compute()
        for metric_name, metric_value in metrics.items():
            self.log(
                f"train/{metric_name}",
                metric_value,
                sync_dist=True,
                on_step=False,
                on_epoch=True,
                prog_bar=True        
            )

        self.train_epoch_mean = torch.stack(self.train_loss).mean()
        self.log(
            "train/loss", 
            self.train_epoch_mean, 
            on_step=False, 
            on_epoch=True, 
            sync_dist=True, 
            logger=True
            )
        # free up the memory
        self.train_loss.clear()
        
        

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

    def on_train_end(self):
        test_values = self.test_metrics.compute()
        val_values = self.val_metrics.compute()
        train_values = self.train_metrics.compute()

        scores = train_values | val_values | test_values
        final_losses = dict(train_loss=self.train_epoch_mean, val_loss=self.val_epoch_mean)

        self.logger.log_hyperparams(
            params={
                "task_name": self.task_name,
            },
            metrics= scores | final_losses
        )

    def configure_optimizers(self):
        if self.optimizer is not None:
            optimizer = self.optimizer(params=self.parameters())
        else:
            # ensure old runs working
            optimizer = load_default_opt(self)

        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


def load_default_opt(task):
    return torch.optim.Adam(params=task.parameters(), lr=1e-4)