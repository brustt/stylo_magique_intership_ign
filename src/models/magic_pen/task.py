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

    def __init__(self, network, params):

        super().__init__()
        self.params = params
        self.model = network

        # let's prevent checkpoint forgetting
        if not params.get("sam_ckpt_path", None):
            raise ValueError("Please provide sam checkpoint")
        
        if not params.get("ft_mode", None):
            raise ValueError("Please provide ft mode")
        
        
        self.load_weights(params.get("sam_ckpt_path"), params.get("use_weights"))
        self.freeze_weigts(params.get("ft_mode"))
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
        
        self.train_loss = []
        self.val_loss = []
    
    def load_weights(self, checkpoint: str, use_weights: Union[Any, List]) ->None:
        pretrained_weights = torch.load(checkpoint)
        if use_weights is None:
            # we use all weights
            self.model.load_state_dict(pretrained_weights, strict=True)
        else:
            # we select weights to load 
            model_dict = self.model.state_dict()
            pretrained_weights = {k: v for k, v in pretrained_weights.items() if any([k.startswith(m) for m in use_weights])}
            model_dict.update(pretrained_weights)
            self.model.load_state_dict(model_dict, strict=False)
            logger.info(f"Weights loaded for : {use_weights}")

    def freeze_weigts(self, ft_mode: str):
        """
        # TODO: 
        #  - freeze layer on key layer selection / name
            - set init weights based on known distrib
        """
        def match_name(trainable_name: str, layer_name: str) -> bool:
            return bool(re.search(trainable_name, layer_name))
        
        if ft_mode == "probing":
            #self.model.image_encoder.requires_grad_(False)
            for param in self.model.image_encoder.parameters():
                param.requires_grad_(False)

        elif ft_mode == "adapter":
            #  ImageEncoderAdapterVit has adapter layer
            for name, l in self.model.image_encoder.named_parameters():
                if not match_name(ft_mode, name):
                    l.requires_grad_(False)

        # freeze layer not contributing to backpropagation
        for name, l in self.model.named_parameters():
            if match_name("iou_prediction_head", name):
                l.requires_grad_(False)
            if match_name("prompt_encoder.mask_downscaling", name):
                l.requires_grad_(False)
            self.model.prompt_encoder.point_embeddings[2].requires_grad_(False)
            self.model.prompt_encoder.point_embeddings[3].requires_grad_(False)

    def on_before_backward(self, loss):
        """
        for name, param in self.named_parameters():
            print(f'on before backward, param name: {name}, grad status: {param.grad.shape if param.grad is not None else None}, grad require: {param.requires_grad}')
        pass
        """
        ...

    def on_after_backward(self):
        """
        for name, param in self.named_parameters():
            print(f'on after backward, param name: {name}, grad status: {param.grad.shape if param.grad is not None else None}, grad require: {param.requires_grad}')
        pass
        """
        ...

    def forward(self, x):
        # try with multimask_output == True and select best one
        # bisam_diff modified dirt and quick
        preds, ious = self.model(x, multimask_output=self.multimask_output)
        # to be updated : current out : B x 1 x 1 x 1024 x 1024
        return preds, ious
    
    def _step(self, batch):
        preds, ious  = self.forward(batch)
        preds = preds.squeeze()
        # align dim - case batch 1
        preds = preds.expand(*batch["label"].shape)
        loss = self.loss(preds, batch["label"])
        return preds, loss

    def training_step(self, batch, batch_idx):
        # print(f"{self.current_epoch }/{batch_idx}") # batch_idx is wrong in ddp 
        preds, loss = self._step(batch)
        self.train_metrics.update(preds, batch["label"])
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
        self.train_metrics.update(preds, batch["label"])

        if self.current_epoch % 10 == 0:
            if batch_idx % 10 == 0:
                bs = preds.shape[0]
                for b_i in range(bs):
                    fig = show_prediction_sample((outputs|dict(batch=batch)), idx=b_i)
                    self.logger.experiment.add_figure(
                            f"sample_{self.current_epoch}_{batch_idx}",
                            fig,
                        )
        if self.current_epoch % 10 == 0:
            if batch_idx == 0:
                sm= nn.Sigmoid()
                self.logger.experiment.add_histogram(
                        f"hist_preds_{self.current_epoch}",
                        sm(preds[0].flatten()),
                    )

    def on_validation_batch_end(self, outputs, batch, batch_idx):
        loss, preds = outputs["loss"], outputs["pred"]
        self.val_metrics.update(preds, batch["label"])
        self.val_loss.append(loss)

    def on_test_batch_end(self, outputs, batch, batch_idx):
        loss, preds = outputs["loss"], outputs["pred"]
        self.test_metrics.update(preds, batch["label"])


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
        
        epoch_mean = torch.stack(self.val_loss).mean()
        self.log(
            "val/loss", 
            epoch_mean, 
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

        epoch_mean = torch.stack(self.train_loss).mean()
        self.log(
            "train/loss", 
            epoch_mean, 
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

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        
        return {"optimizer": optimizer}
