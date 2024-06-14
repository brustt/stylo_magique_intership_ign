import os
from typing import Any, Dict, List
import pytorch_lightning as pl
import torch
from segment_any_change.config_run import ExperimentParams
from segment_any_change.eval import MetricEngine
from torchmetrics.classification import (
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
    BinaryJaccardIndex,
    BinaryConfusionMatrix
)

class CDModule(pl.LightningModule):
    def __init__(self, model, metrics: List, params: ExperimentParams):
        super().__init__()
        self.model = model
        self.metrics_predict = MetricEngine(
            metrics, prefix="pred_", **params.engine_metric
        )
        self.confmat = MetricEngine(
            [BinaryConfusionMatrix(normalize="true")], prefix="pred_", **params.engine_metric
        )
        self.params = params

        self.n_core = int(
            (os.cpu_count() - self.params.num_worker) / self.params.n_job_by_node
        )
        torch.set_num_threads(self.n_core)

    def forward(self, batch):
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx) -> Dict[str, Any]:
        # add inference and metric() in function share by others hook
        preds = self.forward(batch)

        # update and compute - not efficient to compute for each batch - compute in loggercallback if needed
        pred_metrics = self.metrics_predict(preds, batch["label"])
        self.confmat.update(preds, batch["label"]) 

        return {"metrics": pred_metrics, "pred": preds, "batch_idx": batch_idx}

    def on_test_epoch_end(self) -> None:
        self.metrics_predict.reset()

    def configure_optimizers(self):
        pass
