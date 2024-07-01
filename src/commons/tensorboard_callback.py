from dataclasses import asdict
from typing import Any, Tuple
import numpy as np
import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import BasePredictionWriter
import os
import logging
from torchmetrics.classification import (
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
    BinaryJaccardIndex,
    BinaryConfusionMatrix
)
from torchmetrics.detection import MeanAveragePrecision
from src.models.segment_any_change.config_run import ExperimentParams
from commons.eval import MetricEngine, UnitsMetricCounts
from src.commons.utils import (
    create_grid_batch,
    plot_confusion_matrix,

)


# TO DO : define globally
logging.basicConfig(format="%(asctime)s - %(levelname)s ::  %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# TO DO : rewrite each return (units counts, metric, confusion matrix) as different Callbacks ?


class TensorBoardCallbackLogger(Callback):
    """
    doc summaryWriter Tensorboard : https://tensorboardx.readthedocs.io/en/stable/tensorboard.html#tensorboardX.SummaryWriter

    TO DO :
    - ADD Confusion Matrix done
    - ADD PR CURVE

    Split in different callbacks
    """

    _units_key_name = "UnitsMetricCounts"

    def __init__(self, params: ExperimentParams):
        super().__init__()
        self.metrics_classif = MetricEngine(
            [
                BinaryF1Score(),
                BinaryPrecision(),
                BinaryRecall(),
                BinaryJaccardIndex(),
            ], prefix="", name="classif", **params.engine_metric
        )
        self.map = MetricEngine(
            [
                MeanAveragePrecision(
                    iou_type=params.engine_metric.get("iou_type_mAP", "segm"), 
                    max_detection_thresholds=params.engine_metric.get("max_detection_thresholds", None)
                    ),
            ], prefix="", name="map", **params.engine_metric
        )
        self.confmat = MetricEngine(
            [BinaryConfusionMatrix(normalize="true")], prefix="", name="confmat", **params.engine_metric
        )

        self.confmat_units = MetricEngine(
            [UnitsMetricCounts()], prefix="", name="units", **params.engine_metric
        )

    def add_metric(self, key, value, pl_module, batch_idx):
        pl_module.logger.experiment.add_scalar(
            key, value, global_step=batch_idx
        )

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        
        preds, label = outputs["pred"], batch["label"]
        # does a compute is getting previous scores ? Seems so
        out_classif = self.metrics_classif(preds, label)
        out_map = self.map(preds, label)
        self.confmat.update(preds, label)
        pred_cls  = self.confmat_units(preds, label)
        tp, fp, fn, tn = pred_cls.values()

        pl_module.log_dict(out_classif, on_step=True, on_epoch=False, prog_bar=True)

        _out_map = {k:v for k,v in out_map.items() if k in ["map", "mar_1000"]}
        pl_module.log_dict(_out_map, on_step=True, on_epoch=False, prog_bar=True)

        if batch_idx % 15 == 0:
            img_sample = create_grid_batch(preds["masks"], batch, tp, fp, fn)
            pl_module.logger.experiment.add_image(
                f"batch_{batch_idx}",
                img_sample,
                batch_idx,
                dataformats="CHW",
            )

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:

        # should I re-compute ?
        confmat = self.confmat.compute()
        out_map = self.map.compute()
        out_classif = self.metrics_classif.compute()

        print(out_map)
        conf_mat_fig_px = plot_confusion_matrix(
            confmat["BinaryConfusionMatrix"].cpu().numpy(), fig_return=True
        )
        pl_module.logger.experiment.add_figure(
             "binary_confusion_matrix_pixel", conf_mat_fig_px, pl_module.global_step
        )

        pl_module.logger.experiment.add_hparams(
            hparam_dict={
                "model_type": pl_module.params.model_type,
                "batch_size": pl_module.params.batch_size,
                "th_change_proposals": pl_module.params.th_change_proposals,
            },
            metric_dict={
                "Precision":out_classif[f"BinaryPrecision"],
                "Recall":out_classif["BinaryRecall"],
                "F1-Score":out_classif["BinaryF1Score"],
                "IoU":out_classif["BinaryJaccardIndex"],
                "mAP": out_map["map"],
                "mAR_1000":out_map["mar_1000"]
                }
        )

        self.confmat.reset()
        self.map.reset()
        self.metrics_classif.reset()


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
        pass

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        to_write = ["predictions", "batch_idx"]
        torch.save(predictions, os.path.join(self.output_dir, "predictions.pt"))
        logger.info(os.path.join(self.output_dir, "predictions.pt"))
        # output = [{k: v for k, v in p.items()} for p in predictions]
        # torch.save(output, os.path.join(self.output_dir, "predictions.pt"))
        # logger.info(os.path.join(self.output_dir, "predictions.pt"))
