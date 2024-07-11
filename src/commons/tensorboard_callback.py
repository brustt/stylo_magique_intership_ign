import os
from typing import Any, Dict, Tuple, Union
from commons.utils_io import make_path
import numpy as np
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback, BasePredictionWriter
import logging
from torchmetrics.classification import (
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
    BinaryJaccardIndex,
    BinaryConfusionMatrix,
)
from torchmetrics.detection import MeanAveragePrecision
from hydra.core.hydra_config import HydraConfig

from src.models.segment_any_change.config_run import ExperimentParams
from .eval import MetricEngine, UnitsMetricCounts, _factory_metric_processing
from .utils import (
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

    def __init__(self, params: DictConfig):
        super().__init__()
        
        # recursively convert OmegaConfDictConfig to plain python object for MetricEngine compliance
        params = OmegaConf.to_object(params)

        self.tracking_instance_metrics = []
        self.tracking_instance = []
        self.metrics_classif = MetricEngine(
            [
                BinaryF1Score(),
                BinaryPrecision(),
                BinaryRecall(),
                BinaryJaccardIndex(),
            ],
            prefix="",
            name="classif",
            **params.get("engine_metric"),
        )
        self.metrics_classif_instance = MetricEngine(
            [
                BinaryF1Score(),
                BinaryPrecision(),
                BinaryRecall(),
                BinaryJaccardIndex(),
            ],
            prefix="",
            name="classif",
            **params.get("engine_metric"),
        )
        self.map = MetricEngine(
            [
                MeanAveragePrecision(
                    iou_type=params.get("engine_metric").get("iou_type_mAP"),
                    max_detection_thresholds=params.get("engine_metric").get("max_detection_thresholds")
                ),
            ],
            prefix="",
            name="map",
            **params.get("engine_metric"),
        )
        self.map_instance = MetricEngine(
            [
                MeanAveragePrecision(
                    iou_type=params.get("engine_metric").get("iou_type_mAP"),
                    max_detection_thresholds=params.get("engine_metric").get("max_detection_thresholds")
                ),
            ],
            prefix="",
            name="map",
            **params.get("engine_metric"),
        )
        self.confmat = MetricEngine(
            [BinaryConfusionMatrix(normalize="true")],
            prefix="",
            name="confmat",
            **params.get("engine_metric"),
        )

        self.confmat_units = MetricEngine(
            [UnitsMetricCounts()], prefix="", name="units", **params.get("engine_metric")
        )

        self._register_engine_instance = [
            self.metrics_classif_instance,
            self.map_instance,
        ]

        # logger.info("Metric device : ", next(iter(self.confmat.metrics.values())).device)

    def add_metric(self, key, value, pl_module, batch_idx):
        pl_module.logger.experiment.add_scalar(key, value, global_step=batch_idx)

    def update_metrics(self, preds: Dict, labels: torch.Tensor):

        batch_metrics = [{}] * labels.shape[0]
        format_input_engine = {
            "map": lambda x, y: ([x], [y]),
            "classif": lambda x, y: (x, y),
        }

        # global metrics
        self.confmat.update(preds, labels)
        self.metrics_classif.update(preds, labels)

        # pre processing is done twice in regards of global metrics :(
        for engine in self._register_engine_instance:
            preds_, labels_ = _factory_metric_processing(
                engine.check_processing(engine.name), preds, labels, **engine.params
            )
            # loop over batch instances
            for i, (p, l) in enumerate(zip(preds_, labels_)):
                engine.metrics.update(*format_input_engine[engine.name](p, l))
                out_metric = {k:np.round(_.item(), 3) for k,_ in engine.compute().items()} 
                batch_metrics[i] = batch_metrics[i] | out_metric
            engine.reset()
        # save instance metric level
        self.tracking_instance_metrics.extend(batch_metrics)

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

        self.update_metrics(preds, label)
        self.tracking_instance.append(batch["label"])

        if batch_idx % 15 == 0:
            pred_cls = self.confmat_units(preds, label)
            tp, fp, fn, tn = pred_cls.values()
            img_sample = create_grid_batch(preds["masks"], batch, tp, fp, fn)
            pl_module.logger.experiment.add_image(
                f"batch_{batch_idx}",
                img_sample,
                batch_idx,
                dataformats="CHW",
            )

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:

        confmat = self.confmat.compute()
        out_map = self.map.compute()
        out_classif = self.metrics_classif.compute()

        conf_mat_fig_px = plot_confusion_matrix(
            confmat["BinaryConfusionMatrix"].cpu().numpy(), fig_return=True
        )
        pl_module.logger.experiment.add_figure(
            "binary_confusion_matrix_pixel", conf_mat_fig_px, pl_module.global_step
        )

        pl_module.logger.log_hyperparams(
            params={
                "model_type": pl_module.params.model_type,
                "batch_size": pl_module.params.batch_size,
                "th_change_proposals": pl_module.params.th_change_proposals,
            },
            metrics={
                "Precision": out_classif[f"BinaryPrecision"],
                "Recall": out_classif["BinaryRecall"],
                "F1-Score": out_classif["BinaryF1Score"],
                "IoU": out_classif["BinaryJaccardIndex"],
                "mAR_1000": out_map["mar_1000"],
                "mAP": out_map["map"],
            },
        )

        self.confmat.reset()
        self.map.reset()
        self.metrics_classif.reset()

    def on_test_end(self, trainer: Trainer, pl_module: LightningModule) -> None:

        _rank_metric = "BinaryF1Score"
        _top_k = 10

        self.tracking_instance_metrics = pd.DataFrame(self.tracking_instance_metrics)
        # TODO : save to logs
        logger.info(self.tracking_instance_metrics.sort_values(by=_rank_metric).head(20))

        # top_metric = self.tracking_instance_metrics.sort_values(by=_rank_metric)[
        #     :_top_k
        # ]
        # selected_idx = top_metric.index

        # dloader = trainer.test_dataloaders
        # TODO : get best predictions and worst predictions
        # store predictions in memory vs re-run model on indices vs store 10 best and 10 worst

        hydra_output_dir = HydraConfig.get().run.dir
        self.tracking_instance_metrics.to_csv(make_path("instances_metrics.csv", hydra_output_dir), index=False)

"""
# CustomWriter : 
- save model (checkpoint)
- parameters
"""


class PredictionWriter(BasePredictionWriter):
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
        output = {"pred": prediction["pred"], "batch": batch}
        torch.save(output, os.path.join(self.output_dir, f"{batch_idx}.pt"))
