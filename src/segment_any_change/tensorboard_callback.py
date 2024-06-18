from dataclasses import asdict
from typing import Any, Tuple
import numpy as np
import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import BasePredictionWriter
import torchvision
import os
import logging

from segment_any_change.utils import (
    create_overlay_outcome_cls,
    get_units_cnt_obj,
    get_units_cnt_px,
    plot_confusion_matrix,
    rm_substring,
    shift_range_values,
    substring_present,
    to_numpy,
)
from segment_any_change.eval import (
    _register_metric_classif_px,
    _register_metric_classif_obj,
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

    def __init__(self, log_n_steps=5):
        super().__init__()
        self.log_n_steps = log_n_steps
        self.tp_tracking = []
        self.fp_tracking = []
        self.fn_tracking = []
        self.tn_tracking = []
        self.metrics_px_tracking = {
            f"pred_{k}": [] for k in _register_metric_classif_px
        }

    def add_metric(self, key, value, pl_module, batch_idx):
        pl_module.logger.experiment.add_scalar(
            key, value, global_step=batch_idx
        )

    def create_grid_batch(self, preds, batch, tp, fp, fn) -> np.ndarray:
        """create image grid from sample (imgA, imgB), label and masks predictions"""
        sample = []
        images_A = batch["img_A"].cpu()
        images_B = batch["img_B"].cpu()
        labels = batch["label"].cpu()
        img_outcome_cls = torch.zeros(images_A.shape[-2:])

        # to batchify ?
        for i in range(images_A.size(0)):

            img_A = images_A[i]
            img_B = images_B[i]
            # Align to 3 channels
            label = labels[i].unsqueeze(0).repeat(3, 1, 1)
            print(f"UNITS : {tp[i].sum()} - {fp[i].sum()} - {fn[i].sum()}")
            img_outcome_cls = create_overlay_outcome_cls(tp[i], fp[i], fn[i])

            # Stack individual masks and align to 3 channels
            pred = (
                torch.sum(preds[i, ...], axis=0)
                .unsqueeze(0)
                .repeat(3, 1, 1)
                .to(torch.uint8)
            )
            pred = shift_range_values(pred, new_bounds=[0, 255]).to(torch.uint8)
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

    def extract_preds_cls(
        self, outputs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        preds = outputs["pred"]["masks"].cpu()
        tp, fp, fn, tn = None, None, None, None
        # pass key pred_UnitsMetricCounts to params
        # maybe keep track of count instead
        if any(
            [
                substring_present(self._units_key_name, key)
                for key in list(outputs["metrics"])
            ]
        ):
            tp = outputs["metrics"]["pred_UnitsMetricCounts"]["tp_indices"]
            fp = outputs["metrics"]["pred_UnitsMetricCounts"]["fp_indices"]
            fn = outputs["metrics"]["pred_UnitsMetricCounts"]["fn_indices"]
            tn = outputs["metrics"]["pred_UnitsMetricCounts"]["tn_indices"]

        # flush units tracking - do it in Metric cls
        del outputs["metrics"]["pred_UnitsMetricCounts"]

        return preds, tp, fp, fn, tn

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:

        preds, tp, fp, fn, tn = self.extract_preds_cls(outputs)
        # print(f"TP sum {tp.sum()}", tp.shape)
        # print(f"FP sum {fp.sum()}", fp.shape)
        # print(f"FN sum {fn.sum()}")
        print(f"Label 1 sum {batch['label'].sum(dim=(1, 2))}")

        # self.tp_tracking.append(tp)
        # self.fp_tracking.append(tp)
        # self.fn_tracking.append(fn)
        # self.tn_tracking.append(tn)

        for key, metric in outputs["metrics"].items():
            # if it is scalar px classification metric
            if any(
                [
                    rm_substring(key, substring="pred_") == ref_metric
                    for ref_metric in _register_metric_classif_px
                ]
            ):
                # self.add_metric(key, metric, pl_module, trainer)
                self.metrics_px_tracking[key].append(metric)
                self.add_metric(key, metric, pl_module, batch_idx)
                #pl_module.log(key, metric, prog_bar=True, on_step=True)

            if rm_substring(key, substring="pred_") == "BinaryConfusionMatrix":
                confmat = metric.cpu().numpy()
            if any(
                [
                    rm_substring(key, substring="pred_") == ref_metric
                    for ref_metric in _register_metric_classif_obj
                ]
            ):
                for skey, smetric in metric.items():
                    #pl_module.log(skey, smetric, prog_bar=True, on_step=True)
                    self.add_metric(skey, smetric, pl_module, batch_idx)

        # TODO : review
        if batch_idx == 2000000:
            for key, metric in outputs["metrics"].items():
                if any(
                    [
                        rm_substring(key, substring="pred_") == ref_metric
                        for ref_metric in _register_metric_classif_px
                    ]
                ):
                    self.add_metric(key, metric, pl_module, trainer)
                    # self.metrics_px_tracking[key].append(metric)

                if any(
                    [
                        rm_substring(key, substring="pred_") == ref_metric
                        for ref_metric in _register_metric_classif_obj
                    ]
                ):
                    for skey, smetric in metric.items():
                        self.add_metric(skey, smetric, pl_module, trainer)

        if batch_idx % 1 == 0:
            print(f"CHECK label : {tp.sum() + fn.sum()}")
            img_sample = self.create_grid_batch(preds, batch, tp, fp, fn)
            pl_module.logger.experiment.add_image(
                f"batch_{batch_idx}",
                img_sample,
                batch_idx,
                dataformats="CHW",
            )

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:

        # TODO : put compute() metrics here

        # need to change for multiclass
        # very expensive, stack the whole dataset !
        # self.tp_tracking = torch.cat(self.tp_tracking, dim=0)
        # self.fp_tracking = torch.cat(self.fp_tracking, dim=0)
        # self.fn_tracking = torch.cat(self.fn_tracking, dim=0)
        # self.tn_tracking = torch.cat(self.tn_tracking, dim=0)

        # we could count before units on  batch
        # confusion matrix
        # tp_px_cnt, fp_px_cnt, fn_px_cnt, tn_px_cnt = get_units_cnt_px(
        #     self.tp_tracking, self.fp_tracking, self.fn_tracking, self.tn_tracking
        # )
        # tp_obj_cnt, fp_obj_cnt, fn_obj_cnt, tn_obj_cnt = get_units_cnt_obj(self.tp_tracking , self.fp_tracking , self.fn_tracking, self.tn_tracking)

        # conf_mat_fig_px = plot_confusion_matrix(
        #     tp_px_cnt, fp_px_cnt, fn_px_cnt, tn_px_cnt, fig_return=True
        # )
        # conf_mat_fig_obj = plot_confusion_matrix(tp_obj_cnt, fp_obj_cnt, fn_obj_cnt, tn_obj_cnt, fig_return=True)



         # Plot confusion matrix
        confmat = pl_module.confmat.compute()
        conf_mat_fig_px = plot_confusion_matrix(
            confmat["pred_BinaryConfusionMatrix"].cpu().numpy(), fig_return=True
        )
        pl_module.logger.experiment.add_figure(
             "binary_confusion_matrix_pixel", conf_mat_fig_px, pl_module.global_step
        )

        # print(self.metrics_px_tracking)
        pl_module.logger.experiment.add_hparams(
            hparam_dict={
                "model_type": pl_module.params.model_type,
                "batch_size": pl_module.params.batch_size,
                "th_change_proposals": pl_module.params.th_change_proposals,
            },
            metric_dict={
                k: torch.mean(torch.stack(v))
                for k, v in self.metrics_px_tracking.items()
            },
        )

        # pl_module.logger.experiment.add_figure(
        #         "binary_confusion_matrix_object", conf_mat_fig_obj, pl_module.global_step
        #     )


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
