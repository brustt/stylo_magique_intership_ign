from abc import ABC, abstractmethod
from typing import Any, Callable, List, Dict, Tuple, Union
from deprecated import deprecated
import torch
from torchmetrics import Metric, MetricCollection
from torch.utils.data import Dataset, DataLoader

from magic_pen.config import DEVICE
from segment_any_change.masks.mask_process import (
    _bbox_processing,
    _mask_processing,
)
from torch.utils.data import DataLoader
import re
import logging

from segment_any_change.utils import timeit

# TO DO : define globally
logging.basicConfig(format="%(asctime)s - %(levelname)s ::  %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


_register_metric_processing = {
    "BinaryF1Score": "flat",
    "BinaryPrecision": "flat",
    "BinaryRecall": "flat",
    "BinaryJaccardIndex": "iou",
    "MeanAveragePrecision": "mAP",
    "UnitsMetricCounts": "identity",
}

_register_metric_classif_px = ["BinaryF1Score", "BinaryPrecision", "BinaryRecall"]
_register_metric_classif_obj = ["MeanAveragePrecision"]


class UnitsMetricCounts(Metric):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.add_state("tp_indices", default=[], dist_reduce_fx="cat")
        self.add_state("fp_indices", default=[], dist_reduce_fx="cat")
        self.add_state("fn_indices", default=[], dist_reduce_fx="cat")
        self.add_state("tn_indices", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, label: torch.Tensor) -> None:

        if preds.ndim > 2:
            preds = torch.sum(preds, axis=1).to(torch.int8)

        if preds.shape != label.shape:
            raise ValueError("preds and target must have the same shape")

        # ensure [0, 1] values
        MAX = torch.max(label)
        label = label / MAX

        self.tp_indices.append((preds * label))
        self.fp_indices.append((preds * (1 - label)))
        self.fn_indices.append(((1 - preds) * label))
        self.tn_indices.append(((1 - preds) * (1 - label)))

    def compute(self) -> Dict[str, torch.Tensor]:
        # does we need to keep discrimate objects; i.e not only keep binary indices ?
        # 1 x B x H x W
        tp_indices = torch.cat(self.tp_indices, dim=0)
        fp_indices = torch.cat(self.fp_indices, dim=0)
        fn_indices = torch.cat(self.fn_indices, dim=0)
        tn_indices = torch.cat(self.fn_indices, dim=0)

        return dict(
            tp_indices=tp_indices,
            fp_indices=fp_indices,
            fn_indices=fn_indices,
            tn_indices=tn_indices,
        )

    def __call__(
        self, preds: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor]:
        self.update(preds, target)
        return self.compute(self)


def _factory_metric_processing(name: str, preds, labels, **kwargs) -> Tuple[Any, Any]:
    match name:
        case "identity":
            return ProcessingEval().identity_mask_processing(preds, labels)
        case "flat":
            return ProcessingEval().flat_processing(preds, labels)
        case "mAP":
            # add similarity scores as values threshold
            return ProcessingEval().mAP_processing(preds, labels, **kwargs)
        case "iou":
            preds = ProcessingEval().iou_processing(preds, None)
            return preds, labels
        case _:
            raise ValueError("Processing name not found")


class ProcessingEval:

    def get_bitemporal_matching_outputs(
        self, preds
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        assert len(preds) == 3, "Incorrect dictionary length"
        masks, iou, ci = preds.values()
        return masks, iou, ci

    def identity_mask_processing(self, preds, labels) -> Tuple[torch.Tensor]:
        return preds["masks"], labels

    def flat_processing(self, preds, labels) -> Tuple[torch.Tensor]:

        masks = preds["masks"]
        # aggregates masks
        if masks.ndim > 3:
            masks = torch.sum(masks, axis=1)
        # binarize
        labels = (labels > 1) * 1
        masks = (masks > 1) * 1
        # flat
        masks = masks.view(masks.shape[0], -1)
        labels = labels.view(labels.shape[0], -1)

        return masks, labels

    @timeit
    def mAP_processing(
        self, preds: Dict[str, torch.Tensor], labels: torch.Tensor, **kwargs
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Return flat list over imgs batch compliant to input format Torchmetrics mAP
        """

        type_decision_mAP = kwargs.get("type_decision_mAP", "ci")
        iou_type = kwargs.get("iou_type_mAP", "segm")

        match iou_type:
            case "segm":
                preds = _mask_processing(
                    data_type="pred", preds=preds, type_decision_mAP=type_decision_mAP
                )
                labels = _mask_processing(
                    data_type="label",
                    labels=labels,
                    type_decision_mAP=type_decision_mAP,
                )
                return preds, labels
            case "bbox":
                preds = _bbox_processing(
                    data_type="pred", preds=preds, type_decision_mAP=type_decision_mAP
                )
                labels = _bbox_processing(
                    data_type="label",
                    labels=labels,
                    type_decision_mAP=type_decision_mAP,
                )
                return preds, labels
            case _:
                raise ValueError("iou_type not valid")

    def iou_processing(
        self, x: Union[Dict[str, torch.Tensor], torch.Tensor], data_type: str
    ) -> List[Dict]:
        # check input format
        raise RuntimeError("Not implemented yet")


class MetricEngine:
    def __init__(self, in_metrics: List[Metric] = None, prefix="", **kwargs) -> None:
        self.prefix = prefix
        self.metrics = MetricCollection(in_metrics, prefix=prefix)
        self.params = kwargs

    def check_processing(self, name: str) -> str:
        """Extract processing function key name from _register_metric_processing based on metric name"""
        raw_name = re.sub(self.prefix, "", name)
        if not _register_metric_processing.get(raw_name, None):
            raise KeyError(f"Please register metric : {name} to use it")
        return _register_metric_processing[raw_name]

    def compute(self) -> Dict[str, torch.Tensor]:
        res_metric = {}
        # maybe not the most efficient for metrics sharing group feature
        for name, metric in self.metrics.items():
            print(f"compute : {name}")
            res_metric[name] = metric.compute()
        return res_metric

    def update(self, preds: torch.Tensor, labels: torch.Tensor) -> None:
        # maybe not the most efficient for metrics sharing group feature
        for name, metric in self.metrics.items():
            print(f"update : {name}")
            proc_method = self.check_processing(name)
            preds_, labels_ = _factory_metric_processing(
                proc_method, preds, labels, **self.params
            )
            print(f"-")

            metric.update(
                preds_, labels_
            )  # to() will not work on list for bbox- convert in processing

    def reset(self) -> None:
        self.metrics.reset()

    def __call__(self, preds, targets) -> Dict[str, torch.Tensor]:
        self.update(preds, targets)
        return self.compute()


@torch.no_grad()
def evaluate(
    model: Any,
    dataset: Union[str, Dataset],
    batch_size: int,
    eval_engine: Any,
    device="cpu",
    **model_params,
):

    # model.eval() # check if it is nn.Module first
    # model.to(device)

    if isinstance(dataset, str):
        raise TypeError("Evaluation from dataset directory is not implemented yet")

    dataloader = DataLoader(dataset, batch_size)
    for i_batch, input_batch in enumerate(dataloader):
        preds = model(input_batch, **model_params)
        eval_engine.update(preds, input_batch["label"])
    metrics = eval_engine.compute()
    eval_engine.reset()
    return metrics
