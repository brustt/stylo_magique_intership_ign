from abc import ABC, abstractmethod
from typing import Any, Callable, List, Dict, Tuple, Union
from deprecated import deprecated
import torch
from torchmetrics import Metric, MetricCollection
from torchmetrics.segmentation import MeanIoU
from torchmetrics.detection import MeanAveragePrecision
from torchmetrics.classification import (
    BinaryF1Score,
    BinaryAccuracy,
    BinaryAveragePrecision,
    BinaryPrecision,
    BinaryRecall,
)
from torch.utils.data import Dataset, DataLoader

from magic_pen.config import DEVICE
from magic_pen.data.loader import BiTemporalDataset
from magic_pen.data.process import DefaultTransform
from segment_any_change.masks.mask_process import (
    _bbox_processing_labels,
    _bbox_processing_preds,
)
from segment_any_change.matching import BitemporalMatching
from segment_any_change.model import BiSam
from segment_any_change.utils import flush_memory, load_sam
from torch.utils.data import DataLoader
import logging

# TO DO : define globally
logging.basicConfig(format="%(asctime)s - %(levelname)s ::  %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


"""
new engine
"""

_register_metric_group = {
    "BinaryF1Score": "px_classif",
    "BinaryPrecision": "px_classif",
    "BinaryRecall": "px_classif",
    "MeanIou": "px_classif_iou",
}

_register_metric_processing = {
    "BinaryF1Score": "flat",
    "BinaryPrecision": "flat",
    "BinaryRecall": "flat",
    "MeanIou": "iou",
    "MeanAveragePrecision": "bbox",
}

PX_METRICS = [
    BinaryF1Score(),
    BinaryPrecision(),
    BinaryRecall(),
]


def _factory_metric_processing(name: str, preds, labels):
    match name:
        case "flat":
            return ProcessingEval().flat(preds, labels)
        case "bbox":
            preds = ProcessingEval().bbox_processing(preds, "pred")
            labels = ProcessingEval().bbox_processing(labels, "label")
            return preds, labels
        case "iou":
            preds = ProcessingEval().iou_processing(preds, "pred")
            return preds, labels


class ProcessingEval:

    def flat(self, preds, labels):

        masks, iou = preds.values()
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

    def bbox_processing(
        self, x: Union[Dict[str, torch.Tensor], torch.Tensor], data_type: str
    ) -> List[Dict]:
        if data_type == "pred":
            return _bbox_processing_preds(x)
        elif data_type == "label":
            return _bbox_processing_labels(x)
        else:
            raise ValueError("data_type not valid. Valid data_type : pred, label")

    def iou_processing(
        self, x: Union[Dict[str, torch.Tensor], torch.Tensor], data_type: str
    ) -> List[Dict]:
        # check input format
        raise RuntimeError("Not implemented yet")


class MetricEngine:
    def __init__(self, in_metrics: List[Metric] = None, prefix="") -> None:

        self.name = self.check_homogeneity(in_metrics)
        self.metrics = MetricCollection(in_metrics, prefix=prefix)

    def check_homogeneity(self, in_metrics):
        names = set([_register_metric_processing[type(m).__name__] for m in in_metrics])
        if len(names) > 1:
            print(names)
            raise RuntimeError(
                "Cannot instantiate metrics with differents input format under same MetricCollection"
            )
        return names.pop()

    def compute(self) -> None:
        return self.metrics.compute()

    def update(self, preds: torch.Tensor, labels: torch.Tensor) -> None:
        preds, labels = _factory_metric_processing(self.name, preds, labels)
        self.metrics.update(
            preds.to(DEVICE), labels.to(DEVICE)
        )  # to() will not work on list - convert in processing

    def reset(self) -> None:
        self.metrics.reset()

    def __call__(self, preds, targets):
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


"""
DEPRECATED
"""


_metrics_registry = {
    "pixel": {
        "px_classif": {
            "f1_score": BinaryF1Score,
            "precision": BinaryPrecision,
            "recall": BinaryRecall,
        },
        "px_segmentation": {"iou": MeanIoU},  # cannot be instantiate with current imp
    },
    "object": {
        "obj_segmentation": {
            "mAP": MeanAveragePrecision  # extended_summary=True return Precision, Recall, IoU
        }
    },
}


def get_px_metrics(mtype: str = "px_classif") -> List[Metric]:
    """Return px default metrics"""
    if mtype not in _metrics_registry["pixel"]:
        raise ValueError(
            f"Invalid metric type mtype. Valid : {list(_metrics_registry['pixel'])}"
        )
    return [m().to(DEVICE) for m in _metrics_registry["pixel"][mtype].values()]


def get_obj_metrics(mtype: str = "obj_segmentation") -> List[Metric]:
    """Return obj default metrics"""
    if mtype not in _metrics_registry["object"]:
        raise ValueError(
            f"Invalid metric type mtype. Valid : {list(_metrics_registry['object'])}"
        )
    return [m().to(DEVICE) for m in _metrics_registry["object"][mtype].values()]


def find_register_group(d: Dict, key: str) -> Dict:
    """Extract metrics from key name"""
    if key in d:
        return d[key]
    for k, v in d.items():
        if isinstance(v, dict):
            result = find_register_group(v, key)
            if result:
                return result
    return {}


def get_root_group_registry(key: str, registry: Dict = _metrics_registry) -> str:
    """Extract group name from metric registry from nested key"""

    def paths(tree: Dict, cur=()):
        if not isinstance(tree, dict):
            yield cur
        else:
            for n, s in tree.items():
                for path in paths(s, cur + (n,)):
                    yield path

    for p in paths(registry):
        if key in p:
            return p[0]


@deprecated
class MetricGroupGeneric(ABC):

    def __init__(
        self,
        in_metrics: List[Dict[str, Any]],
        metric_key: str = None,
        default_init: Callable = None,
    ) -> None:
        self._metrics = self._build_metrics(in_metrics, metric_key, default_init)

    def _build_metrics(
        self,
        in_metrics: List[Dict[str, Any]],
        metric_key: str = None,
        default_init: Callable = None,
    ) -> MetricCollection:

        metrics: List = []

        if not any([in_metrics, metric_key]):
            # select default metrics
            return MetricCollection(default_init()).to(DEVICE)
        elif in_metrics:
            # select by metrics names
            for metric in in_metrics:
                if metric.get("params", []):
                    metrics.append(
                        find_register_group(_metrics_registry, metric["name"])(
                            **metric["params"]
                        ).to(DEVICE)
                    )
                else:
                    metrics.append(
                        find_register_group(_metrics_registry, metric["name"])().to(
                            DEVICE
                        )
                    )
        else:
            # select metrics by group name
            register_group = find_register_group(_metrics_registry, metric_key)
            if isinstance(register_group, dict):
                metrics = [v().to(DEVICE) for k, v in register_group.items()]
            else:
                metrics = [register_group()]

        return MetricCollection(metrics).to(DEVICE)

    @abstractmethod
    def processing(self, **kwargs):
        raise NotImplementedError("Please provide processing implementation")


@deprecated
class PxGroupMetric(MetricGroupGeneric):
    def __init__(
        self, in_metrics: List[Dict[str, Any]] = None, metric_key=None
    ) -> None:
        super().__init__(in_metrics, metric_key, default_init=get_px_metrics)

    def processing(
        self,
        preds: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        metric_name: str = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        masks, iou = preds.values()

        if masks.ndim > 3:
            masks = torch.sum(masks, axis=1)

        labels = (labels > 1) * 1
        masks = (masks > 1) * 1

        masks = masks.view(masks.shape[0], -1)
        labels = labels.view(labels.shape[0], -1)

        return masks, labels


@deprecated
class ObjGroupMetric(MetricGroupGeneric):
    def __init__(
        self, in_metrics: List[Dict[str, Any]] = None, metric_key=None
    ) -> None:
        super().__init__(in_metrics, metric_key, default_init=get_obj_metrics)

    def processing(
        self, preds: Dict[str, torch.Tensor], labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # preds = bbox_processing_factory(preds, "pred") # deactivate
        # labels = bbox_processing_factory(labels, "label")

        return preds, labels


@deprecated
class MetricEngine_:

    _builder = {"pixel": PxGroupMetric, "object": ObjGroupMetric}

    def __init__(
        self, in_metrics: List[Dict[str, Any]] = None, metric_key=None
    ) -> None:

        self.engine = self.init_metrics_group(in_metrics, metric_key)

    def init_metrics_group(
        self,
        in_metrics: List[Dict[str, Any]] = None,
        metric_key: Union[str, List] = None,
    ) -> None:

        metrics_group = {"pixel": [], "object": []}
        engine = {}

        if in_metrics:
            logger.info("dispach metrics")
            # dispach metrics in PX or OBJ builders
            for metric in in_metrics:
                name = metric["name"] if isinstance(metric, dict) else metric
                gp_name = get_root_group_registry(name)
                metrics_group[gp_name].append(metric)

            for gp_name, l_metrics in metrics_group.items():
                if l_metrics:
                    engine[gp_name] = self._builder[gp_name](in_metrics=l_metrics)

        elif metric_key:
            logger.info("group metric")
            # build group metrics
            gp_name = get_root_group_registry(metric_key)
            engine[gp_name] = self._builder[gp_name](metric_key=metric_key)

        else:
            logger.info("default run metric")
            # run default metrics
            for gp_name in self._builder:
                engine[gp_name] = self._builder[gp_name]()

        return engine

    def compute(self) -> None:
        res = {}
        logger.info("let's compute metrics")  # draft doesn't compute
        for group in self.engine:
            logger.info(f"compute for {group}")
            res[group] = self.engine[group]._metrics.compute()
        return res

    def update(self, preds: torch.Tensor, labels: torch.Tensor) -> None:
        for group in self.engine:
            preds_, labels_ = self.engine[group].processing(preds=preds, labels=labels)
            self.engine[group]._metrics.update(
                preds_.to(DEVICE), labels_.to(DEVICE)
            )  # to() will not work on list

    def reset(self) -> None:
        for group in self.engine:
            self.engine[group]._metrics.reset()
