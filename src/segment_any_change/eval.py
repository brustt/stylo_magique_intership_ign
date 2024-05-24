from typing import Any, List, Dict, Union
import torch
from torchmetrics import MetricCollection, F1Score
from torchmetrics.detection.iou import IntersectionOverUnion

from torch.utils.data import Dataset, DataLoader

_metrics_registry = {
    "f1_score":F1Score,
    "iou": IntersectionOverUnion
}
class CDEvaluator:
    def __init__(self, metrics: List[Dict[str, Any]]) -> None: 
        self._metrics = self.build_metrics(metrics)

    def build_metrics(self, metrics: List[Dict[str, Any]]) -> MetricCollection:
        global _metrics_registry
        metrics = []
        for metric in metrics:
            if metric.get("params", []): 
                metrics.append(_metrics_registry[metric["name"]](**metric["params"]))
        return MetricCollection(metrics)
    
    def compute(self) -> None:
        return self._metrics.compute()

    def update(self, preds: torch.Tensor, labels: torch.Tensor) -> None:
        self._metrics.update(preds, labels)

    def reset(self) -> None:
        self._metrics.reset()

@torch.no_grad()
def evaluate(model, 
             dataset: Union[str, Dataset], 
             batch_size: int,
             evaluator=CDEvaluator, 
             device="cpu",
             **model_params):
    
    # model.eval() # check if it is nn.Module first
    # model.to(device)

    if isinstance(dataset, str):
        raise TypeError("Evaluation from dataset directory is not implemented yet")
    
    dataloader = DataLoader(dataset, batch_size)
    for i_batch, input_batch in enumerate(dataloader):
        preds = model(input_batch, **model_params)
        evaluator.update(preds, input_batch["label"])
    metrics = evaluator.compute()
    evaluator.reset()
    return metrics
