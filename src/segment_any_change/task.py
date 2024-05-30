import pytorch_lightning as pl
import torchvision

from magic_pen.config import DEVICE
from torchmetrics.classification import BinaryF1Score

from segment_any_change.eval import MetricEngine

class CDModule(pl.LightningModule):
    def __init__(self, model, metrics):
        super().__init__()
        self.model = model
        self.metrics_predict = MetricEngine(metrics, prefix="pred_")

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass

    def predict_step(self, batch, batch_idx):
        preds = self.model(batch)
        pred_metrics = self.metrics_predict(preds, batch["label"])
        #self.log_dict(pred_metrics)
        return pred_metrics
    
    def on_predict_epoch_end(self) -> None:
        self.metrics_predict.reset()

    def configure_optimizers(self):
        pass