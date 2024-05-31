import pytorch_lightning as pl
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
        return {"metrics": pred_metrics, "pred": preds, "batch_idx": batch_idx}

    def on_predict_epoch_end(self) -> None:
        self.metrics_predict.reset()

    def configure_optimizers(self):
        pass
