import os
from typing import Any, Dict, List
import pytorch_lightning as pl
import torch
from .config_run import ExperimentParams


class CDModule(pl.LightningModule):
    def __init__(self, model, metrics: List, params: ExperimentParams):
        super().__init__()
        self.model = model

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
        preds = self.forward(batch)
        return {"pred": preds, "batch_idx": batch_idx}

    def on_test_epoch_end(self) -> None:
        pass
    def configure_optimizers(self):
        pass
