import numpy as np
import lightning.pytorch as pl
import torch
from torch.utils import data
from .loader import BiTemporalDataset
from .process import DefaultTransform
from src.models.segment_any_change.config_run import ExperimentParams


class CDDataModule(pl.LightningDataModule):
    def __init__(self, name, params: ExperimentParams) -> None:
        super().__init__()
        self.name = name
        self.batch_size = params.batch_size
        self.num_worker = params.num_worker
        self.params = params

    def prepare_data(self):
        # download, IO, etc. Useful with shared filesystems
        # only called on 1 GPU/TPU in distributed
        pass

    def setup(self, stage):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        self.train = BiTemporalDataset(
            name=self.name,
            dtype="train",
            transform=DefaultTransform(),
            params=self.params,
        )
        # self.val = BiTemporalDataset(
        #     name=self.name, dtype="val", transform=DefaultTransform()
        # ) # not implement for SECOND
        self.test = BiTemporalDataset(
            name=self.name,
            dtype="test",
            transform=DefaultTransform(),
            params=self.params,
        )

    def train_dataloader(self):
        return data.DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_worker,
        )

    def val_dataloader(self):
        return data.DataLoader(
            self.val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_worker,
        )

    def test_dataloader(self):
        subset = torch.utils.data.Subset(self.test, np.random.randint(0, 100, 4))
        return data.DataLoader(
            subset,  # self.test,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_worker,
        )

        # return data.DataLoader(self.test, batch_size=self.batch_size, shuffle=False)

    def predict_dataloader(self):
        # sample first n paires
        subset = torch.utils.data.Subset(self.test, np.arange(2))
        return data.DataLoader(
            subset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_worker,
        )
