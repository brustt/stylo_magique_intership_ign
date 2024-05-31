import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils import data
from magic_pen.data.loader import BiTemporalDataset
from magic_pen.data.process import DefaultTransform


class CDDataModule(pl.LightningDataModule):
    def __init__(self, name, batch_size: int, num_workers: int = 0) -> None:
        super().__init__()
        self.name = name
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        # download, IO, etc. Useful with shared filesystems
        # only called on 1 GPU/TPU in distributed
        pass

    def setup(self, stage):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        self.train = BiTemporalDataset(
            name=self.name, dtype="train", transform=DefaultTransform()
        )
        self.val = BiTemporalDataset(
            name=self.name, dtype="val", transform=DefaultTransform()
        )
        self.test = BiTemporalDataset(
            name=self.name, dtype="test", transform=DefaultTransform()
        )

    def train_dataloader(self):
        return data.DataLoader(self.train, batch_size=self.batch_size, shuffle=False)

    def val_dataloader(self):
        return data.DataLoader(self.val, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return data.DataLoader(self.test, batch_size=self.batch_size, shuffle=False)

    def predict_dataloader(self):
        # sample first 2 paires
        subset = torch.utils.data.Subset(self.test, np.arange(2))
        return data.DataLoader(subset, batch_size=self.batch_size, shuffle=False)
