from typing import Any
import numpy as np
import lightning.pytorch as pl
import torch
from torch.utils import data
from .loader import BiTemporalDataset
from .process import DefaultTransform


class CDDataModule(pl.LightningDataModule):
    def __init__(self, name, params: Any) -> None:
        super().__init__()

        if name == "second":
            raise NotImplementedError('Please implement validation set before')
        
        self.name = name
        self.batch_size = params.batch_size
        self.num_worker = 0 #params.num_worker # compute this parameter instead
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
        self.val = BiTemporalDataset(
            name=self.name,
            dtype="train",
            transform=DefaultTransform(),
            params=self.params,
        ) 

        self.test = BiTemporalDataset(
            name=self.name,
            dtype="test",
            transform=DefaultTransform(),
            params=self.params,
        )

        self.ds_dict_type = dict(
            train=self.train,
            test=self.test,
            val=self.val
        )
    
    def check_dataset_mode(self, dtype: str) -> BiTemporalDataset:
        
        if self.params.get("ds_sample", None):
            subset = torch.utils.data.Subset(
                self.ds_dict_type[dtype], 
                np.random.randint(0, len(self.ds_dict_type[dtype]), self.params.get("ds_sample"))
                )
            return subset
        return self.ds_dict_type[dtype]

    def train_dataloader(self):
        ds = self.check_dataset_mode("train")
        return data.DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_worker,
        )

    def val_dataloader(self):
        ds = self.check_dataset_mode("val")
        return data.DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_worker,
        )

    def test_dataloader(self):
        ds = self.check_dataset_mode("test")
        return data.DataLoader(
            ds,  
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_worker,
            # sampler=data.SequentialSampler(subset),
        )

    # def predict_dataloader(self):
    #     # sample first n paires
    #     subset = torch.utils.data.Subset(self.test, np.arange(2))
    #     return data.DataLoader(
    #         subset,
    #         batch_size=self.batch_size,
    #         shuffle=False,
    #         num_workers=self.num_worker,
    #     )
