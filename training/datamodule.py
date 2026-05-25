from typing import Optional

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig


class NetCDFDataModule(pl.LightningDataModule):
    def __init__(self, dataset_cfg: DictConfig, batch_size: int = 1, num_workers: int = 0):
        super().__init__()
        self.dataset_cfg = dataset_cfg
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset = None
        self.in_channels = None

    def setup(self, stage: Optional[str] = None):
        # instantiate dataset using Hydra/OmegaConf
        self.train_dataset = instantiate(self.dataset_cfg)
        # compute in_channels from first sample
        sample = self.train_dataset[0]
        img = sample.get("image")
        if img is None:
            raise RuntimeError("Dataset must return an 'image' key in sample")
        # assume CHW
        self.in_channels = int(img.shape[0])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers)
