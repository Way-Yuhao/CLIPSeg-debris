from typing import Any, Dict, Optional, Tuple, List
import torch
from lightning import LightningDataModule
# from lightning.pytorch.utilities.types import EVAL_DATALOADERS
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from src.data.components.debris import DebrisDataset


__author__ = 'Yuhao Liu'

class DebrisDataModule(LightningDataModule):

    def __init__(self, batch_size: int, num_workers: int, pin_memory: bool,
                 dataset_dir: str, debris_free_dataset_dir: str, resize_to: tuple, negative_prob: float,
                 *args, **kwargs):
        super().__init__()

        self.dataset_dir = dataset_dir
        self.debris_free_dataset_dir = debris_free_dataset_dir
        self.resize_to = resize_to
        self.negative_prob = negative_prob

        self.save_hyperparameters(logger=False,
                                  ignore=("dataset_dir", "debris_free_dataset_dir", "resize_to", "negative_prob"))
        # to be defined elsewhere
        self.train_dataset = None
        self.validate_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = DebrisDataset(dataset_dir=self.dataset_dir,
                                           debris_free_dataset_dir=self.debris_free_dataset_dir,
                                           resize_to=self.resize_to, negative_prob=self.negative_prob)
        self.validate_dataset = DebrisDataset(dataset_dir=self.dataset_dir,
                                              debris_free_dataset_dir=self.debris_free_dataset_dir,
                                              resize_to=self.resize_to, negative_prob=self.negative_prob)
        self.test_dataset = DebrisDataset(dataset_dir=self.dataset_dir,
                                          debris_free_dataset_dir=self.debris_free_dataset_dir,
                                          resize_to=self.resize_to, negative_prob=self.negative_prob)


    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True,
                          num_workers=self.hparams.num_workers, pin_memory=self.hparams.pin_memory)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.validate_dataset, batch_size=self.hparams.batch_size, shuffle=False,
                          num_workers=self.hparams.num_workers, pin_memory=self.hparams.pin_memory)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size, shuffle=False,
                          num_workers=self.hparams.num_workers, pin_memory=self.hparams.pin_memory)
