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
        # full dataset used for both training and validation
        full_dataset = DebrisDataset(dataset_dir=self.dataset_dir,
                                     debris_free_dataset_dir=self.debris_free_dataset_dir,
                                     resize_to=self.resize_to, negative_prob=self.negative_prob)

        # split dataset: 80% for training, 20% for validation
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        self.train_dataset, self.validate_dataset = random_split(full_dataset, [train_size, val_size])
        # If you need to setup a test dataset, you can do it here, but it's ignored as per your request
        self.test_dataset = None  # You can set this up as needed


    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True,
                          num_workers=self.hparams.num_workers, pin_memory=self.hparams.pin_memory)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.validate_dataset, batch_size=self.hparams.batch_size, shuffle=False,
                          num_workers=self.hparams.num_workers, pin_memory=self.hparams.pin_memory)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size, shuffle=False,
                          num_workers=self.hparams.num_workers, pin_memory=self.hparams.pin_memory)


