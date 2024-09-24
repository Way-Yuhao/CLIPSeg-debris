from typing import Any, Dict, Optional, Tuple, List
import torch
from lightning import LightningDataModule
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from src.utils.multi_annotators_utils.utilis import CustomDataset_punet

__author__ = 'Yuhao Liu'


class MNISTExampleDataModule(LightningDataModule):

    def __init__(self, labeler_tags: List[str], *args, **kwargs):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=("labeler_tags",))

        self.labeler_tags = labeler_tags
        self.num_labelers = len(labeler_tags)

        # to be defined elsewhere
        self.train_dataset = None
        self.validate_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = CustomDataset_punet(dataset_location=self.hparams.train_path,
                                            dataset_tag=self.hparams.dataset_tag, noisylabel=self.hparams.label_mode,
                                            augmentation=True)
        self.validate_dataset = CustomDataset_punet(dataset_location=self.hparams.validate_path,
                                               dataset_tag=self.hparams.dataset_tag,
                                               noisylabel=self.hparams.label_mode, augmentation=False)
        self.test_dataset = CustomDataset_punet(dataset_location=self.hparams.test_path, dataset_tag=self.hparams.dataset_tag,
                                           noisylabel=self.hparams.label_mode,
                                           augmentation=False)

    def train_dataloader(self) -> DataLoader[Any]:
        train_loader = DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True,
                                  num_workers=self.hparams.num_workers, drop_last=True)
        return train_loader

    def val_dataloader(self) -> DataLoader[Any]:
        val_loader = DataLoader(self.validate_dataset, batch_size=1, shuffle=False, drop_last=False)
        return val_loader

    def test_dataloader(self) -> DataLoader[Any]:
        test_loader = DataLoader(self.test_dataset, batch_size=1, shuffle=False, drop_last=False)
        return test_loader