from typing import Any, Dict, Optional, Tuple, List
import torch
from lightning import LightningDataModule
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms as T
from src.utils.multi_annotators_utils.utilis import CustomDataset_punet

__author__ = 'Yuhao Liu'


class MNISTExampleDataModule(LightningDataModule):

    def __init__(self, labeler_tags: List[str], gt_labeler_tag: str, transforms: Optional[T.Compose] = None,
                 *args, **kwargs):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=("labeler_tags", "gt_labeler_tag", "transforms"))

        self.labeler_tags = labeler_tags
        self.gt_labeler_tag = gt_labeler_tag
        self.num_labelers = len(labeler_tags)
        self.transforms = transforms

        # to be defined elsewhere
        self.train_dataset = None
        self.validate_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = CustomDataset_punet(dataset_location=self.hparams.train_path,
                                                dataset_tag=self.hparams.dataset_tag, noisylabel=self.hparams.label_mode,
                                                augmentation=True, transforms=self.transforms)
        self.validate_dataset = CustomDataset_punet(dataset_location=self.hparams.validate_path,
                                                    dataset_tag=self.hparams.dataset_tag,
                                                    noisylabel=self.hparams.label_mode, augmentation=False,
                                                    transforms=self.transforms)
        self.test_dataset = CustomDataset_punet(dataset_location=self.hparams.test_path,
                                                dataset_tag=self.hparams.dataset_tag, noisylabel=self.hparams.label_mode,
                                                augmentation=False, transforms=self.transforms)

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

    def unpack_batch(self, batch: List[torch.Tensor]) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor, Any]:
        input_images = batch[0]
        labels = []
        for i in range(self.num_labelers):
            labels.append(batch[i + 1])
        image_name = batch[-1]
        gt_label_idx = self.labeler_tags.index(self.gt_labeler_tag)
        gt_label = labels[gt_label_idx]
        return input_images, labels, gt_label, image_name