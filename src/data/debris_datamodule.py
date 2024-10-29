from typing import Any, Dict, Optional, Tuple, List
import os
import numpy as np
from lightning import LightningDataModule
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
# from lightning.pytorch.utilities.types import EVAL_DATALOADERS
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split, Subset
from natsort import natsorted
from src.data.components.debris import DebrisDataset


__author__ = 'Yuhao Liu'

class DebrisDataModule(LightningDataModule):

    def __init__(self, batch_size: int, num_workers: int, pin_memory: bool, dataset: Dataset,
                 *args, **kwargs):
        super().__init__()

        # self.dataset_dir = dataset_dir
        # self.debris_free_dataset_dir = debris_free_dataset_dir
        # self.resize_to = resize_to
        # self.negative_prob = negative_prob
        self.full_dataset = dataset

        self.save_hyperparameters(logger=False, ignore=("dataset",))
        # to be defined elsewhere
        self.train_dataset = None
        self.validate_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None) -> None:
        self.delete_dot_underscore_files(self.full_dataset.dataset_dir)
        # split dataset
        if self.hparams.slit_dataset == "random":
            self._split_dataset_random()
        elif self.hparams.slit_dataset == "hurricane":
            self._split_dataset_by_hurricane()
        elif self.hparams.slit_dataset == "hurricane_test":
            self._split_dataset_by_hurricane_test()
        else:
            raise ValueError(f"Invalid dataset split method: {self.hparams.slit_dataset}")
        self.print_dataset_stats() # print dataset stats

    def _split_dataset_random(self):
        """
        Split the dataset randomly into training and validation sets. No test set.
        """
        print('Splitting dataset randomly via 80/20 split...')
        # split dataset: 80% for training, 20% for validation
        train_size = int(0.8 * len(self.full_dataset))
        val_size = len(self.full_dataset) - train_size
        self.train_dataset, self.validate_dataset = random_split(self.full_dataset, [train_size, val_size])
        # If you need to setup a test dataset, you can do it here, but it's ignored as per your request
        self.test_dataset = None  # You can set this up as needed

    def _split_dataset_by_hurricane(self):
        """
        Split the dataset by hurricane, with no test set.
        """
        print('Splitting dataset by hurricane...')
        if self.hparams.remove_ike:
            print('Removing hurricane "ike" from the dataset...')
        if self.hparams.hurricane_id_ranges is None:
            raise ValueError("hurricane_id_ranges must be provided to split the dataset by hurricane.")
        hurricane_id_ranges = self.hparams.hurricane_id_ranges
        # Initialize lists to store the indices of the train, val, and test sets
        train_indices = []
        val_indices = []
        # Iterate over each image ID and determine which set it belongs to
        for idx, img_id in enumerate(self.full_dataset.img_ids):
            img_id_int = int(img_id)  # Convert image ID string to integer
            if hurricane_id_ranges['ian'][0] <= img_id_int <= hurricane_id_ranges['ian'][1]:
                train_indices.append(idx)  # Assign to 'ian' range (for training)
            elif hurricane_id_ranges['ida'][0] <= img_id_int <= hurricane_id_ranges['ida'][1]:
                val_indices.append(idx) # Assign to 'ida' range (for validation and test)
            elif hurricane_id_ranges['ike'][0] <= img_id_int <= hurricane_id_ranges['ike'][1]:
                if not self.hparams.remove_ike:
                    train_indices.append(idx) # Assign to 'ike' range (for training)
            else:
                raise ValueError(f"Image ID {img_id} does not fall into any of the specified hurricane ranges.")
        # Create train and validation datasets using the Subset class
        self.train_dataset = Subset(self.full_dataset, train_indices)
        self.validate_dataset = Subset(self.full_dataset, val_indices)

    def _split_dataset_by_hurricane_test(self):
        """
        Split the dataset by hurricane, with a test set.
        """
        print('Splitting dataset by hurricane with a test set...')
        if self.hparams.remove_ike:
            print('Removing hurricane "ike" from the dataset...')
        if self.hparams.hurricane_id_ranges is None:
            raise ValueError("hurricane_id_ranges must be provided to split the dataset by hurricane.")
        hurricane_id_ranges = self.hparams.hurricane_id_ranges
        # Initialize lists to store the indices of the train, val, and test sets
        train_indices, val_indices, test_indices = [], [], []
        train_val_split_ratio = self.hparams.train_val_split_ratio


        # Iterate over each image ID and determine which set it belongs to
        for idx, img_id in enumerate(self.full_dataset.img_ids):
            img_id_int = int(img_id)  # Convert image ID string to integer
            if hurricane_id_ranges['ian'][0] <= img_id_int <= hurricane_id_ranges['ian'][1]:
                # assign to train or val set
                if np.random.rand() < train_val_split_ratio:
                    train_indices.append(idx)
                else:
                    val_indices.append(idx)
            elif hurricane_id_ranges['ida'][0] <= img_id_int <= hurricane_id_ranges['ida'][1]:
                test_indices.append(idx)  # Assign to 'ida' range (for test)
            elif hurricane_id_ranges['ike'][0] <= img_id_int <= hurricane_id_ranges['ike'][1]:
                if not self.hparams.remove_ike:
                    # assign to train or val set
                    if np.random.rand() < train_val_split_ratio:
                        train_indices.append(idx)
                    else:
                        val_indices.append(idx)
            else:
                raise ValueError(f"Image ID {img_id} does not fall into any of the specified hurricane ranges.")
        # Create train, validation, and test datasets using the Subset class
        self.train_dataset = Subset(self.full_dataset, train_indices)
        self.validate_dataset = Subset(self.full_dataset, val_indices)
        self.test_dataset = Subset(self.full_dataset, test_indices)


    def print_dataset_stats(self):
        print('-------------- Dataset Statistics --------------------')
        print(f"Train dataset size: {len(self.train_dataset)}")
        print(f"Validation dataset size: {len(self.validate_dataset)}")
        print(f"Test dataset size: {len(self.test_dataset) if self.test_dataset is not None else 0}")
        print('------------------------------------------------------')

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True,
                          num_workers=self.hparams.num_workers, pin_memory=self.hparams.pin_memory,
                          persistent_workers=self.hparams.persistent_workers)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.validate_dataset, batch_size=1, shuffle=False,
                          num_workers=1, pin_memory=self.hparams.pin_memory,
                          persistent_workers=self.hparams.persistent_workers)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=1, shuffle=False,
                          num_workers=1, pin_memory=self.hparams.pin_memory,
                          persistent_workers=self.hparams.persistent_workers)

    @staticmethod
    def delete_dot_underscore_files(directory: str):
        """
        Delete all files that start with ._ in the specified directory and its subdirectories.
        Args:
        directory (str): The root directory in which to search for ._ files.
        """
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.startswith('._'): # Check if the file starts with ._
                    file_path = os.path.join(root, file)
                    try:
                        os.remove(file_path)
                        print(f"Deleted: {file_path}")
                    except Exception as e:
                        print(f"Failed to delete {file_path}: {e}")


class DebrisPredictDataModule(LightningDataModule):

    def __init__(self, batch_size: int, num_workers: int, pin_memory: bool, dataset: Dataset,
                 *args, **kwargs):
        super().__init__()

        self.full_dataset = dataset
        self.save_hyperparameters(logger=False, ignore=("dataset",))
        self.predict_dataset = None

    def setup(self, stage: Optional[str] = None) -> None:
        self.predict_dataset = self.full_dataset

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(self.predict_dataset, batch_size=1, shuffle=False,
                          num_workers=1, pin_memory=self.hparams.pin_memory)

    @staticmethod
    def delete_dot_underscore_files(directory: str):
        """
        Delete all files that start with ._ in the specified directory and its subdirectories.
        Args:
        directory (str): The root directory in which to search for ._ files.
        """
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.startswith('._'): # Check if the file starts with ._
                    file_path = os.path.join(root, file)
                    try:
                        os.remove(file_path)
                        print(f"Deleted: {file_path}")
                    except Exception as e:
                        print(f"Failed to delete {file_path}: {e}")


