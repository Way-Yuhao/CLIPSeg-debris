import os
from typing import Any, Dict, Tuple, List
import cv2
import numpy as np
import torch
from torch.nn import Softmax
from lightning import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler, STEP_OUTPUT
from matplotlib import pyplot as plt
from src.models.multi_annotators.models import UNet_CMs
from src.utils.multi_annotators_utils.loss import noisy_label_loss
from src.utils.multi_annotators_utils.utilis import segmentation_scores, generalized_energy_distance

__author__ = 'Yuhao Liu'


class PrecomputedLitModule(LightningModule):

    def __init__(self, precomputed_output_dir: str, output_type: str,
                 *args, **kwargs):
        super().__init__()
        # self.save_hyperparameters(logger=False)
        self.precomputed_dir = precomputed_output_dir
        self.output_type = output_type
        self.precomputed_files = None

        self.found_files = []
        self.missing_files = []

    def setup(self, stage: str) -> None:
        files = os.listdir(self.precomputed_dir)
        files = [f for f in files if f.endswith(self.output_type)]
        self.precomputed_files = files
        print(f"Found {len(files)} precomputed files from {self.precomputed_dir}")

    def training_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        raise NotImplementedError()

    def test_step(self, batch: List[Any], batch_idx: int) -> Dict:
        data_x, data_y = batch  # unpack
        img_id = data_x[4]
        self.find_output(img_id[0])
        # pred_class = self.find_output(img_id[0])
        # gt_class = torch.argmax(data_y[1], dim=1) # shape [1, h, w]
        pass

    def find_output(self, img_id: str):
        f = [f for f in self.precomputed_files if img_id in f]
        if len(f) == 0:
            print(f"Image {img_id} not found in precomputed files!")
            self.missing_files.append(img_id)
        elif len(f) > 1:
            raise ValueError(f"Multiple images found for {img_id} in precomputed files!\n"
                             f"Files: {f}")
        else:
            output = cv2.imread(os.path.join(self.precomputed_dir, f[0]))
            self.found_files.append(img_id)
            return f[0]

    def teardown(self, stage: str) -> None:
        print(f"Found {len(self.found_files)} files and {len(self.missing_files)} missing files")
        print(f"Missing files: {self.missing_files}")
        print(f"Found files: {self.found_files}")