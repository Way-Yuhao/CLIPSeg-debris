from typing import Any, Dict, Tuple, List
import numpy as np
import torch
from torch.nn import Softmax
from lightning import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler, STEP_OUTPUT
from matplotlib import pyplot as plt
from src.models.clipseg.clipseg import CLIPDensePredT

__author__ = 'Yuhao Liu'


class CLIPSegLitModule(LightningModule):

    def __init__(self, net: torch.nn.Module, optimizer: torch.optim.Optimizer, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.net = net

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        return {"optimizer": optimizer}


    def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        data_x, data_y = batch # unpack
        return None
