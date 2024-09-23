import torch
from lightning import LightningModule
from src.models.multi_annotators.models import UNet_CMs

__author__ = 'Yuhao Liu'

class UNetCMsLitModule(LightningModule):

    def __init__(self, net: torch.nn.Module, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.net = net

        # TODO: define loss functions




