from typing import Any, Dict, Tuple
import numpy as np
import torch
from lightning import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler, STEP_OUTPUT
from matplotlib import pyplot as plt

from src.models.multi_annotators.models import UNet_CMs

__author__ = 'Yuhao Liu'

class UNetCMsLitModule(LightningModule):

    def __init__(self, net: torch.nn.Module, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.net = net

        # TODO: define loss functions


    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        # if self.hparams.scheduler is not None:
        #     raise NotImplementedError("Scheduler is not implemented yet.")
        return {"optimizer": optimizer}


    def on_fit_start(self) -> None:
        # demonstrate the training samples:
        Image_index_to_demonstrate = 6
        (images, labels_over, labels_under,
         labels_wrong, labels_good, imagename) = self.trainer.datamodule.validate_dataset[Image_index_to_demonstrate]
        images = np.mean(images, axis=0)

        # plot the labels:
        fig = plt.figure(figsize=(9, 13))
        columns = 5
        rows = 1
        ax = []
        labels = []
        labels_names = []
        labels.append(images)
        labels.append(labels_over)
        labels.append(labels_under)
        labels.append(labels_wrong)
        labels.append(labels_good)
        labels_names.append('Input')
        labels_names.append('Over label')
        labels_names.append('Under label')
        labels_names.append('Wrong label')
        labels_names.append('Good label')

        for i in range(columns * rows):
            if i != 0:
                label_ = labels[i][0, :, :]
            else:
                label_ = labels[i]
            ax.append(fig.add_subplot(rows, columns, i + 1))
            ax[-1].set_title(labels_names[i])
            plt.imshow(label_, cmap='gray')
        plt.show()

    def training_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        pass


    def forward(self, x: torch.Tensor) -> Any:
        return self.net(x)