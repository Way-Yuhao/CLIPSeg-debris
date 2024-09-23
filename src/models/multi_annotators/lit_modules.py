from typing import Any, Dict, Tuple
import numpy as np
import torch
from lightning import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler, STEP_OUTPUT
from matplotlib import pyplot as plt
from src.models.multi_annotators.models import UNet_CMs
from src.utils.multi_annotators_utils.loss import noisy_label_loss
from src.utils.multi_annotators_utils.utilis import segmentation_scores

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
        # Image_index_to_demonstrate = 6
        # (images, labels_over, labels_under,
        #  labels_wrong, labels_good, imagename) = self.trainer.datamodule.validate_dataset[Image_index_to_demonstrate]
        # images = np.mean(images, axis=0)
        #
        # # plot the labels:
        # fig = plt.figure(figsize=(9, 13))
        # columns = 5
        # rows = 1
        # ax = []
        # labels = []
        # labels_names = []
        # labels.append(images)
        # labels.append(labels_over)
        # labels.append(labels_under)
        # labels.append(labels_wrong)
        # labels.append(labels_good)
        # labels_names.append('Input')
        # labels_names.append('Over label')
        # labels_names.append('Under label')
        # labels_names.append('Wrong label')
        # labels_names.append('Good label')
        #
        # for i in range(columns * rows):
        #     if i != 0:
        #         label_ = labels[i][0, :, :]
        #     else:
        #         label_ = labels[i]
        #     ax.append(fig.add_subplot(rows, columns, i + 1))
        #     ax[-1].set_title(labels_names[i])
        #     plt.imshow(label_, cmap='gray')
        # plt.show()
        pass

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> STEP_OUTPUT:
        # unpack the batch
        images, labels_over, labels_under, labels_wrong, labels_good, imagename = batch

        labels_all = []
        labels_all.append(labels_over)
        labels_all.append(labels_under)
        labels_all.append(labels_wrong)
        labels_all.append(labels_good)

        # model has two outputs:
        # first one is the probability map for true ground truth
        # second one is a list collection of probability maps for different noisy ground truths
        outputs_logits, outputs_logits_noisy = self.forward(images)

        # calculate loss:
        # loss: total loss
        # loss_ce: main cross entropy loss
        # loss_trace: regularisation loss
        loss, loss_ce, loss_trace = noisy_label_loss(outputs_logits, outputs_logits_noisy, labels_all,
                                                     self.hparams.alpha)

        _, train_output = torch.max(outputs_logits, dim=1)
        train_iou = segmentation_scores(labels_good.cpu().detach().numpy(), train_output.cpu().detach().numpy(),
                                        self.hparams.class_no)
        self.log('train/loss', loss, on_epoch=True)
        self.log('train/loss_ce', loss_ce, on_epoch=True)
        self.log('train/loss_trace', loss_trace, on_epoch=True)
        self.log('train/iou', train_iou, on_epoch=True)


    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:

    def forward(self, x: torch.Tensor) -> Any:
        return self.net(x)