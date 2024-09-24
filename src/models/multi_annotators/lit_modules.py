from typing import Any, Dict, Tuple
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


class UNetCMsLitModule(LightningModule):

    def __init__(self, net: torch.nn.Module, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.net = net

        # TODO: define loss functions

        # to be defined elsewhere
        self.test_dice = None
        self.test_dice_all = None

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        # if self.hparams.scheduler is not None:
        #     raise NotImplementedError("Scheduler is not implemented yet.")
        return {"optimizer": optimizer}

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
        self.log('train/loss', loss, on_epoch=True, on_step=False)
        self.log('train/loss_ce', loss_ce, on_epoch=True, on_step=False)
        self.log('train/regularization', loss_trace, on_epoch=True, on_step=False)
        self.log('train/iou', train_iou, on_epoch=True, on_step=False)

        return loss

    # def on_validation_epoch_start(self) -> None:
    #     self.test_dice = 0
    #     self.test_dice_all = []

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        # unpack the batch
        v_images, v_labels_over, v_labels_under, v_labels_wrong, v_labels_good, v_imagename = batch
        v_outputs_logits, cms = self.forward(v_images)
        b, c, h, w = v_outputs_logits.size()
        v_outputs_logits = Softmax(dim=1)(v_outputs_logits)

        _, v_output = torch.max(v_outputs_logits, dim=1)
        v_outputs_noisy = []

        v_outputs_logits = v_outputs_logits.view(b, c, h * w)
        v_outputs_logits = v_outputs_logits.permute(0, 2, 1).contiguous().view(b * h * w, c)
        v_outputs_logits = v_outputs_logits.view(b * h * w, c, 1)

        for cm in cms:
            cm = cm.reshape(b, c ** 2, h * w).permute(0, 2, 1).contiguous().view(b * h * w, c * c).view(b * h * w, c, c)
            cm = cm / cm.sum(1, keepdim=True)
            v_noisy_output = torch.bmm(cm, v_outputs_logits).view(b * h * w, c)
            v_noisy_output = v_noisy_output.view(b, h * w, c).permute(0, 2, 1).contiguous().view(b, c, h, w)
            _, v_noisy_output = torch.max(v_noisy_output, dim=1)
            v_outputs_noisy.append(v_noisy_output.cpu().detach().numpy())

        v_dice = segmentation_scores(v_labels_good.cpu().detach(), v_output.cpu().detach().numpy(),
                                     self.hparams.class_no)

        epoch_noisy_labels = [v_labels_over.cpu().detach().numpy(), v_labels_under.cpu().detach().numpy(),
                              v_labels_wrong.cpu().detach().numpy(), v_labels_good.cpu().detach().numpy()]
        v_ged = generalized_energy_distance(epoch_noisy_labels, v_outputs_noisy, self.hparams.class_no)

        self.log('val/dice', v_dice, on_epoch=True)
        self.log('val/ged', v_ged, on_epoch=True)
        return

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> dict:
        # unpack the batch
        v_images, labels_over, labels_under, labels_wrong, labels_good, imagename = batch
        v_outputs_logits_original, v_outputs_logits_noisy = self.forward(v_images)
        b, c, h, w = v_outputs_logits_original.size()
        # plot the final segmentation map
        v_outputs_logits_original = Softmax(dim=1)(v_outputs_logits_original)
        _, v_outputs_logits = torch.max(v_outputs_logits_original, dim=1)

        # save_name = save_path_visual_result + '/test_' + str(i) + '_seg.png'
        # save_name_label = save_path_visual_result + '/test_' + str(i) + '_label.png'
        # save_name_slice = save_path_visual_result + '/test_' + str(i) + '_img.png'

        # plt.imsave(save_name_slice, v_images[:, 1, :, :].reshape(h, w).cpu().detach().numpy(), cmap='gray')
        # plt.imsave(save_name, v_outputs_logits.reshape(h, w).cpu().detach().numpy(), cmap='gray')
        # plt.imsave(save_name_label, labels_good.reshape(h, w).cpu().detach().numpy(), cmap='gray')

        step_output = {'img': v_images[:, 1, :, :].reshape(h, w).cpu().detach().numpy(),
                       'seg': v_outputs_logits.reshape(h, w).cpu().detach().numpy(),
                       'label': labels_good.reshape(h, w).cpu().detach().numpy()}
        # plot the noisy segmentation maps:
        v_outputs_logits_original = v_outputs_logits_original.reshape(b, c, h * w)
        v_outputs_logits_original = v_outputs_logits_original.permute(0, 2, 1).contiguous()
        v_outputs_logits_original = v_outputs_logits_original.view(b * h * w, c).view(b * h * w, c, 1)

        for j, cm in enumerate(v_outputs_logits_noisy):
            cm = cm.view(b, c ** 2, h * w).permute(0, 2, 1).contiguous().view(b * h * w, c * c).view(b * h * w, c, c)
            cm = cm / cm.sum(1, keepdim=True)
            v_noisy_output_original = torch.bmm(cm, v_outputs_logits_original).view(b * h * w, c)
            v_noisy_output_original = v_noisy_output_original.view(b, h * w, c).permute(0, 2, 1).contiguous().view(b, c,
                                                                                                                   h, w)
            _, v_noisy_output = torch.max(v_noisy_output_original, dim=1)
            step_output[f'noisy_{j}_seg'] = v_noisy_output.reshape(h, w).cpu().detach().numpy()
            # save_name = save_path_visual_result + '/test_' + str(i) + '_noisy_' + str(j) + '_seg.png'
            # plt.imsave(save_name, v_noisy_output.reshape(h, w).cpu().detach().numpy(), cmap='gray')
        return step_output

    def forward(self, x: torch.Tensor) -> Any:
        return self.net(x)
