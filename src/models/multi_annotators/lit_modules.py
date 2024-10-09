from typing import Any, Dict, Tuple, List
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
        self.labeler_tags = None
        self.num_labelers = None
        self.gt_labeler_tag = None

    def setup(self, stage: str) -> None:
        self.labeler_tags = self.trainer.datamodule.labeler_tags
        self.gt_labeler_tag = self.trainer.datamodule.gt_labeler_tag
        self.num_labelers = len(self.labeler_tags)
        # self.net = self.net(noisy_labels_no=self.num_labelers)
        return

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        # if self.hparams.scheduler is not None:
        #     raise NotImplementedError("Scheduler is not implemented yet.")
        return {"optimizer": optimizer}

    def training_step(self, batch: List[Any], batch_idx: int) -> STEP_OUTPUT:
        images, labels_all, gt_label, image_name = self.unpack_batch(batch) # unpack the batch
        # model has two outputs: first one is the probability map for true ground truth;
        # second one is a list cms for each annotator
        outputs_logits, cms = self.forward(images) # Modified by Yuhao: 2nd output is cms

        # calculate loss: loss: total loss; loss_ce: main cross entropy loss; loss_trace: regularisation loss
        loss, loss_ce, loss_trace = noisy_label_loss(outputs_logits, cms, labels_all, self.hparams.alpha)
        _, train_output = torch.max(outputs_logits, dim=1)
        train_iou = segmentation_scores(gt_label.cpu().detach().numpy(), train_output.cpu().detach().numpy(),
                                        self.hparams.class_no) # w.r.t. "expert" label, not used in loss calculation
        self.log('train/loss', loss, on_epoch=True, on_step=False)
        self.log('train/loss_ce', loss_ce, on_epoch=True, on_step=False)
        self.log('train/regularization', loss_trace, on_epoch=True, on_step=False)
        self.log('train/iou', train_iou, on_epoch=True, on_step=False)
        return loss

        # b, c, h, w = outputs_logits.size()
        # outputs_logits = Softmax(dim=1)(outputs_logits)
        # _, output = torch.max(outputs_logits, dim=1)
        # step_output = {'img': images[0, :, :, :].squeeze().permute(1, 2, 0).cpu().detach().numpy(),
        #                'seg': output[0, :, :].reshape(h, w).cpu().detach().numpy(),
        #                'label': gt_label[0, :, :].reshape(h, w).cpu().detach().numpy()}
        # step_output_noisy, _ = self.compute_noisy_pred(outputs_logits, cms, labels_all)
        # step_output.update(step_output_noisy)
        # step_output['loss'] = loss
        # return step_output

    def validation_step(self, batch: List[Any], batch_idx: int) -> Dict:
        v_images, v_labels, gt_label, image_name = self.unpack_batch(batch) # unpack the batch
        v_outputs_logits, cms = self.forward(v_images)
        b, c, h, w = v_outputs_logits.size()
        v_outputs_logits = Softmax(dim=1)(v_outputs_logits)
        _, v_output = torch.max(v_outputs_logits, dim=1)

        step_output = {'img': v_images.squeeze().permute(1, 2, 0).cpu().detach().numpy(),
                       'seg': v_output.reshape(h, w).cpu().detach().numpy(),
                       'label': gt_label.reshape(h, w).cpu().detach().numpy()}
        step_output_noisy, v_outputs_noisy = self.compute_noisy_pred(v_outputs_logits, cms, v_labels)
        step_output.update(step_output_noisy)
        v_dice = segmentation_scores(gt_label.cpu().detach(), v_output.cpu().detach().numpy(), self.hparams.class_no)
        epoch_noisy_labels = [label.cpu().detach().numpy() for label in v_labels]
        v_ged = generalized_energy_distance(epoch_noisy_labels, v_outputs_noisy, self.hparams.class_no)
        self.log('val/dice', v_dice, on_epoch=True)
        self.log('val/ged', v_ged, on_epoch=True)
        return step_output

    def test_step(self, batch: List[Any], batch_idx: int) -> Dict:
        v_images, labels_all, gt_label, image_name = self.unpack_batch(batch)  # unpack the batch
        v_outputs_logits, cms = self.forward(v_images) # Modified by Yuhao: 2nd output is cms
        b, c, h, w = v_outputs_logits.size()
        v_outputs_logits = Softmax(dim=1)(v_outputs_logits)
        _, v_outputs = torch.max(v_outputs_logits, dim=1)

        step_output = {'img': v_images.squeeze().permute(1, 2, 0).cpu().detach().numpy(),
                       'seg': v_outputs.reshape(h, w).cpu().detach().numpy(),
                       'label': gt_label.reshape(h, w).cpu().detach().numpy()}
        step_output_noisy, v_outputs_noisy = self.compute_noisy_pred(v_outputs_logits, cms, labels_all)
        step_output.update(step_output_noisy)
        return step_output

    def forward(self, x: torch.Tensor) -> Any:
        return self.net(x)

    def unpack_batch(self, batch: List[Any]) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor, Any]:
        return self.trainer.datamodule.unpack_batch(batch)

    @staticmethod
    def compute_noisy_pred(output_logits: torch.Tensor, cms: List[torch.Tensor], labels: List[torch.Tensor]) \
            -> Tuple[Dict[str, torch.Tensor], List[torch.Tensor]]:
        return_dict = {}
        all_noisy_outputs = []
        b, c, h, w = output_logits.size()
        output_logits = output_logits.reshape(b, c, h * w)
        output_logits = output_logits.permute(0, 2, 1).contiguous()
        output_logits = output_logits.view(b * h * w, c).view(b * h * w, c, 1)
        for j, cm in enumerate(cms):
            cm = cm.view(b, c ** 2, h * w).permute(0, 2, 1).contiguous().view(b * h * w, c * c).view(b * h * w, c, c)
            cm = cm / cm.sum(1, keepdim=True)
            v_noisy_output = torch.bmm(cm, output_logits).view(b * h * w, c)
            v_noisy_output = v_noisy_output.view(b, h * w, c).permute(0, 2, 1).contiguous().view(b, c, h, w)
            _, v_noisy_output = torch.max(v_noisy_output, dim=1)
            all_noisy_outputs.append(v_noisy_output.cpu().detach().numpy())
            return_dict[f'noisy_{j}_seg'] = v_noisy_output.reshape(h, w).cpu().detach().numpy()
            return_dict[f'{j}_label'] = labels[j].reshape(h, w).cpu().detach().numpy()
        return return_dict, all_noisy_outputs

