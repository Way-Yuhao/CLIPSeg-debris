from typing import Any, Dict, Tuple, List
from abc import ABC, abstractmethod
import numpy as np
import torch
from torch.nn import Softmax
from lightning import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler, STEP_OUTPUT
from matplotlib import pyplot as plt
from sklearn.metrics import jaccard_score, precision_score, recall_score, f1_score
from src.utils import RankedLogger

__author__ = 'Yuhao Liu'

logger = RankedLogger(__name__, rank_zero_only=False)


class GenericDebrisSegmentationModule(LightningModule, ABC):

    def __init__(self, model: torch.nn.Module, optimizer:  torch.optim.Optimizer,
                 criterion: torch.nn.Module, compile: bool = False) -> None:
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.save_hyperparameters(logger=False, ignore=("model", "criterion"))
        return

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)
            logger.info("Model compiled.")
        elif self.hparams.compile and stage == "test":
            logger.warning("torch_compile is only available during the fit stage.")

        self.dataset_mask = self.trainer.datamodule.hparams.mask
        if hasattr(self.trainer.datamodule, 'train_dataset') and self.trainer.datamodule.train_dataset is not None:
            self.dataset_class = self.trainer.datamodule.train_dataset.__class__.__name__
        else:
            self.dataset_class = self.trainer.datamodule.predict_dataset.__class__.__name__
        self.num_classes = self.trainer.datamodule.full_dataset.num_classes
        self.all_text_prompts = self.trainer.datamodule.full_dataset.text_prompts
        self.all_densities = self.trainer.datamodule.full_dataset.densities
        self.logger_class_labels = {i: prompt for i, prompt in enumerate(self.all_text_prompts)}
        self.one_hot_labels = [i for i in range(self.num_classes)]
        return

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        else:
            return {"optimizer": optimizer}

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        pass

    @staticmethod
    def unpack(batch: Any):
        data_x, data_y = batch  # unpack
        query_img = data_x[0]
        label = data_y[1]
        return query_img, label

    def compute_metric_single_class_2(self, pred_class: torch.Tensor, gt_class: torch.Tensor, stage: str) -> torch.Tensor:
        pred_class = pred_class.flatten().cpu().numpy()
        gt_class = gt_class.flatten().cpu().numpy()

        iou_per_class = jaccard_score(gt_class, pred_class, zero_division=1, average=None, labels=self.one_hot_labels)
        iou = jaccard_score(gt_class, pred_class, zero_division=1, average='macro', labels=self.one_hot_labels)
        precision_per_class = precision_score(gt_class, pred_class, zero_division=1, average=None, labels=self.one_hot_labels)
        recall_per_class = recall_score(gt_class, pred_class, zero_division=1, average=None, labels=self.one_hot_labels)
        f1_score_per_class = f1_score(gt_class, pred_class, zero_division=1, average=None, labels=self.one_hot_labels)
        f1 = f1_score(gt_class, pred_class, zero_division=1, average='macro', labels=self.one_hot_labels)
        f1_debris_only = f1_score(gt_class, pred_class, zero_division=1, average='macro', labels=self.one_hot_labels[1:])
        dice_debris = segmentation_scores(gt_class, pred_class, self.num_classes, filter_background=True)
        dice = segmentation_scores(gt_class, pred_class, self.num_classes, filter_background=False)

        self.log(f"{stage}/iou_no_debris", iou_per_class[0], on_epoch=True, prog_bar=False, batch_size=gt_class.shape[0])
        self.log(f"{stage}/iou_debris_low", iou_per_class[1], on_epoch=True, prog_bar=False, batch_size=gt_class.shape[0])
        self.log(f"{stage}/iou_debris_high", iou_per_class[2], on_epoch=True, prog_bar=False, batch_size=gt_class.shape[0])
        self.log(f"{stage}/iou_macro", iou, on_epoch=True, prog_bar=False, batch_size=gt_class.shape[0])
        self.log(f"{stage}/precision_no_debris", precision_per_class[0], on_epoch=True, prog_bar=False, batch_size=gt_class.shape[0])
        self.log(f"{stage}/precision_debris_low", precision_per_class[1], on_epoch=True, prog_bar=False, batch_size=gt_class.shape[0])
        self.log(f"{stage}/precision_debris_high", precision_per_class[2], on_epoch=True, prog_bar=False, batch_size=gt_class.shape[0])
        self.log(f"{stage}/recall_no_debris", recall_per_class[0], on_epoch=True, prog_bar=False, batch_size=gt_class.shape[0])
        self.log(f"{stage}/recall_debris_low", recall_per_class[1], on_epoch=True, prog_bar=False, batch_size=gt_class.shape[0])
        self.log(f"{stage}/recall_debris_high", recall_per_class[2], on_epoch=True, prog_bar=False, batch_size=gt_class.shape[0])
        self.log(f"{stage}/f1_score_no_debris", f1_score_per_class[0], on_epoch=True, prog_bar=False, batch_size=gt_class.shape[0])
        self.log(f"{stage}/f1_score_debris_low", f1_score_per_class[1], on_epoch=True, prog_bar=False, batch_size=gt_class.shape[0])
        self.log(f"{stage}/f1_score_debris_high", f1_score_per_class[2], on_epoch=True, prog_bar=False, batch_size=gt_class.shape[0])
        self.log(f"{stage}/f1_macro", f1, on_epoch=True, prog_bar=True, batch_size=gt_class.shape[0])
        self.log(f"{stage}/f1_macro_debris_only", f1_debris_only, on_epoch=True, prog_bar=False, batch_size=gt_class.shape[0])
        self.log(f"{stage}/dice_debris", dice_debris, on_epoch=True, prog_bar=True, batch_size=gt_class.shape[0])
        self.log(f"{stage}/dice", dice, on_epoch=True, prog_bar=True, batch_size=gt_class.shape[0])

def segmentation_scores(label_trues, label_preds, n_class, filter_background=True):
    """
    Computes the Dice score for segmentation tasks.

    :param label_trues: Ground truth labels (tensor-like, [1, h, w])
    :param label_preds: Predicted labels (tensor-like, [1, h, w])
    :param n_class: Total number of classes (including background)
    :param filter_background: Boolean flag to exclude background class (0)
    :return: Mean Dice score (optionally excluding background)
    """
    assert len(label_trues) == len(label_preds)

    # Convert to numpy arrays and flatten
    label_trues = np.asarray(label_trues, dtype='int8').copy().flatten()
    label_preds = np.asarray(label_preds, dtype='int8').copy().flatten()

    # Apply mask if filtering background class
    if filter_background:
        mask = label_trues > 0  # Keep only non-background pixels
        label_trues = label_trues[mask]
        label_preds = label_preds[mask]

    # Compute Dice score for each class
    dice_scores = []
    for cls in range(1 if filter_background else 0, n_class):
        pred_mask = (label_preds == cls)
        true_mask = (label_trues == cls)

        intersection = np.sum(pred_mask & true_mask)
        union = np.sum(pred_mask) + np.sum(true_mask)

        dice = (2 * intersection + 1e-6) / (union + 1e-6)  # Avoid division by zero
        dice_scores.append(dice)

    # Return the mean Dice score across all relevant classes
    return np.mean(dice_scores)

