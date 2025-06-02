from typing import Any, Dict, Tuple, List
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import torch
from torch.nn import Softmax
from lightning import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler, STEP_OUTPUT
from matplotlib import pyplot as plt
from sklearn.metrics import jaccard_score, precision_score, recall_score, f1_score
from src.utils import RankedLogger

__author__ = 'Yuhao Liu'

logger = RankedLogger(__name__, rank_zero_only=False)


class AnnotationReaderLitModule(LightningModule):

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model
        self.save_hyperparameters(logger=False, ignore=("model"))
        return

    def setup(self, stage: str) -> None:
        self.model.setup()
        self.num_classes = self.trainer.datamodule.full_dataset.num_classes
        self.one_hot_labels = [i for i in range(self.num_classes)]
        # set up metric file


    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        pass
        img_id = batch[0][4][0]
        query_img = batch[0][0]
        ground_truth = batch[1][1]
        # convert one-hot to binary
        ground_truth = ground_truth.argmax(dim=1)
        is_debris_positive = torch.any(ground_truth).item()
        if not is_debris_positive:
            return
        individual_annotation = self.model(query_img, img_id)
        self.compute_metric(individual_annotation, ground_truth, img_id)


    def compute_metric(self, pred_class: torch.Tensor, gt_class: torch.Tensor, img_id: str) -> None:
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
        row_dict = {
                   'iou': iou,
                   'f1_score': f1,
                   'f1_debris_only': f1_debris_only,
                   'dice_debris': dice_debris,
                   'dice': dice,
                   'precision_no_debris': precision_per_class[0],
                   'precision_debris_low': precision_per_class[1],
                   'precision_debris_high': precision_per_class[2],
                   'recall_no_debris': recall_per_class[0],
                   'recall_debris_low': recall_per_class[1],
                   'recall_debris_high': recall_per_class[2],
                   'annotator': self.model.annotator,
                   'img_id': img_id,
                   }
        self.logger.experiment.log_metrics(row_dict, step=None)
        return


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

