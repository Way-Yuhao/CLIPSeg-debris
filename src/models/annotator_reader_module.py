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
from src.models.components.read_from_annotator import AnnotationReader
from src.utils import RankedLogger

__author__ = 'Yuhao Liu'

logger = RankedLogger(__name__, rank_zero_only=False)


class AnnotationReaderLitModule(LightningModule):
    """
    Reads annotation from a specified annotator
    """

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
        img_id = batch[0][4][0]
        query_img = batch[0][0]
        ground_truth = batch[1][1]
        # convert one-hot to binary
        ground_truth = ground_truth.argmax(dim=1)
        is_debris_positive = torch.any(ground_truth).item()
        if not is_debris_positive:
            return
        individual_annotation = self.model(query_img, img_id)
        if torch.any(individual_annotation):
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

class MultiAnnotationReaderLitModule(LightningModule):
    """
    Reads annotation from multiple annotators.
    """

    def __init__(self, annotator_ids: List[str], annotation_parent_dir: str):
        super().__init__()
        self.save_hyperparameters()
        self.models = {}
        return

    def setup(self, stage: str) -> None:
        self.num_classes = self.trainer.datamodule.full_dataset.num_classes
        self.one_hot_labels = [i for i in range(self.num_classes)]
        for annotator in self.hparams.annotator_ids:
            model = AnnotationReader(annotation_parent_dir=self.hparams.annotation_parent_dir, annotator=annotator)
            model.setup()
            self.models[annotator] = model
        logger.info("Total annotators found: %d", len(self.models))
        print("Total annotators found: %d" % len(self.models))
        return

    def forward(self, query_img, img_id):
        return_code = 0
        cur_annotations = {}
        for annotator, model in self.models.items():
            individual_annotation = model(query_img, img_id)
            if torch.any(individual_annotation):
               cur_annotations[annotator] = individual_annotation
        if len(cur_annotations) != 3:
            print(f'Only found {len(cur_annotations)} annotators for image {img_id}: {list(cur_annotations.keys())}')
            return_code = -1
        ordered_keys = [ann for ann in self.models.keys() if ann in cur_annotations]
        stack_t = torch.stack([cur_annotations[ann].squeeze(0) for ann in ordered_keys], dim=0)
        return cur_annotations, stack_t, return_code

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        img_id = batch[0][4][0]
        query_img = batch[0][0]
        ground_truth = batch[1][1]
        # convert one-hot to binary
        ground_truth = ground_truth.argmax(dim=1)
        is_debris_positive = torch.any(ground_truth).item()
        if not is_debris_positive:
            return
        annotation_dict, annotation_t, return_code = self.forward(query_img, img_id)
        if return_code == 0:
            row_dict = self.probe_vote_stats(annotation_t, img_id)
            class_counts = self.get_class_count(annotation_dict, ground_truth)
            row_dict.update(class_counts)
            self.logger.experiment.log_metrics(row_dict, step=0)

    @staticmethod
    def probe_vote_stats(annotation_t: torch.Tensor, img_id: str) -> dict:
        """
        Given annotation_t of shape [3, h, w], returns:
        (unanimous_count, two_vs_one_count, all_different_count)
        """
        # split channels
        a, b, c = annotation_t[0], annotation_t[1], annotation_t[2]
        # unanimous: all three equal
        unanimous_mask = (a == b) & (b == c)
        unanimous = int(unanimous_mask.sum())
        # two-vs-one: exactly two equal, third different
        two_vs_one_mask = (
            (a == b) & (c != a)
            | (a == c) & (b != a)
            | (b == c) & (a != b)
        )
        two_vs_one = int(two_vs_one_mask.sum())
        # all different: neither unanimous nor two-vs-one
        total_pixels = annotation_t.shape[1] * annotation_t.shape[2]
        all_different = total_pixels - unanimous - two_vs_one
        row_dict = {'img_id': img_id,
                    'unanimous': unanimous,
                    'two_vs_one': two_vs_one,
                    'all_different': all_different}
        return row_dict

    @staticmethod
    def get_class_count(annotation_dict: Dict[str, torch.Tensor], ground_truth: torch.Tensor) -> Dict[str, int]:
        """
        Returns a dict of pixel counts for:
         - gt_no, gt_low, gt_high
         - {annotator}_no, {annotator}_low, {annotator}_high
        """
        # flatten ground truth
        gt_flat = ground_truth.squeeze().flatten()
        counts: Dict[str, int] = {}
        class_labels = {0: 'no', 1: 'low', 2: 'high'}

        # gt counts
        for cls, label in class_labels.items():
            counts[f'gt_{label}'] = int((gt_flat == cls).sum().item())

        # annotator counts
        for annotator, ann in annotation_dict.items():
            ann_flat = ann.squeeze().flatten()
            for cls, label in class_labels.items():
                key = f'{annotator}_{label}'
                counts[key] = int((ann_flat == cls).sum().item())

        return counts


