import os
import io
from typing import Any, Dict, Optional
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT
from lightning.pytorch.utilities import rank_zero_only
import wandb
from PIL import Image
from matplotlib import pyplot as plt
from lightning.pytorch.callbacks import RichProgressBar, Callback


class AnnotationLogger(Callback):

    def __init__(self, save_dir: str, *args, **kwargs):
        super().__init__()
        self.save_dir = save_dir
        self.demonstrate_on_set = 'val'
        self.demonstrate_idx = 6
        self.test_idx = 15

        self.results_dir = os.path.join(self.save_dir, 'results')

        # to be defined later
        self.labeler_tags = None
        self.num_labelers = None

    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str) -> None:
        os.makedirs(self.results_dir, exist_ok=True)
        self.labeler_tags = trainer.datamodule.labeler_tags
        self.num_labelers = len(self.labeler_tags)

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.demonstrate_data(trainer)

    def on_test_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT,
                          batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        self.save_prediction(outputs, 'test', batch_idx)
        return

    def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.log_prediction(self.test_idx)
        return

    def demonstrate_data(self, trainer: "pl.trianer"):
        """Demonstrate the training samples and log via wandb"""
        if self.demonstrate_on_set == 'train':
            dataset_ = trainer.datamodule.train_dataset
        elif self.demonstrate_on_set == 'val':
            dataset_ = trainer.datamodule.validate_dataset
        elif self.demonstrate_on_set == 'test':
            dataset_ = trainer.datamodule.test_dataset
        else:
            raise NotImplementedError()
        images, labels_over, labels_under, labels_wrong, labels_good, imagename = dataset_[self.demonstrate_idx]
        images = np.mean(images, axis=0)

        # plot the labels:
        fig = plt.figure()
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
            ax[-1].axis('off')

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        image = Image.open(buf)
        wandb.log({"dataloader/samples":
                       wandb.Image(image, caption=f"{self.demonstrate_on_set} sample {self.demonstrate_idx}")})
        buf.close()
        plt.close(fig)
        return

    def save_prediction(self, outputs: dict[str, np.ndarray], phase: str, idx: int):
        for k, v in outputs.items():
            plt.imsave(os.path.join(self.results_dir, f'{phase}_{idx}_{k}.png'), v, cmap='gray')

    def log_prediction(self, test_data_index: int):
        """
        Plot image, ground truth, and final segmentation
        """
        n = self.num_labelers
        title_font_size =20

        # File paths
        img_path = os.path.join(self.results_dir, f'test_{test_data_index}_img.png')
        label_path = os.path.join(self.results_dir, f'test_{test_data_index}_label.png')
        seg_path = os.path.join(self.results_dir, f'test_{test_data_index}_seg.png')

        # Row 1: Input, Ground Truth, and Consensus Prediction
        fig, ax = plt.subplots(2, max(3, n), figsize=(4 * max(3, n), 8))
        row1_imgs = [img_path, label_path, seg_path]
        row1_titles = ['Input Image', 'Ground Truth', 'Prediction of Consensus']

        for i, (img_path, title) in enumerate(zip(row1_imgs, row1_titles)):
            img = Image.open(img_path)
            ax[0, i].imshow(np.array(img), cmap='gray')
            ax[0, i].set_title(title, fontsize=title_font_size)
            ax[0, i].axis('off')

        ax[0, 3].axis('off')

        # Row 2: Noisy Predictions
        for i in range(n):
            noisy_seg_path = os.path.join(self.results_dir, f'test_{test_data_index}_noisy_{i}_seg.png')
            img = Image.open(noisy_seg_path)
            ax[1, i].imshow(np.array(img), cmap='gray')
            if self.labeler_tags is not None:
                ax[1, i].set_title(f'Prediction of {self.labeler_tags[i]}', fontsize=title_font_size)
            else:
                ax[1, i].set_title(f'Prediction of {i}', fontsize=title_font_size)
            ax[1, i].axis('off')

        # Remove empty subplots if n < 3
        if n < 3:
            for i in range(n, 3):
                ax[0, i].axis('off')
            for i in range(n, max(3, n)):
                ax[1, i].axis('off')

        # Log the figure to wandb
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        image = Image.open(buf)
        wandb.log({"test/test_results": wandb.Image(image, caption=f'Test {test_data_index} Results')})
        buf.close()
        plt.close(fig)

