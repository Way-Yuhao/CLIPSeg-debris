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
        self.composite_results_dir = os.path.join(self.save_dir, 'composite_results')

        # to be defined later
        self.labeler_tags = None
        self.num_labelers = None
        self.gt_labeler_tag = None

    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str) -> None:
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.composite_results_dir, exist_ok=True)
        self.labeler_tags = trainer.datamodule.labeler_tags
        self.gt_labeler_tag = trainer.datamodule.gt_labeler_tag
        self.num_labelers = len(self.labeler_tags)
        return

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.demonstrate_data(trainer)

    def on_test_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT,
                          batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        self.save_prediction(outputs, 'test', batch_idx)
        log_wandb = True if batch_idx == self.test_idx else False
        self.save_prediction_composite(outputs, 'test', batch_idx, log_wandb)
        return

    def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        # self.log_prediction_from_disk(self.test_idx)
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
        # images, labels_over, labels_under, labels_wrong, labels_good, imagename = dataset_[self.demonstrate_idx]
        images, labels, gt_label, imagename = trainer.datamodule.unpack_batch(dataset_[self.demonstrate_idx])
        # images = np.mean(images, axis=0)

        # plot the labels:
        fig = plt.figure()
        columns = self.num_labelers + 1
        rows = 1
        ax = []
        images_to_display = [images]
        images_to_display += labels
        labels_names = ['Input'] + self.labeler_tags

        for i in range(columns * rows):
            if i == 0: # input RGB
                im = images_to_display[i].transpose(1, 2, 0) # / 255.
            else:  # labels
                im = images_to_display[i][0, :, :]
            ax.append(fig.add_subplot(rows, columns, i + 1))
            ax[-1].set_title(labels_names[i])
            plt.imshow(im, cmap='gray')
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

    def save_prediction(self, outputs: Dict[str, np.ndarray], phase: str, idx: int):
        for k, v in outputs.items():
            plt.imsave(os.path.join(self.results_dir, f'{phase}_{idx}_{k}.png'), v, cmap='gray')


    def save_prediction_composite(self, outputs: Dict, phase: str, idx: int, log_wandb: bool):
        n = self.num_labelers
        title_font_size = 20

        # Row 1: Input, Ground Truth, and Consensus Prediction
        fig, ax = plt.subplots(3, max(3, n), figsize=(4 * max(3, n), 12))
        row1_imgs_keys = ['img', 'label', 'seg']
        row1_titles = ['Input Image', 'Ground Truth', 'Prediction of Consensus']

        for i, (k, title) in enumerate(zip(row1_imgs_keys, row1_titles)):
            img = outputs[k]
            ax[0, i].imshow(np.array(img), cmap='gray' if k != 'img' else None)
            ax[0, i].set_title(title, fontsize=title_font_size)
            ax[0, i].axis('off')

        ax[0, -1].axis('off')

        # Row 2: Predictions
        for i in range(n):
            img = outputs[f'noisy_{i}_seg']
            ax[1, i].imshow(np.array(img), cmap='gray')
            if self.labeler_tags is not None:
                ax[1, i].set_title(f'Prediction of {self.labeler_tags[i]}', fontsize=title_font_size)
            else:
                ax[1, i].set_title(f'Prediction of {i}', fontsize=title_font_size)
            ax[1, i].axis('off')

        # Row 3: Labels of each annotator
        for i in range(n):
            img = outputs[f'{i}_label']
            ax[2, i].imshow(np.array(img), cmap='gray')
            if self.labeler_tags is not None:
                ax[2, i].set_title(f'Label of {self.labeler_tags[i]}', fontsize=title_font_size)
            else:
                ax[2, i].set_title(f'Label of {i}', fontsize=title_font_size)
            ax[2, i].axis('off')

        # Remove empty subplots if n < 3
        if n < 3:
            for i in range(n, 3):
                ax[0, i].axis('off')
            for i in range(n, max(3, n)):
                ax[1, i].axis('off')
                ax[2, i].axis('off')

        # Log the figure to wandb
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.savefig(os.path.join(self.composite_results_dir, f'{phase}_{idx}_results.png'),
                    format='png', bbox_inches='tight')
        buf.seek(0)
        image = Image.open(buf)
        if log_wandb:
            wandb.log({f"{phase}/results": wandb.Image(image, caption=f'{phase} {idx} Results')})
        buf.close()
        plt.close(fig)

    def log_prediction_from_disk(self, test_data_index: int):
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
        plt.savefig(os.path.join(self.composite_results_dir, f'test_{test_data_index}_results.png'),
                    format='png', bbox_inches='tight')
        buf.seek(0)
        image = Image.open(buf)
        wandb.log({"test/test_results": wandb.Image(image, caption=f'Test {test_data_index} Results')})
        buf.close()
        plt.close(fig)

