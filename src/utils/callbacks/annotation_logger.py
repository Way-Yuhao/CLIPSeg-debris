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

    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str) -> None:
        os.makedirs(self.results_dir, exist_ok=True)

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.demonstrate_data(trainer)

    def on_fit_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        pass

    def on_test_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT,
                          batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        self.save_prediction(outputs, 'test', batch_idx)
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


    # def log_prediction(self):
    #     # Plot image, ground truth, and final segmentation
    #     fig = plt.figure(figsize=(6.7, 13))
    #     columns = 3
    #     rows = 1
    #
    #     ax = []
    #     imgs = [img, label, seg]
    #     imgs_names = ['Test img', 'GroundTruth', 'Pred of true seg']
    #
    #     for i in range(columns * rows):
    #         img_ = imgs[i]
    #         ax.append(fig.add_subplot(rows, columns, i + 1))
    #         ax[-1].set_title(imgs_names[i])
    #         img_ = Image.open(img_)
    #         img_ = np.array(img_, dtype='uint8')
    #         plt.imshow(img_, cmap='gray')
    #
    #     # Log the first set of images to wandb
    #     log_plot_to_wandb(fig, "test_results/main_comparison")
    #
    #     # Plot the segmentation for noisy labels:
    #     fig = plt.figure(figsize=(9, 13))
    #     columns = 4
    #     rows = 1
    #
    #     ax = []
    #     noisy_segs = [over_seg, under_seg, wrong_seg, good_seg]
    #     noisy_segs_names = ['Pred of over', 'Pred of under', 'Pred of wrong', 'Pred of good']
    #
    #     for i in range(columns * rows):
    #         noisy_seg_ = noisy_segs[i]
    #         ax.append(fig.add_subplot(rows, columns, i + 1))
    #         ax[-1].set_title(noisy_segs_names[i])
    #         noisy_seg_ = Image.open(noisy_seg_)
    #         noisy_seg_ = np.array(noisy_seg_, dtype='uint8')
    #         plt.imshow(noisy_seg_, cmap='gray')
    #
    #     # Log the noisy segmentation results to wandb
    #     log_plot_to_wandb(fig, "test_results/noisy_label_predictions")
    #
    #
    # @staticmethod
    # def log_plot_to_wandb(fig, log_name):
    #     """Function to save plot to a BytesIO object for logging in wandb"""
    #     buf = io.BytesIO()
    #     plt.savefig(buf, format='png', bbox_inches='tight')
    #     buf.seek(0)
    #     image = Image.open(buf)
    #     wandb.log({log_name: wandb.Image(image)})
    #     buf.close()
    #     plt.close(fig)
    #     return
    #
