import os
import io
from typing import Any, Dict, Optional, List
from collections import OrderedDict
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT
import wandb
from matplotlib import pyplot as plt
from PIL import Image
import cv2
from src.utils.clipseg_utils.gen_debris_vis_prompt import one_hot_encode_segmentation


class DebrisWandbLogger(Callback):

    def __init__(self, train_log_img_freq: int = 10, val_log_img_freq: int = 10, check_freq_via: str = 'epoch',
                 enable_save_ckpt: bool = False, add_reference_artifact: bool = False,
                 show_train_batches: bool = True, show_val_ids: List[int] = None, save_dir: str = None,
                 log_test_on_wandb: bool = False):
        super().__init__()
        self.train_log_img_freq = train_log_img_freq
        self.val_log_img_freq = val_log_img_freq
        self.check_freq_via = check_freq_via
        self.enable_save_ckpt = enable_save_ckpt
        self.add_reference_artifact = add_reference_artifact
        self.show_train_batches = show_train_batches
        self.show_val_ids = show_val_ids
        self.save_dir = save_dir
        self.log_test_on_wandb = log_test_on_wandb

        self.freqs = {'train_img': train_log_img_freq, 'val_img': val_log_img_freq}
        self.next_log_idx = {'train_img': 0, 'val_img': 0}

        # to be defined elsewhere
        self.num_classes = None
        self.all_text_prompts = None
        self.all_densities = None
        self.logger_class_labels = None
        self.show_val_at_idx = []

    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str) -> None:
        self.num_classes = trainer.datamodule.full_dataset.num_classes
        self.all_text_prompts = trainer.datamodule.full_dataset.text_prompts
        self.all_densities = trainer.datamodule.full_dataset.densities
        self.logger_class_labels = {i: prompt for i, prompt in enumerate(self.all_text_prompts)}
        self.scan_for_val_ids(trainer)

        wandb.run.summary['logdir'] = trainer.default_root_dir

        if self.save_dir is not None:
            # wandb.run.summary['save_test_dir'] = self.save_dir
            os.makedirs(os.path.join(self.save_dir, 'test_predictions'), exist_ok=True)
            os.makedirs(os.path.join(self.save_dir, 'test_predictions_onehot'), exist_ok=True)

    def scan_for_val_ids(self, trainer: "pl.Trainer"):
        ids_remaining = self.show_val_ids
        val_indices = trainer.datamodule.validate_dataset.indices
        val_ids = [trainer.datamodule.full_dataset.img_ids[idx] for idx in val_indices]
        assert val_ids is not None
        for idx, id in enumerate(val_ids):
            if id in ids_remaining:
                self.show_val_at_idx.append(idx)
                ids_remaining.remove(id)
            if len(ids_remaining) == 0:
                break
        if len(ids_remaining) > 0:
            raise ValueError(f"Could not find validation image {ids_remaining}")

    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: Any,
                           batch: Any, batch_idx: int) -> None:
        if self.show_train_batches and self._check_frequency(trainer, 'train_img'):
            self.log_img_pair(outputs, mode='train', batch_idx=batch_idx)

    def on_validation_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT,
                                batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        if batch_idx in self.show_val_at_idx and self._check_frequency(trainer, 'val_img', update=False):
            self.log_predicted_class(outputs, mode='val', batch_idx=batch_idx)

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if trainer.current_epoch > 0:  # skip sanity check
            self._check_frequency(trainer, 'val_img', update=True)
        wandb.log({}, commit=True) # sync wandb step for all logs in this callhook

    def on_test_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT,
                          batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        self.save_png(batch, outputs, batch_idx)
        self.save_one_hot_rgb(batch, outputs, batch_idx)
        self.log_predicted_class(outputs, mode='test', batch_idx=batch_idx)

    def log_predicted_class(self, outputs: STEP_OUTPUT, mode: str, batch_idx: int):
        pred_class = outputs["pred_class"].squeeze().detach().cpu().numpy()
        gt_class = outputs["gt_class"].squeeze().detach().cpu().numpy()
        img = outputs["data_x"][0][0, :, :, :].detach().cpu().numpy().transpose(1, 2, 0)
        img_rescaled = (img - img.min()) / (img.max() - img.min())
        masked_img = wandb.Image(
            img_rescaled,
            masks=OrderedDict([
                ('predictions', {'mask_data': pred_class, 'class_labels': self.logger_class_labels}),
                ('ground_truth', {'mask_data': gt_class, 'class_labels': self.logger_class_labels})
            ])
        )
        wandb.log({f"{mode}/images_{batch_idx}": masked_img}, commit=False)

    @staticmethod
    def log_img_pair(outputs: STEP_OUTPUT, mode: str, batch_idx: int):
        data_x, data_y, pred = outputs["data_x"], outputs["data_y"], outputs["pred"]
        shortened_prompt = data_x[1][0][10:]
        if len(shortened_prompt) == 0:
            shortened_prompt = 'no debris'
        # Create a new figure
        fig, axs = plt.subplots(1, 3, figsize=(10, 3))
        # First subplot for the query image
        query_img_rescaled = data_x[0][0, :, :, :].detach().cpu().numpy().transpose(1, 2, 0)
        query_img_rescaled = (query_img_rescaled - query_img_rescaled.min()) / (
                query_img_rescaled.max() - query_img_rescaled.min())
        axs[0].imshow(query_img_rescaled)
        axs[0].set_title(f'query image')
        # Second subplot for the prediction
        axs[1].imshow(pred[0][0, :, :].detach().cpu().numpy())
        axs[1].set_title(f'prediction ({shortened_prompt})')
        # Third subplot for the ground truth
        axs[2].imshow(data_y[0][0, 0, :, :].detach().cpu().numpy())
        axs[2].set_title(f'gt ({shortened_prompt})')
        plt.tight_layout()
        # Save the figure to a BytesIO object
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        # Close the figure
        plt.close(fig)
        # Convert buffer to a PIL Image
        buf.seek(0)
        img = Image.open(buf)
        # Log the PIL Image to wandb
        wandb.log({f"{mode}/images_{batch_idx}": wandb.Image(img)})

    def _check_frequency(self, trainer: "pl.trainer", key: str, update: bool = True) -> bool:
        if self.freqs[key] == -1:
            return False
        if self.check_freq_via == 'global_step':
            check_idx = trainer.global_step
        elif self.check_freq_via == 'epoch':
            check_idx = trainer.current_epoch
        if check_idx >= self.next_log_idx[key]:
            if update:
                self.next_log_idx[key] = check_idx + self.freqs[key]
            return True
        else:
            return False

    def save_png(self, batch: Any, outputs: Any, batch_idx: int):
        fname = os.path.join(self.save_dir, 'test_predictions', f"test_{batch_idx}.png")
        pred_class = outputs["pred_class"].squeeze().detach().cpu().numpy()
        pred_class = pred_class.astype('uint8')
        cv2.imwrite(fname, pred_class)

    def save_one_hot_rgb(self, batch: Any, outputs: Any, batch_idx: int):
        pred_class = outputs["pred_class"].squeeze().detach().cpu().numpy()
        seg_low = pred_class == 1
        seg_high = pred_class == 2
        one_hot_segmentation = one_hot_encode_segmentation(seg_low, seg_high)
        fname = os.path.join(self.save_dir, 'test_predictions_onehot', f"test_{batch_idx}.png")
        cv2.imwrite(fname, one_hot_segmentation)


