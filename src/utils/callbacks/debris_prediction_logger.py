import os
from typing import Any, Dict, Optional, List
import numpy as np
from lightning import LightningModule, Trainer
from lightning.pytorch.utilities.types import STEP_OUTPUT
from matplotlib import pyplot as plt
from PIL import Image
import cv2
from lightning.pytorch.callbacks import Callback
from src.utils.clipseg_utils.gen_debris_vis_prompt import one_hot_encode_segmentation


class DebrisPredictionLogger(Callback):

    def __init__(self, save_dir: str, cmap: Dict[str, List[int]]):
        """
        Requires batch_size = 1
        Args:
            save_dir: directory to save the predictions
            cmap: RGBA color map for the labels
        """
        self.save_dir = os.path.join(save_dir, 'predictions')
        self.onehot_save_dir = os.path.join(save_dir, 'onehot_rbg')
        self.cmap = cmap
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.onehot_save_dir, exist_ok=True)


    def on_predict_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: Any,
                             batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        # self.save_png(batch, outputs)
        # self.save_one_hot_rgb(batch, outputs)
        self.save_cmap_output(batch, outputs)

    def on_test_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT,
                          batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        # self.save_png(batch, outputs)
        # self.save_one_hot_rgb(batch, outputs)
        self.save_cmap_output(batch, outputs)

    def save_png(self, batch: Any, outputs: Any):
        fname = os.path.join(self.save_dir, batch[1][2][0])
        pred_class = outputs["pred_class"].squeeze().detach().cpu().numpy()
        pred_class = pred_class.astype('uint8')
        cv2.imwrite(fname, pred_class)

    def save_one_hot_rgb(self, batch: Any, outputs: Any):
        pred_class = outputs["pred_class"].squeeze().detach().cpu().numpy()
        seg_low = pred_class == 1
        seg_high = pred_class == 2
        one_hot_segmentation = one_hot_encode_segmentation(seg_low, seg_high)
        fname = os.path.join(self.onehot_save_dir, batch[1][2][0])
        cv2.imwrite(fname, one_hot_segmentation)

    def save_cmap_output(self, batch: Any, outputs: Any):
        # extract rgb image
        original_image = outputs["data_x"][0].squeeze().detach().cpu().numpy()
        original_image = np.moveaxis(original_image, 0, -1)
        original_image = (original_image * 255).astype(np.uint8)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
        original_image = Image.fromarray(original_image).convert("RGBA")

        # extract predicted class
        pred_class = outputs["pred_class"].squeeze().detach().cpu().numpy()


        # Convert labels to a semi-transparent color overlay
        overlay_img = self.label_to_color_image(pred_class)

        final_visualization = Image.alpha_composite(original_image, overlay_img)
        fname = os.path.join(self.onehot_save_dir, f"{batch[1][2][0]}.png")
        final_visualization.save(fname, 'PNG')



        # final_visualization.save(final_merged_output_path, 'PNG')
        #
        # print(f"Resized original image saved at {resized_original_output_path}")
        # print(f"Overlay image saved at {final_merged_output_path}")

    def label_to_color_image(self, label):
        """
        Convert a label array to an RGBA color image using the defined colors.
        Label encoding:
          0: no debris
          1: low debris
          2: high debris
        """
        # extract data
        h, w = label.shape
        no_debris_color = self.cmap['no_debris']
        low_debris_color = self.cmap['low_density']
        high_debris_color = self.cmap['high_density']

        color_img = np.zeros((h, w, 4), dtype=np.uint8)
        color_img[label == 0] = no_debris_color
        color_img[label == 1] = low_debris_color
        color_img[label == 2] = high_debris_color
        return Image.fromarray(color_img, mode='RGBA')



