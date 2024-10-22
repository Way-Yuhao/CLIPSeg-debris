import os
from typing import Any, Dict, Optional, List
from lightning import LightningModule, Trainer
from lightning.pytorch.utilities.types import STEP_OUTPUT
from matplotlib import pyplot as plt
from PIL import Image
import cv2
from lightning.pytorch.callbacks import Callback
from src.utils.clipseg_utils.gen_debris_vis_prompt import one_hot_encode_segmentation

class DebrisPredictionLogger(Callback):

    def __init__(self, save_dir: str):
        self.save_dir = os.path.join(save_dir, 'predictions')
        self.onehot_save_dir = os.path.join(save_dir, 'onehot_rbg')
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.onehot_save_dir, exist_ok=True)

    def on_predict_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: Any,
                             batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        self.save_png(batch, outputs)
        self.save_one_hot_rgb(batch, outputs)

    def save_png(self, batch: Any, outputs: Any):
        fname = os.path.join(self.save_dir, batch[1][2][0])
        pred_class = outputs["pred_class"].squeeze().detach().cpu().numpy()
        pred_class = pred_class.astype('uint8')
        cv2.imwrite(fname, pred_class)
        # print(0)
        # # save the prediction
        # Image.fromarray(pred_class).save(fname)

    def save_one_hot_rgb(self, batch: Any, outputs: Any):
        pred_class = outputs["pred_class"].squeeze().detach().cpu().numpy()
        seg_low = pred_class == 1
        seg_high = pred_class == 2
        one_hot_segmentation = one_hot_encode_segmentation(seg_low, seg_high)
        fname = os.path.join(self.onehot_save_dir, batch[1][2][0])
        cv2.imwrite(fname, one_hot_segmentation)



