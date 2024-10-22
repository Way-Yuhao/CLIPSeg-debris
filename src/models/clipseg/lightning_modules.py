from typing import Any, Dict, Tuple, List
import io
import numpy as np
import torch
from torch.nn import Softmax
from lightning import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler, STEP_OUTPUT
from matplotlib import pyplot as plt
from sklearn.metrics import jaccard_score, precision_score, recall_score, f1_score
from src.utils.multi_annotators_utils.utilis import segmentation_scores, generalized_energy_distance


__author__ = 'Yuhao Liu'


class CLIPSegLitModule(LightningModule):

    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                 loss_fn: torch.nn.Module, scheduler: OptimizerLRScheduler, compile: bool = False,
                 *args, **kwargs):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.model = model
        self.loss_fn = loss_fn

        # to be defined elsewhere
        self.dataset_mask = None
        self.dataset_class = None
        self.num_classes = None
        # debug
        # self.example_input_array = [torch.rand(4, 3, 256, 256), torch.rand(4, 512)]

    def setup(self, stage: str) -> None:
        if self.hparams.pretrained_clipseg_ckpt is not None:
            print(f'Loading pretrained model from {self.hparams.pretrained_clipseg_ckpt}')
            loaded_state_dict = torch.load(self.hparams.pretrained_clipseg_ckpt)
            self.model.load_state_dict(loaded_state_dict, strict=False)
            # loaded_state_dict = checkpoint['state_dict']
            # new_state_dict = {}
            # prefix = 'clip_model.'
            # for key, value in loaded_state_dict.items():
            #     new_key = prefix + key
            #     new_state_dict[new_key] = value
            # self.model.load_state_dict(new_state_dict)
        self.dataset_mask = self.trainer.datamodule.hparams.mask
        if hasattr(self.trainer.datamodule, 'train_dataset') and self.trainer.datamodule.train_dataset is not None:
            self.dataset_class = self.trainer.datamodule.train_dataset.__class__.__name__
        else:
            self.dataset_class = self.trainer.datamodule.predict_dataset.__class__.__name__
        self.num_classes = self.trainer.datamodule.full_dataset.num_classes
        self.all_text_prompts = self.trainer.datamodule.full_dataset.text_prompts
        self.all_densities = self.trainer.datamodule.full_dataset.densities
        self.logger_class_labels = {i: prompt for i, prompt in enumerate(self.all_text_prompts)}

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {  # TODO: check if this is correct
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

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """for debug"""
        return self.model(*args, **kwargs)

    def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        data_x, data_y = batch # unpack
        if self.hparams.mix:
            assert self.dataset_mask.startswith('text_and')
            # with autocast_fn() # FIXME: this might be an issue
            prompts = self.model.sample_prompts(data_x[1], prompt_list=('a photo of {}',))

            text_cond = self.model.compute_conditional(prompts)
            if self.model.__class__.__name__ == 'CLIPDensePredTMasked':
                # when mask=='separate'
                visual_s_cond, _, _ = self.model.visual_forward_masked(data_x[2].cuda(), data_x[3].cuda())
            else:
                # data_x[2] = visual prompt
                visual_s_cond, _, _ = self.model.visual_forward(data_x[2].cuda())
            # end of autocast

            max_txt = self.hparams.mix_text_max if self.hparams.mix_text_max is not None else 1
            batch_size = text_cond.shape[0]

            # sample weights for each element in batch
            text_weights = torch.distributions.Uniform(self.hparams.mix_text_min, max_txt).sample((batch_size,))[:, None]
            text_weights = text_weights.cuda()

            if self.dataset_class == 'PhraseCut':
                # give full weight to text where support_image is invalid
                visual_is_valid = data_x[4] if self.model.__class__.__name__ == 'CLIPDensePredTMasked' else data_x[3]
                text_weights = torch.max(text_weights[:, 0], 1 - visual_is_valid.float().cuda()).unsqueeze(1)
            cond = text_cond * text_weights + visual_s_cond * (1 - text_weights)
        else:  # no mix
            if self.model.__class__.__name__ == 'CLIPDensePredTMasked':
                # compute conditional vector using CLIP masking
                # with autocast_fn():
                assert self.dataset_mask == 'separate'
                cond, _, _ = self.model.visual_forward_masked(data_x[1].cuda(), data_x[2].cuda())
                # end of autocast
            else:
                cond = data_x[1]
                if isinstance(cond, torch.Tensor):
                    cond = cond.cuda()

        # with autocast_fn():
        # visual_q = None
        pred, visual_q, _, _ = self.model(data_x[0].cuda(), cond, return_features=True)
        loss = self.loss_fn(pred, data_y[0].cuda())
        # TODO monitor loss
        # if torch.isnan(loss) or torch.isinf(loss):
        #     # skip if loss is nan
        #     log.warning('Training stopped due to inf/nan loss.')
        #     sys.exit(-1)
        extra_loss = 0
        loss += extra_loss
        # end of autocast
        # TODO scale loss with GradScaler()

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=data_x[0].shape[0])
        step_output = {"loss": loss, "pred": pred, "data_x": data_x, "data_y": data_y}
        return step_output

    def validation_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        data_x, data_y = batch  # unpack

        prompts = self.model.sample_prompts(self.all_text_prompts, prompt_list=('a photo of {}',))
        stacked_rgb_inputs = data_x[0].repeat(len(prompts), 1, 1, 1)
        pred, visual_q, _, _ = self.model(stacked_rgb_inputs, prompts, return_features=True)
        pred_class = torch.argmax(pred, dim=0) # prediction
        gt_class = torch.argmax(data_y[1], dim=1) # ground truth

        # binary masks for debris_low and debris_high
        mask_debris_low = (gt_class == 1).flatten().cpu().numpy()
        pred_debris_low = (pred_class == 1).flatten().cpu().numpy()
        mask_debris_high = (gt_class == 2).flatten().cpu().numpy()
        pred_debris_high = (pred_class == 2).flatten().cpu().numpy()

        ## report metrics
        # IoU, foreground
        iou_debris_low = jaccard_score(mask_debris_low, pred_debris_low, zero_division=0)
        iou_debris_high = jaccard_score(mask_debris_high, pred_debris_high, zero_division=0)
        # precision vs. recall, foreground
        precision_debris_low = precision_score(mask_debris_low, pred_debris_low, zero_division=0)
        recall_debris_low = recall_score(mask_debris_low, pred_debris_low, zero_division=0)
        precision_debris_high = precision_score(mask_debris_high, pred_debris_high, zero_division=0)
        recall_debris_high = recall_score(mask_debris_high, pred_debris_high, zero_division=0)
        # dice
        dice_score = segmentation_scores(gt_class.cpu().detach(), pred_class.cpu().detach().numpy(),  self.num_classes)
        self.log("val/iou_debris_low", iou_debris_low, on_epoch=True, prog_bar=True)
        self.log("val/iou_debris_high", iou_debris_high, on_epoch=True, prog_bar=True)
        self.log("val/precision_debris_low", precision_debris_low, on_epoch=True, prog_bar=True)
        self.log("val/recall_debris_low", recall_debris_low, on_epoch=True, prog_bar=True)
        self.log("val/precision_debris_high", precision_debris_high, on_epoch=True, prog_bar=True)
        self.log("val/recall_debris_high", recall_debris_high, on_epoch=True, prog_bar=True)
        self.log("val/dice", dice_score, on_step=False, on_epoch=True, prog_bar=True, batch_size=data_x[0].shape[0])

        step_output = {"data_x": data_x, "data_y": data_y, "pred": pred, "gt_one_hot": data_y[1],
                       "pred_class": pred_class, "gt_class": gt_class}
        return step_output

    def test_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        data_x, data_y = batch  # unpack

        prompts = self.model.sample_prompts(self.all_text_prompts, prompt_list=('a photo of {}',))
        stacked_rgb_inputs = data_x[0].repeat(len(prompts), 1, 1, 1)
        pred, visual_q, _, _ = self.model(stacked_rgb_inputs, prompts, return_features=True)
        pred_class = torch.argmax(pred, dim=0)
        gt_class = torch.argmax(data_y[1], dim=1)

        # report metrics
        dice_score = segmentation_scores(gt_class.cpu().detach(), pred_class.cpu().detach().numpy(), self.num_classes)

        self.log("test/dice", dice_score, on_step=False, on_epoch=True, prog_bar=True, batch_size=data_x[0].shape[0])

        step_output = {"data_x": data_x, "data_y": data_y, "pred": pred, "gt_one_hot": data_y[1],
                       "pred_class": pred_class, "gt_class": gt_class}
        return step_output

    def predict_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        data_x, data_y = batch  # TODO: unpack data accordingly
        prompts = self.model.sample_prompts(self.all_text_prompts, prompt_list=('a photo of {}',))
        stacked_rgb_inputs = data_x[0].repeat(len(prompts), 1, 1, 1)
        pred, visual_q, _, _ = self.model(stacked_rgb_inputs, prompts, return_features=True)
        pred_class = torch.argmax(pred, dim=0)
        step_output = {'pred_class': pred_class}
        return step_output
