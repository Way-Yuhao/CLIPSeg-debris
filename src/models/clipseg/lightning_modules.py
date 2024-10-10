from typing import Any, Dict, Tuple, List
import io
import numpy as np
import torch
from torch.nn import Softmax
from lightning import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler, STEP_OUTPUT
from matplotlib import pyplot as plt
from PIL import Image
import wandb
# from src.models.clipseg.clipseg import CLIPDensePredT

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

    def setup(self, stage: str) -> None:
        self.dataset_mask = self.trainer.datamodule.hparams.mask
        self.dataset_class = self.trainer.datamodule.train_dataset.__class__.__name__

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

    def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        data_x, data_y = batch # unpack
        if self.hparams.mix:
            assert self.dataset_mask.startswith('text_and')
            # with autocast_fn() # FIXME: this might be an issue
            prompts = self.model.sample_prompts(data_x[1])

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
        visual_q = None
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

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        step_output = {"loss": loss, "pred": pred, "data_x": data_x, "data_y": data_y}
        return step_output

    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        if batch_idx == 0:
            data_x, data_y, pred = outputs["data_x"], outputs["data_y"], outputs["pred"]
            sample_density = data_x[1][0][11:]
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
            axs[1].set_title(f'prediction ({sample_density})')
            # Third subplot for the ground truth
            axs[2].imshow(data_y[0][0, 0, :, :].detach().cpu().numpy())
            axs[2].set_title(f'gt ({sample_density})')
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
            wandb.log({"train/images": wandb.Image(img)})
