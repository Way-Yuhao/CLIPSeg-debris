from typing import Any, Dict, Tuple, Optional
import torch
from src.models.generic_derbis_module import GenericDebrisSegmentationModule

class YOLOv8LitModule(GenericDebrisSegmentationModule):
    """
    A PyTorch Lightning module for a simple CNN.
    """

    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                 criterion, compile: bool = False, *args, **kwargs) -> None:
        super().__init__(model=model, optimizer=optimizer, criterion=criterion, compile=compile)
        self.save_hyperparameters(logger=False, ignore=("model", "optimizer", "criterion", "compile"))
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)['out']

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        query_img, label = self.unpack(batch)
        prediction = self.forward(query_img)
        loss = self.criterion(prediction, label)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=query_img.shape[0])
        step_output = {"loss": loss, "pred": prediction, "data_x": batch[0], "data_y": batch[1]}
        return step_output

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Perform a single validation step on a batch of data from the validation set.
        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.
        :param batch_idx: The index of the current batch.
        """
        query_img, label = self.unpack(batch)
        prediction = self.forward(query_img)
        pred_class = torch.argmax(prediction, dim=1)
        gt_class = torch.argmax(label, dim=1)
        self.compute_metric_single_class_2(pred_class, gt_class, 'val')
        step_output = {"data_x": batch[0], "data_y": batch[1], "pred": prediction, "gt_one_hot": batch[1][1],
                       "pred_class": pred_class, "gt_class": gt_class}
        return step_output