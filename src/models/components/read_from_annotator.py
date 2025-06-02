import os
from typing import Any
import torch
from torch import nn
import numpy as np
import cv2


from src.utils import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=False)

class AnnotationReader(nn.Module):
    """
    A simple module to read annotations from a specified directory.
    """

    def __init__(self, annotation_parent_dir: str, annotator: str):
        super().__init__()
        self.annotator_dir = os.path.join(annotation_parent_dir, annotator)
        self.annotator = annotator
        self.density_levels = ['low', 'high']
        self.resize_to = (256, 256)
        self.annotations = []

    def setup(self):
        if not os.path.exists(self.annotator_dir):
            raise FileNotFoundError(f"Annotation directory {self.annotation_dir} does not exist.")
        # scan for scenes
        scenes = [d for d in os.listdir(self.annotator_dir) if os.path.isdir(os.path.join(self.annotator_dir, d))]
        logger.debug('Found scenes: %s', scenes)
        for scene in scenes:
            self.scan_scenes(scene)
        logger.info("Total binary annotation files found: %d", len(self.annotations))
        return

    def scan_scenes(self, scene_dir: str) -> None:
        """
        Scan the annotation directory for available scenes.
        :return: List of scene names.
        """
        for d in self.density_levels:
            file_path = os.path.join(self.annotator_dir, scene_dir, 'output', d)
            files = [f for f in os.listdir(file_path) if f.endswith('.png') and 'binary' in f]
            files = [os.path.join(file_path, f) for f in files]
            if files:
                logger.debug('Found binary annotation files: %s', files)
                self.annotations += files
            else:
                logger.warning('No binary annotation files found in %s for density %s', file_path, d)
        return

    def forward(self, query_img: torch.tensor, img_id: str) -> Any:
        matches = [a for a in self.annotations if img_id in a]
        if not matches:
            matches = []
        empty_img = torch.zeros(self.resize_to, dtype=torch.float32)
        empty_img = empty_img.unsqueeze(0).repeat(3, 1, 1)
        low_density = [m for m in matches if 'low' in m]
        high_density = [m for m in matches if 'high' in m]
        if low_density:
            low_density_fpath = low_density[0]
            low_density = self.get_annotation(low_density_fpath)
        else:
            logger.warning("No low density annotation found for image ID: %s", img_id)
            low_density = empty_img
        if high_density:
            high_density_fpath = high_density[0]
            high_density = self.get_annotation(high_density_fpath)
        else:
            logger.warning("No high density annotation found for image ID: %s", img_id)
            high_density = empty_img
        # merge
        # 5. Reduce each 3×H×W mask to an H×W boolean map
        low_any = (low_density > 0).any(dim=0)  # H×W bool, True where low annotation is present  [oai_citation:32‡PyTorch Forums](https://discuss.pytorch.org/t/how-to-combine-separate-annotations-for-multiclass-semantic-segmentation/121232?utm_source=chatgpt.com) [oai_citation:33‡Stack Overflow](https://stackoverflow.com/questions/67465227/pytorch-two-binary-masks-union?utm_source=chatgpt.com)
        high_any = (high_density > 0).any(dim=0)  # H×W bool, True where high annotation is present  [oai_citation:34‡PyTorch Forums](https://discuss.pytorch.org/t/how-to-combine-separate-annotations-for-multiclass-semantic-segmentation/121232?utm_source=chatgpt.com) [oai_citation:35‡Stack Overflow](https://stackoverflow.com/questions/67465227/pytorch-two-binary-masks-union?utm_source=chatgpt.com)

        # 6. Ensure high overrides low where they overlap
        high_mask = high_any  # H×W bool, True for high-density pixels  [oai_citation:36‡PyTorch Forums](https://discuss.pytorch.org/t/how-to-combine-separate-annotations-for-multiclass-semantic-segmentation/121232?utm_source=chatgpt.com) [oai_citation:37‡PyTorch Forums](https://discuss.pytorch.org/t/how-to-apply-weighted-loss-to-a-binary-segmentation-problem/35317?utm_source=chatgpt.com)
        low_mask = low_any & (~high_mask)  # H×W bool, True for low only if not high  [oai_citation:38‡PyTorch Forums](https://discuss.pytorch.org/t/how-to-combine-separate-annotations-for-multiclass-semantic-segmentation/121232?utm_source=chatgpt.com) [oai_citation:39‡PyTorch Forums](https://discuss.pytorch.org/t/how-to-apply-weighted-loss-to-a-binary-segmentation-problem/35317?utm_source=chatgpt.com)

        # 7. Convert each boolean mask to an integer mask
        high_int = high_mask.long()  # H×W LongTensor, 1 where high, 0 elsewhere  [oai_citation:40‡PyTorch Forums](https://discuss.pytorch.org/t/how-to-set-mask-labels-for-mask-r-cnn-so-that-i-can-fine-tune-it-into-a-3-classes-classification-and-segementation-model/66164?utm_source=chatgpt.com) [oai_citation:41‡GitHub](https://github.com/bodokaiser/piwise/issues/10?utm_source=chatgpt.com)
        low_int = low_mask.long()  # H×W LongTensor, 1 where low,  0 elsewhere  [oai_citation:42‡PyTorch Forums](https://discuss.pytorch.org/t/how-to-set-mask-labels-for-mask-r-cnn-so-that-i-can-fine-tune-it-into-a-3-classes-classification-and-segementation-model/66164?utm_source=chatgpt.com) [oai_citation:43‡GitHub](https://github.com/bodokaiser/piwise/issues/10?utm_source=chatgpt.com)

        # 8. Combine into a single-channel label map: {0=no, 1=low, 2=high}
        label = high_int * 2 + low_int
        return label.unsqueeze(0)

    def get_annotation(self, filepath: str) -> torch.Tensor:
        img = cv2.imread(filepath)
        img = cv2.resize(img, self.resize_to, cv2.INTER_LINEAR)
        # convert one-hot encoded image to binary mask
        t = self.cvt_img_to_tensor(img)
        t[t > 0] = 1.0  # convert to binary mask
        return t

    def cvt_img_to_tensor(self, img: np.ndarray) -> torch.Tensor:
        """
        Convert an image to a tensor.
        :param img: Input image as a numpy array.
        :return: Tensor representation of the image.
        """
        t = torch.tensor(np.array(img).transpose(2, 0, 1), dtype=torch.float32)
        t = t / 255.0
        return t


