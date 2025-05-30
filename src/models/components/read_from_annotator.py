import os
from typing import Any

import torch
from torch import nn

from src.utils import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=False)

class AnnotationReader(nn.Module):
    """
    A simple module to read annotations from a specified directory.
    """

    def __init__(self, annotation_parent_dir: str, annotator: str):
        super().__init__()
        self.annotator_dir = os.path.join(annotation_parent_dir, annotator)
        self.density_levels = ['low', 'high']
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

    def forward(self, x: torch.Tensor) -> Any:
        pass