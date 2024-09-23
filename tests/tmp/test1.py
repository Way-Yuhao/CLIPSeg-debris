import os
import autoroot
import errno
import torch
import timeit
import imageio
import numpy as np
import torch.nn as nn
from torch.utils import data
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from PIL import Image
from src.utils.multi_annotators_utils.adamW import AdamW
from src.utils.multi_annotators_utils.loss import noisy_label_loss
from src.utils.multi_annotators_utils.utilis import segmentation_scores, CustomDataset_punet, calculate_cm
from src.utils.multi_annotators_utils.utilis import evaluate_noisy_label_4, evaluate_noisy_label_5, evaluate_noisy_label_6
# our proposed model:
from src.models.Models import UNet_CMs