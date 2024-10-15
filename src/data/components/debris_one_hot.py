from typing import List
import os
import os.path as p
# import sys
import numpy as np
# from PIL import Image
import cv2
import torch
from torch.utils.data import Dataset
from natsort import natsorted
from matplotlib import pyplot as plt
from torchvision import transforms


__author__ = 'yuhao liu'

class DebrisOneHotDataset(Dataset):

    def __init__(self, dataset_dir: str, resize_to: tuple, text_prompts: List, densities: List,
                 *args, **kwargs):
        self.dataset_dir = dataset_dir
        self.text_prompts = text_prompts
        self.densities = densities
        self.resize_to = resize_to

        self.num_classes = len(text_prompts)
        assert len(densities) == self.num_classes
        self.original_image_dir = p.join(dataset_dir, 'original')
        self.original_imgs = natsorted([p.join(self.original_image_dir, f)
                                        for f in os.listdir(self.original_image_dir) if f.endswith('.png')])
        self.img_ids = [f.split('-')[2].split('_')[0] for f in self.original_imgs]

        self.annotation_dir = p.join(dataset_dir, 'segmentation_merged')
        self.annotation_files = natsorted([p.join(self.annotation_dir, f) for f in os.listdir(self.annotation_dir)
                                           if f.endswith('.png')])
        self.vis_prompt_dir = p.join(dataset_dir, 'vis_prompts')
        self.vis_prompts = {}

        for density in densities:
            dir_ = p.join(self.vis_prompt_dir, density)
            self.vis_prompts[density] = natsorted([p.join(dir_, f) for f in os.listdir(dir_) if f.endswith('.png')])

        self.normalize = transforms.Normalize((0.57784108, 0.5724125, 0.5619426),
                                              (0.24724819, 0.24302182, 0.23344601))

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img = self.find_input_img(img_id)
        img = self.cvt_img_to_tensor(img)

        density, text_prompt = self.randomly_select_density()
        vis_prompt = self.find_visual_prompt(density)
        vis_prompt = self.cvt_img_to_tensor(vis_prompt)
        vis_s = [text_prompt, vis_prompt, True]

        annotation_one_hot = self.find_annotation_one_hot(img_id)
        annotation_one_hot = self.cvt_annotation_to_tensor(annotation_one_hot)
        annotation = self.find_annotation_for_density(img_id, density)
        annotation = self.cvt_annotation_to_tensor(annotation)

        data_x = (img, ) + tuple(vis_s)
        # FIXME: annotation_one_hot is not used used to be torch.zeros(0)
        data_y = (annotation, annotation_one_hot, idx)

        return data_x, data_y

    def find_input_img(self, img_id: str):
        # find the input RGB image
        img_files = [f for f in self.original_imgs if img_id in f]
        assert len(img_files) == 1
        img = cv2.imread(img_files[0], cv2.IMREAD_COLOR)
        img = cv2.resize(img, self.resize_to, cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        # img = self.normalize(img)
        return img

    def randomly_select_density(self):
        # randomly select a density
        rand_idx = np.random.randint(0, self.num_classes)
        density = self.densities[rand_idx]
        text_prompt = self.text_prompts[rand_idx]
        return density, text_prompt

    def find_visual_prompt(self, density: str):
        # randomly select a visual prompt
        vis_prompts = self.vis_prompts[density]
        vis_prompt_file = vis_prompts[np.random.randint(0, len(vis_prompts))]
        vis_prompt = cv2.imread(vis_prompt_file, cv2.IMREAD_COLOR)
        vis_prompt = cv2.resize(vis_prompt, self.resize_to, cv2.INTER_LINEAR)
        vis_prompt = cv2.cvtColor(vis_prompt, cv2.COLOR_BGR2RGB)
        return vis_prompt

    def find_annotation_one_hot(self, img_id: str):
        # find one-hot annotation file
        annotation_files = [f for f in self.annotation_files if img_id in f]
        assert len(annotation_files) == 1
        annotation_one_hot = cv2.imread(annotation_files[0], cv2.IMREAD_COLOR)
        # annotation_one_hot = cv2.cvtColor(annotation_one_hot, cv2.COLOR_BGR2RGB)
        annotation_one_hot = cv2.resize(annotation_one_hot, self.resize_to, cv2.INTER_LINEAR)
        return annotation_one_hot

    def find_annotation_for_density(self, img_id: str, density: str):
        # find one-hot annotation file, return the annotation for the given density
        one_hot_annotation = self.find_annotation_one_hot(img_id)
        density_idx = self.densities.index(density)
        annotation = one_hot_annotation[:, :, density_idx]
        return annotation

    def cvt_img_to_tensor(self, img: np.ndarray):
        t = torch.tensor(np.array(img).transpose(2, 0, 1), dtype=torch.float32)
        t = t / 255.0
        t = self.normalize(t)
        return t

    @staticmethod
    def cvt_annotation_to_tensor(annotation: np.ndarray):
        if len(annotation.shape) == 3:
           annotation = annotation.transpose(2, 0, 1)
        t = torch.tensor(annotation, dtype=torch.float32)
        t = t / 255.0
        # t = t.unsqueeze(0)
        return t


if __name__ == '__main__':
    d = DebrisOneHotDataset('/home/yuhaoliu/Data/HIDeAI/multi_labeler_onehot/union',
                            (256, 256),
                            text_prompts=['no debris', 'debris at low density', 'debris at high density'],
                            densities=['no', 'low', 'high'])
    # get a sample
    data_x, data_y = d[0]