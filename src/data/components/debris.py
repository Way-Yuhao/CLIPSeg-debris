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

class DebrisDataset(Dataset):

    def __init__(self, dataset_dir: str, debris_free_dataset_dir: str, resize_to: tuple, negative_prob: float):
        # attributes
        self.negative_prob = negative_prob
        # debris density segmentation dataset
        self.dataset_dir = dataset_dir
        self.original_image_dir = p.join(dataset_dir, 'original')
        self.original_imgs = natsorted([p.join(self.original_image_dir, f)
                                        for f in os.listdir(self.original_image_dir) if f.endswith('.png')])
        # filename has the format post-rgb-000126_merged_50m.png, where id here is 000126
        self.img_ids = [f.split('-')[2].split('_')[0] for f in self.original_imgs]
        self.annotation_dir = p.join(dataset_dir, 'segmentation_hl')
        self.vis_prompt_dir = p.join(dataset_dir, 'vis_prompts_hl')
        # a sample is a tuple of (img_id, annotation_file),
        # where annotation_file is one of the annotations for the img_id
        self.sample_ids = [(i, f)
                           for i in self.img_ids
                           for f in os.listdir(self.annotation_dir) if i in f]

        # debris free dataset (manually classified)
        self.debris_free_dataset_dir = debris_free_dataset_dir
        self.negative_imgs = natsorted([p.join(self.debris_free_dataset_dir, f)
                                        for f in os.listdir(self.debris_free_dataset_dir) if f.endswith('.png')])
        negative_img_ids = [f.split('-')[2].split('_')[0] for f in self.negative_imgs]
        self.img_ids += negative_img_ids
        # create an empty annotation image holder
        self.empty_annotation = np.zeros((resize_to[0], resize_to[1]))
        self.sample_ids += [(i, self.empty_annotation) for i in negative_img_ids]

        # Define the normalization transform
        self.normalize = transforms.Normalize((0.57784108, 0.5724125,  0.5619426),
                                              (0.24724819, 0.24302182, 0.23344601))
        self.resize_to = resize_to
        return

    def __len__(self):
        return len(self.sample_ids)

    def get_query_img(self, img_id):
        # img_file = p.join(self.original_image_dir, f'post-rgb-{img_id}_merged_50m.png')
        original_img_file = [f for f in self.original_imgs if img_id in f]
        negative_img_file = [f for f in self.negative_imgs if img_id in f]
        if original_img_file:
            img_file = original_img_file[0]
        elif negative_img_file:
            img_file = negative_img_file[0]
        else:
            raise FileNotFoundError(f'Image file for img_id {img_id} not found.')
        if img_file is None:
            raise FileNotFoundError(f'Image file for img_id {img_id} not found.')
        img = cv2.imread(img_file, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.resize_to)
        return img

    def get_annotation(self, annotation_file):
        if isinstance(annotation_file, np.ndarray):  # negative sample, in which case annotation_file is an empty array
            return annotation_file
        else:  # positive sample
            annotation_file = p.join(self.annotation_dir, annotation_file)
            annotation = cv2.imread(annotation_file, cv2.IMREAD_COLOR)
            annotation = cv2.cvtColor(annotation, cv2.COLOR_BGR2GRAY)
            annotation = cv2.resize(annotation, self.resize_to)  # resize
            # convert to binary mask with threshold at 100
            # _, annotation = cv2.threshold(annotation, 70, 255, cv2.THRESH_BINARY)
            return annotation

    @staticmethod
    def extract_prompt(annotation_file):
        if isinstance(annotation_file, np.ndarray):
            density = 'negative'
            prompt = 'a photo of no debris'
        else:
            # extract the density level from the annotation file
            density = annotation_file.split('_')[-2].split('.')[0].lower()
            prompt = f'a photo of debris at {density} density'
        return prompt, density

    def find_visual_prompt(self, density):
        # assuming self.with_visual is true
        if density == 'negative':
            other_annotated_files = [f for f in os.listdir(self.vis_prompt_dir)]
            # randomly select one of the other annotated files
            other_annotation_file = np.random.choice(other_annotated_files)
            vis_p_density = other_annotation_file.split('_')[-1].split('.')[0].lower()
            other_annotation_file = p.join(self.vis_prompt_dir, other_annotation_file)
            other_annotation = cv2.imread(other_annotation_file, cv2.IMREAD_COLOR)
            other_annotation = cv2.cvtColor(other_annotation, cv2.COLOR_BGR2RGB)
            other_annotation = cv2.resize(other_annotation, self.resize_to)  # resize
            override_prompt = 'a photo of debris at ' + vis_p_density + ' density'
        else:
            # find the corresponding visual image
            other_annotated_files = [f for f in os.listdir(self.vis_prompt_dir) if density in f.lower()]
            # randomly select one of the other annotated files
            other_annotation_file = np.random.choice(other_annotated_files)
            other_annotation_file = p.join(self.vis_prompt_dir, other_annotation_file)
            other_annotation = cv2.imread(other_annotation_file, cv2.IMREAD_COLOR)
            other_annotation = cv2.cvtColor(other_annotation, cv2.COLOR_BGR2RGB)
            other_annotation = cv2.resize(other_annotation, self.resize_to)  # resize
            override_prompt = None
        return other_annotation, override_prompt

    def __getitem__(self, i):
        img_id, annotation_file = self.sample_ids[i]
        img = self.get_query_img(img_id)
        annotation = self.get_annotation(annotation_file)
        prompt, label_density = self.extract_prompt(annotation_file)
        other_annotation, override_prompt = self.find_visual_prompt(label_density)
        label_add = [override_prompt] if override_prompt is not None else [prompt]  # used to be density
        masked_img_s = torch.tensor(np.array(other_annotation).transpose(2, 0, 1), dtype=torch.float32)
        masked_img_s = masked_img_s / 255.0
        masked_img_s = self.normalize(masked_img_s)
        vis_s = label_add + [masked_img_s, True]

        # convert to tensor
        img = torch.tensor(np.array(img).transpose(2, 0, 1), dtype=torch.float32)
        img = img / 255.0  # Scale pixel values to [0, 1]
        img = self.normalize(img)  # Normalize the image
        annotation = torch.tensor(np.array(annotation), dtype=torch.float32)
        annotation = annotation / 255.0
        annotation = annotation.unsqueeze(0)
        # return img, annotation, prompt

        data_x = (img, ) + tuple(vis_s)

        return data_x, (annotation, torch.zeros(0), i)

    def old_get_item(self, i):
        img_id, annotation_file = self.sample_ids[i]
        img_file = p.join(self.original_image_dir, f'post-rgb-{img_id}_merged_50m.png')
        img = cv2.imread(img_file, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.resize_to)  # resize
        annotation_file = p.join(self.annotation_dir, annotation_file)
        annotation = cv2.imread(annotation_file, cv2.IMREAD_COLOR)
        annotation = cv2.cvtColor(annotation, cv2.COLOR_BGR2GRAY)
        annotation = cv2.resize(annotation, self.resize_to)  # resize
        # convert to binary mask with threshold at 100
        _, annotation = cv2.threshold(annotation, 70, 255, cv2.THRESH_BINARY)

        # extract the density level from the annotation file
        density = annotation_file.split('_')[-1].split('.')[0].lower()
        prompt = f'a photo of debris at {density} density'

        # assuming self.with_visual is true
        # find the corresponding visual image
        other_annotated_files = [f for f in os.listdir(self.annotation_dir) if density in f.lower()]
        # randomly select one of the other annotated files
        other_annotation_file = np.random.choice(other_annotated_files)
        other_annotation_file = p.join(self.annotation_dir, other_annotation_file)
        other_annotation = cv2.imread(other_annotation_file, cv2.IMREAD_COLOR)
        other_annotation = cv2.cvtColor(other_annotation, cv2.COLOR_BGR2RGB)
        other_annotation = cv2.resize(other_annotation, self.resize_to)  # resize
        label_add = [prompt] # used to be density
        masked_img_s = torch.tensor(np.array(other_annotation).transpose(2, 0, 1), dtype=torch.float32)
        masked_img_s = masked_img_s / 255.0
        masked_img_s = self.normalize(masked_img_s)
        vis_s = label_add + [masked_img_s, True]

        # convert to tensor
        img = torch.tensor(np.array(img).transpose(2, 0, 1), dtype=torch.float32)
        img = img / 255.0  # Scale pixel values to [0, 1]
        img = self.normalize(img)  # Normalize the image
        annotation = torch.tensor(np.array(annotation), dtype=torch.float32)
        annotation = annotation / 255.0
        annotation = annotation.unsqueeze(0)
        # return img, annotation, prompt

        data_x = (img, ) + tuple(vis_s)

        return data_x, (annotation, torch.zeros(0), i)


# if __name__ == '__main__':
#     dataset_dir = '/home/yuhaoliu/Data/HIDeAI/LMH_labeled'
#     datset = DebrisDataset(dataset_dir)
#     datset[145]
