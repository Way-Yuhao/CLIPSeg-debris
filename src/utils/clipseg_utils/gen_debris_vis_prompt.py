import os
import sys
import cv2
import numpy as np
from PIL import Image
import json
from matplotlib import pyplot as plt
from natsort import natsorted
from tqdm import tqdm
import re


def apply_ev_comp_with_mask(img, mask, ev_comp, invert=False):
    img = img.astype(np.int32)
    # Ensure mask is in the correct format and stacked
    mask = cv2.merge([mask, mask, mask])
    if invert:
        mask = ~mask.astype(bool)
    else:
        mask = mask.astype(bool)
    # Create a copy of img to avoid modifying the original
    out = img.copy()
    # Apply exposure compensation
    out[mask] += ev_comp
    # Clip values to ensure they remain within 0-255
    np.clip(out, 0, 255, out=out)
    return out.astype(np.uint8)


def apply_blur_with_mask(img, mask, kernel_size, invert=True):
    # Ensure mask is in the correct format and stacked
    mask = cv2.merge([mask, mask, mask])
    if invert:
        mask = ~mask.astype(bool)
    else:
        mask = mask.astype(bool)
    # Create a copy of img to avoid modifying the original
    out = img.copy()
    # Apply blur
    out = cv2.GaussianBlur(out, (kernel_size, kernel_size), 0)
    out[~mask] = img[~mask]
    # Clip values to ensure they remain within 0-255
    np.clip(out, 0, 255, out=out)
    return out.astype(np.uint8)

def generate_vis_prompts(rgb_img_dir: str, segmentation_dir: str, vis_prompt_output_dir: str):
    assert os.path.exists(segmentation_dir), FileNotFoundError(f'Segmentation directory not found: {segmentation_dir}')
    os.makedirs(vis_prompt_output_dir, exist_ok=True)


    rgb_imgs = os.listdir(rgb_img_dir)
    rgb_imgs = natsorted([f for f in rgb_imgs if f.endswith('.png') and '._' not in f])
    annotation_files = os.listdir(segmentation_dir)
    annotation_files = natsorted([f for f in annotation_files if f.endswith('.png') and '._' not in f])

    for annotation_file in tqdm(annotation_files):
        # extract first integer from annotation file name
        match = re.search(r'\d+', annotation_file)
        if match:
            idx = match.group()
        else:
            raise ValueError(f'No integer found in annotation file name: {annotation_file}')
        img_file = [f for f in rgb_imgs if idx in f]
        assert len(img_file) == 1, FileNotFoundError(f'Image file not found for annotation file: {annotation_file}')
        img_file = img_file[0]

        img = cv2.imread(os.path.join(rgb_img_dir, img_file), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        annotation = cv2.imread(os.path.join(segmentation_dir, annotation_file), cv2.IMREAD_GRAYSCALE)
        # flip annotation
        annotation = cv2.bitwise_not(annotation)
        assert img is not None, FileNotFoundError(f'Image file not found.')
        assert annotation is not None, FileNotFoundError(f'Annotation file not found.')

        # apply adjustments
        img = img.copy()
        img = apply_blur_with_mask(img, annotation, 7, invert=False)
        img = apply_ev_comp_with_mask(img, annotation, -200, invert=False)
        # save altered image
        base_name = os.path.splitext(annotation_file)[0]  # Get the base name without extension
        modified_base_name = base_name.replace('_binary', '')  # Remove '_binary'
        output_file = os.path.join(vis_prompt_output_dir, f'{modified_base_name}.png')
        cv2.imwrite(output_file, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


if __name__ == '__main__':
    rgb_img_dir = '/home/yuhaoliu/Data/HIDeAI/merged_multi_labler/union/original'
    segmentation_dir = '/home/yuhaoliu/Data/HIDeAI/merged_multi_labler/union/segmentation_hl'
    vis_prompt_output_dir = '/home/yuhaoliu/Data/HIDeAI/merged_multi_labler/union/vis_prompts_hl'
    generate_vis_prompts(rgb_img_dir, segmentation_dir, vis_prompt_output_dir)