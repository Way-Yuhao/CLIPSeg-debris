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


def one_hot_encode_segmentation(seg_low: np.ndarray, seg_high: np.ndarray):
    """
    One-hot encode segmentation maps with background (no debris), low density, and high density.

    Args:
    - seg_low (np.ndarray): Low-density annotation mask.
    - seg_high (np.ndarray): High-density annotation mask.

    Returns:
    - one_hot (np.ndarray): One-hot encoded mask of shape [h, w, 3].
    """
    height, width = seg_low.shape

    # Initialize empty one-hot encoded mask: [h, w, 3] for 3 classes (background, low, high)
    one_hot = np.zeros((height, width, 3), dtype=np.uint8)

    # Background: set all pixels to background by default
    one_hot[:, :, 0] = 1

    # If seg_low exists, encode low-density pixels as [0, 1, 0]
    if seg_low is not None:
        one_hot[seg_low > 0, 0] = 0  # Remove background where there's low density
        one_hot[seg_low > 0, 1] = 1  # Set low density

    # If seg_high exists, encode high-density pixels as [0, 0, 1]
    if seg_high is not None:
        one_hot[seg_high > 0, 0] = 0  # Remove background where there's high density
        one_hot[seg_high > 0, 1] = 0  # Remove low density where there's high density
        one_hot[seg_high > 0, 2] = 1  # Set high density
    # convert to uint8, multiply by 255
    one_hot = (one_hot * 255).astype(np.uint8)
    return one_hot


def generated_merged_annotations(rgb_img_dir: str, segmentation_dir: str, merged_segmentation_output_dir: str):
    assert os.path.exists(segmentation_dir), FileNotFoundError(f'Segmentation directory not found: {segmentation_dir}')
    os.makedirs(merged_segmentation_output_dir, exist_ok=True)

    rgb_imgs = os.listdir(rgb_img_dir)
    rgb_imgs = natsorted([f for f in rgb_imgs if f.endswith('.png') and '._' not in f])
    annotation_files = os.listdir(segmentation_dir)
    annotation_files = natsorted([f for f in annotation_files if f.endswith('.png') and '._' not in f])
    indices = [re.search(r'\d+', rgb_img).group() for rgb_img in rgb_imgs]
    print(len(indices))

    for idx in tqdm(indices):
        rgb_img = [f for f in rgb_imgs if idx in f][0]
        rgb_img = cv2.imread(os.path.join(rgb_img_dir, rgb_img), cv2.IMREAD_COLOR)
        seg_low_path = os.path.join(segmentation_dir, f'post-rgb-{idx}_merged_50m_low_binary.png')
        seg_high_path = os.path.join(segmentation_dir, f'post-rgb-{idx}_merged_50m_high_binary.png')
        if os.path.exists(seg_low_path):
            seg_low = cv2.imread(seg_low_path, cv2.IMREAD_GRAYSCALE)
        else:
            seg_low = np.zeros(rgb_img.shape[:2], dtype=np.uint8)
        if os.path.exists(seg_high_path):
            seg_high = cv2.imread(seg_high_path, cv2.IMREAD_GRAYSCALE)
        else:
            seg_high = np.zeros(rgb_img.shape[:2], dtype=np.uint8)
        one_hot_segmentation = one_hot_encode_segmentation(seg_low, seg_high)

        # Save the one-hot encoded segmentation as a .npy file (or choose another format if preferred)
        # output_file = os.path.join(merged_segmentation_output_dir, f'post-rgb-{idx}_merged_onehot.npy')
        # np.save(output_file, one_hot_segmentation)
        # save as png
        output_file = os.path.join(merged_segmentation_output_dir, f'post-rgb-{idx}_merged_onehot.png')
        cv2.imwrite(output_file, one_hot_segmentation)

def generate_one_hot_for_negative(rgb_img_dir: str, seg_output_dir: str):
    os.makedirs(seg_output_dir, exist_ok=True)
    rgb_imgs = os.listdir(rgb_img_dir)
    rgb_imgs = natsorted([f for f in rgb_imgs if f.endswith('.png') and '._' not in f])
    indices = [re.search(r'\d+', rgb_img).group() for rgb_img in rgb_imgs]
    print(len(indices))
    for idx in tqdm(indices):
        # rgb_img = [f for f in rgb_imgs if idx in f][0]
        one_hot_segmentation = np.zeros((512, 512, 3), dtype=np.uint8)
        one_hot_segmentation[:, :, 0] = 255 # no debris
        output_file = os.path.join(seg_output_dir, f'post-rgb-{idx}_merged_onehot.png')
        cv2.imwrite(output_file, one_hot_segmentation)


def generate_one_hot_from_hw(hw_dir: str, seg_output_dir: str):
    os.makedirs(seg_output_dir, exist_ok=True)
    hw_segs = os.listdir(hw_dir)
    hw_segs = natsorted([f for f in hw_segs if f.endswith('.png') and '._' not in f])
    indices = [re.search(r'\d+', hw_seg).group() for hw_seg in hw_segs]
    print(len(indices))
    for idx in tqdm(indices):
        hw_segmentation = cv2.imread(os.path.join(hw_dir, f'post-rgb-{idx}_merged_50m.png'), cv2.IMREAD_GRAYSCALE) # shape [h, w]
        one_hot_segmentation = np.zeros((hw_segmentation.shape[0], hw_segmentation.shape[1], 3), dtype=np.uint8)
        # convert to one-hot
        one_hot_segmentation[hw_segmentation == 0, 0] = 255
        one_hot_segmentation[hw_segmentation == 1, 1] = 255
        one_hot_segmentation[hw_segmentation == 2, 2] = 255
        output_file = os.path.join(seg_output_dir, f'post-rgb-{idx}_merged_onehot.png')
        cv2.imwrite(output_file, one_hot_segmentation)


if __name__ == '__main__':
    # rgb_img_dir = '/scratch/yl241/data/HIDeAI/multi_labeler_onehot/majority_vote_no_negative/original'
    # segmentation_dir = '/home/yuhaoliu/Data/HIDeAI/multi_labeler_onehot/kooshan_no_negative/segmentation_hl'
    # vis_prompt_output_dir = '/home/yuhaoliu/Data/HIDeAI/multi_labeler_onehot/kooshan_no_negative/vis_prompts_hl'
    # merged_annotations = '/scratch/yl241/data/HIDeAI/multi_labeler_onehot/majority_vote_no_negative/segmentation_merged'
    # hw_annotations = '/scratch/yl241/data/HIDeAI/multi_labeler_onehot/majority_vote_no_negative/segmentation_hw'
    # generated_merged_annotations(rgb_img_dir, segmentation_dir, merged_annotations)
    # generate_vis_prompts(rgb_img_dir, segmentation_dir, vis_prompt_output_dir)
    # generate_one_hot_from_hw(hw_annotations, merged_annotations)

    negative_rgb_dir = '/scratch/yl241/data/HIDeAI/negative_734/rgb'
    negative_seg_dir = '/scratch/yl241/data/HIDeAI/negative_734/one_hot_seg'
    generate_one_hot_for_negative(negative_rgb_dir, negative_seg_dir)
