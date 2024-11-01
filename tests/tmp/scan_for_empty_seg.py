import os
import cv2
from natsort import natsorted

def scan_dir(path: str = None):
    files = os.listdir(path)
    files = [f for f in files if f.endswith('.png')]
    files = natsorted([f for f in files if '._' not in f])
    print(f'Found {len(files)} images')
    empty_segs = []
    for f in files:
        if '002470' in f:
            pass
        img = cv2.imread(os.path.join(path, f))
        assert img is not None, f'Failed to read image: {f}'
        no_debris = img[:, :, 0]
        low_debris = img[:, :, 1]
        high_debris = img[:, :, 2]
        if low_debris.sum() == 0 and high_debris.sum() == 0:
            empty_segs.append(f)
    print(f'Found {len(empty_segs)} empty segmentations')
    print(empty_segs)

if __name__ == '__main__':
    union_no_negative = '/scratch/yl241/data/HIDeAI/multi_labeler_onehot/majority_vote_no_negative/segmentation_merged/'
    scan_dir(union_no_negative)