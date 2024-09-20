import os
import shutil
import random
from PIL import Image

# Specify the source directories for each annotator
annotator_dirs = {
    'Annotator_4': r'./test_data_kooshan/raw_dataset/Annotator_4/low',
    'Annotator_5': r'./test_data_kooshan/raw_dataset/Annotator_5/low',
    'Annotator_6': r'./test_data_kooshan/raw_dataset/Annotator_6/low'
}

# Directory for original images
image_dir = r'./test_data_kooshan/raw_dataset/Image'

# Specify the destination root directory where the train, test, and validate folders will be created
destination_root = r'./test_data_kooshan'

# Create the destination subdirectories if they don't exist
for subset in ['train', 'test', 'validate']:
    os.makedirs(os.path.join(destination_root, subset, 'Image'), exist_ok=True)
    for annotator in annotator_dirs.keys():
        os.makedirs(os.path.join(destination_root, subset, annotator), exist_ok=True)

# Get the list of files common to all annotators and match them with original images
annotator_files = set(os.listdir(annotator_dirs['Annotator_4'])) \
                  & set(os.listdir(annotator_dirs['Annotator_5'])) \
                  & set(os.listdir(annotator_dirs['Annotator_6']))

# Remove "_low_binary" suffix to match original image filenames
image_files = set(f.replace('_low_binary', '') for f in annotator_files)

# Filter out files that are not images (optional, adjust the extensions as needed)
image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
filtered_image_files = [f for f in image_files if any(f.lower().endswith(ext) for ext in image_extensions)]

# Randomly shuffle and split the data into train, test, and validate sets
random.shuffle(filtered_image_files)
total_files = len(filtered_image_files)
train_split = int(0.7 * total_files)
test_split = int(0.2 * total_files)

train_files = filtered_image_files[:train_split]
test_files = filtered_image_files[train_split:train_split + test_split]
validate_files = filtered_image_files[train_split + test_split:]

# Function to resize and copy files to the respective directories
def resize_and_copy_files(files, subset, annotator=None):
    for file in files:
        if annotator:
            # Annotator files include the "_low_binary" suffix
            annotator_file = file.replace('.jpg', '_low_binary.jpg').replace('.jpeg', '_low_binary.jpeg').replace('.png', '_low_binary.png')
            src_path = os.path.join(annotator_dirs[annotator], annotator_file)
            dest_path = os.path.join(destination_root, subset, annotator, annotator_file)
        else:
            # Original images do not include the "_low_binary" suffix
            src_path = os.path.join(image_dir, file)
            dest_path = os.path.join(destination_root, subset, 'Image', file)

        # Open the image, resize it to 256x256, and save it to the destination path
        with Image.open(src_path) as img:
            resized_img = img.resize((256, 256), Image.LANCZOS)
            resized_img.save(dest_path)

# Copy and resize the files for each annotator and the original images
for annotator in annotator_dirs.keys():
    resize_and_copy_files(train_files, 'train', annotator)
    resize_and_copy_files(test_files, 'test', annotator)
    resize_and_copy_files(validate_files, 'validate', annotator)

# Copy and resize the original images
resize_and_copy_files(train_files, 'train')
resize_and_copy_files(test_files, 'test')
resize_and_copy_files(validate_files, 'validate')

print(f"Files have been resized to 256x256 and copied to train, test, and validate folders for each annotator and original images.")
