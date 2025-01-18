import glob
import os
import sys
from pathlib import Path

import numpy as np
from color_constancy import color_constancy
from joblib import Parallel, delayed
from PIL import Image, ImageOps
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))
sys.path.append(parent_dir)

from flamby.utils import read_config, write_value_in_config

dir = str(Path(os.path.realpath(__file__)).parent.resolve())
config_file = os.path.join(dir, "dataset_location.yaml")
dict_config = read_config(config_file)  # renamed variable to avoid overshadowing built-in dict
if not (dict_config["download_complete"]):
    raise ValueError("Download incomplete. Please relaunch the download script")
if dict_config["preprocessing_complete"]:
    print("You have already ran the preprocessing, aborting.")
    sys.exit()
input_path = dict_config["dataset_path"]

dic = {
    "inputs": "ISIC_2019_Training_Input",
    "inputs_preprocessed": "ISIC_2019_Training_Input_preprocessed",
}
input_folder = os.path.join(input_path, dic["inputs"])
output_folder = os.path.join(input_path, dic["inputs_preprocessed"])
os.makedirs(output_folder, exist_ok=True)

def resize_crop_pad(path, output_path, target_size: int, cc: bool):
    """Preprocess image to fixed 224x224 size.
    
    Steps:
    1. Resize the image so that the shorter side equals target_size while maintaining aspect ratio.
    2. If the resized image is larger than target_size in both dimensions, perform a center crop to target_size×target_size.
    3. If the resized image is smaller than target_size in any dimension, pad it to reach target_size×target_size.
    4. Optionally apply color constancy.
    """
    fn = os.path.basename(path)
    img = Image.open(path).convert("RGB")  # ensure image is in RGB format
    original_size = img.size  # (width, height)
    
    # 1. Resize to maintain aspect ratio with shorter side = target_size
    short_edge = min(original_size)
    ratio = float(target_size) / short_edge
    new_size = tuple([int(dim * ratio) for dim in original_size])
    img = img.resize(new_size, resample=Image.BILINEAR)

    # 2. Crop to 224x224 if possible
    width, height = img.size
    if width >= target_size and height >= target_size:
        # Center crop
        left = (width - target_size) // 2
        top = (height - target_size) // 2
        right = left + target_size
        bottom = top + target_size
        img = img.crop((left, top, right, bottom))
    else:
        # 3. Pad if needed (padding has lower priority than cropping)
        # Calculate required padding on each side to reach target_size
        pad_width = max(target_size - width, 0)
        pad_height = max(target_size - height, 0)
        
        # Distribute padding equally on both sides
        padding = (
            pad_width // 2,             # left
            pad_height // 2,            # top
            pad_width - pad_width // 2, # right
            pad_height - pad_height // 2# bottom
        )
        img = ImageOps.expand(img, border=padding, fill=0)  # using black for padding

        # In case padding overshoots due to odd differences, crop final to exact size
        width, height = img.size
        if width > target_size or height > target_size:
            left = (width - target_size) // 2
            top = (height - target_size) // 2
            right = left + target_size
            bottom = top + target_size
            img = img.crop((left, top, right, bottom))

    # 4. Apply color constancy if needed
    if cc:
        img_np = np.array(img)
        img_np = color_constancy(img_np)
        img = Image.fromarray(img_np)

    # Save final image
    img.save(os.path.join(output_path, fn))

if __name__ == "__main__":

    target_size = 224
    cc = True

    images = glob.glob(os.path.join(input_folder, "*.jpg"))

    print(f"Resizing, cropping, and padding images to fixed size {target_size}x{target_size} px.")

    Parallel(n_jobs=32)(
        delayed(resize_crop_pad)(i, output_folder, target_size, cc)
        for i in tqdm(images)
    )

    write_value_in_config(config_file, "preprocessing_complete", True)
