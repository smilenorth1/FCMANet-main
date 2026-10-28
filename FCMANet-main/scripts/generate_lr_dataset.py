"""
Reproduction script: Generate bicubic down‑sampled LR‑HR infrared image pairs
Corresponds to data preprocessing described in manuscript and README.md
This script generates low‑resolution images from original high‑resolution infrared images
for FCMANet infrared image super‑resolution training.
"""

import os
import argparse
import cv2
from PIL import Image
from tqdm import tqdm


def generate_lr_hr_pairs(hr_dir, save_hr_dir, save_lr_dir, scale=4):
    """
    Generate LR images by bicubic down‑sampling from HR source images
    Args:
        hr_dir: folder path of original high‑resolution infrared images
        save_hr_dir: output folder for copied HR ground‑truth images
        save_lr_dir: output folder for generated low‑resolution images
        scale: super‑resolution scale factor (x2 / x4)
    """
    os.makedirs(save_hr_dir, exist_ok=True)
    os.makedirs(save_lr_dir, exist_ok=True)

    img_list = [f for f in os.listdir(hr_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    for img_name in tqdm(img_list, desc="Generating LR‑HR pairs"):
        hr_path = os.path.join(hr_dir, img_name)
        hr_img = Image.open(hr_path).convert("RGB")
        w, h = hr_img.size

        # Save original HR ground truth
        hr_img.save(os.path.join(save_hr_dir, img_name))

        # Bicubic down‑sampling to obtain LR image
        lr_w, lr_h = w // scale, h // scale
        lr_img = hr_img.resize((lr_w, lr_h), resample=Image.BICUBIC)
        lr_img.save(os.path.join(save_lr_dir, img_name))

    print(f"Finished! Total processed images: {len(img_list)}")
    print(f"HR GT saved to: {save_hr_dir}")
    print(f"Generated LR saved to: {save_lr_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate bicubic down‑sampled LR‑HR infrared dataset pairs for FCMANet")
    parser.add_argument("--hr_source", type=str, default="../datasets/train/hr_source",
                        help="Directory path for original HR infrared images")
    parser.add_argument("--save_hr", type=str, default="../datasets/train/HR",
                        help="Output directory for HR ground‑truth images")
    parser.add_argument("--save_lr", type=str, default="../datasets/train/LR_bicubic_x4",
                        help="Output directory for bicubic LR images")
    parser.add_argument("--scale", type=int, default=4, choices=[2, 4],
                        help="Super‑resolution scale factor: 2 or 4")

    args = parser.parse_args()
    generate_lr_hr_pairs(
        hr_dir=args.hr_source,
        save_hr_dir=args.save_hr,
        save_lr_dir=args.save_lr,
        scale=args.scale
    )
