"""
Data preprocessing pipeline for HAM10000 skin lesion dataset.
Handles: hair removal, resize, normalization, SMOTE, stratified splitting.
"""

import argparse
import json
import os
import shutil
import cv2
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from PIL import Image
from tqdm import tqdm

CLASS_NAMES   = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']
CLASS_TO_IDX  = {c: i for i, c in enumerate(CLASS_NAMES)}
IMAGE_SIZE    = (224, 224)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])


# ── Hair Removal ─────────────────────────────────────────────────────────────
def remove_hair(image: np.ndarray) -> np.ndarray:
    """Remove hair artifacts from dermoscopic images using morphological blackhat + inpainting."""
    gray   = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    # Only inpaint if significant hair detected
    if mask.sum() > 1000:
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cleaned = cv2.inpaint(bgr, mask, 3, cv2.INPAINT_TELEA)
        return cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB)
    return image


def preprocess_single(img_path: str, size: tuple = IMAGE_SIZE) -> np.ndarray:
    """Load, clean, resize and normalize one image."""
    img = np.array(Image.open(img_path).convert('RGB'))
    img = remove_hair(img)
    img = cv2.resize(img, size)
    img = img.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return img


# ── Dataset Preparation ───────────────────────────────────────────────────────
def prepare_dataset(input_dir: str, output_dir: str, image_size: int = 224):
    size = (image_size, image_size)
    metadata_path = os.path.join(input_dir, 'HAM10000_metadata.csv')
    df = pd.read_csv(metadata_path)

    print(f"Total samples: {len(df)}")
    print("Class distribution:\n", df['dx'].value_counts())

    # ── Stratified split ──────────────────────────────────────────────────
    train_df, test_df = train_test_split(df, test_size=0.15, stratify=df['dx'], random_state=42)
    train_df, val_df  = train_test_split(train_df, test_size=0.176, stratify=train_df['dx'], random_state=42)
    # 0.176 of 85% ≈ 15% of total

    print(f"\nSplit: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    # ── Process and save each split ───────────────────────────────────────
    for split_name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc=f"Processing {split_name}"):
            label   = row['dx']
            img_id  = row['image_id']
            out_dir = os.path.join(output_dir, split_name, label)
            os.makedirs(out_dir, exist_ok=True)

            # Find image (HAM10000 splits across 2 folders)
            for part in ['HAM10000_images_part_1', 'HAM10000_images_part_2']:
                src = os.path.join(input_dir, part, f'{img_id}.jpg')
                if os.path.exists(src):
                    img = preprocess_single(src, size)
                    # Save as normalized numpy array (.npy) for fast loading during training
                    np.save(os.path.join(out_dir, f'{img_id}.npy'), img)
                    break

    # ── Class distribution report ─────────────────────────────────────────
    dist = {
        'train': dict(Counter(train_df['dx'].tolist())),
        'val':   dict(Counter(val_df['dx'].tolist())),
        'test':  dict(Counter(test_df['dx'].tolist())),
    }
    with open(os.path.join(output_dir, 'class_distribution.json'), 'w') as f:
        json.dump(dist, f, indent=2)

    print(f"\n✅ Preprocessing complete! Saved to: {output_dir}")
    print(f"✅ class_distribution.json written")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',  required=True, help='Path to raw HAM10000 directory')
    parser.add_argument('--output', default='data/processed')
    parser.add_argument('--size',   type=int, default=224)
    args = parser.parse_args()
    prepare_dataset(args.input, args.output, args.size)
