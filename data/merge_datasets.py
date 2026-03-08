"""
Merge Rico and WebUI YOLO datasets into a single unified dataset.

Creates a combined dataset with balanced sampling and generates
the data.yaml configuration file for YOLO training.

Usage:
    # Merge Rico only (if WebUI not yet available):
    python merge_datasets.py --rico_dir ./unified/rico

    # Merge both Rico and WebUI:
    python merge_datasets.py --rico_dir ./unified/rico --webui_dir ./unified/webui

    # Custom output:
    python merge_datasets.py --rico_dir ./unified/rico --output_dir ./unified/combined
"""

import os
import sys
import shutil
import random
import argparse
from pathlib import Path
from collections import Counter

import yaml

sys.path.insert(0, os.path.dirname(__file__))
from class_mapping import UNIFIED_CLASSES

SEED = 42


def count_annotations(labels_dir):
    """Count annotations per class in a YOLO labels directory."""
    counter = Counter()
    total_images = 0
    total_annotations = 0

    for txt_file in Path(labels_dir).glob("*.txt"):
        total_images += 1
        with open(txt_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    if 0 <= class_id < len(UNIFIED_CLASSES):
                        counter[UNIFIED_CLASSES[class_id]] += 1
                        total_annotations += 1

    return counter, total_images, total_annotations


def merge_split(source_dirs, output_dir, split_name):
    """
    Merge a specific split (train/val/test) from multiple source datasets.

    Copies images and labels from each source into the combined output.
    """
    out_img_dir = os.path.join(output_dir, split_name, "images")
    out_lbl_dir = os.path.join(output_dir, split_name, "labels")
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    copied = 0
    for src_dir in source_dirs:
        src_img_dir = os.path.join(src_dir, split_name, "images")
        src_lbl_dir = os.path.join(src_dir, split_name, "labels")

        if not os.path.isdir(src_img_dir):
            print(f"  Warning: {src_img_dir} not found, skipping")
            continue

        for img_file in Path(src_img_dir).iterdir():
            if img_file.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
                lbl_file = Path(src_lbl_dir) / f"{img_file.stem}.txt"

                if lbl_file.exists():
                    dst_img = os.path.join(out_img_dir, img_file.name)
                    dst_lbl = os.path.join(out_lbl_dir, lbl_file.name)

                    if not os.path.exists(dst_img):
                        shutil.copy2(str(img_file), dst_img)
                        shutil.copy2(str(lbl_file), dst_lbl)
                        copied += 1

    return copied


def generate_data_yaml(output_dir):
    """Generate data.yaml for YOLO training."""
    yaml_data = {
        "path": os.path.abspath(output_dir),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(UNIFIED_CLASSES),
        "names": UNIFIED_CLASSES,
    }

    yaml_path = os.path.join(output_dir, "data.yaml")
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_data, f, default_flow_style=False,
                  allow_unicode=True, sort_keys=False)

    return yaml_path


def main():
    parser = argparse.ArgumentParser(
        description="Merge GUI detection datasets and generate data.yaml")

    parser.add_argument("--rico_dir", type=str, default=None,
                        help="Path to Rico YOLO dataset (from prepare_rico.py)")
    parser.add_argument("--webui_dir", type=str, default=None,
                        help="Path to WebUI YOLO dataset (from prepare_webui.py)")
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(os.path.dirname(__file__),
                                             "unified", "combined"),
                        help="Output directory for merged dataset")

    args = parser.parse_args()

    # Collect source directories
    source_dirs = []
    if args.rico_dir and os.path.isdir(args.rico_dir):
        source_dirs.append(args.rico_dir)
        print(f"Rico source: {args.rico_dir}")
    if args.webui_dir and os.path.isdir(args.webui_dir):
        source_dirs.append(args.webui_dir)
        print(f"WebUI source: {args.webui_dir}")

    # Default paths if none specified
    if not source_dirs:
        default_rico = os.path.join(os.path.dirname(__file__), "unified", "rico")
        default_webui = os.path.join(os.path.dirname(__file__), "unified", "webui")

        if os.path.isdir(default_rico):
            source_dirs.append(default_rico)
            print(f"Rico source (default): {default_rico}")
        if os.path.isdir(default_webui):
            source_dirs.append(default_webui)
            print(f"WebUI source (default): {default_webui}")

    if not source_dirs:
        print("ERROR: No source datasets found.")
        print("Run prepare_rico.py and/or prepare_webui.py first.")
        sys.exit(1)

    output_dir = args.output_dir
    print(f"\nOutput directory: {output_dir}")
    print(f"Unified classes ({len(UNIFIED_CLASSES)}): {UNIFIED_CLASSES}")

    # Merge each split
    print("\nMerging datasets...")
    for split in ["train", "val", "test"]:
        n = merge_split(source_dirs, output_dir, split)
        print(f"  {split}: {n} images copied")

    # Generate data.yaml
    yaml_path = generate_data_yaml(output_dir)
    print(f"\ndata.yaml saved to: {yaml_path}")

    # Print statistics
    print("\n" + "=" * 50)
    print("DATASET STATISTICS")
    print("=" * 50)

    total_images = 0
    total_annotations = 0
    global_counter = Counter()

    for split in ["train", "val", "test"]:
        lbl_dir = os.path.join(output_dir, split, "labels")
        if os.path.isdir(lbl_dir):
            counter, n_imgs, n_annots = count_annotations(lbl_dir)
            global_counter += counter
            total_images += n_imgs
            total_annotations += n_annots
            print(f"\n{split}:")
            print(f"  Images: {n_imgs}")
            print(f"  Annotations: {n_annots}")

    print(f"\nTotal: {total_images} images, {total_annotations} annotations")
    print(f"\nClass distribution (all splits):")
    for i, cls in enumerate(UNIFIED_CLASSES):
        count = global_counter.get(cls, 0)
        pct = count / total_annotations * 100 if total_annotations > 0 else 0
        bar = "#" * int(pct / 2)
        print(f"  [{i:2d}] {cls:15s}: {count:>7d} ({pct:5.1f}%) {bar}")

    print(f"\nDone! To train YOLO:")
    print(f"  from ultralytics import YOLO")
    print(f"  model = YOLO('yolo11n.pt')")
    print(f"  model.train(data='{yaml_path}', epochs=60, imgsz=640)")


if __name__ == "__main__":
    main()
