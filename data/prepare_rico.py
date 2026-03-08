"""
Prepare the Rico dataset for GUI element detection in YOLO format.

Downloads two files from interactionmining.org:
  1. unique_uis.tar.gz (6 GB) — 66K screenshots + view hierarchies
  2. semantic_annotations.zip (150 MB) — componentLabel annotations

Then merges them: screenshots from (1) + componentLabel from (2),
and converts to YOLO annotation format.

Rico images are 1440x2560 pixels (Android screenshots).

Usage:
    # Download + convert in one step:
    python prepare_rico.py --download --convert

    # Download only:
    python prepare_rico.py --download

    # Convert only (if already downloaded):
    python prepare_rico.py --convert
"""

import os
import sys
import json
import shutil
import random
import argparse
import zipfile
import tarfile
from pathlib import Path
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
from class_mapping import UNIFIED_CLASSES, map_rico_label

# Rico screenshot resolution
RICO_WIDTH = 1440
RICO_HEIGHT = 2560

# Minimum bounding box area (in pixels) to keep
MIN_AREA = 100

# Train/Val/Test split ratios
SPLIT_RATIOS = (0.8, 0.1, 0.1)

SEED = 42

# Direct download URLs from Google Cloud Storage
SCREENSHOTS_URL = (
    "https://storage.googleapis.com/crowdstf-rico-uiuc-4540/"
    "rico_dataset_v0.1/unique_uis.tar.gz"
)
SEMANTIC_URL = (
    "https://storage.googleapis.com/crowdstf-rico-uiuc-4540/"
    "rico_dataset_v0.1/semantic_annotations.zip"
)

# Base output directory (relative to this script)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RICO_DIR = os.path.join(BASE_DIR, "rico")


def download_file(url, dest_path, description=""):
    """Download a file with progress bar."""
    import requests

    if os.path.exists(dest_path):
        print(f"  Already exists: {dest_path}")
        return

    print(f"  Downloading {description}...")
    print(f"  URL: {url}")

    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()

    total = int(resp.headers.get('content-length', 0))
    downloaded = 0

    with open(dest_path, 'wb') as f:
        for chunk in resp.iter_content(chunk_size=65536):
            f.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                pct = downloaded / total * 100
                mb = downloaded / 1024 / 1024
                total_mb = total / 1024 / 1024
                print(f"\r  {pct:.1f}% ({mb:.0f}/{total_mb:.0f} MB)", end="",
                      flush=True)
    print()


def download_rico():
    """Download Rico screenshots and semantic annotations."""
    os.makedirs(RICO_DIR, exist_ok=True)

    screenshots_dir = os.path.join(RICO_DIR, "combined")
    semantic_dir = os.path.join(RICO_DIR, "semantic_annotations")

    # Check if already extracted
    if (os.path.isdir(screenshots_dir)
            and len(list(Path(screenshots_dir).glob("*.jpg"))) > 1000):
        print(f"Screenshots already extracted ({screenshots_dir})")
    else:
        # Download and extract screenshots
        tar_path = os.path.join(RICO_DIR, "unique_uis.tar.gz")
        try:
            download_file(SCREENSHOTS_URL, tar_path,
                          "Rico screenshots (6 GB)")
            print("  Extracting screenshots...")
            with tarfile.open(tar_path, 'r:gz') as tar:
                tar.extractall(RICO_DIR)
            print(f"  Extracted to {RICO_DIR}")
        except Exception as e:
            print(f"\n  Download failed: {e}")
            print(f"\n  MANUAL DOWNLOAD:")
            print(f"  1. Download from: {SCREENSHOTS_URL}")
            print(f"  2. Extract to: {RICO_DIR}/")
            if not os.path.isdir(screenshots_dir):
                return False

    # Check if semantic annotations already extracted
    if (os.path.isdir(semantic_dir)
            and len(list(Path(semantic_dir).glob("*.json"))) > 1000):
        print(f"Semantic annotations already extracted ({semantic_dir})")
    else:
        # Download and extract semantic annotations
        zip_path = os.path.join(RICO_DIR, "semantic_annotations.zip")
        try:
            download_file(SEMANTIC_URL, zip_path,
                          "Semantic annotations (150 MB)")
            print("  Extracting annotations...")
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(RICO_DIR)
            print(f"  Extracted to {RICO_DIR}")
        except Exception as e:
            print(f"\n  Download failed: {e}")
            print(f"\n  MANUAL DOWNLOAD:")
            print(f"  1. Download from: {SEMANTIC_URL}")
            print(f"  2. Extract to: {RICO_DIR}/")
            if not os.path.isdir(semantic_dir):
                return False

    # Report what we have
    n_jpg = len(list(Path(screenshots_dir).glob("*.jpg"))) if os.path.isdir(screenshots_dir) else 0
    n_json_combined = len(list(Path(screenshots_dir).glob("*.json"))) if os.path.isdir(screenshots_dir) else 0
    n_json_semantic = len(list(Path(semantic_dir).glob("*.json"))) if os.path.isdir(semantic_dir) else 0

    print(f"\nRico data downloaded:")
    print(f"  Screenshots (.jpg):              {n_jpg}")
    print(f"  View hierarchies (combined/):    {n_json_combined}")
    print(f"  Semantic annotations:            {n_json_semantic}")

    return True


def extract_elements_from_tree(node, elements=None):
    """
    Recursively traverse Rico view hierarchy tree.
    Extract elements that have a valid componentLabel.

    Returns list of (componentLabel, [left, top, right, bottom])
    """
    if elements is None:
        elements = []

    if not isinstance(node, dict):
        return elements

    label = node.get("componentLabel", "")
    bounds = node.get("bounds", None)

    if label and bounds and len(bounds) == 4:
        left, top, right, bottom = bounds
        width = right - left
        height = bottom - top
        area = width * height

        # Filter: valid bounds, minimum area, within screen
        if (width > 0 and height > 0
                and area >= MIN_AREA
                and left >= 0 and top >= 0
                and right <= RICO_WIDTH and bottom <= RICO_HEIGHT):
            elements.append((label, bounds))

    # Recurse into children
    for child in node.get("children", []):
        extract_elements_from_tree(child, elements)

    return elements


def bounds_to_yolo(bounds, img_w=RICO_WIDTH, img_h=RICO_HEIGHT):
    """Convert [left, top, right, bottom] pixel bounds to YOLO normalized format."""
    left, top, right, bottom = bounds
    center_x = max(0.0, min(1.0, (left + right) / 2.0 / img_w))
    center_y = max(0.0, min(1.0, (top + bottom) / 2.0 / img_h))
    width = max(0.0, min(1.0, (right - left) / img_w))
    height = max(0.0, min(1.0, (bottom - top) / img_h))
    return center_x, center_y, width, height


def convert_rico_to_yolo():
    """
    Convert Rico data to YOLO format.

    Looks for semantic annotations first (have componentLabel).
    Falls back to view hierarchies in combined/ if no semantic annotations.
    """
    screenshots_dir = os.path.join(RICO_DIR, "combined")
    semantic_dir = os.path.join(RICO_DIR, "semantic_annotations")
    output_dir = os.path.join(BASE_DIR, "unified", "rico")

    # Determine annotation source
    # Priority: semantic_annotations/ (has componentLabel) > combined/ .json
    if os.path.isdir(semantic_dir):
        json_dir = semantic_dir
        json_files = sorted(Path(semantic_dir).glob("*.json"))
        print(f"Using semantic annotations: {len(json_files)} files")
    else:
        json_dir = screenshots_dir
        json_files = sorted(Path(screenshots_dir).glob("*.json"))
        print(f"Using view hierarchies from combined/: {len(json_files)} files")

    if not json_files:
        print("ERROR: No JSON annotation files found.")
        print(f"  Checked: {semantic_dir}")
        print(f"  Checked: {screenshots_dir}")
        print("  Run with --download first.")
        sys.exit(1)

    # Check for screenshots
    if not os.path.isdir(screenshots_dir):
        print(f"ERROR: Screenshots directory not found: {screenshots_dir}")
        sys.exit(1)

    screen_data = []
    class_counter = Counter()
    skipped_labels = Counter()
    total_elements = 0
    no_image = 0

    for i, json_path in enumerate(json_files):
        screen_id = json_path.stem

        # Find corresponding screenshot
        img_path = None
        for ext in [".jpg", ".jpeg", ".png"]:
            candidate = Path(screenshots_dir) / f"{screen_id}{ext}"
            if candidate.exists():
                img_path = candidate
                break

        if img_path is None:
            no_image += 1
            continue

        # Parse JSON
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue

        # Extract elements: handle both tree format and flat format
        elements = []
        if "children" in data or "componentLabel" in data:
            # Tree format (view hierarchy with semantic annotations)
            elements = extract_elements_from_tree(data)
        elif "screen_elements" in data:
            # Flat format (Rico Semantics from Google Research)
            for elem in data["screen_elements"]:
                label = elem.get("componentLabel", elem.get("component_label", ""))
                bounds = elem.get("bounds", None)
                if label and bounds:
                    # Handle normalized [0,1] bounds
                    if all(0 <= b <= 1 for b in bounds):
                        bounds = [
                            bounds[0] * RICO_WIDTH,
                            bounds[1] * RICO_HEIGHT,
                            bounds[2] * RICO_WIDTH,
                            bounds[3] * RICO_HEIGHT,
                        ]
                    elements.append((label, bounds))

        # Map to unified classes
        yolo_annotations = []
        for label, bounds in elements:
            unified_name, class_id = map_rico_label(label)
            if class_id is not None:
                cx, cy, w, h = bounds_to_yolo(bounds)
                if w > 0 and h > 0:
                    yolo_annotations.append((class_id, cx, cy, w, h))
                    class_counter[unified_name] += 1
                    total_elements += 1
            else:
                skipped_labels[label] += 1

        if yolo_annotations:
            screen_data.append({
                "screen_id": screen_id,
                "img_path": str(img_path),
                "annotations": yolo_annotations,
            })

        if (i + 1) % 5000 == 0:
            print(f"  Processed {i + 1}/{len(json_files)} files...")

    print(f"\nProcessed {len(json_files)} annotation files")
    print(f"Screens with matching images: {len(json_files) - no_image}")
    print(f"Screens with valid annotations: {len(screen_data)}")
    print(f"Total elements extracted: {total_elements}")

    if not screen_data:
        print("\nERROR: No valid annotations found.")
        print("The JSON files may not contain componentLabel fields.")
        print("Make sure you downloaded the SEMANTIC annotations version.")
        sys.exit(1)

    print(f"\nClass distribution:")
    for cls in UNIFIED_CLASSES:
        count = class_counter.get(cls, 0)
        if count > 0:
            print(f"  {cls:15s}: {count:>6d}")

    if skipped_labels:
        print(f"\nSkipped labels (not mapped):")
        for label, count in skipped_labels.most_common(10):
            print(f"  {label:20s}: {count:>6d}")

    # Create train/val/test splits
    random.seed(SEED)
    random.shuffle(screen_data)

    n = len(screen_data)
    n_train = int(n * SPLIT_RATIOS[0])
    n_val = int(n * SPLIT_RATIOS[1])

    splits = {
        "train": screen_data[:n_train],
        "val": screen_data[n_train:n_train + n_val],
        "test": screen_data[n_train + n_val:],
    }

    for split_name, split_data in splits.items():
        img_dir = os.path.join(output_dir, split_name, "images")
        lbl_dir = os.path.join(output_dir, split_name, "labels")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        for item in split_data:
            screen_id = item["screen_id"]
            src_img = item["img_path"]
            ext = Path(src_img).suffix

            dst_img = os.path.join(img_dir, f"rico_{screen_id}{ext}")
            dst_lbl = os.path.join(lbl_dir, f"rico_{screen_id}.txt")

            if not os.path.exists(dst_img):
                shutil.copy2(src_img, dst_img)

            with open(dst_lbl, 'w') as f:
                for class_id, cx, cy, w, h in item["annotations"]:
                    f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

        print(f"\n{split_name}: {len(split_data)} images → {img_dir}")

    print(f"\nRico YOLO dataset saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare Rico dataset for GUI element detection (YOLO format)")

    parser.add_argument("--download", action="store_true",
                        help="Download Rico screenshots + semantic annotations")
    parser.add_argument("--convert", action="store_true",
                        help="Convert Rico to YOLO format")

    args = parser.parse_args()

    if not args.download and not args.convert:
        print("Specify --download, --convert, or both. Use -h for help.")
        sys.exit(1)

    if args.download:
        success = download_rico()
        if not success and args.convert:
            print("\nDownload incomplete. Fix issues above and retry.")
            sys.exit(1)

    if args.convert:
        convert_rico_to_yolo()


if __name__ == "__main__":
    main()
