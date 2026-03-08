"""
Prepare the WebUI dataset for GUI element detection in YOLO format.

Processes the WebUI dataset (CHI 2023) which contains web UI screenshots
at multiple viewports with bounding boxes extracted from accessibility trees.

Prerequisites:
    1. Download the WebUI dataset from https://uimodeling.github.io/
    2. Run the generate_dataset_web.py script from the WebUI repo to produce
       processed JSON files with labels + contentBoxes
    OR
    3. Use the raw data directly (axtree.json.gz + screenshot files)

Usage:
    # From processed WebUI data (labels + contentBoxes JSON files):
    python prepare_webui.py --webui_dir /path/to/webui/processed --output_dir ./unified/webui

    # From raw WebUI sample/crawl data:
    python prepare_webui.py --raw --webui_dir /path/to/webui/data --output_dir ./unified/webui
"""

import os
import sys
import json
import gzip
import shutil
import random
import argparse
from pathlib import Path
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
from class_mapping import UNIFIED_CLASSES, map_webui_label

# Minimum bounding box area in pixels
MIN_AREA = 100

# Train/Val/Test split ratios
SPLIT_RATIOS = (0.8, 0.1, 0.1)

SEED = 42

# WebUI viewport dimensions (device → (width, height, scale))
VIEWPORT_SIZES = {
    "default_1280-720":  (1280, 720, 1),
    "default_1366-768":  (1366, 768, 1),
    "default_1536-864":  (1536, 864, 1),
    "default_1920-1080": (1920, 1080, 1),
    "iPad-Pro":          (1024, 1366, 2),
    "iPhone-13 Pro":     (390, 844, 3),
}


def load_gzip_json(path):
    """Load a gzip-compressed JSON file."""
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        return json.load(f)


def extract_elements_from_raw(axtree_data, bb_data):
    """
    Extract GUI elements by joining axtree nodes with bounding boxes.

    axtree_data: dict with 'nodes' list (Chrome DevTools Protocol format).
                 Each node has 'role', 'backendDOMNodeId', etc.
    bb_data:     dict mapping DOM node ID (str) → {x, y, width, height}.

    Returns list of (role_value, [x1, y1, x2, y2]).
    """
    elements = []

    nodes = axtree_data.get("nodes", [])
    if not isinstance(nodes, list):
        return elements

    for node in nodes:
        if node.get("ignored", False):
            continue

        role_info = node.get("role", {})
        role_value = role_info.get("value", "") if isinstance(role_info, dict) else ""
        if not role_value:
            continue

        # Join with bounding box via backendDOMNodeId
        dom_id = str(node.get("backendDOMNodeId", ""))
        if dom_id not in bb_data:
            continue

        box = bb_data[dom_id]
        if not isinstance(box, dict):
            continue
        x = box.get("x", 0)
        y = box.get("y", 0)
        w = box.get("width", 0)
        h = box.get("height", 0)

        if w > 0 and h > 0:
            elements.append((role_value, [x, y, x + w, y + h]))

    return elements


def convert_processed_webui(webui_dir, output_dir):
    """
    Convert pre-processed WebUI data to YOLO format.

    Expects JSON files with structure:
    {
        "labels": [["button"], ["link", "navigation"], ...],
        "contentBoxes": [[x1, y1, x2, y2], ...],
        "key_name": "device-resolution"
    }
    """
    json_files = sorted(Path(webui_dir).glob("**/*.json"))
    print(f"Found {len(json_files)} JSON annotation files")

    screen_data = []
    class_counter = Counter()
    total_elements = 0

    for json_path in json_files:
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue

        labels_list = data.get("labels", [])
        boxes = data.get("contentBoxes", data.get("boxes", []))
        key_name = data.get("key_name", "")

        if not labels_list or not boxes:
            continue

        # Find corresponding screenshot
        img_path = None
        parent = json_path.parent
        for ext in [".webp", ".png", ".jpg", ".jpeg"]:
            candidate = parent / f"{json_path.stem}{ext}"
            if candidate.exists():
                img_path = candidate
                break
            # Try screenshot naming convention
            candidate = parent / f"{key_name}-screenshot{ext}"
            if candidate.exists():
                img_path = candidate
                break

        if img_path is None:
            continue

        # Determine image dimensions from viewport key
        img_w, img_h = _get_dimensions_from_key(key_name, img_path)
        if img_w <= 0 or img_h <= 0:
            continue

        yolo_annotations = []
        for label_set, box in zip(labels_list, boxes):
            if len(box) != 4:
                continue

            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1

            if w * h < MIN_AREA:
                continue

            # Use the first (primary) label
            primary_label = label_set[0] if isinstance(label_set, list) else label_set
            unified_name, class_id = map_webui_label(primary_label)

            if class_id is not None:
                cx = (x1 + x2) / 2.0 / img_w
                cy = (y1 + y2) / 2.0 / img_h
                nw = w / img_w
                nh = h / img_h

                # Clamp to [0, 1]
                cx = max(0.0, min(1.0, cx))
                cy = max(0.0, min(1.0, cy))
                nw = max(0.0, min(1.0, nw))
                nh = max(0.0, min(1.0, nh))

                yolo_annotations.append((class_id, cx, cy, nw, nh))
                class_counter[unified_name] += 1
                total_elements += 1

        if yolo_annotations:
            screen_data.append({
                "screen_id": f"{json_path.parent.name}_{json_path.stem}",
                "img_path": str(img_path),
                "annotations": yolo_annotations,
            })

    _save_yolo_dataset(screen_data, output_dir, class_counter, total_elements,
                       prefix="webui")


def convert_raw_webui(webui_dir, output_dir):
    """
    Convert raw WebUI crawl data to YOLO format.

    Expects directory structure:
    webui_dir/
        {crawl_id}/
            {device}-axtree.json.gz   (accessibility tree — flat node list)
            {device}-bb.json.gz       (bounding boxes keyed by DOM node ID)
            {device}-screenshot.webp  (screenshot)
    """
    crawl_dirs = [d for d in Path(webui_dir).iterdir() if d.is_dir()]
    print(f"Found {len(crawl_dirs)} crawl directories")

    screen_data = []
    class_counter = Counter()
    skipped_roles = Counter()
    total_elements = 0
    no_bb = 0

    for i, crawl_dir in enumerate(crawl_dirs):
        axtree_files = list(crawl_dir.glob("*-axtree.json.gz"))

        for axtree_path in axtree_files:
            device_prefix = axtree_path.name.replace("-axtree.json.gz", "")

            # Find screenshot
            img_path = None
            for ext in [".webp", ".png", ".jpg"]:
                candidate = crawl_dir / f"{device_prefix}-screenshot{ext}"
                if candidate.exists():
                    img_path = candidate
                    break

            if img_path is None:
                continue

            # Find bounding box file
            bb_path = crawl_dir / f"{device_prefix}-bb.json.gz"
            if not bb_path.exists():
                no_bb += 1
                continue

            # Load accessibility tree and bounding boxes
            try:
                axtree = load_gzip_json(axtree_path)
                bb_data = load_gzip_json(bb_path)
            except Exception:
                continue

            # Get image dimensions
            img_w, img_h = _get_dimensions_from_key(device_prefix, img_path)
            if img_w <= 0 or img_h <= 0:
                continue

            # Extract elements by joining axtree roles with bb coordinates
            elements = extract_elements_from_raw(axtree, bb_data)

            yolo_annotations = []
            for role_value, box in elements:
                x1, y1, x2, y2 = box
                w = x2 - x1
                h = y2 - y1

                if w * h < MIN_AREA:
                    continue

                unified_name, class_id = map_webui_label(role_value)
                if class_id is not None:
                    cx = (x1 + x2) / 2.0 / img_w
                    cy = (y1 + y2) / 2.0 / img_h
                    nw = w / img_w
                    nh = h / img_h

                    cx = max(0.0, min(1.0, cx))
                    cy = max(0.0, min(1.0, cy))
                    nw = max(0.0, min(1.0, nw))
                    nh = max(0.0, min(1.0, nh))

                    yolo_annotations.append((class_id, cx, cy, nw, nh))
                    class_counter[unified_name] += 1
                    total_elements += 1
                else:
                    skipped_roles[role_value] += 1

            if yolo_annotations:
                screen_id = f"{crawl_dir.name}_{device_prefix}"
                screen_data.append({
                    "screen_id": screen_id,
                    "img_path": str(img_path),
                    "annotations": yolo_annotations,
                })

        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{len(crawl_dirs)} crawl dirs...")

    if no_bb > 0:
        print(f"Skipped {no_bb} viewports (no bb.json.gz)")

    if skipped_roles:
        print(f"\nSkipped roles (not mapped):")
        for role, count in skipped_roles.most_common(15):
            print(f"  {role:20s}: {count:>6d}")

    _save_yolo_dataset(screen_data, output_dir, class_counter, total_elements,
                       prefix="webui")


def _get_dimensions_from_key(key_name, img_path):
    """Get image dimensions from viewport key or by reading the image."""
    # Try matching known viewport keys
    for vp_key, (w, h, scale) in VIEWPORT_SIZES.items():
        if vp_key in key_name:
            return w * scale, h * scale

    # Fallback: try to read image dimensions
    try:
        from PIL import Image
        with Image.open(img_path) as img:
            return img.size
    except Exception:
        return 0, 0


def _save_yolo_dataset(screen_data, output_dir, class_counter, total_elements,
                       prefix="webui"):
    """Save processed data as YOLO dataset with train/val/test splits."""
    print(f"\nValid screens with annotations: {len(screen_data)}")
    print(f"Total elements extracted: {total_elements}")

    print(f"\nClass distribution:")
    for cls in UNIFIED_CLASSES:
        count = class_counter.get(cls, 0)
        if count > 0:
            print(f"  {cls:15s}: {count:>6d}")

    # Split
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

            dst_img = os.path.join(img_dir, f"{prefix}_{screen_id}{ext}")
            dst_lbl = os.path.join(lbl_dir, f"{prefix}_{screen_id}.txt")

            if not os.path.exists(dst_img):
                shutil.copy2(src_img, dst_img)

            with open(dst_lbl, 'w') as f:
                for class_id, cx, cy, w, h in item["annotations"]:
                    f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

        print(f"{split_name}: {len(split_data)} images → {img_dir}")

    print(f"\nWebUI YOLO dataset saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare WebUI dataset for GUI detection (YOLO format)")

    parser.add_argument("--webui_dir", type=str, required=True,
                        help="Path to WebUI data directory")
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(os.path.dirname(__file__),
                                             "unified", "webui"),
                        help="Output directory")
    parser.add_argument("--raw", action="store_true",
                        help="Process raw crawl data (axtree.json.gz files)")

    args = parser.parse_args()

    if not os.path.isdir(args.webui_dir):
        print(f"ERROR: Directory not found: {args.webui_dir}")
        sys.exit(1)

    if args.raw:
        convert_raw_webui(args.webui_dir, args.output_dir)
    else:
        convert_processed_webui(args.webui_dir, args.output_dir)


if __name__ == "__main__":
    main()
