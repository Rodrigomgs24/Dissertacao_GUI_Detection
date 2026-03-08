"""
Download and extract the WebUI-7k-balanced dataset.

Uses gdown to download from Google Drive and 7z to extract multi-part zips.
"""

import os
import sys
import glob
import shutil
import subprocess

import gdown

# Google Drive folder URL for webui-7k-balanced
DATASET_URL = (
    "https://drive.google.com/drive/folders/"
    "1F8W7OoMnpFGFHMK8m01r8zXb5765AB-N?usp=share_link"
)

# Find 7z executable
SEVEN_ZIP_PATHS = [
    "7z",
    "7z.exe",
    r"C:\Program Files\7-Zip\7z.exe",
    r"C:\Program Files (x86)\7-Zip\7z.exe",
    r"C:\Program Files\NVIDIA Corporation\NVIDIA App\7z.exe",
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TMP_DIR = os.path.join(BASE_DIR, "webui_tmp")
OUTPUT_DIR = os.path.join(BASE_DIR, "webui_raw")


def find_7z():
    """Find a working 7z executable."""
    for path in SEVEN_ZIP_PATHS:
        try:
            result = subprocess.run(
                [path, "--help"],
                capture_output=True, timeout=5
            )
            if result.returncode == 0:
                print(f"Found 7z: {path}")
                return path
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return None


def main():
    seven_zip = find_7z()
    if not seven_zip:
        print("ERROR: 7z not found. Install 7-Zip and add to PATH.")
        print("Checked:", SEVEN_ZIP_PATHS)
        sys.exit(1)

    os.makedirs(TMP_DIR, exist_ok=True)

    # Step 1: Download from Google Drive
    download_dir = os.path.join(TMP_DIR, "webui-7k-balanced")
    if not os.path.exists(download_dir):
        print("Downloading webui-7k-balanced from Google Drive...")
        gdown.download_folder(
            DATASET_URL,
            output=download_dir,
            use_cookies=False
        )
    else:
        print(f"Already downloaded: {download_dir}")

    # Step 2: Find the multi-part zip and split JSON
    zip_parts = glob.glob(os.path.join(download_dir, "*.zip.001"))
    json_files = glob.glob(os.path.join(download_dir, "*.json"))

    if not zip_parts:
        print("ERROR: No .zip.001 file found in download.")
        print(f"Contents: {os.listdir(download_dir)}")
        sys.exit(1)

    zip_file = zip_parts[0]
    print(f"\nFound archive: {zip_file}")

    # Copy split JSON to data dir
    for jf in json_files:
        dst = os.path.join(BASE_DIR, os.path.basename(jf))
        if not os.path.exists(dst):
            shutil.copy2(jf, dst)
            print(f"Copied split file: {os.path.basename(jf)}")

    # Step 3: Extract
    extract_dir = os.path.join(TMP_DIR, "extract")
    os.makedirs(extract_dir, exist_ok=True)

    print(f"\nExtracting with 7z...")
    result = subprocess.run(
        [seven_zip, "x", zip_file, f"-o{extract_dir}", "-y"],
        capture_output=False
    )
    if result.returncode != 0:
        print("ERROR: 7z extraction failed.")
        sys.exit(1)

    # Step 4: Move crawl folders to output
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    crawl_folders = glob.glob(os.path.join(extract_dir, "*", "*"))
    if not crawl_folders:
        crawl_folders = glob.glob(os.path.join(extract_dir, "*"))

    moved = 0
    for folder in crawl_folders:
        if os.path.isdir(folder):
            dst = os.path.join(OUTPUT_DIR, os.path.basename(folder))
            if not os.path.exists(dst):
                shutil.move(folder, dst)
                moved += 1

    print(f"\nMoved {moved} crawl folders to: {OUTPUT_DIR}")

    # Cleanup
    print("Cleaning up temp files...")
    shutil.rmtree(TMP_DIR, ignore_errors=True)

    # Report
    n_dirs = len([d for d in os.listdir(OUTPUT_DIR)
                  if os.path.isdir(os.path.join(OUTPUT_DIR, d))])
    print(f"\nDone! {n_dirs} crawl directories in {OUTPUT_DIR}")
    print(f"\nNext steps:")
    print(f"  python prepare_webui.py --raw --webui_dir {OUTPUT_DIR}")
    print(f"  python merge_datasets.py --rico_dir ./unified/rico --webui_dir ./unified/webui")


if __name__ == "__main__":
    main()
