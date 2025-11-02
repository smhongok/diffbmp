#!/usr/bin/env python3
"""
Script to create FPS-dropped versions of image sequence datasets.
Keeps only 1 in every 4 images from each sequence (FPS is 4x lower).
"""

import os
import shutil
from pathlib import Path
from typing import List
import argparse


def get_sorted_image_files(directory: Path) -> List[Path]:
    """Get sorted list of image files in a directory."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = [f for f in directory.iterdir() 
                   if f.is_file() and f.suffix.lower() in image_extensions]
    return sorted(image_files)


def apply_fps_drop(src_dir: Path, dst_dir: Path, keep_pattern: int = 4):
    """
    Copy images from src_dir to dst_dir, keeping only 1 in every keep_pattern images.
    
    Args:
        src_dir: Source directory containing image sequence
        dst_dir: Destination directory for FPS-dropped sequence
        keep_pattern: Keep 1 image every N images (default: 4, meaning keep 1, drop 3)
    """
    # Create destination directory
    dst_dir.mkdir(parents=True, exist_ok=True)
    
    # Get sorted image files
    image_files = get_sorted_image_files(src_dir)
    
    if not image_files:
        print(f"  Warning: No images found in {src_dir}")
        return
    
    # Copy images, keeping only 1 in every keep_pattern
    copied_count = 0
    dropped_count = 0
    
    for idx, img_file in enumerate(image_files):
        # Keep only every 4th image (indices 0, 4, 8, 12, ...)
        if idx % keep_pattern == 0:
            # Copy the image
            dst_file = dst_dir / img_file.name
            shutil.copy2(img_file, dst_file)
            copied_count += 1
        else:
            dropped_count += 1
    
    print(f"  {src_dir.name}: Copied {copied_count} images, dropped {dropped_count} images")


def process_dataset(dataset_name: str, base_dir: Path):
    """
    Process a single dataset by creating FPS-dropped version.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'DAVIS', 'Sintel-final')
        base_dir: Base directory containing all image datasets
    """
    src_dataset_dir = base_dir / dataset_name
    dst_dataset_dir = base_dir / f"{dataset_name}_fps_drop"
    
    if not src_dataset_dir.exists():
        print(f"Warning: Source directory {src_dataset_dir} does not exist. Skipping.")
        return
    
    print(f"\nProcessing {dataset_name}...")
    print(f"  Source: {src_dataset_dir}")
    print(f"  Destination: {dst_dataset_dir}")
    
    # Get all subdirectories (sequence folders)
    subdirs = [d for d in src_dataset_dir.iterdir() if d.is_dir()]
    
    if not subdirs:
        print(f"  Warning: No subdirectories found in {src_dataset_dir}")
        return
    
    print(f"  Found {len(subdirs)} sequence folders")
    
    # Process each sequence folder
    for subdir in sorted(subdirs):
        src_seq_dir = subdir
        dst_seq_dir = dst_dataset_dir / subdir.name
        apply_fps_drop(src_seq_dir, dst_seq_dir)
    
    print(f"  ✓ Completed {dataset_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Create FPS-dropped versions of image sequence datasets"
    )
    parser.add_argument(
        '--base-dir',
        type=str,
        default='/data/insoo_chung/repo/xy_dynamics/circle_art/images',
        help='Base directory containing image datasets'
    )
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=['DAVIS', 'Sintel-final', 'Sintel-albedo', 'VIRAT'],
        help='List of dataset names to process'
    )
    
    args = parser.parse_args()
    base_dir = Path(args.base_dir)
    
    if not base_dir.exists():
        print(f"Error: Base directory {base_dir} does not exist.")
        return
    
    print("=" * 60)
    print("Creating FPS-Dropped Image Sequence Datasets")
    print("=" * 60)
    print(f"Drop pattern: Keep 1 image, drop 3 images (FPS is 4x lower)")
    print(f"Base directory: {base_dir}")
    print(f"Datasets to process: {', '.join(args.datasets)}")
    
    # Process each dataset
    for dataset_name in args.datasets:
        process_dataset(dataset_name, base_dir)
    
    print("\n" + "=" * 60)
    print("✓ All datasets processed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
