"""
Frame PNG Exporter Utility

Exports frames from video files to individual PNG images.
"""

import cv2
import os
from pathlib import Path
from typing import Optional
import argparse


def export_video_frames(
    video_path: str,
    output_dir: str,
    frame_prefix: Optional[str] = None,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    frame_step: int = 1,
    verbose: bool = True
) -> int:
    """
    Export frames from a video file to PNG images.
    
    Args:
        video_path: Path to the input video file
        output_dir: Directory to save the exported frames
        frame_prefix: Prefix for frame filenames (default: video filename)
        start_frame: First frame to export (0-indexed)
        end_frame: Last frame to export (None = all frames)
        frame_step: Export every Nth frame (1 = all frames)
        verbose: Print progress information
        
    Returns:
        Number of frames exported
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if verbose:
        print(f"Video: {video_path}")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps:.2f}")
        print(f"  Total frames: {total_frames}")
    
    # Determine frame prefix
    if frame_prefix is None:
        frame_prefix = Path(video_path).stem
    
    # Determine end frame
    if end_frame is None:
        end_frame = total_frames
    else:
        end_frame = min(end_frame, total_frames)
    
    # Export frames
    frame_count = 0
    exported_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Check if we should export this frame
        if frame_count >= start_frame and frame_count < end_frame:
            if (frame_count - start_frame) % frame_step == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Save frame
                frame_filename = f"{frame_prefix}_{frame_count:06d}.png"
                frame_path = output_path / frame_filename
                cv2.imwrite(str(frame_path), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
                
                exported_count += 1
                
                if verbose and exported_count % 10 == 0:
                    print(f"  Exported {exported_count} frames...", end='\r')
        
        frame_count += 1
        
        if frame_count >= end_frame:
            break
    
    cap.release()
    
    if verbose:
        print(f"  Exported {exported_count} frames to {output_dir}")
    
    return exported_count


def export_video_directory(
    video_dir: str,
    output_base_dir: str,
    video_extensions: tuple = ('.mp4', '.avi', '.mov', '.mkv'),
    **kwargs
) -> dict:
    """
    Export frames from all videos in a directory.
    
    Args:
        video_dir: Directory containing video files
        output_base_dir: Base directory for output (subdirs created per video)
        video_extensions: Tuple of video file extensions to process
        **kwargs: Additional arguments passed to export_video_frames
        
    Returns:
        Dictionary mapping video paths to number of frames exported
    """
    video_path = Path(video_dir)
    output_base = Path(output_base_dir)
    
    results = {}
    
    # Find all video files
    video_files = []
    for ext in video_extensions:
        video_files.extend(video_path.glob(f"*{ext}"))
    
    video_files = sorted(video_files)
    
    if not video_files:
        print(f"No video files found in {video_dir}")
        return results
    
    print(f"Found {len(video_files)} video(s) to process\n")
    
    # Process each video
    for i, video_file in enumerate(video_files, 1):
        print(f"[{i}/{len(video_files)}] Processing: {video_file.name}")
        
        # Create output directory for this video
        output_dir = output_base / video_file.stem
        
        try:
            num_frames = export_video_frames(
                str(video_file),
                str(output_dir),
                **kwargs
            )
            results[str(video_file)] = num_frames
        except Exception as e:
            print(f"  Error processing {video_file.name}: {e}")
            results[str(video_file)] = 0
        
        print()
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Export video frames to PNG images"
    )
    parser.add_argument(
        "input",
        help="Input video file or directory containing videos"
    )
    parser.add_argument(
        "output",
        help="Output directory for exported frames"
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Prefix for frame filenames (default: video filename)"
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="First frame to export (default: 0)"
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="Last frame to export (default: all frames)"
    )
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="Export every Nth frame (default: 1)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output"
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Single video file
        export_video_frames(
            args.input,
            args.output,
            frame_prefix=args.prefix,
            start_frame=args.start,
            end_frame=args.end,
            frame_step=args.step,
            verbose=not args.quiet
        )
    elif input_path.is_dir():
        # Directory of videos
        export_video_directory(
            args.input,
            args.output,
            frame_prefix=args.prefix,
            start_frame=args.start,
            end_frame=args.end,
            frame_step=args.step,
            verbose=not args.quiet
        )
    else:
        print(f"Error: {args.input} is not a valid file or directory")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
