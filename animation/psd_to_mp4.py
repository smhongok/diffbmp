#!/usr/bin/env python3
"""
PSD to MP4 Animation Converter

This script converts PSD layers into an animated MP4 video.
Each layer becomes a frame in the animation sequence.

Usage:
    python psd_to_mp4.py input.psd -o output.mp4 --fps 24 --duration 1.0
    python psd_to_mp4.py input.psd --layer-start 2 --layer-end 10 --fps 30
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image

try:
    from psd_tools import PSDImage
except ImportError:
    print("Error: psd_tools is required. Install with: pip install psd-tools")
    sys.exit(1)

try:
    from moviepy import ImageClip, CompositeVideoClip, concatenate_videoclips
except ImportError as e:
    print(f"Error: moviepy is required. Install with: pip install moviepy")
    print(f"Import error details: {e}")
    sys.exit(1)


def extract_layer_images(psd_path, layer_start=None, layer_end=None, verbose=False):
    """
    Extract layer images from PSD file.
    
    Args:
        psd_path (str): Path to PSD file
        layer_start (int): Starting layer index (0-based, inclusive)
        layer_end (int): Ending layer index (0-based, inclusive)
        verbose (bool): Print layer information
        
    Returns:
        list: List of PIL Images from layers
    """
    try:
        psd = PSDImage.open(psd_path)
        layers = list(psd)
        
        if verbose:
            print(f"Total layers found: {len(layers)}")
            for i, layer in enumerate(layers):
                print(f"  Layer {i}: {layer.name} (visible: {layer.visible})")
        
        # Apply layer range filtering
        if layer_start is not None:
            layer_start = max(0, layer_start)
        else:
            layer_start = 0
            
        if layer_end is not None:
            layer_end = min(len(layers) - 1, layer_end)
        else:
            layer_end = len(layers) - 1
            
        selected_layers = layers[layer_start:layer_end + 1]
        
        if verbose:
            print(f"Using layers {layer_start} to {layer_end} ({len(selected_layers)} layers)")
        
        # Extract layer images with cumulative compositing
        layer_images = []
        
        # Start with white canvas
        canvas_width = psd.width
        canvas_height = psd.height
        cumulative_canvas = Image.new('RGBA', (canvas_width, canvas_height), (255, 255, 255, 255))
        
        for i, layer in enumerate(selected_layers):
            try:
                # Handle different layer types
                if hasattr(layer, 'compose'):
                    # For group layers and some other types
                    layer_image = layer.compose()
                elif hasattr(layer, 'topil'):
                    # For pixel layers - use topil() method
                    layer_image = layer.topil()
                else:
                    # Fallback: try to get the layer as PIL image directly
                    layer_image = layer.as_PIL()
                
                # Skip if layer_image is None
                if layer_image is None:
                    if verbose:
                        print(f"  Skipping layer {layer_start + i}: {layer.name} (empty or invisible)")
                    continue
                
                # Get layer positioning information
                layer_left = getattr(layer, 'left', 0)
                layer_top = getattr(layer, 'top', 0)
                layer_width = getattr(layer, 'width', layer_image.width if layer_image else 0)
                layer_height = getattr(layer, 'height', layer_image.height if layer_image else 0)
                
                # Convert layer image to RGBA if necessary
                if layer_image.mode != 'RGBA':
                    layer_image = layer_image.convert('RGBA')
                
                # Paste the layer image onto the cumulative canvas with alpha blending
                cumulative_canvas.paste(layer_image, (layer_left, layer_top), layer_image)
                
                # Save a copy of the current cumulative state for this frame
                frame_image = cumulative_canvas.copy()
                layer_images.append(frame_image)
                
                if verbose:
                    print(f"  Composited layer {layer_start + i}: {layer.name} (pos: {layer_left},{layer_top}, size: {layer_width}x{layer_height})")
                    
            except Exception as e:
                if verbose:
                    print(f"Warning: Failed to extract layer {layer_start + i}: {e}")
                continue
        
        return layer_images
        
    except Exception as e:
        print(f"Error opening PSD file: {e}")
        return []


def create_animation(layer_images, output_path, fps=24, frames_per_layer=1, verbose=False):
    """
    Create MP4 animation from layer images.
    
    Args:
        layer_images (list): List of PIL Images
        output_path (str): Output MP4 file path
        fps (int): Frames per second
        frames_per_layer (int): Number of frames each layer is shown
        verbose (bool): Print progress information
    """
    if not layer_images:
        print("Error: No layer images to process")
        return False
    
    try:
        clips = []
        
        # Each layer shown sequentially
        for i, layer_image in enumerate(layer_images):
            # Convert PIL Image to numpy array
            img_array = np.array(layer_image)
            
            # Calculate duration in seconds from frame count
            duration_seconds = frames_per_layer / fps
            
            # Create ImageClip from numpy array
            clip = ImageClip(img_array, duration=duration_seconds)
            clips.append(clip)
            
            if verbose:
                print(f"  Created clip {i+1}/{len(layer_images)}")
        
        # Concatenate clips sequentially
        final_video = concatenate_videoclips(clips, method="compose")
        
        # Write video file
        if verbose:
            print(f"Writing video to: {output_path}")
            print(f"  Duration: {final_video.duration:.2f}s")
            print(f"  FPS: {fps}")
        
        final_video.write_videofile(
            output_path, 
            fps=fps,
            codec='libx264',
            audio=False
        )
        
        # Clean up
        final_video.close()
        for clip in clips:
            clip.close()
        
        return True
        
    except Exception as e:
        print(f"Error creating animation: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Convert PSD layers to MP4 animation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python psd_to_mp4.py input.psd
  
  # Custom output and settings
  python psd_to_mp4.py input.psd -o animation.mp4 --fps 30 --duration 0.5
  
  # Use specific layer range
  python psd_to_mp4.py input.psd --layer-start 2 --layer-end 8
  
  # Composite mode (overlay layers)
  python psd_to_mp4.py input.psd --transition composite --duration 3.0
        """
    )
    
    parser.add_argument('psd_file', help='Input PSD file path')
    
    parser.add_argument('-o', '--output', 
                       help='Output MP4 file path (default: input_name.mp4)')
    
    parser.add_argument('--fps', type=int, default=24,
                       help='Frames per second (default: 24)')
    
    parser.add_argument('--frames', type=int, default=1,
                       help='Number of frames per layer (default: 1)')
    
    parser.add_argument('--layer-start', type=int,
                       help='Starting layer index (0-based, inclusive)')
    
    parser.add_argument('--layer-end', type=int,
                       help='Ending layer index (0-based, inclusive)')
    
    
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.psd_file):
        print(f"Error: PSD file not found: {args.psd_file}")
        sys.exit(1)
    
    # Generate output filename if not provided
    if args.output is None:
        psd_path = Path(args.psd_file)
        args.output = str(psd_path.with_suffix('.mp4'))
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"Converting PSD to MP4:")
    print(f"  Input:  {args.psd_file}")
    print(f"  Output: {args.output}")
    print(f"  FPS:    {args.fps}")
    print(f"  Frames per layer: {args.frames}")
    
    if args.layer_start is not None or args.layer_end is not None:
        print(f"  Layer range: {args.layer_start} to {args.layer_end}")
    
    # Extract layer images
    print("\nExtracting layers...")
    layer_images = extract_layer_images(
        args.psd_file, 
        args.layer_start, 
        args.layer_end, 
        args.verbose
    )
    
    if not layer_images:
        print("Error: No layers could be extracted")
        sys.exit(1)
    
    # Create animation
    print(f"\nCreating animation with {len(layer_images)} layers...")
    success = create_animation(
        layer_images, 
        args.output, 
        fps=args.fps,
        frames_per_layer=args.frames,
        verbose=args.verbose
    )
    
    if success:
        print(f"\n✅ Animation created successfully: {args.output}")
        
        # Show file info
        if os.path.exists(args.output):
            file_size = os.path.getsize(args.output) / (1024 * 1024)  # MB
            print(f"   File size: {file_size:.2f} MB")
    else:
        print("\n❌ Failed to create animation")
        sys.exit(1)


if __name__ == "__main__":
    main()
