#!/usr/bin/env python3
"""
PSD Parallax Animation Generator

Creates parallax scrolling animations from PSD layers where:
- Front layers (lower indices) have shallow depth (move more with camera)
- Back layers (higher indices) have deeper depth (move less with camera)
- Camera movement creates realistic depth perception

Usage:
    python psd_parallax_mp4.py input.psd -o output.mp4
"""

import os
import sys
import argparse
import math
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
try:
    from psd_tools import PSDImage
except ImportError:
    print("Error: psd_tools is required. Install with: pip install psd-tools")
    sys.exit(1)

try:
    from moviepy import ImageClip, concatenate_videoclips
except ImportError as e:
    print(f"Error: moviepy is required. Install with: pip install moviepy")
    print(f"Import error: {e}")
    sys.exit(1)


def extract_psd_layers_with_depth(psd_path, layer_start=None, layer_end=None, verbose=False):
    """
    Extract layers from PSD file with depth information.
    
    Args:
        psd_path: Path to PSD file
        layer_start: Starting layer index (0-based, inclusive)
        layer_end: Ending layer index (0-based, inclusive)
        verbose: Print extraction details
        
    Returns:
        tuple: (layer_images, layer_positions, layer_depths, canvas_size)
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
        
        # Extract individual layer images
        layer_images = []
        layer_positions = []
        layer_depths = []
        canvas_size = (psd.width, psd.height)
        
        total_layers = len(selected_layers)
        
        for i, layer in enumerate(selected_layers):
            try:
                # Handle different layer types
                if hasattr(layer, 'compose'):
                    layer_image = layer.compose()
                elif hasattr(layer, 'topil'):
                    layer_image = layer.topil()
                else:
                    layer_image = layer.as_PIL()
                
                # Skip if layer_image is None
                if layer_image is None:
                    if verbose:
                        print(f"  Skipped layer {i}: {layer.name} - No image data")
                    continue
                
                # Convert to RGBA if needed
                if layer_image.mode != 'RGBA':
                    layer_image = layer_image.convert('RGBA')
                
                # Get layer position
                position = (layer.left, layer.top)
                
                # Calculate depth based on layer order
                # Front layers (index 0) = depth 0.1 (shallow, move most)
                # Back layers (index max) = depth 1.0 (deep, move least)
                depth = 0.1 + (i / max(1, total_layers - 1)) * 0.9
                
                layer_images.append(layer_image)
                layer_positions.append(position)
                layer_depths.append(depth)
                
                if verbose:
                    print(f"  Extracted layer {i}: {layer.name} (pos: {position[0]},{position[1]}, depth: {depth:.2f})")
                    
            except Exception as e:
                if verbose:
                    print(f"  Skipped layer {i}: {layer.name} - {e}")
                continue
        
        if verbose:
            print(f"Successfully extracted {len(layer_images)} layers")
            print(f"Canvas size: {canvas_size}")
        
        return layer_images, layer_positions, layer_depths, canvas_size
        
    except Exception as e:
        print(f"Error opening PSD file: {e}")
        return [], [], [], (0, 0)


def create_parallax_frame(layer_images, layer_positions, layer_depths, canvas_size, camera_offset):
    """
    Create a single frame with parallax effect.
    
    Args:
        layer_images: List of PIL Images for each layer
        layer_positions: List of (x, y) positions for each layer
        layer_depths: List of depth values (0.1=front, 1.0=back)
        canvas_size: (width, height) of the canvas
        camera_offset: (x, y) camera movement offset
        
    Returns:
        PIL Image of the composed frame
    """
    canvas = Image.new('RGB', canvas_size, (255, 255, 255))
    
    for layer_image, position, depth in zip(layer_images, layer_positions, layer_depths):
        # Calculate parallax offset based on depth
        # Shallow layers (low depth) move more with camera
        # Deep layers (high depth) move less with camera
        parallax_factor = 1.0 - depth  # 0.9 for front, 0.0 for back
        
        parallax_x = camera_offset[0] * parallax_factor
        parallax_y = camera_offset[1] * parallax_factor
        
        # Apply parallax offset to layer position
        adjusted_x = int(position[0] + parallax_x)
        adjusted_y = int(position[1] + parallax_y)
        
        # Paste layer onto canvas
        canvas.paste(layer_image, (adjusted_x, adjusted_y), layer_image)
    
    return canvas


def generate_camera_path(total_frames, movement_type='circular', amplitude=50, verbose=False):
    """
    Generate camera movement path for parallax effect.
    
    Args:
        total_frames: Number of frames in animation
        movement_type: Type of camera movement ('circular', 'horizontal', 'vertical', 'figure8')
        amplitude: Maximum movement distance in pixels
        verbose: Print path generation details
        
    Returns:
        List of (x, y) camera offset tuples
    """
    camera_path = []
    
    for frame_idx in range(total_frames):
        t = frame_idx / total_frames  # 0.0 to 1.0
        
        if movement_type == 'circular':
            # Circular camera movement
            angle = t * 2 * math.pi
            offset_x = amplitude * (math.cos(angle) - 1.0)
            offset_y = amplitude * math.sin(angle)
        
        elif movement_type == 'cardioid':
            angle = t * 2 * math.pi
            offset_x = amplitude * (1 - math.cos(angle)) * math.cos(angle)
            offset_y = amplitude * (1 - math.cos(angle)) * math.sin(angle)
            
        elif movement_type == 'horizontal':
            # Horizontal back-and-forth
            offset_x = amplitude * math.sin(t * 2 * math.pi)
            offset_y = 0
            
        elif movement_type == 'vertical':
            # Vertical back-and-forth
            offset_x = 0
            offset_y = amplitude * math.sin(t * 2 * math.pi)
            
        elif movement_type == 'figure8':
            # Figure-8 pattern
            angle = t * 4 * math.pi
            offset_x = amplitude * math.sin(angle)
            offset_y = amplitude * math.sin(angle * 0.5)
            
        elif movement_type == 'zoom':
            # Zoom in/out effect (simulated with scaling movement)
            zoom_factor = 1.0 + 0.3 * math.sin(t * 2 * math.pi)
            offset_x = amplitude * (1.0 - zoom_factor) * 0.5
            offset_y = amplitude * (1.0 - zoom_factor) * 0.5
            
        else:
            # Default: no movement
            offset_x = 0
            offset_y = 0
        
        camera_path.append((offset_x, offset_y))
    
    if verbose:
        print(f"Generated {movement_type} camera path with {total_frames} frames")
        print(f"Movement amplitude: {amplitude} pixels")
    
    return camera_path


def generate_parallax_sequence(layer_images, layer_positions, layer_depths, canvas_size, 
                              total_frames, movement_type='circular', amplitude=50, verbose=False):
    """
    Generate sequence of frames with parallax effect.
    
    Args:
        layer_images: List of PIL Images for each layer
        layer_positions: List of (x, y) positions for each layer
        layer_depths: List of depth values
        canvas_size: (width, height) of the canvas
        total_frames: Total number of frames to generate
        movement_type: Type of camera movement
        amplitude: Movement amplitude in pixels
        verbose: Print progress information
        
    Returns:
        List of PIL Images for each frame
    """
    frames = []
    
    # Generate camera movement path
    camera_path = generate_camera_path(total_frames, movement_type, amplitude, verbose)
    
    if verbose:
        print(f"Generating {total_frames} parallax frames...")
        print(f"Layers: {len(layer_images)}")
        print(f"Depth range: {min(layer_depths):.2f} to {max(layer_depths):.2f}")
    
    for frame_idx in range(total_frames):
        camera_offset = camera_path[frame_idx]
        
        # Create frame with parallax effect
        frame = create_parallax_frame(
            layer_images, 
            layer_positions, 
            layer_depths, 
            canvas_size, 
            camera_offset
        )
        frames.append(frame)
        
        if verbose and (frame_idx + 1) % 50 == 0:
            print(f"  Generated frame {frame_idx + 1}/{total_frames} (camera: {camera_offset[0]:.1f}, {camera_offset[1]:.1f})")
    
    return frames


def create_parallax_animation(frames, output_path, fps=24, verbose=False):
    """
    Create MP4 animation from parallax frames.
    
    Args:
        frames (list): List of PIL Images
        output_path (str): Output MP4 file path
        fps (int): Frames per second
        verbose (bool): Print progress information
    """
    if not frames:
        print("Error: No frames to process")
        return False
    
    try:
        clips = []
        
        # Convert each frame to a short clip
        for i, frame in enumerate(frames):
            # Convert PIL Image to numpy array
            img_array = np.array(frame)
            
            # Create ImageClip with duration of 1/fps seconds
            clip = ImageClip(img_array, duration=1.0/fps)
            clips.append(clip)
            
            if verbose and (i + 1) % 50 == 0:
                print(f"  Created clip {i + 1}/{len(frames)}")
        
        # Concatenate all clips
        if verbose:
            print(f"Writing video to: {output_path}")
            print(f"  Duration: {len(frames)/fps:.2f}s")
            print(f"  FPS: {fps}")
        
        final_clip = concatenate_videoclips(clips, method="compose")
        final_clip.write_videofile(output_path, fps=fps, logger=None)
        
        # Clean up clips
        final_clip.close()
        for clip in clips:
            clip.close()
        
        return True
        
    except Exception as e:
        print(f"Error creating animation: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Create parallax scrolling animation from PSD layers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic circular parallax
  python psd_parallax_mp4.py input.psd
  
  # Horizontal movement with custom amplitude
  python psd_parallax_mp4.py input.psd --movement horizontal --amplitude 100
  
  # Figure-8 pattern with longer duration
  python psd_parallax_mp4.py input.psd --movement figure8 --duration 10 --fps 30
  
  # Use specific layer range
  python psd_parallax_mp4.py input.psd --layer-start 2 --layer-end 8
        """
    )
    
    parser.add_argument('psd_file', help='Input PSD file path')
    
    parser.add_argument('-o', '--output', 
                       help='Output MP4 file path (default: input_name_parallax.mp4)')
    
    parser.add_argument('--duration', type=float, default=8.0,
                       help='Animation duration in seconds (default: 8.0)')
    
    parser.add_argument('--fps', type=int, default=24,
                       help='Frames per second (default: 24)')
    
    parser.add_argument('--movement', choices=['circular', 'horizontal', 'vertical', 'figure8', 'zoom', 'cardioid'],
                       default='cardioid', help='Camera movement pattern (default: cardioid)')
    
    parser.add_argument('--amplitude', type=float, default=50,
                       help='Movement amplitude in pixels (default: 50)')
    
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
        args.output = str(psd_path.with_suffix('').with_suffix('_parallax.mp4'))
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"Creating parallax animation:")
    print(f"  Input:  {args.psd_file}")
    print(f"  Output: {args.output}")
    print(f"  Duration: {args.duration}s")
    print(f"  FPS: {args.fps}")
    print(f"  Movement: {args.movement}")
    print(f"  Amplitude: {args.amplitude} pixels")
    
    if args.layer_start is not None or args.layer_end is not None:
        print(f"  Layer range: {args.layer_start} to {args.layer_end}")
    
    # Extract layer images with depth
    print("\nExtracting layers with depth information...")
    layer_images, layer_positions, layer_depths, canvas_size = extract_psd_layers_with_depth(
        args.psd_file, 
        args.layer_start, 
        args.layer_end, 
        args.verbose
    )
    
    if not layer_images:
        print("Error: No layers could be extracted")
        sys.exit(1)
    
    # Calculate total frames
    total_frames = int(args.duration * args.fps)
    
    # Generate parallax sequence
    print(f"\nGenerating parallax sequence with {len(layer_images)} layers...")
    frames = generate_parallax_sequence(
        layer_images, 
        layer_positions, 
        layer_depths,
        canvas_size,
        total_frames,
        args.movement,
        args.amplitude,
        args.verbose
    )
    
    # Create animation
    print(f"\nCreating animation with {len(frames)} frames...")
    success = create_parallax_animation(
        frames, 
        args.output, 
        fps=args.fps,
        verbose=args.verbose
    )
    
    if success:
        file_size = os.path.getsize(args.output) / (1024 * 1024)  # MB
        print(f"\n✅ Parallax animation created successfully: {args.output}")
        print(f"   File size: {file_size:.2f} MB")
    else:
        print(f"\n❌ Failed to create animation")
        sys.exit(1)


if __name__ == "__main__":
    main()
