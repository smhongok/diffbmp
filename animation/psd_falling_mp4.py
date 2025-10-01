#!/usr/bin/env python3
"""
PSD Falling Animation Generator

Creates gravity-based falling animations from PSD layers where:
- Layers start in their original positions
- At triggered moments, layers begin falling due to gravity
- Each layer falls with realistic physics (gravity acceleration)
- Layers can fall from top-to-bottom or triggered by position/time

Usage:
    python psd_falling_mp4.py input.psd -o output.mp4
"""

import argparse
import os
import sys
import math
import random
from pathlib import Path
import numpy as np
from PIL import Image

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


def extract_psd_layers(psd_path, layer_start=None, layer_end=None, verbose=False):
    """
    Extract layers from PSD file.
    
    Args:
        psd_path (str): Path to PSD file
        layer_start (int): Starting layer index (0-based, inclusive)
        layer_end (int): Ending layer index (0-based, inclusive)
        verbose (bool): Print layer information
        
    Returns:
        tuple: (list of PIL Images, list of (x,y) positions, canvas_size)
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
        canvas_size = (psd.width, psd.height)
        
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
                
                layer_images.append(layer_image)
                layer_positions.append(position)
                
                if verbose:
                    print(f"  Extracted layer {i}: {layer.name} (pos: {position[0]},{position[1]})")
                    
            except Exception as e:
                if verbose:
                    print(f"  Skipped layer {i}: {layer.name} - {e}")
                continue
        
        if verbose:
            print(f"Successfully extracted {len(layer_images)} layers")
            print(f"Canvas size: {canvas_size}")
        
        return layer_images, layer_positions, canvas_size
        
    except Exception as e:
        print(f"Error opening PSD file: {e}")
        return [], [], (0, 0)


def calculate_optimal_duration(num_layers, fps=24):
    """Calculate optimal animation duration based on number of layers."""
    # Base duration: 8 seconds for 1000 layers
    base_duration = 8.0
    base_layers = 1000
    
    # Scale duration with layer count (with some limits)
    duration = base_duration * (num_layers / base_layers)
    duration = max(5.0, min(30.0, duration))  # Between 5-30 seconds
    
    return duration


def create_falling_frame(layer_images, layer_positions, layer_states, canvas_size):
    """
    Create a single frame with falling layers.
    
    Args:
        layer_images: List of PIL Images for each layer
        layer_positions: List of current (x, y) positions for each layer
        layer_states: List of layer states (0=static, 1=falling)
        canvas_size: (width, height) of the canvas
        
    Returns:
        PIL Image of the composed frame
    """
    canvas = Image.new('RGB', canvas_size, (255, 255, 255))
    
    for layer_image, position, state in zip(layer_images, layer_positions, layer_states):
        if state == 0:  # Static layer - don't render (already fallen off screen)
            continue
            
        # Paste layer onto canvas at current position
        x, y = int(position[0]), int(position[1])
        
        # Only paste if layer is still visible on screen
        if y < canvas_size[1] + layer_image.size[1]:  # Allow some margin for partially visible layers
            canvas.paste(layer_image, (x, y), layer_image)
    
    return canvas


def generate_falling_sequence(layer_images, layer_positions, canvas_size, total_frames, 
                             fall_pattern='top_to_bottom', gravity=9.8, fps=24, verbose=False):
    """
    Generate sequence of frames with falling animation.
    
    Args:
        layer_images: List of PIL Images for each layer
        layer_positions: List of original (x, y) positions for each layer
        canvas_size: (width, height) of the canvas
        total_frames: Total number of frames to generate
        fall_pattern: How layers start falling ('top_to_bottom', 'random', 'wave')
        gravity: Gravity acceleration (pixels per second squared)
        fps: Frames per second
        verbose: Print progress information
        
    Returns:
        List of PIL Images for each frame
    """
    frames = []
    num_layers = len(layer_images)
    
    # Initialize layer physics
    layer_states = [1] * num_layers  # 1 = visible/falling, 0 = fallen off screen
    current_positions = [list(pos) for pos in layer_positions]  # Convert to mutable lists
    layer_velocities = [0.0] * num_layers  # Initial y-velocity for each layer
    layer_fall_start_frame = [-1] * num_layers  # Frame when each layer starts falling
    
    # Calculate when each layer should start falling based on pattern
    if fall_pattern == 'top_to_bottom':
        # Sort layers by y-position (top first)
        layer_indices_by_y = sorted(range(num_layers), key=lambda i: layer_positions[i][1])
        
        # Stagger fall start times
        fall_duration = int(total_frames * 0.6)  # 60% of animation for triggering falls
        for i, layer_idx in enumerate(layer_indices_by_y):
            fall_start = int((i / num_layers) * fall_duration)
            layer_fall_start_frame[layer_idx] = fall_start
            
    elif fall_pattern == 'random':
        # Random fall times
        fall_duration = int(total_frames * 0.7)  # 70% of animation for triggering falls
        for i in range(num_layers):
            layer_fall_start_frame[i] = random.randint(0, fall_duration)
            
    elif fall_pattern == 'wave':
        # Wave pattern from left to right
        layer_indices_by_x = sorted(range(num_layers), key=lambda i: layer_positions[i][0])
        
        fall_duration = int(total_frames * 0.5)  # 50% of animation for triggering falls
        for i, layer_idx in enumerate(layer_indices_by_x):
            fall_start = int((i / num_layers) * fall_duration)
            layer_fall_start_frame[layer_idx] = fall_start
    
    # Physics constants
    gravity_per_frame = gravity * (fps * fps) / (fps * fps)  # Convert to pixels per frame^2
    
    if verbose:
        print(f"Generating falling sequence with {num_layers} layers...")
        print(f"Fall pattern: {fall_pattern}")
        print(f"Gravity: {gravity} pixels/s²")
        print(f"Total frames: {total_frames}")
    
    for frame_idx in range(total_frames):
        # Update physics for each layer
        for layer_idx in range(num_layers):
            if layer_states[layer_idx] == 0:  # Already fallen off screen
                continue
                
            # Check if this layer should start falling
            if (layer_fall_start_frame[layer_idx] >= 0 and 
                frame_idx >= layer_fall_start_frame[layer_idx]):
                
                # Apply gravity to velocity
                layer_velocities[layer_idx] += gravity_per_frame
                
                # Update y position based on velocity
                current_positions[layer_idx][1] += layer_velocities[layer_idx]
                
                # Check if layer has fallen off screen
                if current_positions[layer_idx][1] > canvas_size[1] + 100:  # 100px margin
                    layer_states[layer_idx] = 0  # Mark as fallen off screen
        
        # Create frame with current positions
        frame = create_falling_frame(
            layer_images, 
            current_positions, 
            layer_states, 
            canvas_size
        )
        frames.append(frame)
        
        if verbose and (frame_idx + 1) % 50 == 0:
            falling_count = sum(1 for i in range(num_layers) 
                              if layer_fall_start_frame[i] >= 0 and 
                                 frame_idx >= layer_fall_start_frame[i] and 
                                 layer_states[i] == 1)
            fallen_count = sum(1 for s in layer_states if s == 0)
            static_count = sum(1 for i in range(num_layers) 
                             if layer_fall_start_frame[i] < 0 or 
                                frame_idx < layer_fall_start_frame[i])
            
            print(f"  Frame {frame_idx + 1}/{total_frames}: "
                  f"Static={static_count}, Falling={falling_count}, Fallen={fallen_count}")
    
    return frames


def create_falling_animation(frames, output_path, fps=24, verbose=False):
    """
    Create MP4 animation from falling frames.
    
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
        description="Create falling animation from PSD layers with gravity physics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic top-to-bottom falling
  python psd_falling_mp4.py input.psd
  
  # Random falling pattern with custom gravity
  python psd_falling_mp4.py input.psd --pattern random --gravity 15
  
  # Wave pattern with longer duration
  python psd_falling_mp4.py input.psd --pattern wave --duration 12 --fps 30
  
  # Use specific layer range
  python psd_falling_mp4.py input.psd --layer-start 10 --layer-end 50
        """
    )
    
    parser.add_argument('psd_file', help='Input PSD file path')
    
    parser.add_argument('-o', '--output', 
                       help='Output MP4 file path (default: input_name_falling.mp4)')
    
    parser.add_argument('--duration', type=float,
                       help='Animation duration in seconds (auto-calculated if not provided)')
    
    parser.add_argument('--fps', type=int, default=24,
                       help='Frames per second (default: 24)')
    
    parser.add_argument('--pattern', choices=['top_to_bottom', 'random', 'wave'],
                       default='top_to_bottom', help='Falling pattern (default: top_to_bottom)')
    
    parser.add_argument('--gravity', type=float, default=9.8,
                       help='Gravity acceleration in pixels/s² (default: 9.8)')
    
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
        args.output = str(psd_path.with_suffix('').with_suffix('_falling.mp4'))
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"Creating falling animation:")
    print(f"  Input:  {args.psd_file}")
    print(f"  Output: {args.output}")
    print(f"  FPS: {args.fps}")
    print(f"  Pattern: {args.pattern}")
    print(f"  Gravity: {args.gravity} pixels/s²")
    
    if args.layer_start is not None or args.layer_end is not None:
        print(f"  Layer range: {args.layer_start} to {args.layer_end}")
    
    # Extract layer images
    print("\nExtracting layers...")
    layer_images, layer_positions, canvas_size = extract_psd_layers(
        args.psd_file, 
        args.layer_start, 
        args.layer_end, 
        args.verbose
    )
    
    if not layer_images:
        print("Error: No layers could be extracted")
        sys.exit(1)
    
    # Auto-calculate duration if not provided
    if args.duration is None:
        args.duration = calculate_optimal_duration(len(layer_images), args.fps)
        print(f"  Auto-calculated duration: {args.duration:.1f}s (based on {len(layer_images)} layers)")
    else:
        print(f"  Duration: {args.duration}s")
    
    # Calculate total frames
    total_frames = int(args.duration * args.fps)
    
    # Generate falling sequence
    print(f"\nGenerating falling sequence with {len(layer_images)} layers...")
    frames = generate_falling_sequence(
        layer_images, 
        layer_positions, 
        canvas_size,
        total_frames,
        args.pattern,
        args.gravity,
        args.fps,
        args.verbose
    )
    
    # Create animation
    print(f"\nCreating animation with {len(frames)} frames...")
    success = create_falling_animation(
        frames, 
        args.output, 
        fps=args.fps,
        verbose=args.verbose
    )
    
    if success:
        file_size = os.path.getsize(args.output) / (1024 * 1024)  # MB
        print(f"\n✅ Falling animation created successfully: {args.output}")
        print(f"   File size: {file_size:.2f} MB")
    else:
        print(f"\n❌ Failed to create animation")
        sys.exit(1)


if __name__ == "__main__":
    main()
